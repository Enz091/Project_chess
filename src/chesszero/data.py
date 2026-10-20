"""Préparation des données Lichess : PGN en flux -> shards (position, coup joué, résultat).

Chaîne de traitement (voir `extract`) :
  1. le processus principal lit le PGN en flux (fichier local, ou HTTP + zstd à la volée, donc
     sans jamais stocker la base mensuelle de ~30 Go) et découpe les parties ;
  2. il filtre sur les en-têtes seuls (Elo, cadence, fin de partie...), ce qui est très bon marché ;
  3. il décide train / validation PAR PARTIE (hash de l'identifiant Lichess) ;
  4. un pool de workers rejoue les parties gardées et produit, pour chaque position,
     le format compact de l'étape 1, l'indice du coup joué et le résultat vu du camp au trait ;
  5. les résultats, reçus dans l'ordre, sont écrits en shards `.npz` de taille fixe.

Contenu d'un shard (tableaux alignés, une ligne par position) :
  boards  (N, 67) uint8   position compacte (`encoding.pack_board`)
  moves   (N,)    int16   indice du coup joué (`encoding.move_to_index`), dans [0, 4672)
  results (N,)    int8    +1 / 0 / -1 : résultat final du point de vue du camp AU TRAIT
  games   (N,)    int32   numéro de partie (sert à vérifier qu'aucune partie n'est dans train ET val)
  value_valid (N,) uint8  1 si le résultat est fiable comme cible de valeur (Termination « Normal »),
                          0 si la partie s'est finie au drapeau (« Time forfeit ») : le résultat dit alors
                          qui a manqué de temps, pas qui était mieux. Le coup joué reste une bonne cible
                          de politique ; la perte de valeur ignore ces positions (voir `model.policy_value_loss`).
"""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import re
import time
import tomllib
import urllib.request
import zlib
from collections import Counter, deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import numpy as np
import zstandard
from tqdm import tqdm

from .encoding import PACKED_SIZE, move_to_index, pack_board

SPLITS = ("train", "val")
GAMES_PER_SOURCE = 100_000_000  # intervalle de numéros de partie réservé à chaque source (int32)
# Résultat de l'en-tête -> score du point de vue des Blancs
_RESULT_WHITE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
_HEADER_RE = re.compile(r'\[(\w+) "(.*)"\]')


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class FilterConfig:
    """Critères de sélection des parties (valeurs par défaut = celles demandées pour l'étape 3)."""

    min_elo: int = 2000  # Elo minimum des DEUX joueurs
    min_plies: int = 20  # parties plus courtes (en demi-coups) écartées
    min_estimated_seconds: int = 180  # base + 40 * incrément ; < 180 s = bullet chez Lichess
    allowed_terminations: tuple[str, ...] = ("Normal", "Time forfeit")  # exclut Abandoned, etc.
    exclude_bots: bool = True  # écarte les parties où un joueur a le titre BOT (moteurs)
    val_fraction: float = 0.02  # part des PARTIES réservée à la validation

    @classmethod
    def from_toml(cls, path: str | Path) -> "FilterConfig":
        with open(path, "rb") as f:
            section = tomllib.load(f).get("filter", {})
        known = {f.name for f in fields(cls)}
        unknown = set(section) - known
        if unknown:
            raise ValueError(f"clés inconnues dans [filter] : {sorted(unknown)}")
        if "allowed_terminations" in section:
            section["allowed_terminations"] = tuple(section["allowed_terminations"])
        return cls(**section)


# --------------------------------------------------------------------------- lecture en flux


class _CountingReader(io.RawIOBase):
    """Enveloppe un flux binaire et compte les octets lus (taille réellement téléchargée)."""

    def __init__(self, raw):
        self._raw = raw
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        n = self._raw.readinto(buffer)
        self.bytes_read += n or 0
        return n

    def close(self) -> None:
        self._raw.close()
        super().close()


def open_pgn(source: str) -> tuple[io.TextIOBase, _CountingReader]:
    """Ouvre un PGN (chemin local ou URL http(s), brut ou `.zst`) en flux de texte.

    Retourne le flux de texte et le compteur d'octets (compressés) lus depuis la source.
    """
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(source, headers={"User-Agent": "chesszero-data/0.1"})
        raw = urllib.request.urlopen(request, timeout=60)
    else:
        raw = open(source, "rb")
    counter = _CountingReader(raw)
    binary: io.RawIOBase = counter
    if source.endswith(".zst"):
        # Lichess compresse avec une grande fenêtre : il faut autoriser jusqu'à 2 Gio.
        binary = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(counter)
    text = io.TextIOWrapper(
        io.BufferedReader(binary, 1 << 20), encoding="utf-8", errors="replace"
    )
    return text, counter


def iter_game_texts(stream) -> Iterator[str]:
    """Découpe un flux PGN en parties (texte brut). Une partie commence par `[Event `."""
    buf: list[str] = []
    for line in stream:
        if line.startswith("[Event ") and buf:
            yield "".join(buf)
            buf = []
        buf.append(line)
    if buf:
        yield "".join(buf)


def parse_headers(text: str) -> dict[str, str]:
    """En-têtes `[Clé "valeur"]` d'une partie (s'arrête à la première ligne de coups)."""
    headers = {}
    for line in text.split("\n"):
        if line.startswith("["):
            m = _HEADER_RE.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
        elif line.strip():  # première ligne de coups
            break
    return headers


# --------------------------------------------------------------------------- filtrage


def estimated_seconds(time_control: str) -> int | None:
    """Durée estimée d'une partie (base + 40 * incrément), None si cadence inconnue (« - »)."""
    try:
        base, inc = time_control.split("+")
        return int(base) + 40 * int(inc)
    except ValueError:
        return None


def reject_reason(headers: dict[str, str], cfg: FilterConfig) -> str | None:
    """Raison du rejet d'après les seuls en-têtes, ou None si la partie est gardée."""
    if headers.get("Result") not in _RESULT_WHITE:
        return "result"
    try:
        elo = min(int(headers["WhiteElo"]), int(headers["BlackElo"]))
    except (KeyError, ValueError):  # absent ou « ? »
        return "elo"
    if elo < cfg.min_elo:
        return "elo"
    if headers.get("Termination") not in cfg.allowed_terminations:
        return "termination"
    if headers.get("Variant", "Standard") != "Standard" or "FEN" in headers or "SetUp" in headers:
        return "variant"
    seconds = estimated_seconds(headers.get("TimeControl", "-"))
    if seconds is None:
        return "time_control"
    if seconds < cfg.min_estimated_seconds:
        return "bullet"
    # En dernier : « bot » ne compte que les parties qui auraient été gardées sans ce filtre.
    if cfg.exclude_bots and "BOT" in (headers.get("WhiteTitle"), headers.get("BlackTitle")):
        return "bot"
    return None


def split_of(headers: dict[str, str], game_no: int, val_fraction: float) -> str:
    """« train » ou « val », fixé par un hash de l'identifiant de la partie (donc stable)."""
    key = headers.get("Site") or f"game-{game_no}"
    return "val" if zlib.crc32(key.encode()) % 10000 < val_fraction * 10000 else "train"


# --------------------------------------------------------------------------- conversion


def game_to_arrays(text: str, game_no: int, min_plies: int) -> tuple | str:
    """Rejoue une partie. Retourne (boards, moves, results, games, value_valid) ou une raison de rejet."""
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None or game.errors:  # coup illégal ou illisible : on jette toute la partie
        return "parse_error"
    moves = list(game.mainline_moves())
    if len(moves) < min_plies:
        return "too_short"
    white_score = _RESULT_WHITE[game.headers["Result"]]

    n = len(moves)
    boards = np.empty((n, PACKED_SIZE), dtype=np.uint8)
    indices = np.empty(n, dtype=np.int16)
    results = np.empty(n, dtype=np.int8)
    board = game.board()
    for i, move in enumerate(moves):
        boards[i] = pack_board(board)
        indices[i] = move_to_index(move, board)
        results[i] = white_score if board.turn == chess.WHITE else -white_score
        board.push(move)
    value_valid = np.full(n, game.headers.get("Termination") == "Normal", dtype=np.uint8)
    return boards, indices, results, np.full(n, game_no, dtype=np.int32), value_valid


def process_chunk(args: tuple[list[tuple[int, str, str]], int]):
    """Travail d'un worker : liste de (n° de partie, split, texte) -> tableaux par split."""
    items, min_plies = args
    parts: dict[str, list[tuple]] = {s: [] for s in SPLITS}
    stats: Counter = Counter()
    for game_no, split, text in items:
        out = game_to_arrays(text, game_no, min_plies)
        if isinstance(out, str):
            stats[out] += 1
        else:
            parts[split].append(out)
            stats["kept"] += 1
            termination = parse_headers(text).get("Termination", "?")
            stats[f"games:{termination}"] += 1
            stats[f"positions:{termination}"] += len(out[0])
    arrays = {}
    for split, lst in parts.items():
        if lst:
            arrays[split] = tuple(np.concatenate(col) for col in zip(*lst))
    return arrays, stats


# --------------------------------------------------------------------------- écriture / lecture


def _write_shard(path: Path, boards, moves, results, games, value_valid) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, boards=boards, moves=moves, results=results, games=games, value_valid=value_valid)
    os.replace(tmp, path)  # atomique, comme pour les checkpoints


class _ShardWriter:
    """Accumule les positions d'un split et écrit un shard à chaque `shard_size` positions."""

    def __init__(self, out_dir: Path, split: str, shard_size: int):
        self.out_dir, self.split, self.shard_size = out_dir, split, shard_size
        self.parts: list[tuple] = []
        self.count = 0
        self.shards: list[dict] = []

    def add(self, arrays: tuple) -> None:
        self.parts.append(arrays)
        self.count += len(arrays[0])
        while self.count >= self.shard_size:
            self._flush(self.shard_size)

    def close(self) -> None:
        if self.count:
            self._flush(self.count)

    def _flush(self, n: int) -> None:
        cols = [np.concatenate(col) for col in zip(*self.parts)]
        name = f"{self.split}_{len(self.shards):05d}.npz"
        _write_shard(self.out_dir / name, *(c[:n] for c in cols))
        self.shards.append({"file": name, "positions": n})
        rest = [c[n:] for c in cols]
        self.parts = [tuple(rest)] if len(rest[0]) else []
        self.count = len(rest[0])


def shard_paths(data_dir: str | Path, split: str) -> list[Path]:
    return sorted(Path(data_dir).glob(f"{split}_*.npz"))


def load_split(data_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    """Charge tous les shards d'un split en mémoire (3 M de positions ~ 215 Mo)."""
    paths = shard_paths(data_dir, split)
    if not paths:
        raise FileNotFoundError(f"aucun shard « {split} » dans {data_dir}")
    cols: dict[str, list[np.ndarray]] = {}
    for p in paths:
        with np.load(p) as z:
            for key in z.files:
                cols.setdefault(key, []).append(z[key])
    return {key: np.concatenate(v) for key, v in cols.items()}


# --------------------------------------------------------------------------- extraction


def extract(
    source: str | list[str],
    out_dir: str | Path,
    cfg: FilterConfig = FilterConfig(),
    *,
    target_positions: int | None = None,
    per_source_positions: int | None = None,
    max_games: int | None = None,
    shard_size: int = 250_000,
    workers: int = 1,
    chunk_games: int = 100,
    overwrite: bool = False,
    progress: bool = True,
) -> dict:
    """Lit `source`, filtre, convertit et écrit les shards dans `out_dir`. Retourne le manifeste.

    `source` peut être une liste (plusieurs mois, lus l'un après l'autre) : chaque source est alors
    lue jusqu'à `per_source_positions` positions (train + val), ou jusqu'à sa fin. Les numéros de
    partie continuent d'une source à l'autre, donc restent uniques.
    S'arrête quand `target_positions` (train + val) est atteint, quand `max_games` parties ont été
    LUES (avant filtrage), ou à la fin de la dernière source. workers=1 : tout dans le processus courant.
    La sortie ne dépend pas du nombre de workers (les résultats sont consommés dans l'ordre).
    """
    sources = [source] if isinstance(source, str) else list(source)
    if len(sources) * GAMES_PER_SOURCE > np.iinfo(np.int32).max:
        raise ValueError(f"trop de sources ({len(sources)}) pour des numéros de partie int32")
    out_dir = Path(out_dir)
    existing = [p for s in SPLITS for p in shard_paths(out_dir, s)] + list(out_dir.glob("manifest.json"))
    if existing:
        if not overwrite:
            raise FileExistsError(f"{out_dir} contient déjà des données (utiliser overwrite=True)")
        for p in existing:
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    stats: Counter = Counter()
    rejected: Counter = Counter()
    positions = {s: 0 for s in SPLITS}
    writers = {s: _ShardWriter(out_dir, s, shard_size) for s in SPLITS}
    pending: deque = deque()  # résultats en attente, dans l'ordre d'envoi
    window = max(2 * workers, 1)  # nombre de lots en vol : borne la mémoire
    pool = mp.Pool(workers) if workers > 1 else None
    bar = tqdm(total=target_positions, unit="pos", disable=not progress, smoothing=0.05)

    def done() -> bool:
        return target_positions is not None and sum(positions.values()) >= target_positions

    def consume(result) -> None:
        arrays, chunk_stats = result
        stats.update(chunk_stats)
        for split, cols in arrays.items():
            writers[split].add(cols)
            positions[split] += len(cols[0])
            bar.update(len(cols[0]))
        bar.set_postfix(read=stats["read"], kept=stats["kept"])

    def submit(items: list) -> None:
        job = (items, cfg.min_plies)
        pending.append(pool.apply_async(process_chunk, (job,)) if pool else process_chunk(job))

    def next_result():
        r = pending.popleft()
        return r.get() if pool else r

    per_source: list[dict] = []
    ended = True  # False si la dernière source lue a été arrêtée avant sa fin
    try:
        for src_index, src in enumerate(sources):
            start_pos, start_read, start_kept = sum(positions.values()), stats["read"], stats["kept"]
            stream, counter = open_pgn(src)

            def source_done() -> bool:
                return done() or (
                    per_source_positions is not None
                    and sum(positions.values()) - start_pos >= per_source_positions
                )

            stopped = False
            try:
                items: list[tuple[int, str, str]] = []
                for text in iter_game_texts(stream):
                    stats["read"] += 1
                    headers = parse_headers(text)
                    reason = reject_reason(headers, cfg)
                    if reason:
                        rejected[reason] += 1
                    else:
                        # numéro stable d'une source à l'autre (ne dépend pas des lots lus d'avance)
                        game_no = src_index * GAMES_PER_SOURCE + stats["read"] - start_read
                        items.append((game_no, split_of(headers, game_no, cfg.val_fraction), text))
                    if len(items) >= chunk_games:
                        submit(items)
                        items = []
                        while len(pending) >= window and not source_done():
                            consume(next_result())
                    if source_done() or (max_games is not None and stats["read"] >= max_games):
                        stopped = True
                        break
                if items and not source_done():
                    submit(items)
                while pending and not source_done():  # on jette ce qui dépasserait la cible
                    consume(next_result())
            finally:
                stream.close()
                pending.clear()  # résultats en trop (ou d'une source finie) : jetés
            per_source.append(
                {
                    "source": src,
                    "games_read": stats["read"] - start_read,
                    "games_kept": stats["kept"] - start_kept,
                    "positions": sum(positions.values()) - start_pos,
                    "bytes_read_compressed": counter.bytes_read,
                    "reached_end": not stopped,
                }
            )
            ended = not stopped
            if done() or (max_games is not None and stats["read"] >= max_games):
                break
    finally:
        bar.close()
        if pool:
            pool.terminate()
            pool.join()
    for w in writers.values():
        w.close()

    on_disk = [out_dir / s["file"] for w in writers.values() for s in w.shards]
    manifest = {
        "source": source,
        "sources": per_source,
        "filter": asdict(cfg),
        "target_positions": target_positions,
        "per_source_positions": per_source_positions,
        "max_games": max_games,
        "stopped_early": not ended,
        "games_read": stats["read"],
        "games_rejected_by_header": dict(rejected),
        "games_rejected_after_parse": {k: stats[k] for k in ("parse_error", "too_short") if stats[k]},
        "games_kept": stats["kept"],
        # parties / positions gardées, par type de fin de partie (Normal, Time forfeit)
        "kept_by_termination": {
            "games": {k[6:]: v for k, v in sorted(stats.items()) if k.startswith("games:")},
            "positions": {k[10:]: v for k, v in sorted(stats.items()) if k.startswith("positions:")},
        },
        "positions": positions,
        "shards": {s: w.shards for s, w in writers.items()},
        "bytes_read_compressed": sum(d["bytes_read_compressed"] for d in per_source),
        "bytes_on_disk": sum(p.stat().st_size for p in on_disk),
        "seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest
