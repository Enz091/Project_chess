"""Self-play (étape 6c) : parties de l'IA contre elle-même, enregistrées comme cibles d'entraînement.

Une partie :
  - départ : une position du livre d'ouvertures (`book_fraction`, 90 %) ou la position initiale ;
  - à chaque coup : 200 nouvelles simulations (arbre réutilisé, bruit de Dirichlet neuf à la racine), puis
    enregistrement de la position (TOUS les coups légaux de la racine avec leurs visites, Q de la racine) ;
    un coup forcé (un seul coup légal) est joué sans recherche et n'est pas enregistré ;
  - choix du coup : jusqu'au demi-coup `temperature_plies` (compté depuis le début de la partie), tirage
    selon les visites parmi les coups qui ont au moins `temperature_min_visit_ratio` des visites du meilleur ;
    ensuite, le plus visité ;
  - abandon : le camp au trait abandonne si le Q de la racine ET celui de son coup le plus visité restent
    sous `resign_threshold` pendant `resign_consecutive` de ses coups d'affilée, à partir du demi-coup
    `resign_min_ply`. Dans les parties de contrôle (`no_resign_fraction`), personne n'abandonne : on note le
    demi-coup où chaque camp aurait abandonné, pour mesurer les faux abandons ;
  - fin : règles officielles (mat, pat, matériel insuffisant, 50 coups, triple répétition), abandon,
    ARBITRAGE PAR LES TABLES dès la première position à ≤ `tablebase_pieces` pièces (résultat exact ; gain
    « maudit » / perte « bénie » = nulle), ou nulle à `max_plies` demi-coups ;
  - cible de valeur de chaque position : résultat final vu par le camp au trait de cette position.

Plusieurs parties à la fois : `run_selfplay` fait avancer `games_per_worker` parties ensemble ; leurs feuilles
partent au réseau dans un même appel (`search_batched`).

Fichiers (« shards »), un par lot de parties et par processus, écrits de façon atomique :
  `<session>_w<k>_<n°>.npz` + `.json` (version du réseau, réglages, horodatage, nombre de positions).
  Positions : boards (N, 67) uint8 ; policy_offsets (N+1,) int32 ; policy_index (M,) int16 et policy_visits (M,)
  uint16 = coups légaux de la racine et leurs visites (positions i : [offsets[i], offsets[i+1])) ; policy_prior (M,)
  float16 = prior du réseau SANS bruit et policy_q (M,) float16 = Q de chaque coup (W/N, camp au trait ; NaN si non
  visité), pour débruiter la cible (chesszero/policy_target.py ; absents des shards d'avant 2026-10-24) ; played (N,)
  int16 = indice du coup joué ; result (N,) int8 ; root_q (N,) float16 (camp au trait) ; ply (N,) int16 ;
  game (N,) int32 = numéro de la partie dans le shard.
  Parties : game_uid, game_moves (coups UCI depuis la position initiale, ouverture du livre comprise),
  start_index (n° dans le livre, -1 = position initiale), start_plies, result_white, termination, plies,
  control, would_resign (G, 2) = demi-coup où les Blancs / les Noirs auraient abandonné (-1 = jamais).

Tampon entre sessions Kaggle (`sync_buffer`) : voir la fonction.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import chess
import numpy as np

from chesszero.encoding import pack_board
from chesszero.mcts import CONFIG_DIR, MCTS, MCTSConfig, search_batched, terminal_value

TERMINATIONS = ["mat", "pat", "matériel insuffisant", "50 coups", "triple répétition", "tables", "abandon",
                "limite de demi-coups"]
T_MATE, T_STALEMATE, T_MATERIAL, T_FIFTY, T_REPETITION, T_TABLES, T_RESIGN, T_MAX_PLIES = range(len(TERMINATIONS))


@dataclass(frozen=True)
class SelfPlayConfig:
    """Section [selfplay] de configs/<nom>.toml."""

    simulations: int = 200  # nouvelles simulations par coup
    leaves_per_tree: int = 4  # feuilles posées par arbre à chaque appel au réseau
    games_per_worker: int = 64  # parties menées ensemble par processus
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_plies: int = 30  # tirage selon les visites tant que le demi-coup de la partie est < 30
    temperature_min_visit_ratio: float = 0.1  # coups tirables : visites >= 10 % de celles du meilleur
    book_fraction: float = 0.9  # parties qui partent du livre (les autres : position initiale)
    resign_threshold: float = -0.90
    resign_consecutive: int = 3
    resign_min_ply: int = 20
    no_resign_fraction: float = 0.10  # parties de contrôle (sans abandon)
    max_plies: int = 400
    tablebase_pieces: int = 5
    shard_games: int = 100  # un fichier tous les N parties...
    shard_minutes: float = 10.0  # ... ou toutes les N minutes
    buffer_positions: int = 500_000  # fenêtre du tampon conservée d'une session à l'autre
    stop_new_games_minutes: float = 20.0  # avant la fin du budget : plus de nouvelles parties
    hard_stop_minutes: float = 5.0  # avant la fin du budget : arrêt, parties inachevées abandonnées

    @classmethod
    def from_toml(cls, path: str | Path) -> SelfPlayConfig:
        with open(path, "rb") as f:
            section = tomllib.load(f).get("selfplay", {})
        unknown = set(section) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"clés inconnues dans [selfplay] : {sorted(unknown)}")
        return cls(**section)

    @classmethod
    def named(cls, name: str) -> SelfPlayConfig:
        return cls.from_toml(CONFIG_DIR / f"{name}.toml")

    def mcts_config(self, base: MCTSConfig) -> MCTSConfig:
        """Réglages de recherche du self-play : ceux de [mcts] + bruit de Dirichlet ; la température est
        gérée ici (filtre sur les visites), pas par MCTS.best_move."""
        return replace(base, num_simulations=self.simulations, batch_size=self.leaves_per_tree,
                       dirichlet_alpha=self.dirichlet_alpha, dirichlet_epsilon=self.dirichlet_epsilon, temperature=0.0)


def end_of_game(board: chess.Board, tablebase=None) -> tuple[int, int] | None:
    """(résultat pour les Blancs, code de fin) si la partie s'arrête dans `board`, sinon None.
    Arbitrage par les tables : valeur exacte du camp au trait (gain maudit / perte bénie = 0)."""
    value = terminal_value(board)
    if value is not None:
        if value == -1.0:
            return (-1 if board.turn == chess.WHITE else 1), T_MATE
        if not any(board.generate_legal_moves()):
            code = T_STALEMATE
        elif board.is_insufficient_material():
            code = T_MATERIAL
        elif board.halfmove_clock >= 100:
            code = T_FIFTY
        else:
            code = T_REPETITION
        return 0, code
    if tablebase is not None:
        value = tablebase.value(board)
        if value is not None:
            v = int(round(value))
            return (v if board.turn == chess.WHITE else -v), T_TABLES
    return None


# --------------------------------------------------------------------------- une partie


class Game:
    def __init__(self, uid: str, mcts: MCTS, start_index: int, control: bool):
        self.uid = uid
        self.mcts = mcts
        self.start_index = start_index
        self.start_plies = len(mcts.board.move_stack)
        self.control = control
        # positions enregistrées : (position compacte, indices légaux, visites, coup joué, Q racine, demi-coup,
        # prior sans bruit, Q de chaque coup)
        self.records: list[tuple] = []
        self.low_streak = {chess.WHITE: 0, chess.BLACK: 0}
        self.would_resign = {chess.WHITE: -1, chess.BLACK: -1}
        self.result_white: int | None = None
        self.termination: int | None = None

    @property
    def board(self) -> chess.Board:
        return self.mcts.board

    @property
    def done(self) -> bool:
        return self.result_white is not None

    def finish(self, result_white: int, termination: int) -> None:
        self.result_white, self.termination = result_white, termination

    def check_end(self, cfg: SelfPlayConfig, tablebase) -> None:
        end = end_of_game(self.board, tablebase)
        if end is not None:
            self.finish(*end)
        elif len(self.board.move_stack) >= cfg.max_plies:
            self.finish(0, T_MAX_PLIES)

    def play_forced_moves(self, cfg: SelfPlayConfig, tablebase) -> None:
        """Coups forcés (un seul coup légal) : joués sans recherche ni enregistrement."""
        while not self.done and self.board.legal_moves.count() == 1:
            self.mcts.advance(next(iter(self.board.legal_moves)))
            self.check_end(cfg, tablebase)

    def step(self, cfg: SelfPlayConfig, rng: np.random.Generator, tablebase) -> None:
        """Après la recherche : enregistre la position, teste l'abandon, choisit et joue le coup."""
        root, board = self.mcts.root, self.board
        ply, side = len(board.move_stack), board.turn
        visits = root.N
        best = int(np.lexsort((root.P_raw, visits))[-1])
        if ply < cfg.temperature_plies and cfg.temperature > 0:
            v = visits.astype(np.float64)
            weights = np.where(v >= cfg.temperature_min_visit_ratio * v.max(), v ** (1.0 / cfg.temperature), 0.0)
            a = int(rng.choice(len(v), p=weights / weights.sum()))
        else:
            a = best
        q_root = root.q()
        q_best = float(root.W[best] / root.N[best])
        q_moves = np.where(visits > 0, root.W / np.maximum(visits, 1), np.nan)
        self.records.append((pack_board(board), root.idx.astype(np.int16), visits.astype(np.uint16),
                             int(root.idx[a]), q_root, ply, root.P_raw.astype(np.float16), q_moves.astype(np.float16)))

        if q_root < cfg.resign_threshold and q_best < cfg.resign_threshold:
            self.low_streak[side] += 1
        else:
            self.low_streak[side] = 0
        if self.low_streak[side] >= cfg.resign_consecutive and ply >= cfg.resign_min_ply:
            if self.control:
                if self.would_resign[side] < 0:
                    self.would_resign[side] = ply
            else:
                self.finish(-1 if side == chess.WHITE else 1, T_RESIGN)
                return
        self.mcts.advance(root.moves[a])
        self.check_end(cfg, tablebase)

    def arrays(self) -> dict:
        """Positions de la partie (résultat vu par le camp au trait de chaque position)."""
        n = len(self.records)
        sign = np.array([1 if r[5] % 2 == 0 else -1 for r in self.records], dtype=np.int8)  # demi-coup pair = Blancs
        return {
            "boards": np.stack([r[0] for r in self.records]) if n else np.zeros((0, 67), np.uint8),
            "index": [r[1] for r in self.records], "visits": [r[2] for r in self.records],
            "prior": [r[6] for r in self.records], "q": [r[7] for r in self.records],
            "played": np.array([r[3] for r in self.records], dtype=np.int16),
            "result": (sign * self.result_white).astype(np.int8),
            "root_q": np.array([r[4] for r in self.records], dtype=np.float16),
            "ply": np.array([r[5] for r in self.records], dtype=np.int16),
        }


# --------------------------------------------------------------------------- fichiers


class ShardWriter:
    """Accumule les parties finies d'un processus et écrit un shard tous les `shard_games` parties ou toutes
    les `shard_minutes` minutes (écriture atomique : fichier temporaire puis renommage)."""

    def __init__(self, out_dir: str | Path, prefix: str, meta: dict, shard_games: int, shard_minutes: float):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix, self.meta = prefix, meta
        self.shard_games, self.shard_seconds = shard_games, shard_minutes * 60
        self.games: list[Game] = []
        self.seq = 0
        self.last_flush = time.time()
        self.written: list[Path] = []

    def add(self, game: Game) -> None:
        self.games.append(game)
        if len(self.games) >= self.shard_games or time.time() - self.last_flush >= self.shard_seconds:
            self.flush()

    def flush(self) -> Path | None:
        self.last_flush = time.time()
        if not self.games:
            return None
        parts = [g.arrays() for g in self.games]
        counts = [len(p["played"]) for p in parts]
        index = [i for p in parts for i in p["index"]]
        lengths = np.array([len(i) for i in index], dtype=np.int64)
        offsets = np.zeros(len(index) + 1, dtype=np.int32)
        np.cumsum(lengths, out=offsets[1:])
        data = {
            "boards": np.concatenate([p["boards"] for p in parts]),
            "policy_offsets": offsets,
            "policy_index": np.concatenate(index).astype(np.int16) if index else np.zeros(0, np.int16),
            "policy_visits": (np.concatenate([v for p in parts for v in p["visits"]]).astype(np.uint16)
                              if index else np.zeros(0, np.uint16)),
            "policy_prior": (np.concatenate([v for p in parts for v in p["prior"]]).astype(np.float16)
                             if index else np.zeros(0, np.float16)),
            "policy_q": (np.concatenate([v for p in parts for v in p["q"]]).astype(np.float16)
                         if index else np.zeros(0, np.float16)),
            "played": np.concatenate([p["played"] for p in parts]),
            "result": np.concatenate([p["result"] for p in parts]),
            "root_q": np.concatenate([p["root_q"] for p in parts]),
            "ply": np.concatenate([p["ply"] for p in parts]),
            "game": np.repeat(np.arange(len(parts), dtype=np.int32), counts),
            "game_uid": np.array([g.uid for g in self.games]),
            "game_moves": np.array([" ".join(m.uci() for m in g.board.move_stack) for g in self.games]),
            "start_index": np.array([g.start_index for g in self.games], dtype=np.int32),
            "start_plies": np.array([g.start_plies for g in self.games], dtype=np.int16),
            "result_white": np.array([g.result_white for g in self.games], dtype=np.int8),
            "termination": np.array([g.termination for g in self.games], dtype=np.uint8),
            "plies": np.array([len(g.board.move_stack) for g in self.games], dtype=np.int16),
            "control": np.array([g.control for g in self.games], dtype=bool),
            "would_resign": np.array([[g.would_resign[chess.WHITE], g.would_resign[chess.BLACK]] for g in self.games],
                                     dtype=np.int16),
        }
        name = f"{self.prefix}_{self.seq:05d}"
        path = self.out_dir / f"{name}.npz"
        tmp = self.out_dir / f"{name}.tmp.npz"
        np.savez_compressed(tmp, **data)
        meta = {**self.meta, "shard": name, "created": time.time(),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "positions": int(len(data["played"])), "games": len(self.games)}
        json_tmp = self.out_dir / f"{name}.json.tmp"
        json_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        os.replace(tmp, path)  # le .npz d'abord : un .json présent garantit un .npz complet
        os.replace(json_tmp, self.out_dir / f"{name}.json")
        self.written.append(path)
        self.games, self.seq = [], self.seq + 1
        return path


def load_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- tampon entre sessions


def list_shards(dirs) -> list[dict]:
    """Shards complets (un .json à côté du .npz) de plusieurs dossiers, sans doublon de nom ; le plus récent
    d'abord (horodatage de création écrit dans le .json)."""
    found: dict[str, dict] = {}
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for meta_path in d.glob("*.json"):
            npz = meta_path.with_suffix(".npz")
            if not npz.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            name = meta_path.stem
            if name not in found:
                found[name] = {"name": name, "npz": npz, "json": meta_path, "created": meta["created"],
                               "positions": meta["positions"]}
    return sorted(found.values(), key=lambda s: (s["created"], s["name"]), reverse=True)


def buffer_window(shards: list[dict], max_positions: int) -> list[dict]:
    """Les shards les plus récents jusqu'à `max_positions` positions (le dernier peut dépasser un peu)."""
    out, total = [], 0
    for s in shards:
        if total >= max_positions:
            break
        out.append(s)
        total += s["positions"]
    return out


def sync_buffer(sources, dest: str | Path, max_positions: int) -> dict:
    """Met dans `dest` exactement la fenêtre du tampon : les shards les plus récents (jusqu'à `max_positions`
    positions) trouvés dans `sources` (sorties des sessions précédentes, en lecture seule sur Kaggle) et dans
    `dest` lui-même. Copie ceux qui manquent, supprime de `dest` ceux qui sont sortis de la fenêtre.

    Pourquoi : la sortie d'une version de notebook Kaggle ne contient que le dossier de travail de CETTE session.
    Appelée au début de la session (on récupère la fenêtre précédente) et à la fin (les nouveaux shards entrent
    dans la fenêtre, les plus anciens en sortent), elle garantit que la sortie contient toujours le tampon complet.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    window = buffer_window(list_shards(list(sources) + [dest]), max_positions)
    keep = {s["name"] for s in window}
    copied = removed = 0
    for s in window:
        if s["npz"].parent.resolve() != dest.resolve():
            shutil.copy2(s["npz"], dest / s["npz"].name)
            shutil.copy2(s["json"], dest / s["json"].name)
            copied += 1
    for s in list_shards([dest]):
        if s["name"] not in keep:
            s["npz"].unlink()
            s["json"].unlink()
            removed += 1
    return {"shards": len(window), "positions": sum(s["positions"] for s in window), "copied": copied,
            "removed": removed}


# --------------------------------------------------------------------------- boucle de self-play


def run_selfplay(evaluator, cfg: SelfPlayConfig, mcts_cfg: MCTSConfig, writer: ShardWriter, rng: np.random.Generator,
                 uid_prefix: str, book=None, tablebase=None, max_games: int | None = None,
                 soft_deadline: float | None = None, hard_deadline: float | None = None, log=None,
                 target_positions: int | None = None, min_seconds: float = 0.0) -> dict:
    """Joue des parties jusqu'à `max_games` parties commencées ou jusqu'aux échéances (temps `time.time()`) :
    après `soft_deadline`, plus de nouvelles parties ; à `hard_deadline`, arrêt (parties inachevées perdues).
    `target_positions` (tranche de la boucle) : plus de nouvelles parties dès que les positions des parties finies +
    celles attendues des parties en cours (au moins la longueur moyenne des parties finies) couvrent l'objectif ET que
    `min_seconds` se sont écoulées ; les parties en cours se terminent."""
    stats = {"games": 0, "positions": 0, "simulations": 0, "unfinished_dropped": 0,
             "terminations": [0] * len(TERMINATIONS), "seconds": 0.0, "target_reached_after": None}
    started = 0
    start_time = time.time()

    def new_game() -> Game:
        nonlocal started
        if book is not None and rng.random() < cfg.book_fraction:
            index = book.sample_index(rng)  # hors réserve d'arène si elle a été exclue (indice du livre complet)
            board = book.board(index)
        else:
            index, board = -1, chess.Board()
        game = Game(f"{uid_prefix}-{started}", MCTS(evaluator, mcts_cfg, board, rng=rng, tablebase=tablebase), index,
                    control=bool(rng.random() < cfg.no_resign_fraction))
        started += 1
        return game

    def target_covered() -> bool:
        if target_positions is None or time.time() - start_time < min_seconds:
            return False
        mean = stats["positions"] / stats["games"] if stats["games"] else 0.0
        expected = stats["positions"] + sum(max(len(g.records), mean) for g in games if not g.done)
        if expected >= target_positions and stats["target_reached_after"] is None:
            stats["target_reached_after"] = time.time() - start_time
        return expected >= target_positions

    def may_start() -> bool:
        return ((max_games is None or started < max_games) and (soft_deadline is None or time.time() < soft_deadline)
                and not target_covered())

    games: list[Game] = []
    while True:
        games = [g for g in games if not g.done]
        while len(games) < cfg.games_per_worker and may_start():
            games.append(new_game())
        if not games:
            break
        if hard_deadline is not None and time.time() >= hard_deadline:
            stats["unfinished_dropped"] += len(games)
            break
        for g in games:
            g.play_forced_moves(cfg, tablebase)
        active = [g for g in games if not g.done]
        if active:
            search_batched([g.mcts for g in active], evaluator, cfg.simulations, cfg.leaves_per_tree)
            stats["simulations"] += cfg.simulations * len(active)
            for g in active:
                g.step(cfg, rng, tablebase)
        for g in games:
            if g.done:
                writer.add(g)
                stats["games"] += 1
                stats["positions"] += len(g.records)
                stats["terminations"][g.termination] += 1
                if log is not None and stats["games"] % 10 == 0:
                    rate = stats["simulations"] / max(time.time() - start_time, 1e-9)
                    log(f"{stats['games']} parties, {stats['positions']} positions, {rate:.0f} simulations/s")
    writer.flush()
    stats["seconds"] = time.time() - start_time
    return stats


# --------------------------------------------------------------------------- statistiques de session

PIECE_POINTS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
PLY_BINS = [(0, 60, "< 60"), (60, 100, "60-99"), (100, 150, "100-149"), (150, 200, "150-199"), (200, 10**6, ">= 200")]
MATERIAL_BINS = [(60, 999, ">= 60 (presque complet)"), (40, 60, "40-59"), (20, 40, "20-39"), (0, 20, "< 20 (finale)")]


def material_points(board: chess.Board) -> int:
    """Matériel des deux camps en points (P 1, N/B 3, R 5, Q 9) : 78 dans la position initiale."""
    return sum(PIECE_POINTS[p.piece_type] for p in board.piece_map().values())


def repetition_stats(shards: list[dict]) -> dict:
    """Indicateur des répétitions précoces : parties nulles par triple répétition, réparties par demi-coup de la
    fin et par matériel restant. Sert à décider (après les premières sessions Kaggle) s'il faut traiter la double
    répétition comme une nulle dans l'arbre."""
    games = draws = 0
    reps: list[tuple[int, int]] = []  # (demi-coups, matériel restant)
    for data in shards:
        for g in range(len(data["game_uid"])):
            games += 1
            draws += int(data["result_white"][g]) == 0
            if int(data["termination"][g]) == T_REPETITION:
                board = chess.Board()
                for uci in str(data["game_moves"][g]).split():
                    board.push_uci(uci)
                reps.append((len(board.move_stack), material_points(board)))
    by_ply = {label: sum(lo <= p < hi for p, _ in reps) for lo, hi, label in PLY_BINS}
    by_material = {label: sum(lo <= m < hi for _, m in reps) for lo, hi, label in MATERIAL_BINS}
    return {"games": games, "draws": draws, "repetition_draws": len(reps),
            "share_of_games": len(reps) / games if games else 0.0,
            "share_of_draws": len(reps) / draws if draws else 0.0,
            "by_ply": by_ply, "by_material": by_material,
            "median_ply": float(np.median([p for p, _ in reps])) if reps else None,
            "median_material": float(np.median([m for _, m in reps])) if reps else None}


def describe_repetitions(s: dict) -> str:
    lines = [f"Nulles par triple répétition : {s['repetition_draws']} sur {s['games']} parties "
             f"({s['share_of_games']:.1%} des parties, {s['share_of_draws']:.1%} des {s['draws']} nulles)"]
    if s["repetition_draws"]:
        lines.append("  par demi-coup de la fin : " + ", ".join(f"{k} : {v}" for k, v in s["by_ply"].items()))
        lines.append("  par matériel restant (points, 78 au départ) : "
                     + ", ".join(f"{k} : {v}" for k, v in s["by_material"].items()))
        lines.append(f"  médianes : demi-coup {s['median_ply']:.0f}, matériel {s['median_material']:.0f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- vérification d'un shard


def verify_shard(data: dict, cfg: SelfPlayConfig, tablebase=None) -> list[str]:
    """Vérifie un shard en rejouant chaque partie depuis la position initiale. Renvoie la liste des erreurs.

    Positions : la position compacte enregistrée = celle de la partie rejouée à ce demi-coup ; décodée seule
    (`board_from_packed`), ses coups légaux = la liste enregistrée ; coup joué = coup de la partie, présent dans
    la liste et visité ; visites >= simulations ; résultat = résultat de la partie vu par le camp au trait ; pas
    de position à un seul coup légal. Parties : longueur, départ, fin cohérente avec la position finale (mat,
    pat, matériel, 50 coups, répétition, tables : ≤ 5 pièces, compteur 0, résultat des tables si fournies),
    abandon (partie normale, camp au trait, Q racine sous le seuil sur ses `resign_consecutive` derniers coups),
    demi-coup d'abandon des parties de contrôle.
    """
    from chesszero.encoding import board_from_packed, move_to_index
    from chesszero.mcts import legal_moves_and_indices

    errors: list[str] = []
    off, index, visits = data["policy_offsets"], data["policy_index"], data["policy_visits"]
    rows_by_game: dict[int, list[int]] = {}
    for row, g in enumerate(data["game"]):
        rows_by_game.setdefault(int(g), []).append(row)
    for g, uid in enumerate(data["game_uid"]):
        def err(msg, g=g, uid=uid):
            errors.append(f"partie {g} ({uid}) : {msg}")

        moves = [chess.Move.from_uci(u) for u in str(data["game_moves"][g]).split()]
        plies, result_white, term = int(data["plies"][g]), int(data["result_white"][g]), int(data["termination"][g])
        control, would = bool(data["control"][g]), [int(x) for x in data["would_resign"][g]]
        if plies != len(moves):
            err(f"plies {plies} != {len(moves)} coups")
        rows = sorted(rows_by_game.get(g, []), key=lambda r: int(data["ply"][r]))
        by_ply = {int(data["ply"][r]): r for r in rows}
        if len(by_ply) != len(rows):
            err("deux positions au même demi-coup")
        start = int(data["start_plies"][g])
        if (int(data["start_index"][g]) >= 0) != (start > 0):
            err("départ incohérent (livre / position initiale)")
        board = chess.Board()
        for p in range(len(moves) + 1):
            if p < start and p in by_ply:
                err(f"position enregistrée dans l'ouverture du livre (demi-coup {p})")
            r = by_ply.get(p)
            if r is not None:
                if not np.array_equal(pack_board(board), data["boards"][r]):
                    err(f"demi-coup {p} : position enregistrée différente de la partie")
                legal = index[off[r]:off[r + 1]]
                v = visits[off[r]:off[r + 1]]
                _, ref = legal_moves_and_indices(board_from_packed(data["boards"][r]))
                if sorted(ref.tolist()) != sorted(legal.tolist()):
                    err(f"demi-coup {p} : coups légaux enregistrés != coups légaux de la position décodée")
                if len(legal) < 2:
                    err(f"demi-coup {p} : coup forcé enregistré")
                if int(v.sum()) < cfg.simulations:
                    err(f"demi-coup {p} : {int(v.sum())} visites < {cfg.simulations} simulations")
                if "policy_prior" in data:  # shards depuis le 2026-10-24 : prior sans bruit et Q par coup
                    pr = data["policy_prior"][off[r]:off[r + 1]].astype(np.float64)
                    qm = data["policy_q"][off[r]:off[r + 1]].astype(np.float64)
                    if abs(pr.sum() - 1) > 0.02 or (pr < 0).any():
                        err(f"demi-coup {p} : priors de somme {pr.sum():.3f}")
                    if not np.array_equal(np.isnan(qm), v == 0) or (np.abs(qm[v > 0]) > 1.001).any():
                        err(f"demi-coup {p} : Q des coups incohérents avec les visites")
                played = int(data["played"][r])
                where = np.nonzero(legal == played)[0]
                if len(where) != 1 or v[where[0]] == 0:
                    err(f"demi-coup {p} : coup joué absent de la liste ou jamais visité")
                if p < len(moves) and played != move_to_index(moves[p], board):
                    err(f"demi-coup {p} : coup joué enregistré != coup de la partie")
                if p == len(moves) and term != T_RESIGN:
                    err("position enregistrée après le dernier coup alors que la partie ne finit pas par abandon")
                sign = 1 if board.turn == chess.WHITE else -1
                if int(data["result"][r]) != sign * result_white:
                    err(f"demi-coup {p} : résultat {int(data['result'][r])} != {sign * result_white} attendu")
            if p < len(moves):
                if moves[p] not in board.legal_moves:
                    err(f"demi-coup {p} : coup illégal {moves[p]}")
                    break
                board.push(moves[p])

        # fin de partie, vue depuis la position finale
        side_to_move_sign = 1 if board.turn == chess.WHITE else -1
        if term == T_RESIGN:
            side_rows = [r for r in rows if int(data["ply"][r]) % 2 == len(moves) % 2]
            last = side_rows[-cfg.resign_consecutive:]
            if control:
                err("abandon dans une partie de contrôle")
            if result_white != -side_to_move_sign:
                err("abandon : le résultat ne donne pas la défaite au camp qui abandonne")
            if not rows or int(data["ply"][rows[-1]]) != len(moves):
                err("abandon : la dernière position enregistrée n'est pas celle de l'abandon")
            if len(moves) < cfg.resign_min_ply or len(last) < cfg.resign_consecutive or any(
                    float(data["root_q"][r]) >= cfg.resign_threshold + 1e-3 for r in last):
                err("abandon sans les conditions (demi-coup minimal, Q racine sous le seuil assez longtemps)")
        else:
            end = end_of_game(board, tablebase if term == T_TABLES else None)
            if term == T_MAX_PLIES:
                if len(moves) < cfg.max_plies or result_white != 0:
                    err("limite de demi-coups incohérente")
            elif term == T_TABLES:
                if chess.popcount(board.occupied) > cfg.tablebase_pieces or board.castling_rights or board.halfmove_clock:
                    err("arbitrage par les tables sur une position non couverte (pièces, roque ou compteur)")
                if tablebase is not None and (end is None or end != (result_white, T_TABLES)):
                    err(f"arbitrage par les tables : résultat {result_white} != tables {end}")
            elif end is None or end != (result_white, term):
                err(f"fin enregistrée ({TERMINATIONS[term]}, {result_white}) != position finale {end}")
        for color, ply in zip((chess.WHITE, chess.BLACK), would):
            if ply >= 0:
                if not control:
                    err("demi-coup d'abandon noté dans une partie normale")
                if ply % 2 != (0 if color == chess.WHITE else 1) or ply not in by_ply:
                    err(f"demi-coup d'abandon {ply} incohérent pour {'les Blancs' if color else 'les Noirs'}")
    return errors


def config_dict(cfg: SelfPlayConfig, mcts_cfg: MCTSConfig) -> dict:
    return {"selfplay": asdict(cfg), "mcts": mcts_cfg.to_dict()}
