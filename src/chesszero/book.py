"""Livre d'ouvertures du self-play (étape 6b) : positions de départ tirées de vraies parties Lichess.

Pourquoi : des parties de self-play qui partent toutes de la position initiale (même avec du bruit) se
referment vite sur un petit répertoire. On part donc, la plupart du temps, d'une position atteinte par
des joueurs humains (Elo ≥ 2000), après 8 à 20 demi-coups.

Construction (`scripts/prepare_book.py`) :
  - parties filtrées comme les données supervisées (`configs/data.toml` : Elo des deux joueurs, pas de
    bullet, pas de BOT, fin normale ou au temps), partie d'entraînement seulement (pas de validation) ;
  - chaque partie est coupée à un demi-coup tiré entre `min_ply` et `max_ply` (graine = identifiant de la
    partie : même livre à chaque extraction) ; on garde la SUITE DES COUPS, pas seulement la position,
    pour que l'historique (répétitions) soit le même que dans une vraie partie ;
  - une entrée par position distincte (la première suite rencontrée), avec le nombre de parties qui y
    passent ; le tirage se fait ensuite UNIFORMÉMENT parmi les positions distinctes, pas selon leur
    fréquence, pour ne pas surreprésenter les lignes principales ;
  - positions trop déséquilibrées écartées : |valeur du réseau| > `max_abs_value` ;
  - entrées retirées si leur suite passe par une des 40 positions de départ du protocole de mesure v2
    (comparaison de positions : les transpositions comptent) ou s'arrête sur une de ces lignes avant sa fin :
    le self-play ne s'entraîne pas sur les ouvertures de la mesure.

Format : `books/<nom>.tsv.gz`, une ligne par position : coups UCI séparés par des espaces, tabulation,
nombre de parties, tabulation, valeur du réseau (camp au trait) ; + `books/<nom>.json` (paramètres, stats).
"""

from __future__ import annotations

import gzip
import json
import random
import re
import zlib
from pathlib import Path

import chess
import numpy as np

_COMMENT_RE = re.compile(r"\{[^}]*\}")
_MOVE_NUMBER_RE = re.compile(r"^\d+\.+")
_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def first_moves(text: str, max_plies: int) -> list[chess.Move] | None:
    """Les `max_plies` premiers coups d'une partie PGN (texte brut d'une partie, en-têtes compris), sans
    passer par `chess.pgn` (trop lent pour des millions de parties) : commentaires {...} (pendule, éval)
    retirés, numéros de coups, annotations (?!) et NAG ($n) ignorés. None si un coup est illisible."""
    _, sep, body = text.partition("\n\n")
    if not sep:
        return None
    board, out = chess.Board(), []
    for token in _COMMENT_RE.sub(" ", body).split():
        if token in _RESULTS or token.startswith("$"):
            continue
        token = _MOVE_NUMBER_RE.sub("", token).rstrip("?!")
        if not token:
            continue
        try:
            move = board.parse_san(token)
        except ValueError:
            return None
        out.append(move)
        board.push(move)
        if len(out) >= max_plies:
            break
    return out


def cut_ply(game_key: str, min_ply: int, max_ply: int) -> int:
    """Demi-coup de coupe, tiré uniformément dans [min_ply, max_ply], fixé par l'identifiant de la partie."""
    return random.Random(zlib.crc32(game_key.encode())).randint(min_ply, max_ply)


def book_entry(text: str, game_key: str, min_ply: int, max_ply: int) -> tuple[str, str] | None:
    """(clé de position EPD, coups UCI jusqu'à la coupe) pour une partie, ou None si elle ne convient pas
    (illisible, trop courte : il faut au moins un coup joué APRÈS la position, ou position terminale)."""
    ply = cut_ply(game_key, min_ply, max_ply)
    moves = first_moves(text, ply + 1)
    if moves is None or len(moves) <= ply:
        return None
    board = chess.Board()
    for move in moves[:ply]:
        board.push(move)
    if board.is_game_over(claim_draw=False):
        return None
    return board.epd(), " ".join(m.uci() for m in moves[:ply])


def opening_keys(openings: list[tuple[str, str]]) -> tuple[set, set]:
    """Positions à exclure du livre pour des ouvertures données (nom, coups UCI) : (positions finales,
    positions intermédiaires de leurs lignes). Clé = `Board._transposition_key()` (placement des pièces,
    trait, roques, en passant), donc indépendante de l'ordre des coups."""
    final, path = set(), set()
    for _, moves in openings:
        board = chess.Board()
        for uci in moves.split():
            board.push_uci(uci)
            path.add(board._transposition_key())
        final.add(board._transposition_key())
    return final, path - final


def touches_openings(moves: str, final: set, path: set) -> str | None:
    """« through » si une position de la suite `moves` (transpositions comprises) est une position finale
    exclue, « on_path » si la suite s'arrête sur une ligne exclue avant sa fin, None sinon."""
    board = chess.Board()
    for uci in moves.split():
        board.push(chess.Move.from_uci(uci))
        if board._transposition_key() in final:
            return "through"
    return "on_path" if board._transposition_key() in path else None


class OpeningBook:
    """Positions de départ du self-play (suites de coups UCI depuis la position initiale)."""

    def __init__(self, lines: list[str], counts: np.ndarray | None = None, values: np.ndarray | None = None):
        if not lines:
            raise ValueError("livre vide")
        self.lines = lines
        self.counts = counts if counts is not None else np.ones(len(lines), dtype=np.int64)
        self.values = values if values is not None else np.zeros(len(lines), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.lines)

    @classmethod
    def load(cls, path: str | Path) -> OpeningBook:
        lines, counts, values = [], [], []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for row in f:
                moves, count, value = row.rstrip("\n").split("\t")
                lines.append(moves)
                counts.append(int(count))
                values.append(float(value))
        return cls(lines, np.array(counts, dtype=np.int64), np.array(values, dtype=np.float32))

    def save(self, path: str | Path, meta: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=9) as f:
            for moves, count, value in zip(self.lines, self.counts, self.values):
                f.write(f"{moves}\t{int(count)}\t{float(value):.3f}\n")
        tmp.replace(path)
        meta_path = path.with_name(path.name.removesuffix(".tsv.gz") + ".json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

    def excluding(self, openings: list[tuple[str, str]]) -> tuple[OpeningBook, dict[str, int]]:
        """Livre sans les entrées qui passent par une de ces ouvertures (ou s'arrêtent sur leur ligne).
        Renvoie (nouveau livre, nombre d'entrées retirées par motif)."""
        final, path = opening_keys(openings)
        reasons = [touches_openings(moves, final, path) for moves in self.lines]
        keep = np.array([r is None for r in reasons])
        removed = {"through": reasons.count("through"), "on_path": reasons.count("on_path")}
        return OpeningBook([l for l, k in zip(self.lines, keep) if k], self.counts[keep], self.values[keep]), removed

    def board(self, i: int) -> chess.Board:
        """Position n° `i`, avec tout l'historique de coups (pile de `chess.Board`)."""
        board = chess.Board()
        for uci in self.lines[i].split():
            board.push_uci(uci)
        return board

    def sample(self, rng: np.random.Generator) -> chess.Board:
        """Une position tirée uniformément parmi les positions distinctes du livre."""
        return self.board(int(rng.integers(len(self.lines))))
