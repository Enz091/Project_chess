"""Tables de finales Syzygy : résultat exact des positions à peu de pièces (5 au plus ici).

Deux sortes de fichiers, tous deux nécessaires :
  - WDL (`.rtbw`) : gain / nulle / perte du camp au trait, en supposant le compteur des 50 coups à zéro.
    python-chess renvoie 2 (gain), 1 (gain « maudit » : mat forcé mais trop long, nulle par la règle des
    50 coups), 0 (nulle), -1 (perte « bénie »), -2 (perte).
  - DTZ (`.rtbz`) : distance (en demi-coups) jusqu'au prochain coup qui remet le compteur à zéro (prise ou
    coup de pion, ou mat). Ce n'est PAS la distance au mat : Syzygy ne donne jamais le mat le plus court.
    Jouer le coup qui minimise la DTZ garantit en revanche de convertir un gain en respectant la règle des
    50 coups (on progresse toujours vers la prochaine prise / poussée de pion, puis vers le mat).

Utilisation par la recherche (`MCTS(..., tablebase=...)`) :
  - dans l'arbre, une position couverte (≤ `max_pieces` pièces rois compris, sans droit de roque) devient
    une feuille de valeur exacte (+1 / 0 / -1, règle des 50 coups comprise) : le réseau n'est pas appelé ;
  - à la racine, si la position est couverte, le coup est choisi par `choose` (classement de Stockfish) :
    gain → le mat immédiat s'il existe, sinon le coup gagnant de plus petite DTZ ; nulle → le coup non perdant
    le plus visité par le MCTS ; perte → la défense la plus longue (plus grande DTZ).

Règle des 50 coups avec un compteur non nul (comme Stockfish, `Tablebases::root_probe`) : un gain de DTZ `d`
depuis une position de compteur `c` n'est sûr que si `d + c <= 100`, sinon c'est une nulle.
"""

from __future__ import annotations

from pathlib import Path

import chess
import chess.syzygy

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "tools" / "syzygy"
WDL_VALUE = {2: 1.0, 1: 0.0, 0: 0.0, -1: 0.0, -2: -1.0}  # valeur pour la recherche (gain maudit = nulle)
DTZ_AFTER_ZEROING = {2: 1, 1: 101, 0: 0, -1: -101, -2: -1}  # DTZ d'un coup qui remet le compteur à zéro


class Tablebase:
    """Accès aux tables. `prober` (tests) remplace les fichiers : objet avec `probe_wdl` et `probe_dtz`."""

    def __init__(self, directory: str | Path | None = None, *, max_pieces: int = 5, prober=None,
                 cache_size: int = 500_000):
        if prober is None:
            directory = Path(directory if directory is not None else DEFAULT_DIR)
            prober = chess.syzygy.open_tablebase(str(directory))
            # pièces couvertes = plus grand nombre de pièces pour lequel les DEUX sortes de tables existent
            have = {len(name) - 1 for name in prober.wdl} & {len(name) - 1 for name in prober.dtz}
            max_pieces = min(max_pieces, max(have, default=0))
            self.directory = directory
        else:
            self.directory = None
        self.prober = prober
        self.max_pieces = max_pieces
        self._cache: dict = {}
        self._cache_size = cache_size
        self.probes = 0  # positions de l'arbre résolues par les tables (statistique)

    @classmethod
    def open(cls, directory: str | Path | None = None, max_pieces: int = 5) -> Tablebase | None:
        """Ouvre les tables si le dossier en contient, sinon None."""
        directory = Path(directory if directory is not None else DEFAULT_DIR)
        if not directory.is_dir():
            return None
        tb = cls(directory, max_pieces=max_pieces)
        return tb if tb.max_pieces >= 3 else None

    def close(self) -> None:
        if hasattr(self.prober, "close"):
            self.prober.close()

    def describe(self) -> str:
        return f"Syzygy {self.max_pieces} pièces ({self.directory})"

    # ---------------------------------------------------------------- sondage d'une position

    def covers(self, board: chess.Board) -> bool:
        return chess.popcount(board.occupied) <= self.max_pieces and not board.castling_rights

    def wdl(self, board: chess.Board) -> int | None:
        """Résultat exact pour le camp au trait (2, 1, 0, -1, -2), compteur des 50 coups compris, ou None."""
        if not self.covers(board):
            return None
        key = (board._transposition_key(), board.halfmove_clock)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        try:
            w = self.prober.probe_wdl(board)
            if w in (2, -2) and board.halfmove_clock > 0:
                # gain (ou perte) en supposant le compteur à zéro : encore vrai avec le compteur actuel ?
                if abs(self.prober.probe_dtz(board)) + board.halfmove_clock > 100:
                    w //= 2  # 2 -> 1 (gain maudit), -2 -> -1 (perte bénie)
        except KeyError:  # table absente (MissingTableError)
            return None
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = w
        return w

    def value(self, board: chess.Board) -> float | None:
        """Valeur exacte pour la recherche (+1, 0, -1) du camp au trait, ou None si non couverte."""
        w = self.wdl(board)
        if w is None:
            return None
        self.probes += 1
        return WDL_VALUE[w]

    # ---------------------------------------------------------------- choix du coup à la racine

    def rank_moves(self, board: chess.Board) -> list[tuple[chess.Move, int, bool]] | None:
        """Pour chaque coup légal : (coup, DTZ comptée depuis `board` en incluant ce coup, mat immédiat ?).

        DTZ > 0 : le coup gagne, prochaine remise à zéro dans DTZ demi-coups ; < 0 : il perd ; 0 : nulle.
        Même calcul que Stockfish (`root_probe`). None si la position n'est pas couverte ou si une table manque.
        """
        if not self.covers(board):
            return None
        from chesszero.mcts import terminal_value  # import local : mcts importe ce module

        out = []
        try:
            for move in list(board.legal_moves):
                board.push(move)
                try:
                    mate = board.is_checkmate()
                    if mate:
                        dtz = 1
                    elif terminal_value(board) is not None:  # pat, répétition, 50 coups, matériel insuffisant
                        dtz = 0
                    elif board.halfmove_clock == 0:  # prise ou coup de pion : la WDL est exacte
                        dtz = DTZ_AFTER_ZEROING[-self.prober.probe_wdl(board)]
                    else:  # DTZ de l'adversaire, changée de signe, + le demi-coup joué
                        d = -self.prober.probe_dtz(board)
                        dtz = d + 1 if d > 0 else d - 1 if d < 0 else 0
                finally:
                    board.pop()
                out.append((move, dtz, mate))
        except KeyError:
            return None
        return out

    def choose(self, board: chess.Board, visits: dict[chess.Move, int] | None = None) -> chess.Move | None:
        """Coup des tables pour la racine `board`, ou None si non couverte. `visits` départage les coups
        équivalents (nulles) : le MCTS garde la main là où les tables ne distinguent rien."""
        ranked = self.rank_moves(board)
        if not ranked:
            return None
        visits = visits or {}
        c = board.halfmove_clock
        wins = [r for r in ranked if r[1] > 0 and r[1] + c <= 100]
        if wins:  # mat immédiat d'abord, puis la plus petite DTZ, puis le plus visité
            return min(wins, key=lambda r: (not r[2], r[1], -visits.get(r[0], 0)))[0]
        losses = [r for r in ranked if r[1] < 0 and -r[1] + c <= 100]
        safe = [r for r in ranked if r not in losses]
        if safe:  # nulle : le MCTS choisit parmi les coups qui ne perdent pas (à égalité, un gain maudit)
            return max(safe, key=lambda r: (visits.get(r[0], 0), r[1] > 0))[0]
        return max(losses, key=lambda r: (-r[1], visits.get(r[0], 0)))[0]  # perte : tenir le plus longtemps
