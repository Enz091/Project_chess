"""Tables de finales : logique de sondage et de choix (faux sondeur), puis vraies tables si présentes."""

import chess
import numpy as np
import pytest

from chesszero.mcts import MCTS, MCTSConfig
from chesszero.syzygy import DEFAULT_DIR, Tablebase

KQK_MATE_IN_1 = "7k/8/6K1/8/8/8/8/Q7 w - - 0 1"  # Qa8# ou Qg7#


class FakeProber:
    """Le camp qui a une dame gagne ; DTZ = distance du roi faible au coin a8 + 1 (inventée mais déterministe)."""

    def __init__(self):
        self.calls = 0

    def _strong(self, board):
        for color in chess.COLORS:
            if board.pieces(chess.QUEEN, color):
                return color
        return None

    def probe_wdl(self, board):
        self.calls += 1
        strong = self._strong(board)
        if strong is None:
            return 0
        return 2 if board.turn == strong else -2

    def probe_dtz(self, board):
        w = self.probe_wdl(board)
        if w == 0:
            return 0
        weak_king = board.king(not self._strong(board))
        d = chess.square_distance(weak_king, chess.A8) + 1
        return d if w > 0 else -d


class StubEvaluator:
    def __init__(self):
        self.positions = 0

    def evaluate(self, boards, legal_indices):
        self.positions += len(boards)
        return [np.full(len(i), 1.0 / len(i)) for i in legal_indices], np.zeros(len(boards))


def fake_tb(max_pieces=5):
    return Tablebase(prober=FakeProber(), max_pieces=max_pieces)


# --------------------------------------------------------------------------- sondage


def test_covers_piece_count_and_castling():
    tb = fake_tb(max_pieces=3)
    assert tb.covers(chess.Board(KQK_MATE_IN_1))
    assert not tb.covers(chess.Board("7k/8/6K1/8/8/8/P7/Q7 w - - 0 1"))  # 4 pièces > 3
    assert not tb.covers(chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"))  # droit de roque
    assert tb.wdl(chess.Board()) is None


def test_wdl_and_value():
    tb = fake_tb()
    board = chess.Board(KQK_MATE_IN_1)
    assert tb.wdl(board) == 2 and tb.value(board) == 1.0
    board.turn = chess.BLACK
    assert tb.wdl(board) == -2 and tb.value(board) == -1.0
    assert tb.value(chess.Board("7k/8/6K1/8/8/8/8/8 w - - 0 1")) == 0.0


def test_fifty_move_counter_turns_win_into_draw():
    # DTZ du faux sondeur ici : roi h8, distance à a8 = 7, donc 8 ; gain sûr seulement si 8 + compteur <= 100
    tb = fake_tb()
    assert tb.wdl(chess.Board("7k/8/6K1/8/8/8/8/Q7 w - - 92 80")) == 2
    assert tb.wdl(chess.Board("7k/8/6K1/8/8/8/8/Q7 w - - 93 80")) == 1  # gain maudit
    assert tb.value(chess.Board("7k/8/6K1/8/8/8/8/Q7 w - - 93 80")) == 0.0
    assert tb.wdl(chess.Board("7k/8/6K1/8/8/8/8/Q7 b - - 93 80")) == -1  # perte bénie


def test_cache_avoids_second_probe():
    tb = fake_tb()
    board = chess.Board("7k/8/6K1/8/8/8/8/Q7 w - - 10 80")
    tb.wdl(board)
    calls = tb.prober.calls
    tb.wdl(board)
    assert tb.prober.calls == calls


def test_missing_table_gives_none():
    class Missing:
        def probe_wdl(self, board):
            raise chess.syzygy.MissingTableError("absente")

        probe_dtz = probe_wdl

    tb = Tablebase(prober=Missing(), max_pieces=5)
    board = chess.Board(KQK_MATE_IN_1)
    assert tb.wdl(board) is None and tb.value(board) is None and tb.choose(board) is None


# --------------------------------------------------------------------------- choix à la racine


def test_choose_prefers_mate_in_one():
    board = chess.Board(KQK_MATE_IN_1)
    move = fake_tb().choose(board)
    board.push(move)
    assert board.is_checkmate()


def test_choose_minimises_dtz_when_no_mate():
    board = chess.Board("8/8/8/3k4/8/8/8/Q3K3 w - - 0 1")
    tb = fake_tb()
    ranked = tb.rank_moves(board)
    assert all(dtz > 0 for _, dtz, _ in ranked)
    best = min(dtz for _, dtz, _ in ranked)
    move = tb.choose(board, visits={m: 1000 for m, d, _ in ranked if d != best})  # les visites ne priment pas
    assert dict((m, d) for m, d, _ in ranked)[move] == best


def test_rank_counts_the_move_played():
    # coup non zéroïsant : DTZ adverse (négative) changée de signe + 1
    board = chess.Board("8/8/8/3k4/8/8/8/Q3K3 w - - 0 1")
    tb = fake_tb()
    for move, dtz, mate in tb.rank_moves(board):
        board.push(move)
        expected = 1 if mate else -tb.prober.probe_dtz(board) + 1
        board.pop()
        assert dtz == expected


def test_choose_draw_follows_visits_among_non_losing_moves():
    # Noirs : Kxa1 (prise de la dame) annule, tout le reste perd -> Kxa1 même si un autre coup a plus de visites
    board = chess.Board("8/8/8/8/8/8/1k6/Q3K3 b - - 0 1")
    tb = fake_tb()
    capture = chess.Move.from_uci("b2a1")
    assert capture in board.legal_moves
    move = tb.choose(board, visits={m: (1 if m == capture else 50) for m in board.legal_moves})
    assert move == capture


def test_choose_loss_resists_longest():
    board = chess.Board("8/8/8/8/3k4/8/8/Q3K3 b - - 0 1")  # Noirs perdent, aucune prise possible
    tb = fake_tb()
    ranked = tb.rank_moves(board)
    assert all(d < 0 for _, d, _ in ranked)
    move = tb.choose(board)
    assert dict((m, d) for m, d, _ in ranked)[move] == min(d for _, d, _ in ranked)


# --------------------------------------------------------------------------- dans le MCTS


def test_mcts_tablebase_leaves_are_exact_and_skip_the_network():
    # 6 pièces à la racine (hors tables) ; exd5 mène à 5 pièces : feuille exacte, sans appel au réseau
    board = chess.Board("4k3/7p/8/3p4/4P3/8/8/Q3K3 w - - 0 1")
    tb = fake_tb(max_pieces=5)
    ev = StubEvaluator()
    mcts = MCTS(ev, MCTSConfig(num_simulations=200, batch_size=8), board, check_invariants=True, tablebase=tb)
    mcts.search()
    root = mcts.root
    a = root.moves.index(chess.Move.from_uci("e4d5"))
    child = root.children[a]
    assert child is not None and child.terminal == -1.0  # Noirs au trait, perdants selon le faux sondeur
    assert tb.probes > 0 and mcts.stats["terminal"] >= root.N[a]


def test_mcts_advance_into_tablebase_leaf_reopens_search():
    board = chess.Board("4k3/7p/8/3p4/4P3/8/8/Q3K3 w - - 0 1")
    mcts = MCTS(StubEvaluator(), MCTSConfig(num_simulations=100, batch_size=4), board, tablebase=fake_tb())
    mcts.search()
    mcts.advance(chess.Move.from_uci("e4d5"))
    assert mcts.root.terminal is None  # pas une fin de partie : on peut chercher
    mcts.search(20)
    move = mcts.best_move(0)
    assert move in mcts.board.legal_moves


def test_mcts_best_move_uses_tables_at_root():
    mcts = MCTS(StubEvaluator(), MCTSConfig(num_simulations=50, batch_size=4), chess.Board(KQK_MATE_IN_1),
                tablebase=fake_tb())
    mcts.search()
    mcts.board.push(mcts.best_move(0))
    assert mcts.board.is_checkmate()


# --------------------------------------------------------------------------- vraies tables (si téléchargées)

real = pytest.mark.skipif(Tablebase.open(DEFAULT_DIR) is None, reason="tables Syzygy absentes de tools/syzygy")


@real
def test_real_tables_known_position():
    tb = Tablebase.open()
    assert tb.max_pieces == 5
    board = chess.Board("8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1")  # exemple de python-chess : perte, DTZ -53
    assert tb.wdl(board) == -2
    assert tb.prober.probe_dtz(board) == -53


@real
def test_real_tables_convert_kbnk_against_best_defence():
    """KBN contre K (le mat le plus difficile) : en suivant les tables des deux côtés, mat avant 50 coups."""
    tb = Tablebase.open()
    board = chess.Board("8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1")
    for _ in range(200):
        if board.is_game_over():
            break
        board.push(tb.choose(board))
    assert board.is_checkmate() and board.turn == chess.BLACK
