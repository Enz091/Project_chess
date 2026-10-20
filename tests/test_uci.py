import io
import threading
import time

import chess
import numpy as np
import pytest

from chesszero.uci import GoParams, UCIEngine, parse_go, score_cp, time_budget

BACK_RANK = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"  # Ra8#


class Stub:
    """Priors uniformes, valeur 0 : l'arbre ne dépend que des positions terminales."""

    def evaluate(self, boards, legal_indices):
        return [np.full(len(i), 1.0 / len(i)) for i in legal_indices], np.zeros(len(boards))


def make_engine():
    out = io.StringIO()
    return UCIEngine(out=out, evaluator=Stub()), out


def go(engine, command="go"):
    engine.handle(command)
    if engine._worker is not None:
        engine._worker.join()


def lines(out):
    return out.getvalue().splitlines()


def bestmove(out):
    return [l for l in lines(out) if l.startswith("bestmove")][-1].split()[1]


def test_handshake():
    engine, out = make_engine()
    engine.handle("uci")
    text = lines(out)
    assert text[-1] == "uciok"
    for name in ("Simulations", "MoveTime", "Threads", "BatchSize", "ReuseTree", "Checkpoint"):
        assert any(f"option name {name} " in l for l in text)
    engine.handle("isready")
    assert lines(out)[-1] == "readyok"


def test_go_nodes_returns_legal_move_and_info():
    engine, out = make_engine()
    engine.handle("position startpos moves e2e4")
    go(engine, "go nodes 64")
    move = chess.Move.from_uci(bestmove(out))
    assert move in chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").legal_moves
    info = [l for l in lines(out) if l.startswith("info ")][-1]
    assert " score cp " in info and " pv " in info
    assert engine.mcts.root.n_sum >= 64


def test_finds_mate_in_one():
    engine, out = make_engine()
    engine.handle(f"position fen {BACK_RANK}")
    go(engine, "go nodes 200")
    assert bestmove(out) == "a1a8"


def test_default_go_uses_simulations_option():
    engine, out = make_engine()
    engine.handle("setoption name Simulations value 40")
    go(engine, "go")
    assert 40 <= engine.mcts.root.n_sum < 40 + 8  # exactement 40 simulations, arrondi aux lots


def test_movetime_is_respected():
    engine, out = make_engine()
    start = time.perf_counter()
    go(engine, "go movetime 300")
    assert 0.2 < time.perf_counter() - start < 1.5
    assert bestmove(out)


def test_movetime_option_when_go_has_no_limit():
    engine, out = make_engine()
    engine.handle("setoption name MoveTime value 250")
    start = time.perf_counter()
    go(engine, "go")
    assert 0.15 < time.perf_counter() - start < 1.5


def test_stop_ends_infinite_search():
    engine, out = make_engine()
    engine.handle("go infinite")
    time.sleep(0.2)
    assert not any(l.startswith("bestmove") for l in lines(out))
    engine.handle("stop")
    assert bestmove(out)


def test_tree_reuse_keeps_subtree():
    engine, out = make_engine()
    engine.handle("position startpos")
    go(engine, "go nodes 200")
    first = engine.mcts
    mine = bestmove(out)
    # l'adversaire répond par son coup le plus visité dans l'arbre
    node = first.root.children[first.root.moves.index(chess.Move.from_uci(mine))]
    answer = node.moves[int(node.N.argmax())]
    visits_before = int(node.children[int(node.N.argmax())].n_sum)
    assert visits_before > 0
    engine.handle(f"position startpos moves {mine} {answer.uci()}")
    engine.handle("go nodes 8")
    engine._worker.join()
    assert engine.mcts is first  # même objet : l'arbre est réutilisé
    assert engine.mcts.root.n_sum >= visits_before + 8
    assert engine.mcts.board.move_stack == [chess.Move.from_uci(mine), answer]


def test_tree_not_reused_for_unrelated_position_or_new_game():
    engine, out = make_engine()
    engine.handle("position startpos moves e2e4")
    go(engine, "go nodes 16")
    first = engine.mcts
    engine.handle("position startpos moves d2d4")
    go(engine, "go nodes 16")
    assert engine.mcts is not first
    second = engine.mcts
    engine.handle("setoption name ReuseTree value false")
    engine.handle("position startpos moves d2d4")
    go(engine, "go nodes 16")
    assert engine.mcts is not second
    third = engine.mcts
    engine.handle("ucinewgame")
    engine.handle("position startpos moves d2d4")
    go(engine, "go nodes 16")
    assert engine.mcts is not third


def test_repetition_history_is_kept_through_moves():
    engine, out = make_engine()
    engine.handle("position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1")
    go(engine, "go nodes 16")
    assert len(engine.mcts.board.move_stack) == 7  # l'historique complet est dans l'arbre
    assert chess.Move.from_uci(bestmove(out)) in engine.board.legal_moves


def test_terminal_position_gives_null_move():
    engine, out = make_engine()
    engine.handle("position startpos moves f2f3 e7e5 g2g4 d8h4")  # mat du berger fou
    go(engine, "go nodes 16")
    assert bestmove(out) == "0000"


def test_setoption_parsing_and_unknown_option():
    engine, out = make_engine()
    engine.handle("setoption name Threads value 99")
    assert engine.options["Threads"] == 8  # plafond
    engine.handle("setoption name simulations value 77")
    assert engine.options["Simulations"] == 77
    engine.handle("setoption name Nope value 1")
    assert "option inconnue" in lines(out)[-1]


def test_search_error_still_gives_bestmove():
    class Broken:
        def evaluate(self, boards, legal_indices):
            raise RuntimeError("boom")

    out = io.StringIO()
    engine = UCIEngine(out=out, evaluator=Broken())
    go(engine, "go nodes 8")
    assert chess.Move.from_uci(bestmove(out)) in chess.Board().legal_moves


def test_parse_go_and_time_budget():
    p = parse_go("wtime 60000 btime 30000 winc 1000 binc 500 movestogo 20 nodes 5".split())
    assert (p.wtime, p.btime, p.winc, p.binc, p.movestogo, p.nodes) == (60000, 30000, 1000, 500, 20, 5)
    assert parse_go(["infinite"]).infinite
    assert time_budget(GoParams(), chess.WHITE) is None
    assert time_budget(GoParams(movetime=1000), chess.WHITE) == pytest.approx(0.95)
    # 60 s / 30 + 0,8 * 1 s - 0,05
    assert time_budget(GoParams(wtime=60000, btime=1000, winc=1000), chess.WHITE) == pytest.approx(2.75)
    # le camp au trait compte : Noirs avec 1 s restante -> pas plus de la moitié
    assert time_budget(GoParams(wtime=60000, btime=1000, binc=5000), chess.BLACK) == pytest.approx(0.45)


def test_score_cp_is_odd_and_monotonic():
    assert score_cp(0) == 0
    assert score_cp(0.3) == -score_cp(-0.3) > 0
    assert score_cp(0.5) > score_cp(0.3)
    assert abs(score_cp(1.0)) < 10_000
