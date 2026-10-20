import io
import threading
import time

import chess
import numpy as np
import pytest

from chesszero.uci import GoParams, UCIEngine, main, parse_go, score_cp, search_limits, time_budget

UCI_QUIT = "uci" + chr(10) + "quit" + chr(10)
BACK_RANK = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"  # Ra8#


class Stub:
    """Priors uniformes, valeur 0 : l'arbre ne dépend que des positions terminales."""

    def evaluate(self, boards, legal_indices):
        return [np.full(len(i), 1.0 / len(i)) for i in legal_indices], np.zeros(len(boards))


class SlowStub(Stub):
    """Comme Stub, mais chaque appel dure 10 ms : le temps compte, comme avec le vrai réseau."""

    def evaluate(self, boards, legal_indices):
        time.sleep(0.01)
        return super().evaluate(boards, legal_indices)


def make_engine(evaluator=None, **options):
    out = io.StringIO()
    return UCIEngine(out=out, evaluator=evaluator or Stub(), options=options), out


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


def test_movetime_is_respected_with_clock_mode():
    engine, out = make_engine(UseClock=True)
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


def test_stop_ends_infinite_search_in_clock_mode():
    engine, out = make_engine(UseClock=True)
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


# --------------------------------------------------------------------------- modes de force : UseClock


def test_use_clock_is_off_by_default_and_advertised():
    engine, out = make_engine()
    assert engine.options["UseClock"] is False
    engine.handle("uci")
    assert "option name UseClock type check default false" in lines(out)
    engine.handle("setoption name UseClock value true")
    assert engine.options["UseClock"] is True
    engine.handle("setoption name UseClock value false")
    assert engine.options["UseClock"] is False


@pytest.mark.parametrize("use_clock, go_cmd, expected", [
    # (secondes, simulations) pour Simulations = 200, MoveTime = 0
    (False, "go wtime 300000 btime 300000 winc 3000 binc 3000", (None, 200)),   # pendule ignorée
    (False, "go movetime 5000", (None, 200)),                                      # movetime ignoré aussi
    (False, "go wtime 1000 btime 1000 nodes 50", (None, 50)),                      # nodes explicite
    (False, "go infinite", (None, 200)),                                           # répond quand même
    (False, "go", (None, 200)),
    (True, "go wtime 300000 btime 300000 winc 3000 binc 3000", (300 / 30 + 0.8 * 3 - 0.05, None)),
    (True, "go movetime 5000", (4.95, None)),
    (True, "go wtime 1000 btime 1000 nodes 50", (0.02, 50)),                  # budget plancher + nodes
    (True, "go infinite", (None, None)),                                           # jusqu'à stop
    (True, "go nodes 80", (None, 80)),
    (True, "go", (None, 200)),
])
def test_search_limits_priority(use_clock, go_cmd, expected):
    seconds, sims = search_limits(parse_go(go_cmd.split()[1:]), chess.WHITE, use_clock, 200)
    assert sims == expected[1]
    if expected[0] is None:
        assert seconds is None
    else:
        assert seconds == pytest.approx(expected[0])


def test_move_time_option_only_for_bare_go():
    bare, clock = parse_go([]), parse_go("wtime 60000 btime 60000".split())
    assert search_limits(bare, chess.WHITE, False, 200, 1500) == (pytest.approx(1.45), None)
    assert search_limits(bare, chess.WHITE, True, 200, 1500) == (pytest.approx(1.45), None)
    assert search_limits(clock, chess.WHITE, False, 200, 1500) == (None, 200)  # la pendule du GUI l'emporte sur MoveTime
    assert search_limits(clock, chess.WHITE, True, 200, 1500)[1] is None and search_limits(clock, chess.WHITE, True, 200, 1500)[0] > 1.5


def test_fixed_mode_does_exact_simulations_whatever_the_clock():
    # horloge dérisoire (50 ms) mais 160 simulations demandées : en force fixe, elles sont toutes faites
    engine, out = make_engine(SlowStub(), Simulations=160)
    go(engine, "go wtime 50 btime 50 winc 0 binc 0")
    assert engine.mcts.root.n_sum == 160
    first = engine.mcts
    go(engine, "go movetime 20")  # movetime ignoré lui aussi : encore 160 NOUVELLES simulations
    assert engine.mcts is first and engine.mcts.root.n_sum == 320


def test_clock_mode_uses_the_clock_not_simulations():
    engine, out = make_engine(SlowStub(), Simulations=160, UseClock=True)
    start = time.perf_counter()
    go(engine, "go wtime 50 btime 50 winc 0 binc 0")  # budget minimal : 20 ms
    assert time.perf_counter() - start < 0.5
    assert 0 < engine.mcts.root.n_sum < 160  # Simulations ignoré
    # et avec une grosse pendule : bien plus que les 160 de Simulations
    engine, out = make_engine(Simulations=10, UseClock=True)
    go(engine, "go wtime 3000 btime 3000 winc 0 binc 0")  # ~ 3000/30 - 0,05 = 0,05 s
    assert engine.mcts.root.n_sum > 10


def test_fixed_mode_infinite_clock_gets_an_answer():
    engine, out = make_engine(Simulations=32)
    go(engine, "go infinite")
    assert engine.mcts.root.n_sum == 32 and bestmove(out)


def test_fixed_mode_still_answers_stop_and_quit():
    engine, out = make_engine(SlowStub(), Simulations=1_000_000)
    engine.handle("go wtime 1000 btime 1000")  # l'horloge (20 ms) serait épuisée tout de suite : ignorée
    time.sleep(0.3)
    assert not any(l.startswith("bestmove") for l in lines(out))  # toujours en train de chercher
    t = time.perf_counter()
    engine.handle("stop")
    assert time.perf_counter() - t < 1.0 and bestmove(out)
    assert 0 < engine.mcts.root.n_sum < 1_000_000
    # quit pendant une recherche : s'arrête aussi
    engine.handle("go wtime 1000 btime 1000")
    time.sleep(0.1)
    t = time.perf_counter()
    assert engine.handle("quit") is False
    assert time.perf_counter() - t < 1.0
    assert len([l for l in lines(out) if l.startswith("bestmove")]) == 2


def test_profile_arguments_set_defaults():
    out = io.StringIO()
    main(["--simulations", "50", "--use-clock", "--threads", "99"], stdin=io.StringIO(UCI_QUIT), stdout=out)
    text = lines(out)
    assert "option name Simulations type spin default 50 min 1 max 1000000" in text
    assert "option name UseClock type check default true" in text
    assert "option name Threads type spin default 8 min 1 max 8" in text  # plafond
    out = io.StringIO()
    main([], stdin=io.StringIO(UCI_QUIT), stdout=out)
    assert "option name Simulations type spin default 200 min 1 max 1000000" in lines(out)
    assert "option name UseClock type check default false" in lines(out)
