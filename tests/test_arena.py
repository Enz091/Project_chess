"""Arène (étape 7a) : ouvertures, couleurs, joueurs, abandon, Elo par paires, décision, reprise après coupure."""

import json
import math
import zlib
from dataclasses import asdict, replace
from pathlib import Path

import chess
import numpy as np
import pytest

from chesszero.arena import (ArenaConfig, Player, anchor_openings, append_history, gate_decision, gate_openings,
                             play_games, read_jsonl, run_arena, schedule, summarize)
from chesszero.book import (OpeningBook, arena_reserve_path, exclude_arena_reserve, is_arena_reserved)
from chesszero.mcts import MCTSConfig
from chesszero.rating import elo_from_score

ROOT = Path(__file__).resolve().parents[1]
SCHOLAR = "e2e4 e7e5 f1c4 b8c6 d1h5 g8f6"  # les Blancs matent en un coup (Qxf7#)


class Stub:
    """Priors uniformes, valeur `value_fn(board)` (camp au trait) ; compte les positions évaluées."""

    def __init__(self, value_fn=lambda b: 0.0):
        self.value_fn = value_fn
        self.positions = 0

    def evaluate(self, boards, legal_indices):
        self.positions += len(boards)
        return [np.full(len(i), 1.0 / len(i)) for i in legal_indices], np.array([self.value_fn(b) for b in boards])


def white_is_lost(board):
    return -0.99 if board.turn == chess.WHITE else 0.99


CFG = ArenaConfig(pairs=3, anchor_pairs=2, simulations=16, games_per_worker=4, max_plies=40)
RESERVE = OpeningBook(["e2e4 e7e5", "d2d4 d7d5", "c2c4 g8f6", "g1f3 d7d5", "e2e4 c7c5", "d2d4 g8f6"])


def players(sims_a=16, sims_b=16):
    return {"a": Player("A", None, sims_a), "b": Player("B", None, sims_b)}


def inproc(evaluators, pl, cfg=CFG, reserve=RESERVE, calls=None):
    """`run_games` sans processus (tests) ; `calls` reçoit le nombre de parties demandées à chaque lancement."""
    def run(todo, emit):
        if calls is not None:
            calls.append(len(todo))
        return play_games(todo, reserve, evaluators, pl, cfg, MCTSConfig(), emit)
    return run


def arena(tmp_path, openings=(0, 1, 2), evaluators=None, pl=None, cfg=CFG, **kw):
    pl = pl or players()
    evaluators = evaluators or {"a": Stub(), "b": Stub()}
    return run_arena(tmp_path, pl, list(openings), cfg, MCTSConfig(), RESERVE, "essai", kind="anchor",
                     label="essai", run_games=kw.pop("run_games", inproc(evaluators, pl, cfg)), log=lambda m: None,
                     **kw)


# --------------------------------------------------------------------------- ouvertures et programme


def test_anchor_openings_are_fixed_and_gate_openings_avoid_them():
    assert anchor_openings(1991, 100) == anchor_openings(1991, 100)
    anchor = set(anchor_openings(1991, 100))
    assert len(anchor) == 100
    g7, g8 = gate_openings(1991, 100, 7, 100), gate_openings(1991, 100, 8, 100)
    assert g7 == gate_openings(1991, 100, 7, 100) and g7 != g8  # graine = numéro du candidat
    assert len(set(g7)) == 100 and not anchor & set(g7) and not anchor & set(g8)
    with pytest.raises(ValueError):
        anchor_openings(10, 11)


def test_schedule_plays_each_opening_with_both_colors():
    plan = schedule([5, 2])
    assert [(s["opening"], s["a_white"]) for s in plan] == [(5, True), (5, False), (2, True), (2, False)]
    assert len({s["key"] for s in plan}) == 4
    with pytest.raises(ValueError):
        schedule([1, 1])


# --------------------------------------------------------------------------- réserve d'arène et self-play


def test_committed_arena_reserve_follows_the_rule():
    """La réserve versionnée = entrées du livre dont crc32("arena:" + EPD) % 1000 < 9 (contrôle complet de la
    réserve, échantillon de 20 000 entrées pour le reste du livre ; le livre exclut déjà les ouvertures v2)."""
    path = ROOT / "books" / "lichess_2000.tsv.gz"
    reserve_path = arena_reserve_path(path)
    if not reserve_path.exists():
        pytest.skip("réserve absente")
    book, reserve = OpeningBook.load(path), OpeningBook.load(reserve_path)
    meta = json.loads(reserve_path.with_name("lichess_2000_arena.json").read_text(encoding="utf-8"))
    assert meta["positions"] == len(reserve) and 1500 < len(reserve) < 2500
    in_book = set(book.lines)
    assert all(line in in_book for line in reserve.lines)
    assert all(is_arena_reserved(reserve.board(i).epd()) for i in range(len(reserve)))
    reserved = set(reserve.lines)
    rng = np.random.default_rng(0)
    sample = [i for i in rng.choice(len(book), size=20_000, replace=False) if book.lines[i] not in reserved]
    assert not any(is_arena_reserved(book.board(int(i)).epd()) for i in sample)


def test_selfplay_never_samples_the_reserve_and_keeps_book_indices(tmp_path):
    from chesszero.selfplay import SelfPlayConfig, ShardWriter, load_shard, run_selfplay

    lines = ["e2e4 e7e5", "d2d4 d7d5", "c2c4 g8f6", "g1f3 d7d5"]
    OpeningBook(lines).save(tmp_path / "b.tsv.gz", {})
    OpeningBook([lines[0], lines[2]]).save(arena_reserve_path(tmp_path / "b.tsv.gz"), {})
    book = OpeningBook.load(tmp_path / "b.tsv.gz")
    assert exclude_arena_reserve(book, tmp_path / "b.tsv.gz") == 2
    rng = np.random.default_rng(0)
    assert {book.sample_index(rng) for _ in range(500)} == {1, 3}  # indices du livre complet
    cfg = SelfPlayConfig(simulations=8, games_per_worker=4, max_plies=30, book_fraction=1.0, shard_games=100)
    writer = ShardWriter(tmp_path / "sp", "t_w0", {}, cfg.shard_games, cfg.shard_minutes)
    run_selfplay(Stub(), cfg, cfg.mcts_config(MCTSConfig()), writer, rng, "t", book=book, max_games=12)
    data = load_shard(writer.written[0])
    assert set(data["start_index"].tolist()) <= {1, 3}
    for g in range(len(data["start_index"])):
        assert str(data["game_moves"][g]).startswith(lines[int(data["start_index"][g])])
    assert exclude_arena_reserve(OpeningBook(lines), tmp_path / "sans_reserve.tsv.gz") == 0


# --------------------------------------------------------------------------- couleurs, joueurs, abandon


def test_colors_and_results_are_attributed_to_the_right_player():
    """Les Blancs matent en un coup : A gagne quand il a les Blancs, perd quand il a les Noirs."""
    reserve = OpeningBook([SCHOLAR])
    pl = players(200, 200)
    out = []
    play_games(schedule([0]), reserve, {"a": Stub(), "b": Stub()}, pl, CFG, MCTSConfig(), out.append)
    by_color = {r["a_white"]: r for r in out}
    for r in out:
        assert r["termination"] == "mat" and r["result_white"] == 1 and r["moves"].endswith("h5f7")
        assert r["plies"] == 7 and r["start_plies"] == 6
    assert by_color[True]["score_a"] == 1.0 and by_color[False]["score_a"] == 0.0
    s = summarize(out)
    assert s["score"] == 0.5 and s["pairs_scored_1"] == 1 and s["white_score"] == 1.0


def test_each_player_searches_with_its_own_network_and_simulations():
    """A : 40 simulations, B : 8. Chaque réseau évalue ~ (ses simulations) positions par coup de SON joueur."""
    ev = {"a": Stub(), "b": Stub()}
    pl = players(40, 8)
    cfg = replace(CFG, max_plies=26)
    out = []
    play_games(schedule([0, 1]), RESERVE, ev, pl, cfg, MCTSConfig(), out.append)
    moves = {"a": 0, "b": 0}
    for r in out:
        for ply in range(r["start_plies"], r["plies"]):
            white = ply % 2 == 0
            moves["a" if white == r["a_white"] else "b"] += 1
    assert moves["a"] == moves["b"] > 0
    assert 30 <= ev["a"].positions / moves["a"] <= 42
    assert 5 <= ev["b"].positions / moves["b"] <= 10


def test_resignation_needs_both_sides_to_agree():
    # départ après 20 demi-coups (espagnole) : l'abandon est permis tout de suite
    reserve = OpeningBook(["e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 "
                           "c6a5 b3c2 c7c5"])
    # 2 coups d'affilée (au-delà, des nulles par répétition dans l'arbre du faux réseau font passer le Q du meilleur
    # coup des Noirs sous 0,90 : la règle exige les deux Q, des deux côtés)
    cfg = replace(CFG, max_plies=30, resign_consecutive=2)
    out = []
    play_games(schedule([0]), reserve, {"a": Stub(white_is_lost), "b": Stub(white_is_lost)}, players(), cfg,
               MCTSConfig(), out.append)
    # les deux réseaux voient les Blancs perdus : Blancs bas aux demi-coups 20, 22 ; Noirs hauts aux demi-coups 21, 23
    # -> abandon des Blancs au demi-coup 24 (au demi-coup 22, les Noirs n'ont joué qu'un coup)
    assert all(r["termination"] == "abandon" and r["result_white"] == -1 and r["plies"] == 24 for r in out)
    out = []  # B ne se voit pas gagnant (valeur 0) : A, avec les Blancs, n'abandonne pas
    play_games(schedule([0])[:1], reserve, {"a": Stub(white_is_lost), "b": Stub()}, players(), cfg, MCTSConfig(),
               out.append)
    assert out[0]["termination"] != "abandon" and out[0]["plies"] > 24


# --------------------------------------------------------------------------- score, Elo, décision


def rec(opening, a_white, score):
    rw = (1 if score == 1 else -1 if score == 0 else 0) * (1 if a_white else -1)
    return {"key": f"{opening}:{a_white}", "opening": opening, "a_white": a_white, "score_a": score,
            "result_white": rw, "termination": "mat" if score != 0.5 else "triple répétition", "plies": 50}


def test_summary_uses_pairs_for_the_standard_error():
    pairs = {0: (1, 1), 1: (1, 0), 2: (0.5, 0.5), 3: (1, 0.5)}
    records = [rec(o, w, s) for o, (x, y) in pairs.items() for w, s in ((True, x), (False, y))]
    s = summarize(records)
    means = [1.0, 0.5, 0.5, 0.75]
    assert s["score"] == pytest.approx(0.6875) and s["points"] == 5.5
    assert (s["wins"], s["draws"], s["losses"]) == (4, 3, 1)
    assert s["score_se"] == pytest.approx(np.std(means, ddof=1) / 2)
    assert s["elo"] == pytest.approx(elo_from_score(0.6875))
    assert s["elo_low"] < s["elo"] < s["elo_high"]
    assert s["los"] == pytest.approx(0.5 * (1 + math.erf((0.6875 - 0.5) / s["score_se"] / math.sqrt(2))))
    assert s["pairs_scored_1"] == 2 and s["complete_pairs"] == 4
    assert s["repetitions_before_ply_60"] == 3
    assert elo_from_score(0.52) == pytest.approx(13.9, abs=0.1)


def test_gate_accepts_from_104_points_out_of_200():
    def gate(points):
        return {"score": points / 200, "points": points, "games": 200}
    assert gate_decision(gate(104), ArenaConfig())["accept"]
    d = gate_decision(gate(103), ArenaConfig())
    assert not d["accept"] and "51.5%" in d["reasons"][0]


def test_anchor_veto_at_two_standard_errors():
    cfg, gate = ArenaConfig(), {"score": 0.6, "points": 120, "games": 200}
    accepted = {"elo": 50.0, "elo_se": 20.0}
    se = math.hypot(20, 20)  # 28,3 : veto sous 50 - 56,6
    assert gate_decision(gate, cfg, {"elo": 50 - 2 * se + 1, "elo_se": 20.0}, accepted)["accept"]
    d = gate_decision(gate, cfg, {"elo": 50 - 2 * se - 1, "elo_se": 20.0}, accepted)
    assert not d["accept"] and d["reasons"][0].startswith("veto de l'ancre")
    assert d["anchor_diff_se"] == pytest.approx(se)
    assert gate_decision(gate, cfg, {"elo": -500, "elo_se": 20.0})["accept"]  # sans référence : pas de veto


# --------------------------------------------------------------------------- fichiers, coupures, reprise


def test_truncated_last_line_is_ignored_and_repaired_but_corruption_is_an_error(tmp_path):
    path = tmp_path / "games.jsonl"
    path.write_bytes(b'{"key": "a"}\n{"key": "b"}\n{"key": "c", "sco')
    assert [r["key"] for r in read_jsonl(path)] == ["a", "b"]
    assert path.read_bytes().endswith(b"sco")  # lecture seule : rien de modifié
    assert [r["key"] for r in read_jsonl(path, repair=True)] == ["a", "b"]
    assert path.read_bytes() == b'{"key": "a"}\n{"key": "b"}\n'
    path.write_bytes(b'{"key": "a"}\n{"key": \n{"key": "b"}\n')
    with pytest.raises(ValueError, match="ligne 2"):
        read_jsonl(path)


def test_resume_after_cut_replays_only_missing_games_and_matches_a_straight_run(tmp_path):
    straight = arena(tmp_path / "straight")
    assert straight is not None and straight["summary"]["games"] == 6

    out = tmp_path / "cut"
    calls = []
    run = inproc({"a": Stub(), "b": Stub()}, players(), calls=calls)
    assert arena(out, run_games=run, stop_after=2) is None  # coupure simulée après 2 parties
    assert len(read_jsonl(out / "games.jsonl")) == 2 and not (out / "result.json").exists()
    # coupure pendant l'écriture d'une 3e partie, et fichiers temporaires orphelins d'écritures atomiques
    with open(out / "games.jsonl", "ab") as f:
        f.write(b'{"key": "2:A-blancs", "opening": 2, "sc')
    (out / "result.json.tmp").write_text("{ tronqué", encoding="utf-8")
    (out / "games.pgn.tmp").write_text("[Event", encoding="utf-8")
    resumed = arena(out, run_games=run)
    assert calls == [6, 4]  # la reprise ne rejoue que les 4 couples (ouverture, couleur) manquants
    rows = read_jsonl(out / "games.jsonl")
    assert len(rows) == 6 and len({r["key"] for r in rows}) == 6
    assert resumed["summary"] == straight["summary"]
    assert sorted(rows, key=lambda r: r["key"]) == sorted(read_jsonl(tmp_path / "straight" / "games.jsonl"),
                                                          key=lambda r: r["key"])
    assert not (out / "result.json.tmp").exists() and not (out / "games.pgn.tmp").exists()
    assert (out / "games.pgn").read_text(encoding="utf-8").count("[Event ") == 6


def test_finished_arena_is_not_replayed_and_a_different_arena_is_refused(tmp_path):
    first = arena(tmp_path)

    def must_not_run(todo, emit):
        raise AssertionError("arène déjà terminée")
    assert arena(tmp_path, run_games=must_not_run) == first
    with pytest.raises(ValueError, match="autre arène"):
        arena(tmp_path, pl=players(16, 8))
    with pytest.raises(ValueError, match="autre arène"):
        arena(tmp_path, openings=(0, 1, 3))


def test_history_line_has_git_hash_and_config_and_is_added_once(tmp_path):
    result = arena(tmp_path / "a")
    assert set(result["git"]) == {"commit", "dirty"}
    hist = tmp_path / "history.jsonl"
    assert append_history(hist, {"id": "essai", **result})
    assert not append_history(hist, {"id": "essai", **result})  # reprise : pas de doublon
    (row,) = read_jsonl(hist)
    assert row["config"] == asdict(CFG) and row["mcts"] == MCTSConfig().to_dict()
    assert row["players"]["a"]["simulations"] == 16 and row["summary"]["games"] == 6


def test_process_runner_with_a_real_small_network(tmp_path):
    """Chemin réel : processus (spawn), réseau chargé depuis un checkpoint, file de résultats, coupure puis reprise."""
    import torch

    from chesszero.arena import process_runner
    from chesszero.model import ModelConfig, build_model

    mc = ModelConfig(n_blocks=1, n_filters=8, value_hidden=8)
    torch.manual_seed(0)
    ckpt = tmp_path / "net.pt"
    torch.save({"config": {"model": asdict(mc)}, "model": build_model(mc).state_dict(), "step": 0}, ckpt)
    reserve_path = tmp_path / "r_arena.tsv.gz"
    RESERVE.save(reserve_path, {})
    pl = {"a": Player("A", str(ckpt), 8), "b": Player("B", str(ckpt), 4)}
    cfg = replace(CFG, max_plies=24)
    stats = []
    run = process_runner(pl, cfg, MCTSConfig(), str(reserve_path), None, workers=2, device="cpu", stats_out=stats)
    out = tmp_path / "arena"
    kw = dict(players=pl, openings=[0, 1], cfg=cfg, base_mcts=MCTSConfig(), reserve=RESERVE, reserve_name="r",
              run_games=run, log=lambda m: None)
    assert run_arena(out, stop_after=1, **kw) is None
    assert len(read_jsonl(out / "games.jsonl")) == 1
    result = run_arena(out, **kw)
    assert result["summary"]["games"] == 4 and len(read_jsonl(out / "games.jsonl")) == 4
    assert result["players"]["a"]["sha256"] == result["players"]["b"]["sha256"]
    assert stats and all(s["simulations"]["a"] > 0 for s in stats if s["games"])
