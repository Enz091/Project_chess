"""scripts/diagnose_session.py : pertes en centipions, phases, bascules, débruitage exact / sans Q, règles d'arène."""

import json
import sys
from pathlib import Path

import chess
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_session as ds  # noqa: E402


def test_phase_and_move_loss():
    assert ds.phase_of(29, 70) == "ouverture" and ds.phase_of(30, 20) == "milieu" and ds.phase_of(30, 19) == "finale"
    assert ds.material(chess.Board()) == 78
    assert ds.move_loss(50, -250, True) == 300  # Blancs : +50 -> -250
    assert ds.move_loss(50, 250, False) == 200  # Noirs : (vu des Noirs) -50 -> -250
    assert ds.move_loss(50, 80, True) == 0  # gain : pas de perte négative
    assert ds.move_loss(ds.MATE_CP, 0, True) == 1000  # mat borné à ±1000


def test_analyse_game_swing_repetition_and_conversion():
    # A (candidat) a les Noirs ; livre de 2 demi-coups ; puis 4 coups des réseaux
    game = {"key": "0:A-noirs", "a_white": False, "score_a": 0.0, "start_plies": 2, "termination": "mat",
            "moves": "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6"}
    fens = ds.game_positions(game)
    assert [p for p, _ in fens] == [2, 3, 4, 5, 6]
    cp_white = [20, 30, 400, 420, 450]  # coup des Noirs au demi-coup 3 : +30 -> +400 (gaffe de A)
    sf = {ds.fen_key(f): cp for (_, f), cp in zip(fens, cp_white)}
    out = ds.analyse_game(game, sf)
    a_rows = [r for r in out["rows"] if r["player"] == "A"]
    assert [r["loss"] for r in a_rows] == [370, 30] and all(r["phase"] == "ouverture" for r in a_rows)
    sw = out["swing"]
    assert sw["loser"] == "A" and sw["by"] == "A" and sw["ply"] == 3 and sw["loss"] == 370
    assert sw["loser_cp_before"] == -30 and sw["worst_own"]["loss"] == 370
    assert out["conversion"]["B"] == {"reached_300": True, "won": True}
    assert out["conversion"]["A"]["reached_300"] is False
    rep = dict(game, termination="triple répétition", score_a=0.5)
    r = ds.analyse_game(rep, sf)["repetition"]
    assert r["renouncer"] == "B"  # Blancs (B) à plus de +150 en moyenne sur les 6 dernières positions : B a renoncé
    assert "swing" not in ds.analyse_game(rep, sf)


def _shard(tmp_path, name, visits, prior, q):
    off = np.cumsum([0] + [len(v) for v in visits])
    np.savez(tmp_path / f"{name}.npz", policy_offsets=off.astype(np.int32),
             policy_visits=np.concatenate(visits).astype(np.uint16),
             policy_prior=np.concatenate(prior).astype(np.float16),
             policy_q=np.concatenate(q).astype(np.float16))


def test_prune_counts_discoveries_suppressed_by_the_no_q_approximation(tmp_path):
    # position 0 : 2e coup découvert meilleur que son prior (Q au-dessus du meilleur) -> gardé par la règle exacte,
    # plafonné par l'approximation sans Q ; position 1 : coup gonflé par le bruit, mauvais Q -> retiré par les deux
    nan = np.nan
    _shard(tmp_path, "L0002_00_w0_00000",
           [np.array([150, 80, 0]), np.array([150, 40, 10])],
           [np.array([0.4, 0.2, 0.4]), np.array([0.6, 0.39, 0.01])],
           [np.array([0.10, 0.14, nan]), np.array([0.10, 0.08, -0.5])])
    out = tmp_path / "prune.json"

    class A:
        buffer, prefix, c_puct = tmp_path, "L0002", 1.5
    A.out = str(out)
    assert ds.cmd_prune(A) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["positions"] == 2 and res["positions_with_suppressed_discovery"] == 0.5
    assert res["most_visited_changed"] == 0.0 and res["suppressed_target_mass_mean_affected"] > 0.01
    assert res["positions_with_overkept_move"] == 0.5  # le coup à Q -0,5 garde 2 visites sans Q, 0 avec


def test_simulate_rules_first_decision_exact_then_transitive():
    def c(i, elo, se=19.0, los=None):
        from statistics import NormalDist
        score = 1 / (1 + 10 ** (-elo / 400))
        return {"id": i, "elo": elo, "elo_se": se, "score": score,
                "los": NormalDist().cdf(elo / se) if los is None else los, "elo_high": elo + 1.96 * se}
    cands = [c(0, -17), c(1, -35), c(2, -19)]
    rules = {"s52": ("score", 0.52), "l10": ("los", 0.10), "all": ("all", None)}
    sims = ds.simulate_rules(cands, rules)
    assert [r["accepted"] for r in sims["s52"]["rows"]] == [False] * 3
    assert sims["l10"]["rows"][0]["accepted"] and sims["l10"]["rows"][0]["exact"]
    assert not sims["l10"]["rows"][1]["exact"] and sims["l10"]["rows"][1]["vs"] == 0
    assert sims["all"]["generations"] == 3
    # veto de l'ancre : chute de plus de 2 erreurs types sous l'accepté
    sims = ds.simulate_rules([c(0, 0), c(1, -120)], {"all": ("all", None)})
    assert sims["all"]["rows"][1]["veto"] and sims["all"]["generations"] == 1
    # arrêt « ancre » : générations 3 et 4 avec borne haute < 0
    low = [c(i, -60, se=10) for i in range(4)]
    sims = ds.simulate_rules(low, {"all": ("all", None)})
    assert sims["all"]["status"] == "ancre" and sims["all"]["generations"] == 4
    # compteur de rejets remis à zéro (remise de la ligne) : pas de stagnation au 5e rejet
    bad = [c(i, -40, los=0.01) for i in range(6)]
    assert ds.simulate_rules(bad, {"s": ("score", 0.52)})["s"]["status"] == "stagnation"
    assert ds.simulate_rules(bad, {"s": ("score", 0.52)}, counters_reset_after=(1,))["s"]["status"] == "ok"


def test_selfref_metrics_entropy_spoiled_wins_and_correlation():
    assert ds.entropy([1, 1]) == pytest.approx(np.log(2)) and ds.entropy([5, 0]) == 0.0
    board = chess.Board()
    item = {"fen": board.fen(), "moves": []}
    after = {}
    for uci, cp in (("e2e4", 400), ("g1f3", 150), ("a2a3", 380)):
        b = chess.Board()
        b.push_uci(uci)
        after[uci] = (b.fen(), cp)
    sf = {ds.fen_key(board.fen()): 420, **{ds.fen_key(f): cp for f, cp in after.values()}}
    ref = [{"move": "e2e4", "prior_top_move": "e2e4", "visit_entropy": 0.5, "top_visit_share": 0.8, "prior_entropy": 1.0,
            "prior_top": 0.5, "moves_10pct": 2, "root_q": 0.6}]
    res = [dict(ref[0], move="g1f3", visit_entropy=0.2)]
    m = ds.set_metrics([item], res, ref, sf, np.random.default_rng(0))
    assert m["acpl"] == 270 and m["acpl_diff_vs_ref"] == 250  # 420 -> 150 contre 420 -> 400
    assert m["spoiled_below_200"] == 1.0 and m["kept_300"] == 0.0 and m["same_search_move_as_ref"] == 0.0
    assert m["search_move_is_prior_top"] == 0.0 and m["visit_entropy"] == 0.2
    rows = {"a": {"x": 1.0, "y": 3.0}, "b": {"x": 2.0, "y": 2.0}, "c": {"x": 3.0, "y": 1.0}}
    c = ds.correlations(rows, {"a": 0, "b": 10, "c": 20})
    assert c["x"]["pearson"] == pytest.approx(1.0) and c["y"]["spearman"] == pytest.approx(-1.0)
