"""scripts/analyse_openings.py : classement des paires, statistiques par groupe, ensembles de positions."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyse_openings as ao  # noqa: E402


def pair(w, b):
    """Paire : résultat (vu des Blancs) de la partie où A a les Blancs, puis de celle où A a les Noirs."""
    return {True: {"result_white": w}, False: {"result_white": b}}


def test_pair_kinds():
    assert ao.pair_kind(pair(1, 1)) == "Blancs gagnent les deux (1-1)"
    assert ao.pair_kind(pair(-1, -1)) == "Noirs gagnent les deux (1-1)"
    assert ao.pair_kind(pair(0, 0)) == "deux nulles"
    assert ao.pair_kind(pair(1, -1)) == "A gagne les deux"  # A gagne avec les Blancs puis avec les Noirs
    assert ao.pair_kind(pair(-1, 1)) == "B gagne les deux"
    assert ao.pair_kind(pair(1, 0)) == "une nulle et une victoire" and ao.pair_kind(pair(0, -1)) == "une nulle et une victoire"
    assert ao.pair_kind({True: {"result_white": 1}}) is None  # paire incomplète (arène reprise)


def test_group_stats_shares_are_computed_on_absolute_values():
    s = ao.group_stats([0, 50, -120, 250, -310, 99, 100, -100])
    assert s["n"] == 8 and s["share_gt_100"] == 0.375  # 120, 250, 310 (100 n'est pas > 100)
    assert s["share_gt_200"] == 0.25 and s["share_gt_300"] == 0.125
    assert s["share_white_better_100"] == 0.125 and s["share_black_better_100"] == 0.25


def test_read_games_groups_by_opening(tmp_path):
    rows = [{"opening": 7, "a_white": True, "result_white": 1}, {"opening": 7, "a_white": False, "result_white": 1},
            {"opening": 9, "a_white": True, "result_white": 0}]
    (tmp_path / "games.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    games = ao.read_games(tmp_path)
    assert set(games) == {7, 9} and ao.pair_kind(games[7]) == "Blancs gagnent les deux (1-1)" and ao.pair_kind(games[9]) is None


def test_sets_use_the_versioned_books_with_a_fixed_seed(tmp_path):
    class Args:
        out = str(tmp_path / "sets.json")
        seed, book_size, anchor = 0, 50, 10

    assert ao.cmd_sets(Args) == 0
    first = json.loads(Path(Args.out).read_text(encoding="utf-8"))
    assert ao.cmd_sets(Args) == 0
    again = json.loads(Path(Args.out).read_text(encoding="utf-8"))
    assert first == again  # graine fixe : mêmes positions
    assert len(first["groups"]["book_random"]) == 50 and len(first["groups"]["reserve"]) == 1991
    assert set(first["groups"]["anchor_100"]) <= set(first["groups"]["reserve"]) and len(first["groups"]["anchor_100"]) == 10
    assert len(first["reserve_keys"]) == 1991
    reserve = set(first["groups"]["reserve"])
    assert not (set(first["groups"]["book_random"]) & reserve)  # le self-play ne tire pas la réserve : l'échantillon non plus
