import math
import random

import pytest

from chesszero.rating import EloFit, difference, elo_from_score, expected_score, fit_elo, per_level


def simulate(true_rating, levels, per, rng, draw_rate=0.0):
    games = []
    for lv in levels:
        p = expected_score(true_rating, lv)
        for _ in range(per):
            if rng.random() < draw_rate:
                games.append((lv, 0.5))
            else:  # gagne avec une probabilité telle que le score espéré reste p
                q = (p - 0.5 * draw_rate) / (1 - draw_rate)
                games.append((lv, 1.0 if rng.random() < q else 0.0))
    return games


def test_expected_score_basics():
    assert expected_score(2000, 2000) == pytest.approx(0.5)
    assert expected_score(2400, 2000) == pytest.approx(10 / 11)
    assert elo_from_score(expected_score(2100, 2000)) == pytest.approx(100)


def test_fit_exact_on_balanced_data():
    # 50 % contre un seul niveau : R = ce niveau
    games = [(1900, 1.0), (1900, 0.0)] * 10
    assert fit_elo(games).rating == pytest.approx(1900, abs=0.01)
    # 10 / 11 contre 2000 : R = 2400
    games = [(2000, 1.0)] * 10 + [(2000, 0.0)]
    assert fit_elo(games).rating == pytest.approx(2400, abs=0.1)


def test_fit_is_invariant_to_game_order_and_shift():
    rng = random.Random(1)
    games = simulate(2100, [1800, 2000, 2200, 2400], 20, rng, draw_rate=0.2)
    base = fit_elo(games).rating
    shuffled = games[:]
    rng.shuffle(shuffled)
    assert fit_elo(shuffled).rating == pytest.approx(base, abs=1e-6)
    shifted = fit_elo([(lv + 300, s) for lv, s in games])
    assert shifted.rating == pytest.approx(base + 300, abs=1e-4)
    assert shifted.se == pytest.approx(fit_elo(games).se)


def test_confidence_interval_coverage():
    rng = random.Random(7)
    hits, runs = 0, 300
    for _ in range(runs):
        fit = fit_elo(simulate(2000, [1700, 1900, 2100, 2300], 40, rng, draw_rate=0.2))
        hits += fit.low <= 2000 <= fit.high
    assert 0.90 <= hits / runs <= 0.995  # nominal 95 %


def test_more_games_shrink_the_interval():
    rng = random.Random(3)
    small = fit_elo(simulate(2000, [1900, 2100], 10, rng))
    large = fit_elo(simulate(2000, [1900, 2100], 200, rng))
    assert large.se < small.se / 3


def test_draws_reduce_uncertainty():
    rng = random.Random(5)
    no_draw = fit_elo(simulate(2000, [1900, 2100], 200, rng, draw_rate=0.0))
    many = fit_elo(simulate(2000, [1900, 2100], 200, rng, draw_rate=0.6))
    assert many.se < no_draw.se


def test_all_wins_or_losses_give_bounds_only():
    win = fit_elo([(2500, 1.0)] * 20)
    assert win.bound == "lower" and win.rating == 2500 and win.se is None and "≥" in win.describe()
    loss = fit_elo([(1320, 0.0)] * 20)
    assert loss.bound == "upper" and "≤" in loss.describe()


def test_one_draw_in_all_wins_is_measurable_and_high():
    fit = fit_elo([(2500, 1.0)] * 19 + [(2500, 0.5)])
    assert fit.bound is None and fit.rating > 2500 + 300


def test_difference():
    a = EloFit(2000, 30, 1941, 2059, 100)
    b = EloFit(2100, 40, 2022, 2178, 100)
    d, se = difference(a, b)
    assert d == 100 and se == pytest.approx(50)
    with pytest.raises(ValueError):
        difference(a, EloFit(2500, None, None, None, 20, "lower"))


def test_per_level_table():
    rows = per_level([(1900, 1.0), (1900, 0.5), (1900, 0.0), (2200, 0.0), (2200, 0.0)])
    assert [r["level"] for r in rows] == [1900, 2200]
    assert (rows[0]["wins"], rows[0]["draws"], rows[0]["losses"]) == (1, 1, 1)
    assert rows[0]["score"] == 0.5 and rows[0]["elo"] == pytest.approx(1900)
    assert rows[1]["elo"] is None  # 0 sur 2 : pas mesurable
