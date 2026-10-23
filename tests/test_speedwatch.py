"""Surveillance de la vitesse de l'IA pendant les mesures contre Stockfish (chesszero/speedwatch.py + match_stockfish.py)."""

import json
import sys
from pathlib import Path

import pytest

from chesszero.speedwatch import NOMINAL_SECONDS_PER_MOVE, SpeedWatch, classify_power, describe_power, read_power, verdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import match_stockfish as ms  # noqa: E402


def feed(watch, values):
    return [watch.add(v) for v in values]


def noisy(n, centre=0.8, spread=0.06):
    return [centre * (1 + spread * (((i * 37) % 11) - 5) / 5) for i in range(n)]


def test_stable_measurement():
    w = SpeedWatch(0.8)
    infos = feed(w, noisy(100))
    assert not any(i["alert"] for i in infos) and w.state["episodes"] == []
    assert w.state["series"][0][0] == 20 and len(w.state["series"]) == 5  # une valeur toutes les 20 parties
    state, text = verdict(w.state)
    assert state == "stable" and "STABLE" in text


def test_no_alert_before_the_window_is_full():
    w = SpeedWatch(0.8)
    infos = feed(w, [3.0] * 19)  # 3 fois trop lent, mais seulement 19 parties
    assert not any(i["alert"] for i in infos)
    assert feed(w, [3.0])[0]["alert"]  # la 20e remplit la fenêtre


def test_throttling_that_starts_midway_is_detected_at_the_right_game():
    """Vitesse normale pendant 50 parties, puis 3 fois plus lent : moyenne glissante > 1,0 s (+25 %) dès la 3e partie lente
    ((20 - k) x 0,8 + k x 2,4) / 20 > 1,0  <=>  k >= 3. (Sans arrêt automatique : voir les tests d'arrêt plus bas.)"""
    w = SpeedWatch(0.8, abort_after=None)
    infos = feed(w, [0.8] * 50 + [2.4] * 30)
    first = next(i for i, x in enumerate(infos) if x["alert"])
    assert first == 52 and infos[first]["event"] == "start"  # 53e partie
    assert infos[first]["deviation"] == pytest.approx(0.30, abs=1e-9)
    assert all(x["alert"] for x in infos[first:]) and sum(x["event"] == "start" for x in infos) == 1
    state, text = verdict(w.state)
    assert state == "NON STABLE" and "NON STABLE" in text and "encore en cours" in text
    assert w.state["worst"]["deviation"] == pytest.approx(2.0, abs=1e-9)  # +200 % (2,4 contre 0,8)


def test_brief_episode_is_reported_but_not_invalidating():
    """Quatre parties à 2 s parmi 400 : la moyenne glissante sort de la tolérance sur 17 fenêtres seulement."""
    w = SpeedWatch(0.8)
    values = [0.8] * 200 + [2.0] * 4 + [0.8] * 196
    infos = feed(w, values)
    assert sum(x["event"] == "start" for x in infos) == 1 and sum(x["event"] == "end" for x in infos) == 1
    assert 0 < w.state["alert_games"] <= 0.05 * len(values)
    state, text = verdict(w.state)
    assert state == "perturbée brièvement" and "bref" in text
    assert w.state["episodes"][0]["end"] is not None


def test_faster_than_reference_is_also_flagged():
    w = SpeedWatch(1.6)
    feed(w, [1.0] * 25)
    assert w.state["alert_games"] > 0 and w.state["max_deviation"] < 0


def test_start_bench_checks_nominal_and_reference_speed():
    w = SpeedWatch(0.82)
    assert w.bench(0.82, 4) is None
    slow = SpeedWatch(2.5)  # la nuit du 2026-10-22
    msg = slow.bench(2.5, 4)
    assert msg and "2.50 s par coup contre 0.80" in msg
    resumed = SpeedWatch(0.8)
    assert "bench de départ de cette mesure" in resumed.bench(1.6, 4)  # reprise plus lente que la référence


def test_stable_but_slow_from_the_start_is_not_comparable():
    """Cas de la nuit : 2,5 s par coup du début à la fin = stable dans le temps, mais pas comparable à la référence."""
    w = SpeedWatch(2.55)
    feed(w, noisy(80, centre=2.5, spread=0.05))
    state, text = verdict(w.state)
    assert state == "NON COMPARABLE" and "NON COMPARABLE avec la référence" in text and "+2" in text
    assert NOMINAL_SECONDS_PER_MOVE == 0.80


def test_resume_continues_the_same_state():
    w = SpeedWatch(0.8)
    feed(w, [0.8] * 30)
    saved = json.loads(json.dumps(w.state))  # aller-retour par le fichier JSON de résultats
    again = SpeedWatch(1.7, state=saved)  # le nouveau bench (1,7) ne change pas la référence du premier lancement
    assert again.reference == 0.8 and again.state["games"] == 30
    feed(again, [0.8] * 10)
    assert again.state["games"] == 40 and len(again.recent) == 20


def test_unfollowed_measurement():
    assert verdict(None)[0] == "non suivie" and verdict({"games": 0})[0] == "non suivie"


# --------------------------------------------------------------------------- dans match_stockfish.py


class FakePool:
    """Remplace multiprocessing.Pool : rend les parties préparées, dans l'ordre."""

    def __init__(self, speeds):
        self.speeds = list(speeds)

    def imap_unordered(self, fn, tasks):
        for task, v in zip(tasks, self.speeds):
            yield {**task, "opening_name": "Essai", "score": 1.0, "plies": 60, "termination": "mat", "moves": "e2e4 e7e5",
                   "ai_sec_per_move": v, "ai_tb_moves": 0, "seconds": 10.0, "checkpoint_step": 1, "stockfish": "SF"}


@pytest.fixture(autouse=True)
def quick_diagnostics(monkeypatch):
    """Les bandeaux interrogent le système (processus gourmands 1,5 s, placement 2 s) : remplacés par des valeurs fixes."""
    monkeypatch.setattr(ms.cpu_guard, "busy_processes", lambda *a, **k: [])
    monkeypatch.setattr(ms.cpu_guard, "measure_placement", lambda *a, **k: None)

    class NoProbe:  # la vraie sonde attend 20 s avant de mesurer
        def __init__(self, *a, **k):
            pass

        def start(self):
            return self

        def finish(self, timeout=0):
            return None

    monkeypatch.setattr(ms.cpu_guard, "PlacementProbe", NoProbe)


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "OUT_DIR", tmp_path)
    return ms.Results("v2/essai_vitesse")


def test_play_tasks_records_rolling_speed_and_prints_banner(results, capsys):
    results.watch = SpeedWatch(0.8)
    tasks = [{"level": 1320 + i, "round": 0, "opening": 0, "ai_white": True, "phase": 1} for i in range(40)]
    pool = FakePool([0.8] * 25 + [2.4] * 15)
    ms.play_tasks(pool, results, tasks, "essai")
    out = capsys.readouterr().out
    assert out.count("ATTENTION VITESSE") == 1 and "moy.20" in out
    assert "!" * 100 in out
    saved = json.loads(results.path.read_text(encoding="utf-8"))
    assert saved["speed"]["games"] == 40 and saved["speed"]["alert_games"] > 0
    assert all("ai_rolling_s_per_move" in g for g in saved["games"])
    assert saved["games"][0]["ai_rolling_s_per_move"] == 0.8
    assert saved["games"][-1]["ai_rolling_s_per_move"] > 1.0
    assert [s[:2] for s in saved["speed"]["series"]] == [[20, 0.8], [40, saved["speed"]["series"][1][1]]]


def test_play_tasks_without_watch_is_unchanged(results, capsys):
    tasks = [{"level": 1320, "round": 0, "opening": i, "ai_white": True, "phase": 1} for i in range(3)]
    ms.play_tasks(FakePool([0.8] * 3), results, tasks, "essai")
    assert "moy." not in capsys.readouterr().out
    assert "speed" not in results.data and all("ai_rolling_s_per_move" not in g for g in results.games)


def test_start_speed_watch_uses_unrecorded_bench_games(results, capsys):
    watch = ms.start_speed_watch(FakePool([0.8, 0.82, 0.78, 0.8]), results, 4)
    assert watch.reference == pytest.approx(0.8) and results.watch is watch and not results.games
    assert "ATTENTION VITESSE AU DÉPART" not in capsys.readouterr().out
    slow = ms.Results("v2/essai_vitesse_lent")
    ms.start_speed_watch(FakePool([2.5] * 4), slow, 4)
    out = capsys.readouterr().out
    assert "ATTENTION VITESSE AU DÉPART" in out and "2.50 s par coup contre 0.80" in out


def test_final_report_states_whether_the_measurement_was_stable(results, capsys):
    results.watch = SpeedWatch(0.8, abort_after=None)  # arrêt automatique testé à part
    levels = [1320, 1600, 1900, 2200]
    tasks = [{"level": levels[i % 4], "round": 0, "opening": i, "ai_white": i % 2 == 0, "phase": 1} for i in range(60)]
    ms.play_tasks(FakePool([0.8] * 30 + [2.5] * 30), results, tasks, "essai")
    capsys.readouterr()
    ms.report(results)
    out = capsys.readouterr().out
    assert "MESURE NON STABLE" in out and "ATTENTION : " in out
    stable = ms.Results("v2/essai_stable")
    stable.watch = SpeedWatch(0.8)
    ms.play_tasks(FakePool([0.8] * 60), stable, tasks, "essai")
    capsys.readouterr()
    ms.report(stable)
    assert "STABLE du début à la fin" in capsys.readouterr().out


def test_old_results_without_speed_still_report(capsys):
    ms.report(ms.Results("v2/kaggle_sup"))
    assert "vitesse non suivie" in capsys.readouterr().out


# --------------------------------------------------------------------------- batterie et état de charge (piste du bridage pendant la recharge)


def test_classify_power_states():
    assert classify_power(0, 0, 87) == {"percent": 87, "state": "sur batterie"}
    assert classify_power(1, 8, 71) == {"percent": 71, "state": "en charge"}
    assert classify_power(1, 0, 100) == {"percent": 100, "state": "chargée"}
    assert classify_power(1, 9, 100)["state"] == "en charge"  # le drapeau de charge prime
    assert classify_power(1, 0, 80) == {"percent": 80, "state": "secteur sans charge"}
    assert classify_power(1, 128, 255) == {"percent": None, "state": "secteur (pas de batterie)"}
    assert classify_power(255, 255, 255) == {"percent": None, "state": "inconnu"}


def test_read_power_returns_a_known_state_on_this_machine():
    p = read_power()
    if p is None:
        pytest.skip("alimentation non lisible ici")
    assert p["state"] in {"sur batterie", "en charge", "chargée", "secteur sans charge", "secteur (pas de batterie)",
                          "inconnu"}
    assert p["percent"] is None or 0 <= p["percent"] <= 100


def test_power_log_records_changes_episodes_and_series():
    w = SpeedWatch(0.8)
    states = [{"percent": 71 + i // 10, "state": "en charge"} for i in range(30)]
    states += [{"percent": 100, "state": "chargée"}] * 30
    speeds = [2.4] * 30 + [0.8] * 30  # lent pendant la charge, normal une fois chargée
    infos = [w.add(s, p) for s, p in zip(speeds, states)]
    log = w.state["power"]
    assert log["first"]["state"] == "en charge" and log["first"]["percent"] == 71 and log["first"]["game"] == 1
    assert log["last"] == {"game": 60, "percent": 100, "state": "chargée"}
    assert [(c["state"], c["game"]) for c in log["changes"]] == [("en charge", 1), ("chargée", 31)]  # un seul changement
    assert infos[0]["power"]["percent"] == 71
    ep = w.state["episodes"][0]
    assert ep["power_start"] == {"percent": 71 + 14 // 10, "state": "en charge"}  # l'alerte commence à la 15e partie
    assert ep["power_end"]["state"] == "chargée"  # elle finit une fois chargée
    assert [s[2:] for s in w.state["series"]] == [[72, "en charge"], [100, "chargée"], [100, "chargée"]]
    text = describe_power(w.state)
    assert "départ 71 % (en charge)" in text and "fin 100 % (chargée)" in text and "chargée 100 % (partie 31)" in text
    assert "commencé en charge à 72 %" in text
    assert "alimentation : départ" in verdict(w.state)[1].splitlines()[-1]


def test_power_is_optional_and_old_state_still_works():
    w = SpeedWatch(0.8)
    feed(w, [0.8] * 25)
    assert "power" not in w.state and "alimentation non relevée" in describe_power(w.state)
    assert w.state["series"][0] == [20, 0.8, None, None]
    legacy = {"games": 3, "alert_games": 0, "reference_s_per_move": 0.8, "nominal_s_per_move": 0.8, "window": 20,
              "tolerance": 0.25, "episodes": [], "series": [[20, 0.8]], "worst": None}  # fichier d'avant cette version
    state, text = verdict(legacy)
    assert state == "stable" and "alimentation non relevée" in text


def test_play_tasks_records_battery_and_charge_state_in_every_game(results, monkeypatch, capsys):
    readings = iter([{"percent": 80 + i // 5, "state": "en charge" if i < 10 else "chargée"} for i in range(40)])
    monkeypatch.setattr(ms, "read_power", lambda: next(readings))
    results.watch = SpeedWatch(0.8)
    tasks = [{"level": 1320 + i, "round": 0, "opening": 0, "ai_white": True, "phase": 1} for i in range(10)]
    ms.play_tasks(FakePool([0.8] * 10), results, tasks, "essai")
    out = capsys.readouterr().out
    assert "| 80 % en charge" in out and "| 81 % en charge" in out
    saved = json.loads(results.path.read_text(encoding="utf-8"))
    assert [(g["battery_percent"], g["power_state"]) for g in saved["games"]][:6] == [(80, "en charge")] * 5 + [(81, "en charge")]
    assert saved["speed"]["power"]["first"]["percent"] == 80 and saved["speed"]["power"]["last"]["game"] == 10


def test_speed_banner_and_start_bench_show_the_power_state(results, monkeypatch, capsys):
    monkeypatch.setattr(ms, "read_power", lambda: {"percent": 71, "state": "en charge"})
    ms.start_speed_watch(FakePool([0.8] * 4), results, 4)
    assert "Alimentation au départ : 71 % en charge" in capsys.readouterr().out
    assert results.data["speed"]["benches"][0]["battery_percent"] == 71
    tasks = [{"level": 1320 + i, "round": 0, "opening": 0, "ai_white": True, "phase": 1} for i in range(30)]
    ms.play_tasks(FakePool([0.8] * 15 + [2.4] * 15), results, tasks, "essai")
    out = capsys.readouterr().out
    assert "Alimentation à cet instant : 71 % en charge" in out


# --------------------------------------------------------------------------- arrêt automatique (20 parties d'affilée hors tolérance)


def test_abort_fires_on_the_20th_consecutive_game_out_of_tolerance_and_not_before():
    w = SpeedWatch(0.8)  # abort_after = 20 par défaut
    infos = feed(w, [0.8] * 40 + [2.4] * 40)
    first_alert = next(i for i, x in enumerate(infos) if x["alert"])
    aborts = [i for i, x in enumerate(infos) if x["abort"]]
    assert aborts[0] == first_alert + 19  # la 20e partie d'affilée hors tolérance
    assert not any(x["abort"] for x in infos[:aborts[0]])
    assert w.state["aborted"]["consecutive_games"] == 20 and w.state["aborted"]["game"] == aborts[0] + 1
    state, text = verdict(w.state)
    assert state == "NON STABLE" and "ARRÊTÉE AUTOMATIQUEMENT" in text and "20 parties d'affilée" in text


def test_recovery_resets_the_consecutive_count():
    """La moyenne glissante a de la mémoire : 6 parties 3 fois trop lentes la laissent hors tolérance 23 parties. L'arrêt compte
    donc les parties d'affilée où la moyenne ET la partie elle-même sont hors tolérance : 6, puis retour à la normale, pas 23."""
    w = SpeedWatch(0.8)
    infos = feed(w, [0.8] * 30 + [3.0] * 6 + [0.8] * 40 + [3.0] * 6 + [0.8] * 30)
    streaks, run = [], 0
    for x in infos:
        if x["alert"]:
            run += 1
        elif run:
            streaks.append(run)
            run = 0
    assert streaks[0] >= 20  # la moyenne glissante, elle, reste bien hors tolérance plus de 20 parties...
    assert not any(x["abort"] for x in infos) and "aborted" not in w.state  # ...mais cela n'arrête pas la mesure
    assert w.state["consecutive_alert"] == 0


def test_a_blip_of_a_few_games_never_stops_but_20_slow_games_in_a_row_do():
    blip = SpeedWatch(0.8)
    assert not any(x["abort"] for x in feed(blip, [0.8] * 30 + [2.4] * 5 + [0.8] * 60))
    w = SpeedWatch(0.8)
    infos = feed(w, [0.8] * 30 + [2.4] * 40)
    first_stop = next(i for i, x in enumerate(infos) if x["abort"])
    assert first_stop == 30 + 2 + 19  # la 3e partie lente fait sortir la moyenne ; 20 parties d'affilée à partir de là
    fast = SpeedWatch(1.6)  # plus rapide que la référence, dans la durée : arrêt aussi
    assert any(x["abort"] for x in feed(fast, [1.6] * 25 + [0.8] * 60))


def test_abort_can_be_disabled_and_a_resume_starts_from_zero():
    w = SpeedWatch(0.8, abort_after=None)
    assert not any(x["abort"] for x in feed(w, [0.8] * 20 + [3.0] * 60))
    a = SpeedWatch(0.8)
    feed(a, [0.8] * 20 + [3.0] * 60)
    assert a.state["aborted"]
    saved = json.loads(json.dumps(a.state))
    again = SpeedWatch(0.8, state=saved)  # relance de la même commande
    assert again.state["consecutive_alert"] == 0 and "aborted" not in again.state and len(again.state["past_aborts"]) == 1
    assert not again.add(0.8)["abort"]


def test_reference_measurements_would_not_have_been_stopped():
    """La mesure de référence a un épisode de 19 parties hors tolérance (parties 522-540) : sous le seuil de 20."""
    for name in ("kaggle_sup", "kaggle_sup_r2"):
        path = Path(__file__).resolve().parents[1] / "docs" / "stockfish" / "v2" / f"{name}.json"
        if not path.exists():
            pytest.skip("résultats absents")
        speeds = [g["ai_sec_per_move"] for g in json.loads(path.read_text(encoding="utf-8"))["games"]]
        w = SpeedWatch(sum(speeds[:4]) / 4)
        assert not any(i["abort"] for i in feed(w, speeds)), name


def test_play_tasks_stops_the_measurement_with_a_clear_message_and_keeps_the_games(results, capsys):
    results.watch = SpeedWatch(0.8)
    tasks = [{"level": 1320 + i, "round": 0, "opening": 0, "ai_white": True, "phase": 1} for i in range(120)]
    with pytest.raises(ms.SpeedAbort) as stop:
        ms.play_tasks(FakePool([0.8] * 30 + [2.5] * 90), results, tasks, "essai")
    msg = str(stop.value)
    assert "MESURE ARRÊTÉE AUTOMATIQUEMENT" in msg and "20 parties d'affilée" in msg and "relancer la même commande" in msg
    assert "cœurs E" in msg and "processus gourmands" in msg and "alimentation (" in msg
    saved = json.loads(results.path.read_text(encoding="utf-8"))
    played = len(saved["games"])
    assert 30 < played < 120 and saved["speed"]["aborted"]["game"] == played  # la partie fatale est enregistrée aussi
    assert saved["speed"]["consecutive_alert"] >= 20


def test_main_exits_with_code_3_on_abort(tmp_path, monkeypatch, capsys):
    """Le point d'entrée convertit l'arrêt automatique en code de sortie 3 (parties gardées)."""
    monkeypatch.setattr(ms, "OUT_DIR", tmp_path)
    monkeypatch.setattr(ms, "resolve_syzygy", lambda path: {"syzygy": None})

    class Pool:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ms, "make_pool", lambda *a, **k: Pool())
    monkeypatch.setattr(ms, "startup_checks", lambda: {})
    monkeypatch.setattr(ms, "environment", lambda args: {"power": {"on_ac": True, "power_mode": "meilleures performances"}})

    def boom(pool, results, games):
        raise ms.SpeedAbort("MESURE ARRÊTÉE AUTOMATIQUEMENT (test)")

    monkeypatch.setattr(ms, "start_speed_watch", boom)
    monkeypatch.setattr(sys, "argv", ["match_stockfish.py", "run", "--tag", "essai_arret"])
    with pytest.raises(SystemExit) as e:
        ms.main()
    assert e.value.code == 3 and "MESURE ARRÊTÉE AUTOMATIQUEMENT (test)" in capsys.readouterr().out


def test_start_bench_reports_the_core_placement(results, monkeypatch, capsys):
    class Probe:
        def __init__(self, *a, **k):
            pass

        def start(self):
            return self

        def finish(self, timeout=0):
            return {"p_cores_busy": 0.4, "e_cores_busy": 4.8, "e_share": 0.923, "total_cores_busy": 5.2}

    monkeypatch.setattr(ms.cpu_guard, "PlacementProbe", Probe)
    watch = ms.start_speed_watch(FakePool([0.8] * 4), results, 4)
    out = capsys.readouterr().out
    assert "ATTENTION PLACEMENT" in out and "cœurs E" in out and "92%" in out
    assert watch.state["placements"][0]["state"] == "E_DOMINANT"
