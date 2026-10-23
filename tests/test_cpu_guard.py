"""cpu_guard : protection contre EcoQoS, placement sur les cœurs P/E, processus concurrents."""

import os
import struct
import subprocess
import sys
import time

import pytest

from chesszero import cpu_guard as cg

windows_only = pytest.mark.skipif(not cg.IS_WINDOWS, reason="API Windows")


def cpu_set_buffer(efficiency: list[int]) -> bytes:
    """Tampon au format de GetSystemCpuSetInformation : une entrée de 32 octets par processeur logique."""
    out = b""
    for i, eff in enumerate(efficiency):
        entry = struct.pack("<II", 32, 0) + struct.pack("<IHBBBBB", 256 + i, 0, i, i // 2, 0, 0, eff) + b"\x00" * 13
        assert len(entry) == 32
        out += entry
    return out


def test_layout_decoded_from_cpu_set_information():
    # 6 cœurs P avec deux threads (classe 1) puis 4 cœurs E (classe 0), comme le portable : logiques 0-11 puis 12-15
    layout = cg.layout_from_cpu_sets(cpu_set_buffer([1] * 12 + [0] * 4))
    assert layout == {"p": list(range(12)), "e": [12, 13, 14, 15], "hybrid": True}
    homogeneous = cg.layout_from_cpu_sets(cpu_set_buffer([0] * 8))
    assert homogeneous == {"p": list(range(8)), "e": [], "hybrid": False}
    assert cg.layout_from_cpu_sets(b"") == {"p": [], "e": [], "hybrid": False}
    three = cg.layout_from_cpu_sets(cpu_set_buffer([2, 2, 1, 1, 0, 0]))  # trois classes : la plus haute = P, le reste = E
    assert three["p"] == [0, 1] and three["e"] == [2, 3, 4, 5]


def test_placement_share_and_verdict():
    layout = {"p": [0, 1, 2, 3], "e": [4, 5, 6, 7], "hybrid": True}
    on_p = cg.placement_from_busy([0.9, 0.9, 0.9, 0.9, 0.1, 0.0, 0.0, 0.0], layout)
    assert on_p == {"p_cores_busy": 3.6, "e_cores_busy": 0.1, "e_share": 0.027, "total_cores_busy": 3.7}
    assert cg.placement_verdict(on_p)[0] == "ok"
    on_e = cg.placement_from_busy([0.0, 0.1, 0.0, 0.1, 0.9, 0.9, 0.9, 0.9], layout)  # la nuit : cœurs P endormis
    state, text = cg.placement_verdict(on_e)
    assert state == "E_DOMINANT" and "CŒURS E" in text and "3 fois" in text
    assert cg.placement_verdict(cg.placement_from_busy([0.0] * 8, layout))[0] == "peu chargé"
    assert cg.placement_verdict(None)[0] == "inconnu"
    mixed = cg.placement_from_busy([0.9, 0.9, 0.9, 0.0, 0.9, 0.9, 0.0, 0.0], layout)  # 60 % ou moins sur E : pas d'alerte
    assert cg.placement_verdict(mixed)[0] == "ok"


def test_measure_placement_uses_busy_time_between_two_samples(monkeypatch):
    layout = {"p": [0, 1], "e": [2, 3], "hybrid": True}
    samples = iter([[(0, 0)] * 4, [(10, 100), (10, 100), (90, 100), (90, 100)]])  # (occupé, total) cumulés
    monkeypatch.setattr(cg, "_cpu_times", lambda: next(samples))
    slept = []
    m = cg.measure_placement(5.0, layout, sleep=slept.append)
    assert slept == [5.0] and m["p_cores_busy"] == 0.2 and m["e_cores_busy"] == 1.8 and m["e_share"] == 0.9
    assert cg.measure_placement(5.0, {"p": [0], "e": [], "hybrid": False}, sleep=slept.append) is None  # pas hybride


def test_cores_by_name_groups_a_many_process_browser():
    names = {1: "chrome.exe", 2: "chrome.exe", 3: "chrome.exe", 4: "python.exe", 5: "dwm.exe"}
    found = cg.cores_by_name({1: 1.0, 2: 0.9, 3: 0.8, 4: 0.3, 5: 0.1}, names, threshold=2.0)
    assert found == [{"name": "chrome.exe", "cores": 2.7, "max_single": 1.0, "pids": [1, 2, 3]}]  # somme > 2 cœurs
    one = cg.cores_by_name({1: 2.6, 4: 0.3}, names, threshold=2.0)  # un seul processus au-dessus du seuil
    assert one[0]["name"] == "chrome.exe" and one[0]["max_single"] == 2.6
    assert cg.cores_by_name({4: 0.3, 5: 0.1}, names, threshold=2.0) == []
    assert "chrome.exe ≈ 2.7 cœurs (3 processus)" in cg.describe_busy(found)
    assert "aucun autre programme" in cg.describe_busy([])


@windows_only
def test_ecoqos_is_disabled_on_this_process_and_read_back():
    assert cg.disable_ecoqos() is True
    assert cg.ecoqos_state() == (1, 0)  # contrôle pris (1), bridage désactivé (0)
    assert cg.disable_ecoqos(2**22 + 1) is False  # processus inexistant : échec propre, pas d'exception


@windows_only
def test_protect_covers_children_created_later():
    info = cg.protect(interval=0.5)
    assert info == {"ecoqos_disabled": True, "watching": True}
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert child.pid in cg.descendants()
        deadline = time.time() + 15
        while cg.ecoqos_state(child.pid) != (1, 0) and time.time() < deadline:
            time.sleep(0.2)
        assert cg.ecoqos_state(child.pid) == (1, 0)
        assert cg.protected_children().get(child.pid) is True
    finally:
        child.kill()
        child.wait()
    assert cg.protect(interval=0.5)["watching"] is True  # idempotent


@windows_only
def test_core_layout_matches_this_machine():
    layout = cg.core_layout()
    assert layout["p"], "au moins des cœurs P"
    assert sorted(layout["p"] + layout["e"]) == list(range(os.cpu_count()))
    assert not set(layout["p"]) & set(layout["e"])


@windows_only
def test_busy_processes_finds_competing_load_and_ignores_own_tree():
    loops = [subprocess.Popen([sys.executable, "-c", "import time\nt = time.time()\nwhile time.time() - t < 8: pass"])
             for _ in range(3)]
    try:
        time.sleep(1.5)
        mine = cg.busy_processes(window=2.0, threshold=2.0, exclude_tree=True)  # mes enfants ne comptent pas
        assert not any(g["name"].lower().startswith("python") and set(g["pids"]) <= {p.pid for p in loops} for g in mine)
        everything = cg.busy_processes(window=2.0, threshold=2.0, exclude_tree=False)
        py = [g for g in everything if g["name"].lower() == "python.exe"]
        assert py and py[0]["cores"] > 2.0 and {p.pid for p in loops} <= set(py[0]["pids"])
    finally:
        for p in loops:
            p.kill()
            p.wait()


@windows_only
def test_real_placement_measurement_runs():
    layout = cg.core_layout()
    if not layout["hybrid"]:
        pytest.skip("processeur non hybride")
    m = cg.measure_placement(1.0, layout)
    assert m is not None and 0 <= m["e_share"] <= 1 and m["total_cores_busy"] >= 0


def test_everything_is_a_noop_elsewhere(monkeypatch):
    monkeypatch.setattr(cg, "IS_WINDOWS", False)
    assert cg.disable_ecoqos() is False and cg.ecoqos_state() is None
    assert cg.protect() == {"ecoqos_disabled": False, "watching": False}
    assert cg.core_layout() == {"p": [], "e": [], "hybrid": False}
    assert cg.busy_processes(window=0.0) == [] and cg.process_table() == []
    assert cg.PlacementProbe().start().finish() is None


LONG_SCRIPTS = ["match_stockfish", "ab_value_target", "analyse_generation", "train", "train_selfplay", "selfplay", "arena",
                "bench_selfplay", "selfplay_session", "alphazero_session", "eval_value_target", "prepare_data",
                "prepare_book", "match", "match_sims", "bench_mcts", "bench_inference"]


@pytest.mark.parametrize("name", LONG_SCRIPTS)
def test_every_long_local_script_is_protected(name):
    """Un nouveau script de calcul long doit appeler cpu_guard.protect() : cette liste l'impose aux scripts existants."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py").read_text(encoding="utf-8")
    assert "cpu_guard" in text and ("cpu_guard.protect()" in text or name == "match_stockfish")
    if name == "match_stockfish":
        assert "startup_checks()" in text and "cpu_guard.disable_ecoqos(" in text  # processus principal + workers + Stockfish


def test_worker_processes_disable_ecoqos_themselves():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert "disable_ecoqos()" in (root / "scripts" / "selfplay.py").read_text(encoding="utf-8")
    assert "disable_ecoqos()" in (root / "src" / "chesszero" / "arena.py").read_text(encoding="utf-8")
