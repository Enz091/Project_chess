"""Étape 7e : découverte des entrées Kaggle, mode check, mémoire, commit figé du notebook."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import chesszero.kaggle_inputs as ki
from chesszero.loop import LoopStop, Orchestrator, actions, read_loop_state
from chesszero.memwatch import MemoryWatch, describe_memory, read_memory_gb
from chesszero.selfplay import SelfPlayConfig, list_shards
from chesszero.selfplay_train import SelfPlayTrainer

from test_loop import CONFIGS, LOOP, TRAIN, world  # noqa: F401  (world : fixture de test_loop, sortie de self-play seul)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_selfplay.ipynb"


@pytest.fixture(autouse=True)
def small_networks(monkeypatch):
    monkeypatch.setattr(ki, "MIN_NETWORK_BYTES", 1000)  # les faux réseaux des tests sont petits


def fake_tree(root: Path, world, *, init=True, human=True, outputs=("selfplay_only",), val=True) -> Path:
    """Arborescence /kaggle/input simulée : Dataset du réseau, Dataset Lichess, sorties de notebooks."""
    inp = root / "input"
    inp.mkdir()
    if init:
        (inp / "chesszero-kaggle-sup").mkdir()
        (inp / "chesszero-kaggle-sup" / "best.pt").write_bytes(Path(world["init"]).read_bytes())
        (inp / "decoy").mkdir()
        (inp / "decoy" / "autre.pt").write_bytes(b"x" * 5000)  # un autre .pt, au mauvais SHA-256
    if human:
        h = inp / "lichess-24m"
        h.mkdir()
        (h / "train_00000.npz").write_bytes(b"x")
        if val:
            (h / "val_00000.npz").write_bytes(b"x")
        (h / "manifest.json").write_text(json.dumps({"positions": {"train": 900, "val": 100}}), encoding="utf-8")
    for k, kind in enumerate(outputs):
        base = inp / f"notebooks/compte/sortie-{k}" if k % 2 else inp / f"sortie-{k}"  # deux dispositions de montage
        shutil_copy_selfplay(world["s0"] / "selfplay", base / "selfplay")
        if kind.startswith("loop"):
            n = int(kind.split(":")[1])
            SelfPlayTrainer(base / "sp_train", TRAIN, "cpu", world["init"], log=lambda m: None)
            (base / "loop").mkdir()
            state = {"format": 1, "sessions": [{"n": i} for i in range(1, n + 1)], "controls": {"done": True},
                     "estimates": {}, "events": [], "failures": {}, "phase": None}
            (base / "loop" / "loop_state.json").write_text(json.dumps(state), encoding="utf-8")
    return inp


def shutil_copy_selfplay(src: Path, dst: Path) -> None:
    import shutil

    shutil.copytree(src, dst)


SHA = lambda world: ki.sha256_of(world["init"])  # noqa: E731


# --------------------------------------------------------------------------- découverte


def test_discovery_finds_inputs_by_content(tmp_path, world):
    inp = fake_tree(tmp_path, world)
    found = ki.discover(inp, SHA(world))
    assert found["problems"] == [] and found["init"].name == "best.pt" and found["human"].name == "lichess-24m"
    assert any("1,000 positions" in n or "1000 positions" in n for n in found["notes"])
    assert len(found["previous"]) == 1 and found["previous"][0]["kind"] == "selfplay_only"
    path, problems = ki.select_previous(found, 0)
    assert problems == [] and path == found["previous"][0]["path"]


def test_zero_previous_outputs_is_a_clear_refusal(tmp_path, world):
    inp = fake_tree(tmp_path, world, outputs=())
    found = ki.discover(inp, SHA(world))
    assert found["problems"] == [] and found["previous"] == []
    path, problems = ki.select_previous(found, 0)
    assert path is None and len(problems) == 1 and "Aucune sortie précédente" in problems[0] and "Your Work" in problems[0]


def test_two_previous_outputs_is_refused_and_both_are_listed(tmp_path, world):
    inp = fake_tree(tmp_path, world, outputs=("selfplay_only", "loop:2"))
    found = ki.discover(inp, SHA(world))
    assert {p["kind"] for p in found["previous"]} == {"selfplay_only", "loop"}
    path, problems = ki.select_previous(found, 0)
    assert path is None and "EXACTEMENT une" in problems[0]
    assert "self-play seul" in problems[0] and "boucle, session n°2" in problems[0]
    assert all(str(p["path"]) in problems[0] for p in found["previous"])


def test_expected_session_must_match_the_found_output(tmp_path, world):
    inp = fake_tree(tmp_path, world, outputs=("loop:3",))
    found = ki.discover(inp, SHA(world))
    path, problems = ki.select_previous(found, 1)
    assert path is None and "session de boucle n°3" in problems[0] and "EXPECTED_SESSION = 3" in problems[0]
    assert ki.select_previous(found, 3)[1] == []
    sp_only = ki.discover(fake_tree(tmp_path / "bis", world) if (tmp_path / "bis").mkdir() is None else None, SHA(world))
    path, problems = ki.select_previous(sp_only, 2)
    assert path is None and "self-play seul" in problems[0] and "EXPECTED_SESSION = 2" in problems[0]


def test_wrong_network_missing_lichess_and_missing_validation_are_refused(tmp_path, world):
    inp = fake_tree(tmp_path, world)
    assert "Réseau de départ introuvable" in " ".join(ki.discover(inp, "0" * 64)["problems"])
    (tmp_path / "b").mkdir()
    assert "Données Lichess introuvables" in " ".join(ki.discover(fake_tree(tmp_path / "b", world, human=False),
                                                                  SHA(world))["problems"])
    (tmp_path / "c").mkdir()
    assert "aucun shard de validation" in " ".join(ki.discover(fake_tree(tmp_path / "c", world, val=False),
                                                              SHA(world))["problems"])
    assert "n'existe pas" in ki.discover(tmp_path / "absent", SHA(world))["problems"][0]


def test_incomplete_shards_are_not_an_output(tmp_path, world):
    """Un .npz sans .json (écriture interrompue) ne fait pas une sortie précédente."""
    inp = fake_tree(tmp_path, world, outputs=())
    d = inp / "orphelin" / "selfplay"
    d.mkdir(parents=True)
    (d / "s_w0_00000.npz").write_bytes(b"x")
    assert ki.discover(inp, SHA(world))["previous"] == []


# --------------------------------------------------------------------------- mode check de l'orchestrateur


def orchestrator(world, work, resume, expect=0, configs=CONFIGS):
    return Orchestrator(work, "smoke", world["init"], executor=None, minutes=30, expect_session=expect,
                        resume_from=resume, configs=configs)


def test_check_inputs_describes_the_start_and_writes_nothing(tmp_path, world):
    work = tmp_path / "working"
    lines = orchestrator(world, work, world["s0"]).check_inputs(human=None, syzygy="")
    text = "\n".join(lines)
    assert "SELF-PLAY SEUL" in text and "génération 0" in text and "EXPECTED_SESSION = 0" in text
    assert "Réseau de départ" in text and "pas 0" in text and "Tables Syzygy" in text
    assert not work.exists()  # rien d'écrit : pas même le dossier de travail
    assert all(not p.name.startswith(("loop", "sp_train")) for p in world["s0"].iterdir())


def test_check_inputs_refusals(tmp_path, world):
    with pytest.raises(LoopStop, match="Tampon trop petit"):
        orchestrator(world, tmp_path / "w1", world["s0"], configs={**CONFIGS, "train": TRAIN.__class__(
            **{**TRAIN.__dict__, "min_buffer_positions": 10**9})}).check_inputs()
    with pytest.raises(LoopStop, match="Mauvaise entrée"):
        orchestrator(world, tmp_path / "w2", world["s0"], expect=2).check_inputs()
    with pytest.raises(LoopStop, match="Aucun tampon"):
        (tmp_path / "vide").mkdir()
        orchestrator(world, tmp_path / "w3", tmp_path / "vide").check_inputs()
    shutil_copy_selfplay(world["s0"] / "selfplay", tmp_path / "occupe" / "selfplay")  # dossier de travail déjà utilisé
    with pytest.raises(LoopStop, match="n'est pas vide"):
        orchestrator(world, tmp_path / "occupe", world["s0"]).check_inputs()


def test_check_inputs_on_a_loop_output(tmp_path, world):
    inp = fake_tree(tmp_path, world, outputs=("loop:2",))
    prev = next(p for p in inp.iterdir() if p.name.startswith("sortie"))
    lines = orchestrator(world, tmp_path / "w", prev, expect=2).check_inputs()
    assert "session de boucle n°2" in "\n".join(lines)
    with pytest.raises(LoopStop, match="attendu la session n°1"):
        orchestrator(world, tmp_path / "w2", prev, expect=1).check_inputs()


def test_check_inputs_command_line_exit_codes(tmp_path, world):
    cmd = [sys.executable, str(ROOT / "scripts" / "alphazero_session.py"), "--check-inputs", "--config", "smoke",
           "--work", str(tmp_path / "w"), "--resume-from", str(world["s0"]), "--minutes", "30", "--init",
           str(world["init"]), "--syzygy", "", "--human", str(tmp_path)]
    res = subprocess.run(cmd + ["--expect-session", "5"], capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 4 and "REFUS DE DÉMARRER" in res.stdout and "Mauvaise entrée" in res.stdout
    res = subprocess.run(cmd + ["--expect-session", "0", "--expect-init-sha256", "0" * 64], capture_output=True,
                         text=True, encoding="utf-8")
    assert res.returncode == 4 and "SHA-256 attendu" in res.stdout
    assert not (tmp_path / "w").exists()


# --------------------------------------------------------------------------- mémoire


def test_memory_watch_tracks_minimum_per_phase_and_alerts():
    values = iter([(20.0, 31.0), (12.5, 31.0), (3.2, 31.0), (9.0, 31.0), (3.9, 31.0)])
    phase = ["selfplay"]
    w = MemoryWatch(interval=999, alert_gb=4.0, reader=lambda: next(values), phase=lambda: phase[0])
    w.sample()
    w.sample()
    phase[0] = "train"
    w.sample()
    w.sample()
    phase[0] = "arena_gate"
    w.sample()
    s = w.summary()
    assert s["samples"] == 5 and s["total_gb"] == 31.0 and s["first_available_gb"] == 20.0
    assert s["min_available_gb"] == 3.2 and s["min_phase"] == "train"
    assert s["by_phase"] == {"selfplay": 12.5, "train": 3.2, "arena_gate": 3.9}
    assert s["alert_samples"] == 2 and [a["phase"] for a in s["alerts"]] == ["train", "arena_gate"]
    text = describe_memory(s)
    assert "3.2 Go pendant la phase « train »" in text and "2 relevé(s) sous 4 Go" in text
    assert describe_memory(None) == "mémoire : non relevée"


def test_memory_watch_thread_samples_and_stops():
    n = {"calls": 0}

    def reader():
        n["calls"] += 1
        return 20.0 - n["calls"] * 0.01, 31.0

    w = MemoryWatch(interval=0.01, reader=reader)
    w.start()
    deadline = __import__("time").time() + 5
    while n["calls"] < 5 and __import__("time").time() < deadline:
        __import__("time").sleep(0.01)
    w.stop()
    calls = n["calls"]
    assert calls >= 5 and w.summary()["samples"] == calls
    __import__("time").sleep(0.05)
    assert n["calls"] == calls  # le thread est bien arrêté


def test_memory_reader_returns_plausible_values_here():
    got = read_memory_gb()
    assert got is not None and 0 < got[0] <= got[1]


def test_low_memory_raises_an_action_and_the_report_shows_the_minimum():
    ram = {"interval_seconds": 30, "alert_gb": 4.0, "samples": 9, "total_gb": 31.0, "first_available_gb": 29.0,
           "min_available_gb": 2.5, "min_phase": "train", "by_phase": {"train": 2.5}, "alert_samples": 3, "alerts": []}
    base = {"view": {}, "end_reason": "fin", "notes": [], "v2": [], "selfplay": [], "alerts": [], "gen2_bias": False,
            "prev_sps": None, "sps": None, "session": 1, "timings": []}
    acts = actions({**base, "ram": ram}, LOOP)
    assert any(a.startswith("MÉMOIRE BASSE : 2.5 Go") and "train" in a for a in acts)
    assert not any("MÉMOIRE" in a for a in actions({**base, "ram": {**ram, "min_available_gb": 12.0}}, LOOP))
    assert not any("MÉMOIRE" in a for a in actions(base, LOOP))


def test_orchestrator_records_memory_in_state_and_report(tmp_path, world, monkeypatch):
    """Session complète (faux exécuteur de test_loop) : relevés dans loop_state.json et section « Mémoire » du rapport."""
    from test_loop import FakeExecutor

    values = iter([(25.0, 31.0)] + [(3.0, 31.0)] * 100)
    watch = MemoryWatch(interval=999, alert_gb=4.0, reader=lambda: next(values), phase=lambda: None)
    work = tmp_path / "w"
    ex = FakeExecutor(world, work)
    orch = Orchestrator(work, "smoke", world["init"], ex, 12, 0, resume_from=world["s0"], clock=lambda: ex.clock[0],
                        log=lambda m: None, root=ROOT, configs=CONFIGS, memwatch=watch)
    watch.phase = lambda: orch.state.get("phase")
    result = orch.run()
    state = read_loop_state(work / "loop" / "loop_state.json")
    ram = state["sessions"][-1]["ram"]
    assert ram["samples"] >= 2 and ram["min_available_gb"] == 3.0
    assert "## Mémoire" in result["text"] and "minimum 3.0 Go" in result["text"]
    assert any(a.startswith("MÉMOIRE BASSE") for a in result["actions"])


# --------------------------------------------------------------------------- notebook


def notebook_sources() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def test_notebook_pins_an_exact_commit_that_exists_in_this_repository():
    settings = notebook_sources()[1]
    m = re.search(r'^COMMIT = "([^"]*)"', settings, flags=re.M)
    assert m, "COMMIT absent de la cellule de réglages"
    commit = m.group(1)
    assert re.fullmatch(r"[0-9a-f]{40}", commit), "COMMIT doit être un hash complet, jamais None ni main"
    shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, capture_output=True, text=True)
    if shallow.returncode != 0 or shallow.stdout.strip() == "true":
        pytest.skip("pas de dépôt git complet : existence du commit non vérifiable")
    res = subprocess.run(["git", "cat-file", "-t", commit], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0 and res.stdout.strip() == "commit", f"COMMIT {commit} n'existe pas dans ce dépôt"


def test_pinned_commit_contains_what_the_notebook_needs():
    """Le code figé doit contenir les fichiers et options que le notebook utilise (sinon la session échouerait sur Kaggle)."""
    commit = re.search(r'^COMMIT = "([0-9a-f]{40})"', notebook_sources()[1], flags=re.M).group(1)
    shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, capture_output=True, text=True)
    if shallow.returncode != 0 or shallow.stdout.strip() == "true":
        pytest.skip("pas de dépôt git complet")

    def show(path):
        res = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8")
        assert res.returncode == 0, f"{path} absent du commit figé {commit[:10]}"
        return res.stdout

    assert "def discover" in show("src/chesszero/kaggle_inputs.py")
    session = show("scripts/alphazero_session.py")
    assert "--check-inputs" in session and "--expect-init-sha256" in session
    assert "MemoryWatch" in show("src/chesszero/memwatch.py") and "def agreement_indicator" in show("scripts/selfplay.py")
    assert "start_speed_watch" in show("scripts/match_stockfish.py")
    for needed in ("download_syzygy.py", "selfplay_session.py", "train_selfplay.py", "arena.py"):
        show(f"scripts/{needed}")
    for binary in ("books/lichess_2000.tsv.gz", "books/lichess_2000_arena.tsv.gz"):  # présents dans le clone Kaggle
        assert subprocess.run(["git", "cat-file", "-e", f"{commit}:{binary}"], cwd=ROOT).returncode == 0, binary


def test_notebook_modes_and_safeguards():
    src = notebook_sources()
    joined = "\n".join(src)
    import re

    assert 'MODE = "check"' in src[1]  # le mode par défaut n'écrit rien
    assert re.search(r"^LOOP_MINUTES = \d+ ", src[1], re.M) and re.search(r"^EXPECTED_SESSION = \d+ ", src[1], re.M)
    assert re.search(r"^RESET_LINE = None ", src[1], re.M)  # jamais de remise de la ligne par défaut
    assert '"check", "loop", "bench", "full"' in src[1]
    assert '"checkout"' in src[2] and "assert head == COMMIT" in src[2]
    assert "--check-inputs" in joined and "--expect-session" in joined and "--expect-init-sha256" in joined
    assert "select_previous" in joined and "alphazero_session.py" in joined
    assert 'if MODE != "check"' in src[6]  # pas de téléchargement des tables en mode check
    assert "Aucun GPU" in src[3] and 'MODE == "check"' in src[3]  # GPU non requis en mode check
    for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "cellule", "exec")  # syntaxe valide


def test_notebook_refuses_an_unpinned_commit():
    settings = notebook_sources()[1]
    for bad in ("None", '"main"', '""', '"abc123"'):
        code = re.sub(r'^COMMIT = .*$', f"COMMIT = {bad}", settings, flags=re.M)
        with pytest.raises(ValueError, match="COMMIT doit être un hash complet"):
            exec(compile(code, "cellule 0", "exec"), {"__name__": "nb"})
