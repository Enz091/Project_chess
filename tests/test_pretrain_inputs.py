"""chesszero/pretrain_inputs.py et dataset_check.py : entrées d'une session Kaggle de pré-entraînement B1."""

import json

import numpy as np
import torch

from chesszero.data import _write_shard
from chesszero.dataset_check import check_shards, find_dataset, sha256_file
from chesszero.model import ModelConfig
from chesszero.pretrain_inputs import check_pretrain_inputs, restore_previous


def make_dataset(folder, name="lichess_48m"):
    folder.mkdir(parents=True)
    shards = {}
    for fname, n in (("train_00000.npz", 30), ("train_00001.npz", 20), ("val_00000.npz", 10)):
        _write_shard(folder / fname, np.zeros((n, 67), np.uint8), np.zeros(n, np.int16), np.zeros(n, np.int8),
                     np.zeros(n, np.int32), np.ones(n, np.uint8))
        shards[fname] = {"positions": n, "bytes": (folder / fname).stat().st_size, "sha256": sha256_file(folder / fname)}
    manifest = {"dataset": name, "positions": {"train": 50, "val": 10}, "shards": shards,
                "exclusion": {"train_removed": 7}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def make_previous(folder, step, model=None, throughput_ok=None):
    folder.mkdir(parents=True)
    model = model or ModelConfig.named("kaggle_l").to_dict()
    torch.save({"step": step, "config": {"model": model}, "model": {}}, folder / "last.pt")
    (folder / "metrics.csv").write_text("step\n", encoding="utf-8")
    if throughput_ok is not None:
        (folder / "throughput_check.json").write_text(json.dumps(
            {"ok": throughput_ok, "pos_per_s": 2500.0, "min_pos_per_s": 3000, "step": 2000}), encoding="utf-8")


def test_dataset_is_found_by_its_manifest_and_every_shard_is_checked(tmp_path):
    manifest = make_dataset(tmp_path / "datasets" / "compte" / "lichess-48m")
    make_dataset(tmp_path / "datasets" / "compte" / "autre", name="autre")
    (data,) = find_dataset(tmp_path, "lichess_48m")
    read = lambda n: (data / n).read_bytes()  # noqa: E731
    names = [p.name for p in data.glob("*.npz")]
    assert check_shards(read, names, manifest) == []
    raw = bytearray((data / "train_00001.npz").read_bytes())
    raw[-30] ^= 1  # un octet abîmé
    (data / "train_00001.npz").write_bytes(bytes(raw))
    assert any("empreinte" in e for e in check_shards(read, names, manifest))
    assert any("shards différents" in e for e in check_shards(read, names[:-1], manifest))


def test_start_from_scratch_resume_and_refusals(tmp_path):
    make_dataset(tmp_path / "in" / "lichess-48m")
    r = check_pretrain_inputs(tmp_path / "in", expected_step=0)
    assert r["ok"] and r["previous"] is None and r["total_steps"] == 92000
    assert not check_pretrain_inputs(tmp_path / "in", expected_step=5000)["ok"]  # reprise attendue, rien trouvé
    make_previous(tmp_path / "in" / "notebook" / "b1_12x160", 5000)
    r = check_pretrain_inputs(tmp_path / "in", expected_step=5000)
    assert r["ok"] and r["previous_step"] == 5000
    assert not check_pretrain_inputs(tmp_path / "in", expected_step=0)["ok"]  # mauvais EXPECTED_STEP
    out = tmp_path / "working" / "b1_12x160"
    assert sorted(restore_previous(r["previous"], out)) == ["last.pt", "metrics.csv"]
    assert restore_previous(r["previous"], out) == []  # last.pt déjà présent : rien n'est écrasé
    make_previous(tmp_path / "in" / "notebook2" / "b1_12x160", 6000)
    assert any("sorties précédentes" in e for e in check_pretrain_inputs(tmp_path / "in", expected_step=5000)["errors"])


def test_other_network_or_stop_by_the_throughput_guard_are_refused(tmp_path):
    make_dataset(tmp_path / "in" / "lichess-48m")
    make_previous(tmp_path / "in" / "nb" / "b1_12x160", 2000, model=ModelConfig.named("kaggle").to_dict())
    assert any("autre réseau" in e for e in check_pretrain_inputs(tmp_path / "in", expected_step=2000)["errors"])
    make_dataset(tmp_path / "in2" / "lichess-48m")
    make_previous(tmp_path / "in2" / "nb" / "b1_12x160", 2000, throughput_ok=False)
    r = check_pretrain_inputs(tmp_path / "in2", expected_step=2000)
    assert not r["ok"] and any("garde-fou" in e for e in r["errors"])
    assert check_pretrain_inputs(tmp_path / "in2", expected_step=2000, accept_low_throughput=True)["ok"]
    assert not check_pretrain_inputs(tmp_path / "vide", expected_step=0)["ok"]  # pas de données


def _notebook_cells():
    from pathlib import Path

    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_pretrain_b1.ipynb").read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_check_mode_on_a_fake_kaggle_writes_nothing(tmp_path, capsys):
    cells = _notebook_cells()
    assert 'MODE = "check"' in cells[0] and 'GPUS = "auto"' in cells[0] and "EXPECTED_STEP = 0" in cells[0]
    assert "ok=(0, 6)" in cells[5]  # garde-fou de débit : la version reste « Successful »
    inp, work = tmp_path / "kaggle" / "input", tmp_path / "kaggle" / "working"
    make_dataset(inp / "datasets" / "compte" / "lichess-48m")
    work.mkdir(parents=True)
    ns = {}
    src0 = cells[0].replace('Path("/kaggle/input")', f"Path({str(inp)!r})").replace('Path("/kaggle/working")', f"Path({str(work)!r})")
    exec(src0, ns)
    for src in cells[2:]:  # cellule 1 (clonage au commit figé) sautée : le code est déjà là
        exec(src, ns)
    out = capsys.readouterr().out
    assert "la session démarrerait" in out and "départ de zéro" in out and "chaque shard vérifié" in out
    assert list(work.iterdir()) == []
    # reprise attendue mais absente : refus clair
    ns2 = {}
    exec(src0.replace("EXPECTED_STEP = 0 ", "EXPECTED_STEP = 4000 "), ns2)
    exec(cells[2], ns2)
    import pytest

    with pytest.raises(RuntimeError, match="NE démarrerait PAS"):
        exec(cells[3], ns2)
