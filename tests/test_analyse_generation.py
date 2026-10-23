"""scripts/analyse_generation.py : rejeu des parties, export PGN, accord politique brute / cible MCTS."""

import io
import json
import subprocess
import sys
from pathlib import Path

import chess.pgn

from test_loop import world  # noqa: F401  (fixture : sortie de self-play seul au format exact de l'étape 6)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = str(ROOT / "scripts" / "analyse_generation.py")


def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *map(str, args)], capture_output=True, text=True, encoding="utf-8")


def test_replay_check_succeeds_on_real_shards_and_fails_on_a_corrupted_one(tmp_path, world):
    buf = world["s0"] / "selfplay"
    res = run("check", "--buffer", buf)
    assert res.returncode == 0 and "0 incohérence" in res.stdout and "positions rejouées" in res.stdout
    import numpy as np

    bad = tmp_path / "selfplay"
    bad.mkdir()
    for p in buf.glob("*.json"):
        (bad / p.name).write_bytes(p.read_bytes())
    for p in buf.glob("*.npz"):
        with np.load(p) as z:
            data = {k: z[k] for k in z.files}
        data["played"] = (data["played"] + 1).astype(data["played"].dtype)  # coups joués faux
        np.savez_compressed(bad / p.name, **data)
    res = run("check", "--buffer", bad)
    assert res.returncode == 1 and "0 incohérence" not in res.stdout


def test_pgn_export_is_readable_and_commented(tmp_path, world):
    out = tmp_path / "pgn"
    res = run("pgn", "--buffer", world["s0"] / "selfplay", "--out", out, "--min-material", "0", "--name", "essai")
    assert res.returncode == 0, res.stdout + res.stderr
    assert (out / "README.md").exists()
    games = 0
    for name in ("repetitions_40pts.pgn", "mats_les_plus_courts.pgn", "abandons.pgn"):
        text = (out / name).read_text(encoding="utf-8")
        stream = io.StringIO(text)
        while (g := chess.pgn.read_game(stream)) is not None:
            games += 1
            assert g.headers["Termination"] and g.headers["GameUID"] and g.headers["Event"].startswith("ChessZero self-play essai")
            node = g
            while node.variations:
                node = node.variations[0]
            assert "FIN :" in node.comment and "matériel restant" in node.comment  # raison de fin au dernier coup
            assert int(g.headers["PlyCount"]) == len(list(g.mainline_moves()))
    assert games >= 1


def test_agreement_json_for_two_networks_on_the_same_sample(tmp_path, world):
    out = tmp_path / "accord.json"
    res = run("agreement", "--buffer", world["s0"] / "selfplay", "--checkpoint", world["init"], world["init"],
              "--positions", "50", "--all-positions", "--threads", "2", "--out", out)
    assert res.returncode == 0, res.stdout + res.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["positions"] == 50 and "Dirichlet" in data["note"]
    nets = list(data["networks"].values())
    assert len(nets) == 1 or nets[0] == nets[1]  # même fichier deux fois : même clé, mêmes chiffres
    cells = nets[0]["phases"]
    assert cells["all"]["n"] == 50 and 0 <= cells["all"]["agree"] <= 1 and cells["all"]["kl"] >= 0
