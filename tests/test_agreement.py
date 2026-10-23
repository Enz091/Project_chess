"""Indicateur d'accord politique brute / cible MCTS (suivi des générations)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chesszero.agreement import (describe_agreement, gather, merge_agreement, phase_of, policy_agreement,
                                 sample_positions, session_agreement, summarize)
from chesszero.encoding import legal_move_indices, pack_board
from chesszero.model import ModelConfig, build_model

ROOT = Path(__file__).resolve().parents[1]


def random_shard(n_games=4, plies=14, seed=0, visits_seed=1) -> dict:
    """Shard au format du self-play : parties aléatoires, visites aléatoires sur les coups légaux."""
    rng = np.random.default_rng(seed)
    vrng = np.random.default_rng(visits_seed)
    boards, ply, game, played = [], [], [], []
    offsets, index, visits = [0], [], []
    for g in range(n_games):
        b = chess.Board()
        for p in range(plies):
            moves = legal_move_indices(b)
            if not moves:
                break
            boards.append(pack_board(b))
            ply.append(p)
            game.append(g)
            idx = np.array([i for _, i in moves], dtype=np.int16)
            v = vrng.integers(0, 40, size=len(idx)).astype(np.uint16)
            v[vrng.integers(len(idx))] += 50  # un coup nettement préféré
            index.append(idx)
            visits.append(v)
            offsets.append(offsets[-1] + len(idx))
            played.append(int(idx[np.argmax(v)]))
            b.push(moves[int(rng.integers(len(moves)))][0])
    return {"boards": np.stack(boards), "policy_offsets": np.array(offsets, dtype=np.int32),
            "policy_index": np.concatenate(index), "policy_visits": np.concatenate(visits),
            "played": np.array(played, dtype=np.int16), "ply": np.array(ply, dtype=np.int16),
            "game": np.array(game, dtype=np.int32)}


class OracleNet(torch.nn.Module):
    """Réseau factice : logits fixés par une fonction des positions (pour des attentes exactes)."""

    def __init__(self, logits_for):
        super().__init__()
        self.logits_for = logits_for
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.logits_for(x), torch.zeros(len(x))


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return build_model(ModelConfig(n_blocks=1, n_filters=8, value_hidden=8)).eval()


def naive(model, shard, rows):
    """Calcul position par position, sans lots ni remplissage : référence pour les fonctions vectorisées."""
    from chesszero.encoding import unpack_batch_torch

    out = []
    for i in rows:
        with torch.inference_mode():
            logits, _ = model(unpack_batch_torch(torch.from_numpy(shard["boards"][i:i + 1])))
        a, b = shard["policy_offsets"][i], shard["policy_offsets"][i + 1]
        z = logits[0].double()[torch.from_numpy(shard["policy_index"][a:b].astype(np.int64))]
        logp = torch.log_softmax(z, 0).numpy()
        v = shard["policy_visits"][a:b].astype(np.float64)
        t = v / v.sum()
        kl = sum(tt * (math.log(tt) - lp) for tt, lp in zip(t, logp) if tt > 0)
        out.append((int(np.argmax(logp) == np.argmax(v)), kl))
    return np.array(out)


def test_phases():
    assert phase_of(0, 78) == "opening" and phase_of(29, 10) == "opening"
    assert phase_of(30, 19) == "endgame" and phase_of(30, 20) == "middle" and phase_of(150, 60) == "middle"


def test_sample_is_fixed_and_without_replacement():
    shards = [random_shard(seed=1), random_shard(seed=2)]
    a, b = sample_positions(shards, 30, 7), sample_positions(shards, 30, 7)
    assert a == b and len(set(a)) == 30
    assert sample_positions(shards, 30, 8) != a
    total = sum(len(s["played"]) for s in shards)
    assert len(sample_positions(shards, 10**6, 0)) == total  # moins de positions que demandé : toutes
    assert sample_positions([], 10, 0) == []


def test_vectorised_matches_naive_loop():
    shard = random_shard(n_games=5, plies=16)
    model = tiny_model()
    picks = sample_positions([shard], 60, 3)
    batch = gather([shard], picks)
    agree, kl = policy_agreement(model, batch, batch_size=7)  # petits lots : remplissage différent à chaque lot
    expect = naive(model, shard, [i for _, i in picks])
    assert np.array_equal(agree, expect[:, 0])
    assert np.allclose(kl, expect[:, 1], atol=1e-9)
    assert (kl >= -1e-12).all()


def test_exact_values_with_oracle_networks():
    shard = random_shard(n_games=3, plies=12)
    picks = sample_positions([shard], 20, 0)
    batch = gather([shard], picks)
    best = {int(j): int(batch["legal"][j][np.argmax(batch["visits"][j])]) for j in range(len(picks))}

    def logits(favoured, sharp=40.0):
        def f(x):
            out = torch.zeros(len(x), 4672)
            for r in range(len(x)):  # un seul lot dans ce test : les positions arrivent dans l'ordre de l'échantillon
                out[r, favoured[f.calls + r]] = sharp
            f.calls += len(x)
            return out
        f.calls = 0
        return f

    # réseau qui met toute sa masse sur le coup le plus visité : accord 1 ; la KL vaut alors -log p(coup) de la cible
    good = OracleNet(logits([best[j] for j in range(len(picks))]))
    agree, kl = policy_agreement(good, batch, batch_size=len(picks))
    assert agree.all()
    # réseau qui met toute sa masse sur un coup NON préféré : accord 0
    worst = []
    for j in range(len(picks)):
        others = [m for m in batch["legal"][j] if m != best[j]]
        worst.append(int(others[0]))
    bad = OracleNet(logits(worst))
    agree_bad, kl_bad = policy_agreement(bad, batch, batch_size=len(picks))
    assert not agree_bad.any() and (kl_bad > kl).all()
    # réseau uniforme : KL = log(n) - entropie de la cible, accord = 1 seulement si tous les coups sont à égalité
    flat = OracleNet(lambda x: torch.zeros(len(x), 4672))
    agree_flat, kl_flat = policy_agreement(flat, batch)
    for j in range(len(picks)):
        t = batch["visits"][j] / batch["visits"][j].sum()
        want = math.log(len(t)) + float(np.sum(t[t > 0] * np.log(t[t > 0])))
        assert kl_flat[j] == pytest.approx(want, abs=1e-9)


def test_summary_and_merge_are_weighted_by_positions():
    shard = random_shard(n_games=4, plies=20)
    net = tiny_model()
    a = session_agreement([shard], {"played": net, "reference": net}, n=50, seed=0)
    assert a["sample"] == 50
    for key in ("played", "reference"):
        cells = a["networks"][key]
        assert cells["all"]["n"] == 50 == sum(cells[p]["n"] for p in ("opening", "middle", "endgame"))
        assert cells["opening"]["n"] == 50  # parties de 20 demi-coups : toutes en ouverture
        assert cells["middle"]["agree"] is None  # phase vide
    assert a["networks"]["played"] == a["networks"]["reference"]  # même réseau, même échantillon : écart nul
    # fusion pondérée : deux tranches de tailles différentes
    s1 = session_agreement([random_shard(seed=1)], {"played": net}, n=30, seed=0)
    s2 = session_agreement([random_shard(seed=2, visits_seed=5)], {"played": net}, n=10, seed=0)
    m = merge_agreement([s1, s2, None, {"error": "x"}])
    n1, n2 = s1["networks"]["played"]["all"], s2["networks"]["played"]["all"]
    assert m["sample"] == 40 and m["networks"]["played"]["all"]["n"] == n1["n"] + n2["n"]
    want = (n1["agree"] * n1["n"] + n2["agree"] * n2["n"]) / (n1["n"] + n2["n"])
    assert m["networks"]["played"]["all"]["agree"] == pytest.approx(want)
    assert merge_agreement([None, {"error": "x"}]) is None


def test_description_mentions_both_networks_gap_and_dirichlet_bias():
    shard = random_shard(n_games=3, plies=12)
    net_a, net_b = tiny_model(0), tiny_model(1)
    text = describe_agreement(session_agreement([shard], {"played": net_a, "reference": net_b}, n=30))
    assert "réseau qui a joué" in text and "kaggle_sup" in text and "écart" in text
    assert "Dirichlet" in text and "59.8%" in text.replace(",", ".")
    assert "non calculé" in describe_agreement(None)
    assert "kaggle_sup (départ)" not in describe_agreement(session_agreement([shard], {"played": net_a}, n=30))


def test_selfplay_script_writes_agreement_in_session_json(tmp_path):
    """Bout en bout avec le vrai scripts/selfplay.py (réseau tiny, 1 partie) : l'indicateur est dans sessions/<nom>.json."""
    ckpt = ROOT / "checkpoints" / "tiny_sup" / "best.pt"
    if not ckpt.exists():
        pytest.skip("checkpoints/tiny_sup/best.pt absent")
    out = tmp_path / "selfplay"
    cmd = [sys.executable, str(ROOT / "scripts" / "selfplay.py"), "--out", str(out), "--checkpoint", str(ckpt),
           "--reference", str(ckpt), "--config", "smoke", "--workers", "1", "--threads", "2", "--max-games", "2",
           "--games-per-worker", "2", "--device", "cpu", "--syzygy", "", "--session", "essai",
           "--agreement-positions", "40"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    stats = json.loads((out / "sessions" / "essai.json").read_text(encoding="utf-8"))
    a = stats["agreement"]
    assert a["sample"] > 0 and "error" not in a and a["played_is_reference"] is True
    assert set(a["networks"]) == {"played", "reference"}
    assert a["networks"]["played"] == a["networks"]["reference"]
    assert "accord politique brute / cible MCTS" in res.stdout
    # désactivé : pas d'indicateur, la session fonctionne comme avant
    out2 = tmp_path / "selfplay2"
    cmd2 = [c for c in cmd] + ["--agreement-positions", "0"]
    cmd2[cmd2.index("--out") + 1] = str(out2)
    res2 = subprocess.run(cmd2, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res2.returncode == 0
    assert json.loads((out2 / "sessions" / "essai.json").read_text(encoding="utf-8"))["agreement"] is None
