"""Cibles de politique débruitées (chesszero/policy_target.py) et leur emploi par l'entraîneur."""

import json

import chess
import numpy as np
import pytest

from chesszero.mcts import MCTS, MCTSConfig, legal_moves_and_indices
from chesszero.policy_target import best_moves, prune_one, prune_visits, target_stats

C_PUCT = 1.5


class Stub:
    """Prior fixe à la racine (e4 50 %, d4 30 %, a3 0,5 %, le reste à parts égales), uniforme ailleurs ; valeur +0,5
    pour le camp au trait après a3 (donc a3 vaut -0,5 pour les Blancs), 0 partout ailleurs."""

    def __init__(self):
        self.root_fen = chess.Board().fen()

    def evaluate(self, boards, legal_indices):
        priors, values = [], []
        for b, idx in zip(boards, legal_indices):
            if b.fen() == self.root_fen:
                moves, _ = legal_moves_and_indices(b)
                special = {"e2e4": 0.5, "d2d4": 0.3, "a2a3": 0.005}
                rest = (1 - sum(special.values())) / (len(moves) - len(special))
                priors.append(np.array([special.get(m.uci(), rest) for m in moves]))
            else:
                priors.append(np.full(len(idx), 1 / len(idx)))
            last = b.move_stack[-1].uci() if b.move_stack else None
            values.append(0.5 if last == "a2a3" and len(b.move_stack) == 1 else 0.0)
        return priors, np.array(values)


def root_arrays(tree):
    r = tree.root
    n = r.N.astype(np.float64)
    return n, r.P_raw.copy(), np.where(n > 0, r.W / np.maximum(n, 1), np.nan)


def search(noise_on=None, simulations=200):
    """Recherche depuis la position initiale ; `noise_on` = coup qui reçoit TOUT le bruit de Dirichlet (epsilon 0,25)."""
    eps = 0.25 if noise_on else 0.0

    class NoiseOn:  # générateur dont le « bruit de Dirichlet » tombe entièrement sur `noise_on`
        def dirichlet(self, alpha):
            v = np.zeros(len(alpha))
            v[[m.uci() for m in tree.root.moves].index(noise_on)] = 1.0
            return v

    tree = MCTS(Stub(), MCTSConfig(num_simulations=simulations, batch_size=4, dirichlet_epsilon=eps), chess.Board(),
                rng=NoiseOn())
    tree.search(simulations)
    return tree


def test_noise_boosted_move_loses_its_extra_visits():
    """Le bruit met tout son poids sur a3 (prior 0,5 %, mauvais coup) : la recherche lui donne beaucoup de visites ;
    le débruitage les retire (au niveau de la recherche sans bruit), sans toucher au meilleur coup."""
    clean, noisy = search(), search(noise_on="a2a3")
    a3 = [m.uci() for m in noisy.root.moves].index("a2a3")
    n_clean, _, _ = root_arrays(clean)
    n, p, q = root_arrays(noisy)
    assert n[a3] >= 10 and n_clean[a3] <= 1  # le bruit a gonflé a3
    pruned = prune_one(n, p, q, C_PUCT)
    assert pruned[a3] <= max(n_clean[a3], 1)
    b = int(np.argmax(n))
    assert pruned[b] == n[b] and int(np.argmax(pruned)) == b
    assert (pruned <= n).all() and (pruned[n == 0] == 0).all()
    tv = lambda x: 0.5 * np.abs(x / x.sum() - n_clean / n_clean.sum()).sum()
    assert tv(pruned) < tv(n) / 2  # nettement plus proche de la recherche sans bruit


def test_clean_targets_at_equilibrium_are_unchanged():
    """Sans bruit, des visites à l'équilibre de PUCT (chaque coup a exactement les visites que justifient sa valeur
    et son prior) ne changent pas ; un coup dont la valeur dépasse celle du meilleur n'est jamais réduit."""
    prior = np.array([0.55, 0.25, 0.15, 0.05])
    q = np.array([0.10, 0.02, -0.10, 0.30])
    best_n = 150.0
    visits = np.array([best_n, 0, 0, 7.0])
    for _ in range(50):  # point fixe : visites minimales des coups 1 et 2 pour le total courant
        scale = C_PUCT * np.sqrt(1 + visits.sum())
        s_best = q[0] + scale * prior[0] / (1 + best_n)
        for a in (1, 2):
            visits[a] = np.ceil(scale * prior[a] / (s_best - q[a]) - 1 - 1e-9)
    assert visits[1] > 0 and visits[2] > 0
    assert np.array_equal(prune_one(visits, prior, q, C_PUCT), visits)


def test_clean_search_keeps_best_and_barely_changes():
    clean = search()
    n, p, q = root_arrays(clean)
    pruned = prune_one(n, p, q, C_PUCT)
    assert int(np.argmax(pruned)) == int(np.argmax(n)) and pruned.max() == n.max()
    assert 0.5 * np.abs(pruned / pruned.sum() - n / n.sum()).sum() < 0.08


def test_flat_version_matches_single_positions_and_ties():
    rng = np.random.default_rng(3)
    rows, flat = [], {"v": [], "p": [], "q": []}
    for k in range(20):
        m = int(rng.integers(2, 9))
        v = rng.integers(0, 40, size=m).astype(float)
        v[0] = max(v[0], 1)
        p = rng.dirichlet(np.ones(m))
        q = np.where(v > 0, rng.uniform(-1, 1, size=m), np.nan)
        rows.append((v, p, q))
        for key, x in zip("vpq", (v, p, q)):
            flat[key].append(x)
    offsets = np.concatenate([[0], np.cumsum([len(r[0]) for r in rows])])
    out = prune_visits(offsets, *(np.concatenate(flat[k]) for k in "vpq"), C_PUCT)
    for i, (v, p, q) in enumerate(rows):
        assert np.array_equal(out[offsets[i]:offsets[i + 1]], prune_one(v, p, q, C_PUCT))
    # égalité de visites : le plus grand prior est le meilleur coup (comme MCTS.best_move)
    assert best_moves(np.array([0, 3]), np.array([5.0, 9.0, 9.0]), np.array([0.2, 0.3, 0.5]))[0] == 2
    s = target_stats([(np.array([0, 2, 4]), np.array([1.0, 1.0, 4.0, 0.0]))])
    assert s["visits"] == 6 and s["entropy"] == pytest.approx(np.log(2) / 2) and s["top_share"] == pytest.approx(0.75)


def test_without_q_uses_prior_ratio_to_best():
    """Approximation des anciens shards (sans Q) : chaque coup visité est plafonné à (1 + N_meilleur) x P / P_meilleur."""
    v = np.array([100.0, 30.0, 12.0, 0.0])
    p = np.array([0.5, 0.2, 0.01, 0.29])
    out = prune_one(v, p, np.where(v > 0, 0.0, np.nan), C_PUCT)
    assert out.tolist() == [100.0, 30.0, 2.0, 0.0]  # 101 x 0,2 / 0,5 - 1 = 39,4 >= 30 ; 101 x 0,01 / 0,5 - 1 = 1,02


# --------------------------------------------------------------------------- shards et entraîneur


def write_shard(directory, name, with_fields: bool, generation=0, step=None):
    """Shard minimal : 2 positions (position initiale, coups légaux), visites gonflées sur a3 pour la première."""
    board = chess.Board()
    moves, idx = legal_moves_and_indices(board)
    from chesszero.encoding import pack_board

    a3 = [m.uci() for m in moves].index("a2a3")
    v1 = np.zeros(len(moves))
    v1[0], v1[a3] = 150, 50
    v2 = np.zeros(len(moves))
    v2[0], v2[1] = 120, 80
    prior = np.full(len(moves), 0.4 / (len(moves) - 2))
    prior[0], prior[1], prior[a3] = 0.4, 0.2, 0.001
    prior /= prior.sum()
    q = lambda v: np.where(v > 0, np.where(np.arange(len(v)) == a3, -0.6, 0.0), np.nan)
    data = {"boards": np.stack([pack_board(board)] * 2), "policy_offsets": np.array([0, len(moves), 2 * len(moves)]),
            "policy_index": np.concatenate([idx, idx]).astype(np.int16),
            "policy_visits": np.concatenate([v1, v2]).astype(np.uint16), "played": np.array([idx[0]] * 2, np.int16),
            "result": np.zeros(2, np.int8), "root_q": np.zeros(2, np.float16), "ply": np.zeros(2, np.int16),
            "game": np.array([0, 1], np.int32), "game_uid": np.array([f"{name}-0", f"{name}-1"])}
    if with_fields:
        data["policy_prior"] = np.concatenate([prior, prior]).astype(np.float16)
        data["policy_q"] = np.concatenate([q(v1), q(v2)]).astype(np.float16)
    np.savez(directory / f"{name}.npz", **data)
    meta = {"generation": generation, "checkpoint_step": step, "created": 1.0, "positions": 2,
            "config": {"mcts": {"c_puct": C_PUCT}}}
    (directory / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    return a3, np.concatenate([prior, prior])


def test_selfplay_data_prunes_with_recorded_fields_and_prior_source(tmp_path):
    from chesszero.selfplay import list_shards
    from chesszero.selfplay_train import SelfPlayData

    a3, prior = write_shard(tmp_path, "new", True)
    write_shard(tmp_path, "old", False)
    shards = sorted(list_shards([tmp_path]), key=lambda s: s["name"])
    calls = []

    def source(shard, part):
        calls.append(shard["name"])
        return prior  # prior sans bruit recalculé (ici : le vrai)

    raw = SelfPlayData(shards, 0.0)
    pruned = SelfPlayData(shards, 0.0, "pruned", source)
    assert calls == ["old"]  # les shards récents ont leur prior
    new_row = [i for i, s in enumerate(shards) if s["name"] == "new"][0]
    first = slice(*pruned.offsets[[2 * new_row, 2 * new_row + 1]])
    assert raw.visits[first][a3] == 50 and pruned.visits[first][a3] == 0  # Q -0,6 et prior 0,1 % : bruit retiré
    old_row = 1 - new_row
    old_first = slice(*pruned.offsets[[2 * old_row, 2 * old_row + 1]])
    assert pruned.visits[old_first][a3] == 0  # sans Q : plafond 151 x 0,1 % / 40 % - 1 < 0
    second = slice(*pruned.offsets[[2 * old_row + 1, 2 * old_row + 2]])
    # sans Q, un 2e coup très visité est aussi plafonné par le rapport des priors : 121 x 0,2 / 0,4 - 1 = 59,5 -> 60
    assert pruned.visits[second][1] == 60 and raw.visits[second][1] == 80
    stats = pruned.prune_stats
    assert stats["exact_q_share"] == 0.5 and stats["removed_visit_share"] > 0
    assert stats["entropy_pruned"] < stats["entropy_raw"]
    with pytest.raises(ValueError):
        SelfPlayData(shards, 0.0, "pruned", None)
