import random

import chess
import numpy as np
import pytest
import torch

from chesszero.encoding import (
    N_ACTIONS,
    N_PLANES,
    encode_board,
    legal_move_mask,
    move_to_index,
)
from chesszero.model import (
    ChessNet,
    ModelConfig,
    build_model,
    count_parameters,
    masked_policy,
    policy_value_loss,
)

SMALL = ModelConfig(n_blocks=2, n_filters=16, value_hidden=16)


def random_boards(n: int, seed: int = 0) -> list[chess.Board]:
    """n positions distinctes tirées de parties au hasard (ni fin de partie, ni doublon)."""
    rng = random.Random(seed)
    boards, seen = [], set()
    while len(boards) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, 120)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        key = board.fen()
        if not board.is_game_over() and key not in seen:
            seen.add(key)
            boards.append(board)
    return boards


def batch_from(boards):
    x = torch.from_numpy(np.stack([encode_board(b) for b in boards]))
    mask = torch.from_numpy(np.stack([legal_move_mask(b) for b in boards]))
    return x, mask


@pytest.mark.parametrize("name", ["tiny", "kaggle"])
def test_named_configs_load(name):
    model = build_model(name)
    assert count_parameters(model) > 0


def test_output_shapes_and_value_range():
    model = ChessNet(SMALL).eval()
    x, _ = batch_from(random_boards(5))
    with torch.no_grad():
        logits, value = model(x * 50)  # entrées exagérées : tanh doit rester borné
    assert logits.shape == (5, N_ACTIONS)
    assert value.shape == (5,)
    assert value.abs().max() <= 1.0


def test_policy_output_layout_matches_move_indices():
    """La sortie (rangée, colonne, plan) aplatie doit suivre case_départ*73 + plan.

    On pose un « pic » dans le tenseur de sortie de la dernière conv à (plan p, rangée r, colonne c)
    et on vérifie qu'il atterrit à l'indice (8*r + c)*73 + p.
    """
    model = ChessNet(SMALL).eval()
    captured = {}
    last_conv = model.policy_head[-1]
    last_conv.register_forward_hook(lambda m, i, out: captured.update(out=out))
    x, _ = batch_from(random_boards(1))
    with torch.no_grad():
        logits, _ = model(x)
    out = captured["out"]  # (1, 73, 8, 8)
    for plane, rank, file in [(0, 0, 0), (5, 3, 6), (72, 7, 7), (60, 4, 1)]:
        assert logits[0, (8 * rank + file) * 73 + plane] == out[0, plane, rank, file]

    # Et de façon concrète : le coup e2e4 des Blancs part de e2 = case 12 -> indices 12*73 + plan.
    board = chess.Board()
    idx = move_to_index(chess.Move.from_uci("e2e4"), board)
    assert idx // 73 == chess.E2
    assert logits[0, idx] == out[0, idx % 73, chess.square_rank(chess.E2), chess.square_file(chess.E2)]


def test_masked_probabilities_sum_to_one_on_legal_moves():
    model = ChessNet(SMALL).eval()
    boards = random_boards(8, seed=1)
    x, mask = batch_from(boards)
    with torch.no_grad():
        logits, _ = model(x)
    probs = masked_policy(logits, mask)
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)
    assert probs[~mask].max() == 0.0  # aucune probabilité sur un coup illégal
    assert (probs[mask] > 0).all()


def test_loss_index_target_equals_one_hot_distribution():
    model = ChessNet(SMALL).eval()
    boards = random_boards(6, seed=2)
    x, mask = batch_from(boards)
    rng = np.random.default_rng(0)
    idx = torch.tensor([rng.choice(np.flatnonzero(m)) for m in mask.numpy()])
    v = torch.tensor(rng.uniform(-1, 1, 6), dtype=torch.float32)
    with torch.no_grad():
        logits, value = model(x)
    loss_idx, parts_idx = policy_value_loss(logits, value, idx, v, mask)
    loss_dist, _ = policy_value_loss(logits, value, torch.nn.functional.one_hot(idx, N_ACTIONS).float(), v, mask)
    assert torch.isclose(loss_idx, loss_dist)
    assert torch.isclose(loss_idx, torch.tensor(parts_idx["policy"] + parts_idx["value"]))


def test_loss_ignores_logits_of_illegal_moves():
    boards = random_boards(3, seed=3)
    x, mask = batch_from(boards)
    logits = torch.randn(3, N_ACTIONS)
    idx = torch.tensor([int(np.flatnonzero(m)[0]) for m in mask.numpy()])
    v = torch.zeros(3)
    base, _ = policy_value_loss(logits, v, idx, v, mask)
    perturbed = logits.clone()
    perturbed[~mask] += 100.0
    other, _ = policy_value_loss(perturbed, v, idx, v, mask)
    assert torch.isclose(base, other)


def _value_head_grads(model, x, mask, idx, v, value_valid):
    """Gradients de la tête valeur (value_conv + value_fc) pour une perte donnée. Modèle en mode eval :
    les positions du lot sont alors indépendantes entre elles (pas de statistiques de lot)."""
    model.zero_grad()
    logits, value = model(x)
    loss, parts = policy_value_loss(logits, value, idx, v, mask, value_valid)
    loss.backward()
    head = list(model.value_conv.parameters()) + list(model.value_fc.parameters())
    return [p.grad.clone() for p in head], parts


def test_invalid_value_positions_send_no_gradient_to_value_head():
    """Une position à value_valid = 0 ne contribue pas au gradient de la tête valeur,
    quelle que soit sa cible de valeur ; la perte de valeur est la moyenne sur les positions valides."""
    torch.manual_seed(0)
    model = ChessNet(SMALL).eval()
    x, mask = batch_from(random_boards(8, seed=6))
    rng = np.random.default_rng(1)
    idx = torch.tensor([rng.choice(np.flatnonzero(m)) for m in mask.numpy()])
    v = torch.tensor(rng.uniform(-1, 1, 8), dtype=torch.float32)
    valid = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.uint8)

    grads, parts = _value_head_grads(model, x, mask, idx, v, valid)
    assert any(g.abs().sum() > 0 for g in grads)  # les positions valides, elles, font bien apprendre la tête

    # 1. On change complètement la cible des positions invalides : rien ne bouge.
    v_changed = v.clone()
    v_changed[4:] = -torch.sign(v[4:]) * 1.0 + 0.5
    grads2, parts2 = _value_head_grads(model, x, mask, idx, v_changed, valid)
    assert parts2["value"] == parts["value"]
    assert all(torch.equal(a, b) for a, b in zip(grads, grads2))

    # 2. Même gradient que si le lot ne contenait QUE les positions valides.
    grads3, parts3 = _value_head_grads(model, x[:4], mask[:4], idx[:4], v[:4], None)
    assert parts3["value"] == pytest.approx(parts["value"], rel=1e-5)
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(grads, grads3))

    # 3. Aucune position valide : perte de valeur nulle, gradient nul sur la tête valeur...
    none_valid = torch.zeros(8, dtype=torch.uint8)
    grads4, parts4 = _value_head_grads(model, x, mask, idx, v, none_valid)
    assert parts4["value"] == 0.0 and all(g.abs().sum() == 0 for g in grads4)
    # ... mais la politique, elle, apprend sur toutes les positions.
    assert model.policy_head[-1].weight.grad.abs().sum() > 0

    # 4. La perte politique ne dépend pas de value_valid.
    logits, value = model(x)
    p_all = policy_value_loss(logits, value, idx, v, mask, none_valid)[1]["policy"]
    p_ref = policy_value_loss(logits, value, idx, v, mask)[1]["policy"]
    assert p_all == p_ref


def test_overfit_50_positions():
    """Le réseau tiny doit mémoriser 50 positions (coup et résultat aléatoires) : perte ~ 0.

    Les cibles sont du bruit sans aucune structure : seul le surapprentissage peut les atteindre.
    C'est un test de bon câblage (gradients, masque, perte), pas de généralisation.
    """
    torch.manual_seed(0)
    rng = random.Random(0)
    boards = random_boards(50, seed=4)
    x, mask = batch_from(boards)
    target_move = torch.tensor([move_to_index(rng.choice(list(b.legal_moves)), b) for b in boards])
    target_value = torch.tensor([rng.choice([-1.0, 0.0, 1.0]) for _ in boards])

    model = build_model("tiny")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    model.train()
    first = None
    for step in range(100):
        logits, value = model(x)
        loss, _ = policy_value_loss(logits, value, target_move, target_value, mask)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # On évalue en mode eval (statistiques BatchNorm « running »), comme en inférence réelle.
    model.eval()
    with torch.no_grad():
        logits, value = model(x)
        loss, parts = policy_value_loss(logits, value, target_move, target_value, mask)
    accuracy = (logits.masked_fill(~mask, -1e9).argmax(dim=1) == target_move).float().mean().item()

    assert first > 3.0, "la perte initiale devrait être élevée (~ ln(nb coups légaux))"
    assert loss.item() < 0.05, f"perte finale trop haute : {loss.item():.4f} ({parts})"
    assert accuracy == 1.0
