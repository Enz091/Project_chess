"""chesszero/b1_criteria.py : critère 2 de B1 jugé sur les positions non protégées."""

import random

import chess
import numpy as np
import pytest
import torch

from chesszero import b1_criteria as c
from chesszero.encoding import move_to_index, pack_board
from chesszero.model import ModelConfig, build_model
from chesszero.selfplay_train import eval_human, human_val_set


def test_thresholds_are_the_approved_ones():
    assert (c.TOP1_MIN, c.POLICY_LOSS_MAX) == (0.5429, 1.3961)
    assert (c.INFO_TOP1_MIN_ALL, c.INFO_POLICY_LOSS_MAX_ALL) == (0.542, 1.3949)
    assert (c.LICHESS_VAL, c.LICHESS_SEED) == (50000, 0)


def m(top1, pol):
    return {"top1": top1, "policy_loss": pol, "value_loss": 0.7, "positions": 1}


def test_verdict_uses_the_unprotected_positions_only():
    # rempli sur les non protégées, raté sur les 50 000 : critère rempli (l'information n'y change rien)
    v = c.judge({"unprotected": m(0.5430, 1.3960), "all": m(0.5400, 1.4000)})
    assert v["passed"] and not v["info_all_positions"]["top1_ok"] and not v["info_all_positions"]["policy_loss_ok"]
    # l'inverse : critère non rempli
    assert not c.judge({"unprotected": m(0.5428, 1.3900), "all": m(0.5500, 1.3800)})["passed"]
    assert not c.judge({"unprotected": m(0.5500, 1.3962), "all": m(0.5500, 1.3800)})["passed"]
    assert c.judge({"unprotected": m(0.5429, 1.3961), "all": m(0.5, 1.5)})["passed"]  # seuils inclus


def random_positions(n, seed=0):
    rng, out = random.Random(seed), []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randint(12, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            out.append(board)
    return out


def test_protected_positions_and_weighted_total(tmp_path):
    keys = None  # vraies lignes v2 et ancre
    start = chess.Board()
    late = chess.Board()
    late.halfmove_clock = 7  # même position, compteur des 50 coups différent : toujours protégée
    boards = [start, late] + random_positions(40)
    packed = np.stack([pack_board(b) for b in boards])
    mask = c.protected_mask(packed, keys)
    assert mask[0] and mask[1] and mask[2:].sum() <= 2  # parties au hasard : presque jamais sur une ligne protégée
    rng = np.random.default_rng(0)
    human = {"boards": packed,
             "moves": np.array([move_to_index(next(iter(b.legal_moves)), b) for b in boards], np.int16),
             "results": rng.integers(-1, 2, len(boards)).astype(np.int8),
             "value_valid": (rng.random(len(boards)) < 0.7).astype(np.uint8)}
    val = human_val_set(human, 0, 0, None)
    torch.manual_seed(0)
    model = build_model(ModelConfig.named("tiny"))
    with torch.no_grad():
        got = c.evaluate(model, val, mask, "cpu")
        ref = eval_human(model, val, "cpu", batch_size=7)
    assert got["unprotected"]["positions"] + got["protected"]["positions"] == len(boards) == got["all"]["positions"]
    for k in ("policy_loss", "top1", "value_loss"):
        assert got["all"][k] == pytest.approx(ref[k], rel=1e-5, abs=1e-6)
    with torch.no_grad():
        none = c.evaluate(model, val, np.zeros(len(boards), bool), "cpu")
    assert none["protected"] is None and none["all"]["top1"] == pytest.approx(ref["top1"])
