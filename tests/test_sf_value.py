"""chesszero/sf_value.py (essai B2) : positions exclues, centipions -> valeur, échelle, cible, affinage avec reprise."""

import json
import math

import chess
import numpy as np
import pytest
import torch

from chesszero import sf_value as sv
from chesszero.encoding import legal_masks_from_packed, move_to_index, pack_board


def test_position_keys_ignore_the_fifty_move_counter():
    a, b = chess.Board(), chess.Board()
    b.halfmove_clock = 7
    keys = sv.position_keys(np.stack([pack_board(a), pack_board(b)]))
    assert keys[0] == keys[1]
    c = chess.Board()
    c.push_uci("e2e4")
    assert sv.position_keys(pack_board(c)[None])[0] != keys[0]


def test_protected_openings_cover_every_position_of_v2_and_anchor_lines():
    lines = sv.protected_lines()
    assert len(lines["v2"]) == 40 and len(lines["anchor"]) == 100
    keys = sv.opening_keys()
    for line in lines["v2"][:5] + lines["anchor"][:5]:
        board = chess.Board()
        assert sv.position_keys(pack_board(board)[None])[0] in keys  # position initiale
        for uci in line.split():
            board.push_uci(uci)
            assert sv.position_keys(pack_board(board)[None])[0] in keys
    # tirage : les positions protégées sont écartées, les autres gardées
    board = chess.Board()
    for uci in lines["anchor"][0].split():
        board.push_uci(uci)
    other = chess.Board("8/8/8/4k3/8/8/4K3/7R w - - 0 1")
    boards = np.stack([pack_board(board)] * 5 + [pack_board(other)] * 5 + [pack_board(chess.Board())] * 3)
    rows, dropped = sv.sample_rows(boards, 5, 0, keys)
    assert sorted(rows.tolist()) == [5, 6, 7, 8, 9] and dropped >= 0
    with pytest.raises(ValueError):
        sv.sample_rows(boards, 6, 0, keys)


def test_cp_to_value_mates_and_scale():
    v = sv.cp_to_value(np.array([0, 300, -300, 0, 0]), np.array([0, 0, 0, 1, -1]), 300.0)
    assert v[0] == 0 and v[1] == pytest.approx(math.tanh(1.0)) and v[2] == pytest.approx(-math.tanh(1.0))
    assert v[3] == 1.0 and v[4] == -1.0 and v.dtype == np.float32


def test_fit_scale_recovers_the_scale_of_synthetic_results():
    rng = np.random.default_rng(0)
    cp = rng.normal(0, 400, 200_000)
    p = np.tanh(cp / 450.0)  # espérance ; résultat tiré : +1 avec (1+p)/2, -1 sinon (sans nulle)
    res = np.where(rng.random(len(cp)) < (1 + p) / 2, 1.0, -1.0)
    fit = sv.fit_scale(cp, np.zeros(len(cp), int), res)
    assert abs(fit["scale"] - 450) < 20 and fit["mse"] < fit["mse_zero"]
    assert fit["lichess_scale"] == pytest.approx(543.17, abs=0.1)


def test_value_target_mixture_keeps_stockfish_alone_without_a_valid_result():
    labels = {"cp": np.array([0, 0]), "mate": np.array([0, 0]), "results": np.array([1, 1], np.int8),
              "value_valid": np.array([1, 0], np.uint8)}
    assert sv.value_target(labels, 400, 1.0).tolist() == [0, 0]
    assert sv.value_target(labels, 400, 0.75).tolist() == [0.25, 0.0]


def test_lr_schedule_warmup_then_cosine():
    cfg = sv.FinetuneConfig(steps=1000, warmup_steps=100, lr=1e-4, lr_start=1e-5, lr_end=1e-5)
    assert sv.lr_at(0, cfg) == pytest.approx(1e-5) and sv.lr_at(100, cfg) == pytest.approx(1e-4)
    assert sv.lr_at(550, cfg) == pytest.approx(5.5e-5) and sv.lr_at(1000, cfg) == pytest.approx(1e-5)


def _positions(n, seed):
    rng = np.random.default_rng(seed)
    boards, moves = [], []
    while len(boards) < n:
        b = chess.Board()
        for _ in range(int(rng.integers(0, 30))):
            legal = list(b.legal_moves)
            if not legal:
                break
            b.push(legal[int(rng.integers(len(legal)))])
        legal = list(b.legal_moves)
        if not legal:
            continue
        m = legal[int(rng.integers(len(legal)))]
        boards.append(pack_board(b))
        moves.append(move_to_index(m, b))
    return np.stack(boards), np.array(moves, np.int16)


def _labels(n, seed):
    boards, moves = _positions(n, seed)
    rng = np.random.default_rng(seed)
    return {"boards": boards, "moves": moves, "results": rng.integers(-1, 2, n).astype(np.int8),
            "value_valid": np.ones(n, np.uint8), "cp": rng.integers(-500, 500, n).astype(np.int32),
            "mate": np.zeros(n, np.int8)}


def test_finetune_writes_a_loadable_network_and_resumes_exactly(tmp_path):
    from chesszero.checkpoint import load_for_inference
    from chesszero.model import ModelConfig, build_model

    torch.manual_seed(0)
    mc = ModelConfig.named("tiny")
    init = tmp_path / "init.pt"
    torch.save({"step": 0, "epoch": 0, "model": build_model(mc).state_dict(), "config": {"model": mc.to_dict()},
                "metrics": {}}, init)
    labels, val = _labels(48, 1), _labels(16, 2)
    hb, hm = _positions(16, 3)
    human = {"boards": hb, "moves": hm.astype(np.int64), "results": np.zeros(16, np.float32),
             "value_valid": np.ones(16, np.float32), "legal": legal_masks_from_packed(hb)}
    cfg = sv.FinetuneConfig(steps=6, batch_size=8, warmup_steps=2, eval_every=3, save_every=1)
    quiet = lambda *a: None  # noqa: E731
    last = sv.finetune(init, labels, val, human, tmp_path / "a", cfg, 400.0, workers=0, log=quiet)
    assert last["step"] == 6
    steps = [json.loads(l)["step"] for l in (tmp_path / "a" / "metrics.jsonl").read_text().splitlines()]
    assert steps == [0, 3, 6]
    model, meta = load_for_inference(tmp_path / "a" / "final.pt")
    assert meta["step"] == 6 and meta["config"]["scale"] == 400.0
    # coupure au pas 4 puis reprise : mêmes poids qu'un affinage d'une traite
    assert sv.finetune(init, labels, val, human, tmp_path / "b", cfg, 400.0, workers=0, log=quiet,
                       stop_after=4)["interrupted"]
    sv.finetune(init, labels, val, human, tmp_path / "b", cfg, 400.0, workers=0, log=quiet)
    a = torch.load(tmp_path / "a" / "final.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "final.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    with pytest.raises(ValueError):  # autres réglages sur le même dossier : refus
        sv.finetune(init, labels, val, human, tmp_path / "b", sv.FinetuneConfig(steps=7, batch_size=8), 400.0,
                    workers=0, log=quiet)
