"""Tests de l'entraînement supervisé (étape 4a) sur un mini jeu de données synthétique mais apprenable."""

import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import chess
import numpy as np
import pytest
import torch

from chesszero.data import _write_shard, load_split
from chesszero.encoding import (
    board_from_packed,
    legal_masks_from_packed,
    legal_move_mask,
    pack_board,
)
from chesszero.model import ModelConfig, policy_value_loss
from chesszero.train import (
    CSV_COLUMNS,
    Meter,
    PhaseTimer,
    GpuSampler,
    TrainConfig,
    ValSet,
    lr_factor,
    read_metrics,
    stream_batches,
    timed_batches,
    topk_correct,
    train,
    train_indices,
)

SMALL = ModelConfig(n_blocks=1, n_filters=16, value_hidden=16)
CPU = torch.device("cpu")
CFG = TrainConfig(
    batch_size=32, lr=3e-3, warmup_steps=5, total_steps=30, log_every=5, eval_every=10, save_every=5,
    val_positions=0, num_workers=0, keep_last=2,
)
QUIET = lambda *_: None  # noqa: E731


def make_positions(n: int, seed: int):
    """n positions de parties aléatoires. Cible apprenable : coup = le légal de plus petit indice,
    résultat = +1 si j'ai plus de matériel que l'adversaire, sinon −1."""
    rng = random.Random(seed)
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    boards, moves, results, valid = [], [], [], []
    while len(boards) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, 80)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over():
            continue
        # Les indices de coups sont dans le repère « camp au trait en bas » : on calcule dans ce repère.
        oriented = board_from_packed(pack_board(board))
        mine = sum(values.get(p.piece_type, 0) for p in oriented.piece_map().values() if p.color == chess.WHITE)
        theirs = sum(values.get(p.piece_type, 0) for p in oriented.piece_map().values() if p.color == chess.BLACK)
        boards.append(pack_board(board))
        moves.append(int(np.argmax(legal_move_mask(board))))
        results.append(1 if mine > theirs else -1)
        valid.append(int(rng.random() < 0.7))  # ~30 % de positions « Time forfeit » : sans cible de valeur
    return np.stack(boards), np.array(moves, np.int16), np.array(results, np.int8), np.array(valid, np.uint8)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sup")
    for split, n, seed in (("train", 640, 1), ("val", 96, 2)):
        boards, moves, results, valid = make_positions(n, seed)
        _write_shard(d / f"{split}_00000.npz", boards, moves, results, np.zeros(n, np.int32), valid)
    return d


# --------------------------------------------------------------------------- masques et lots


def test_packed_board_roundtrip_and_masks():
    """Masque calculé depuis le format compact == masque de la vraie position (Blancs ET Noirs au trait)."""
    rng = random.Random(5)
    seen_turns = set()
    for _ in range(80):
        board = chess.Board()
        for _ in range(rng.randint(0, 100)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over():
            continue
        seen_turns.add(board.turn)
        packed = pack_board(board)
        assert np.array_equal(legal_move_mask(board), legal_masks_from_packed(packed[None])[0])
        assert np.array_equal(pack_board(board_from_packed(packed)), packed)
    assert seen_turns == {chess.WHITE, chess.BLACK}


def test_train_indices_epochs_and_resumability():
    n, bs = 100, 32  # 3 pas par époque, 4 exemples jetés
    all_idx = list(train_indices(n, bs, seed=1, start_step=0, end_step=9))
    for epoch in range(3):
        used = np.concatenate(all_idx[3 * epoch : 3 * epoch + 3])
        assert len(set(used.tolist())) == 96  # aucun doublon dans une époque
    assert not np.array_equal(np.concatenate(all_idx[:3]), np.concatenate(all_idx[3:6]))  # mélangé à chaque époque
    resumed = list(train_indices(n, bs, seed=1, start_step=4, end_step=9))  # repartir du pas 4
    assert all(np.array_equal(a, b) for a, b in zip(all_idx[4:], resumed))


def test_batches_identical_with_and_without_workers(data_dir):
    data = load_split(data_dir, "train")
    idx = list(train_indices(len(data["boards"]), 32, 0, 0, 6))
    plain = list(stream_batches(data, idx, None))
    with ProcessPoolExecutor(2) as pool:
        pooled = list(stream_batches(data, idx, pool))
    assert len(plain) == len(pooled) == 6
    for a, b in zip(plain, pooled):
        for x, y in zip(a, b):
            assert np.array_equal(x, y)
    for b in plain:  # le coup joué est toujours légal
        assert b.legal[np.arange(32), b.moves].all()


def test_topk_correct():
    logits = torch.tensor([[0.1, 0.9, 0.5, 0.3], [0.9, 0.1, 0.5, 0.3]])
    legal = torch.tensor([[True, True, True, True], [False, True, True, True]])
    # ligne 0 : ordre 1,2,3,0 ; ligne 1 (0 illégal) : ordre 2,3,1
    assert topk_correct(logits, legal, torch.tensor([1, 2])) == (2, 2)
    assert topk_correct(logits, legal, torch.tensor([3, 1])) == (0, 2)
    assert topk_correct(logits, legal, torch.tensor([0, 0])) == (0, 0)  # 4e choix ; coup illégal


def test_lr_schedule_shape():
    cfg = TrainConfig(lr=1e-3, warmup_steps=100, total_steps=1000, min_lr_ratio=0.01)
    f = [lr_factor(s, cfg) for s in range(1000)]
    assert f[0] == pytest.approx(0.01) and f[99] == pytest.approx(1.0)  # montée linéaire
    assert f[100] == pytest.approx(1.0) and max(f) == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(f[100:], f[101:]))  # puis décroissance monotone
    assert f[-1] == pytest.approx(0.01, abs=1e-3)


def test_config_files_have_train_section():
    for name in ("tiny", "kaggle"):
        cfg = TrainConfig.named(name)
        assert cfg.total_steps > cfg.warmup_steps > 0
    with pytest.raises(ValueError):
        TrainConfig(log_every=7, eval_every=10)


def test_meter_averages_value_loss_over_valid_positions_only():
    m = Meter()
    m.add(n=10, n_value=2, policy=1.0, value=0.5, top1=5, top3=8)  # 2 positions valides, perte valeur moyenne 0,5
    m.add(n=30, n_value=0, policy=2.0, value=0.0, top1=3, top3=9)  # aucune position valide
    m.add(n=10, n_value=8, policy=3.0, value=0.1, top1=1, top3=2)
    out = m.means()
    assert out["policy_loss"] == pytest.approx((10 * 1 + 30 * 2 + 10 * 3) / 50)  # toutes les positions
    assert out["value_loss"] == pytest.approx((2 * 0.5 + 8 * 0.1) / 10)  # positions valides seulement
    assert out["loss"] == pytest.approx(out["policy_loss"] + out["value_loss"])
    assert out["top1"] == pytest.approx(9 / 50) and out["top3"] == pytest.approx(19 / 50)


def test_data_without_value_valid_is_refused(tmp_path):
    boards, moves, results, valid = make_positions(40, 7)
    for split in ("train", "val"):  # ancien format : pas de value_valid
        np.savez(tmp_path / f"{split}_00000.npz", boards=boards, moves=moves, results=results, games=np.zeros(40, np.int32))
    with pytest.raises(ValueError, match="value_valid"):
        train(SMALL, replace(CFG, batch_size=8), tmp_path, tmp_path / "out", CPU, log=QUIET)


def test_valset_subset_is_fixed(data_dir):
    data = load_split(data_dir, "val")
    a, b = ValSet(data, 40, None), ValSet(data, 40, None)
    assert len(a) == 40 and np.array_equal(a.boards, b.boards)
    assert len(ValSet(data, 0, None)) == len(data["boards"])


# --------------------------------------------------------------------------- entraînement


def run_to_end(data_dir, out_dir, cfg=CFG, **kw):
    return train(SMALL, cfg, data_dir, out_dir, CPU, log=QUIET, **kw)


def comparable(rows):
    """Lignes du CSV sans les colonnes de temps."""
    return [{k: v for k, v in r.items() if k not in ("pos_per_s", "elapsed_s", "gpu_util", "data_wait_pct", "mask_wait_pct")} for r in rows]


def test_loss_decreases(data_dir, tmp_path):
    out = run_to_end(data_dir, tmp_path, replace(CFG, total_steps=60, eval_every=60, save_every=60))
    rows = read_metrics(tmp_path / "metrics.csv")
    train_rows = [r for r in rows if r["split"] == "train"]
    first, last = train_rows[0], train_rows[-1]
    assert last["policy_loss"] < 0.5 * first["policy_loss"]  # ~3,4 au départ
    assert last["top1"] > first["top1"] + 0.2
    assert last["top3"] >= last["top1"]
    assert last["value_loss"] < first["value_loss"]
    val = [r for r in rows if r["split"] == "val"][-1]
    assert val["policy_loss"] < 3.0  # positions jamais vues : mieux que le hasard (~3,4)
    assert out["final_val"]["top3"] >= out["final_val"]["top1"]
    assert list(rows[0]) == CSV_COLUMNS


def test_resume_matches_uninterrupted_run(data_dir, tmp_path):
    ref_dir, cut_dir = tmp_path / "ref", tmp_path / "cut"
    run_to_end(data_dir, ref_dir)

    # Coupure au pas 13 : le dernier checkpoint est au pas 10.
    cut = run_to_end(data_dir, cut_dir, stop_after=13)
    assert cut["interrupted"]
    run_to_end(data_dir, cut_dir)  # relance : reprise automatique depuis last.pt

    a = torch.load(ref_dir / "last.pt", weights_only=False)
    b = torch.load(cut_dir / "last.pt", weights_only=False)
    assert a["step"] == b["step"] == CFG.total_steps
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    assert comparable(read_metrics(ref_dir / "metrics.csv")) == comparable(read_metrics(cut_dir / "metrics.csv"))
    assert a["metrics"]["best_val_loss"] == b["metrics"]["best_val_loss"]
    assert (ref_dir / "best.pt").exists() and (cut_dir / "best.pt").exists()


def test_resume_truncates_csv_rows_after_checkpoint(data_dir, tmp_path):
    run_to_end(data_dir, tmp_path, stop_after=13)  # checkpoint au pas 10 ; lignes CSV jusqu'au pas 10
    # Journal « en avance » sur le checkpoint (coupure après l'écriture du CSV, avant la sauvegarde)
    with open(tmp_path / "metrics.csv", "a", encoding="utf-8") as f:
        f.write("15,0.7,train,9,9,9,0,0,0.001,100,5\n")
    run_to_end(data_dir, tmp_path)
    steps = [r["step"] for r in read_metrics(tmp_path / "metrics.csv") if r["split"] == "train"]
    assert steps == [5, 10, 15, 20, 25, 30]  # la ligne parasite du pas 15 a été remplacée par la vraie


def test_best_checkpoint_follows_validation_loss(data_dir, tmp_path):
    out = run_to_end(data_dir, tmp_path)
    vals = [r for r in read_metrics(tmp_path / "metrics.csv") if r["split"] == "val"]
    assert [r["step"] for r in vals] == [10, 20, 30]
    best = min(vals, key=lambda r: r["loss"])
    ck = torch.load(tmp_path / "best.pt", weights_only=False)
    assert ck["step"] == best["step"]
    assert ck["metrics"]["val"]["loss"] == pytest.approx(best["loss"])
    assert out["best_val_loss"] == pytest.approx(best["loss"])
    assert len(list(tmp_path.glob("ckpt_*.pt"))) <= CFG.keep_last
    assert ck["config"]["model"] == SMALL.to_dict()


def test_finished_run_is_not_retrained(data_dir, tmp_path):
    run_to_end(data_dir, tmp_path)
    before = torch.load(tmp_path / "last.pt", weights_only=False)["model"]
    run_to_end(data_dir, tmp_path)  # déjà terminé : ne fait que ré-évaluer
    after = torch.load(tmp_path / "last.pt", weights_only=False)["model"]
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert [r["split"] for r in read_metrics(tmp_path / "metrics.csv")].count("val_full") == 1


def test_resume_with_another_network_is_refused(data_dir, tmp_path):
    run_to_end(data_dir, tmp_path)
    with pytest.raises(ValueError, match="autre réseau"):
        train(replace(SMALL, n_filters=32), CFG, data_dir, tmp_path, CPU, log=QUIET)


# --------------------------------------------------------------------------- perte sans masque, diagnostics, budget de temps


def test_loss_without_mask_is_plain_softmax_cross_entropy():
    torch.manual_seed(0)
    logits, value = torch.randn(6, 4672), torch.zeros(6)
    moves = torch.randint(0, 4672, (6,))
    loss, parts = policy_value_loss(logits, value, moves, torch.zeros(6), None)
    expected = torch.nn.functional.cross_entropy(logits, moves)
    assert parts["policy"] == pytest.approx(expected.item(), rel=1e-5)
    legal = torch.zeros(6, 4672, dtype=torch.bool)
    legal[torch.arange(6), moves] = True  # un seul coup légal : le masque donne une perte nulle, pas l'absence de masque
    assert policy_value_loss(logits, value, moves, torch.zeros(6), legal)[1]["policy"] == pytest.approx(0.0, abs=1e-5)


def test_topk_correct_without_mask():
    logits = torch.tensor([[0.9, 0.1, 0.5, 0.3]])
    assert topk_correct(logits, None, torch.tensor([0])) == (1, 1)
    assert topk_correct(logits, None, torch.tensor([1])) == (0, 0)  # 4e choix


def test_unmasked_batches_have_no_masks_and_same_content(data_dir):
    data = load_split(data_dir, "train")
    idx = list(train_indices(len(data["boards"]), 32, 0, 0, 4))
    with ProcessPoolExecutor(2) as pool:
        plain = list(stream_batches(data, idx, pool, with_masks=False))  # le pool n'est pas utilisé
    masked = list(stream_batches(data, idx, None))
    for a, b in zip(plain, masked):
        assert a.legal is None
        assert np.array_equal(a.x, b.x) and np.array_equal(a.moves, b.moves)


def test_timer_counts_mask_time_only_when_masks_are_computed(data_dir):
    data = load_split(data_dir, "train")
    idx = list(train_indices(len(data["boards"]), 32, 0, 0, 4))
    t_on, t_off = PhaseTimer(), PhaseTimer()
    list(timed_batches(stream_batches(data, idx, None, timer=t_on), t_on))
    list(timed_batches(stream_batches(data, idx, None, with_masks=False, timer=t_off), t_off))
    assert 0 < t_on.mask_s <= t_on.data_s
    assert t_off.mask_s == 0 and t_off.data_s > 0
    t_on.compute_s = t_on.data_s  # compute = data -> 50 % d'attente
    assert t_on.pct(t_on.data_s) == pytest.approx(50.0)


def test_gpu_sampler_averages_injected_readings():
    values = iter([40.0, 60.0, 80.0] + [80.0] * 1000)
    sampler = GpuSampler(CPU, read=lambda: next(values), interval=0.01)
    deadline = time.time() + 5
    while sampler.mean() != sampler.mean() and time.time() < deadline:  # attend le 1er échantillon
        time.sleep(0.01)
    sampler.close()
    assert 40.0 <= sampler.mean() <= 80.0
    sampler.reset()
    assert sampler.mean() != sampler.mean()  # nan après reset
    assert GpuSampler(CPU).mean() != GpuSampler(CPU).mean()  # CPU : pas de mesure


def test_training_without_mask_runs_learns_and_validates_with_masks(data_dir, tmp_path):
    cfg = replace(CFG, total_steps=60, eval_every=60, save_every=60, mask_train_loss=False)
    out = run_to_end(data_dir, tmp_path, cfg)
    rows = read_metrics(tmp_path / "metrics.csv")
    train_rows = [r for r in rows if r["split"] == "train"]
    assert train_rows[-1]["policy_loss"] < 0.6 * train_rows[0]["policy_loss"]  # apprend sans masque
    assert all(r["mask_wait_pct"] == 0 for r in train_rows)
    # validation masquée : top-k parmi les coups légaux, donc au moins aussi bon que sans masque
    assert out["final_val"]["top1"] > 0.1 and out["final_val"]["top3"] >= out["final_val"]["top1"]


def test_diagnostic_columns_are_recorded(data_dir, tmp_path):
    run_to_end(data_dir, tmp_path)
    train_rows = [r for r in read_metrics(tmp_path / "metrics.csv") if r["split"] == "train"]
    for r in train_rows:
        assert r["pos_per_s"] > 0
        assert 0 <= r["data_wait_pct"] <= 100 and 0 <= r["mask_wait_pct"] <= r["data_wait_pct"]
        assert r["gpu_util"] != r["gpu_util"]  # nan : pas de GPU


def test_time_budget_stops_cleanly_and_resumes_identically(data_dir, tmp_path):
    ref_dir, cut_dir = tmp_path / "ref", tmp_path / "cut"
    run_to_end(data_dir, ref_dir)
    cut = run_to_end(data_dir, cut_dir, max_minutes=1e-9)  # budget épuisé dès le 1er pas
    assert cut["timed_out"] and cut["step"] == 1
    assert torch.load(cut_dir / "last.pt", weights_only=False)["step"] == 1  # sauvegardé même hors save_every
    run_to_end(data_dir, cut_dir)
    a = torch.load(ref_dir / "last.pt", weights_only=False)
    b = torch.load(cut_dir / "last.pt", weights_only=False)
    assert a["step"] == b["step"] == CFG.total_steps
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
