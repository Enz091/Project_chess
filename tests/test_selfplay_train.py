"""Entraînement sur le self-play (étape 6d) : nombre de pas, lots mélangés, garde-fous, reprise, rejet."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from chesszero.checkpoint import load_for_inference
from chesszero.mcts import MCTSConfig
from chesszero.model import ModelConfig, build_model
from chesszero.selfplay import SelfPlayConfig, ShardWriter, run_selfplay
from chesszero.selfplay_train import (SelfPlayTrainConfig, SelfPlayTrainer, batch_rows, decide, human_val_set,
                                      plan_steps)


class Stub:
    def evaluate(self, boards, legal_indices):
        return [np.full(len(i), 1.0 / len(i)) for i in legal_indices], np.zeros(len(boards))


CFG = SelfPlayTrainConfig(batch_size=16, lr=1e-3, lr_start=1e-4, warmup_steps=5, log_every=5, save_every=5,
                          min_buffer_positions=0, heldout_fraction=0.3, lichess_val_positions=0, num_workers=0,
                          amp="off", buffer_positions=10**6)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Un tampon de self-play (faux réseau), des données « humaines » et un réseau tiny de départ."""
    root = tmp_path_factory.mktemp("sp")
    sp_cfg = SelfPlayConfig(simulations=16, games_per_worker=4, max_plies=50, book_fraction=0.0)
    writer = ShardWriter(root / "buffer", "t_w0", {}, 3, 60)
    run_selfplay(Stub(), sp_cfg, sp_cfg.mcts_config(MCTSConfig()), writer, np.random.default_rng(0), "t", max_games=8)
    from chesszero.selfplay import load_shard

    data = [load_shard(p) for p in writer.written]
    boards = np.concatenate([d["boards"] for d in data])
    human = {"boards": boards, "moves": np.concatenate([d["played"] for d in data]).astype(np.int16),
             "results": np.concatenate([d["result"] for d in data]),
             "value_valid": (np.arange(len(boards)) % 2).astype(np.uint8)}
    torch.manual_seed(0)
    model_cfg = ModelConfig.named("tiny")
    init = root / "init.pt"
    torch.save({"step": 0, "model": build_model(model_cfg).state_dict(), "config": {"model": asdict(model_cfg)}}, init)
    val = human_val_set(human, 0, 0, None)
    return {"root": root, "buffer": root / "buffer", "human": human, "val": val, "init": init}


def trainer(world, out, cfg=CFG):
    torch.manual_seed(0)
    return SelfPlayTrainer(world["root"] / out, cfg, "cpu", world["init"], log=lambda m: None)


# --------------------------------------------------------------------------- nombre de pas et lots


def test_plan_steps_waits_for_buffer_and_caps_views():
    cfg = SelfPlayTrainConfig()  # lots de 1024, 25 % humain -> 768 positions de self-play par lot
    assert plan_steps(100_000, 100_000, cfg, 4.0, 0.25) == 0  # tampon < 250 000 : pas de tour
    assert plan_steps(250_000, 250_000, cfg, 4.0, 0.25) == 250_000 // 768  # premier tour : plafonné à 1 passage
    assert plan_steps(100_000, 500_000, cfg, 4.0, 0.25) == -(-4 * 100_000 // 768)  # régime normal : 521 pas
    assert plan_steps(100_000, 500_000, cfg, 2.0, 0.10) == -(-2 * 100_000 // 922)


def test_batch_rows_exact_composition_and_deterministic():
    cfg = SelfPlayTrainConfig(batch_size=1024)
    for frac, n_h in ((0.25, 256), (0.10, 102)):
        sp, hu = batch_rows(cfg, frac, round_id=3, step=7, n_selfplay_train=5000, n_human=9000)
        assert len(hu) == n_h and len(sp) == 1024 - n_h
        sp2, hu2 = batch_rows(cfg, frac, round_id=3, step=7, n_selfplay_train=5000, n_human=9000)
        assert np.array_equal(sp, sp2) and np.array_equal(hu, hu2)
    other, _ = batch_rows(cfg, 0.25, round_id=3, step=8, n_selfplay_train=5000, n_human=9000)
    assert not np.array_equal(sp, other)


# --------------------------------------------------------------------------- garde-fous


def m(policy, top1, value=0.73, heldout=0.5, train_value=0.5):
    return {"lichess": {"policy_loss": policy, "top1": top1, "value_loss": value},
            "heldout": {"value_mse": heldout}, "train": {"sp_value": train_value}}


BASE = m(1.40, 0.535)


def test_guards_strict_during_first_generations():
    cfg = SelfPlayTrainConfig()
    d = decide(m(1.46, 0.53), BASE, None, [], generation=0, cfg=cfg)  # perte de politique +0,06
    assert not d["accept"] and d["actions"] == ["halve_lr"]
    d = decide(m(1.42, 0.50), BASE, None, [], generation=2, cfg=cfg)  # top-1 −3,5 points
    assert not d["accept"] and d["actions"] == ["human_25"]
    assert decide(m(1.43, 0.51), BASE, None, [], generation=4, cfg=cfg)["accept"]


def test_guards_after_five_generations_only_catastrophes_reject():
    cfg = SelfPlayTrainConfig()
    d = decide(m(1.48, 0.49), BASE, None, [], generation=5, cfg=cfg)  # dépasse G1 et G2 : alertes seulement
    assert d["accept"] and len(d["alerts"]) == 2
    d = decide(m(1.56, 0.50), BASE, None, [], generation=8, cfg=cfg)  # perte de politique +0,16
    assert not d["accept"] and "halve_lr" in d["actions"]
    d = decide(m(1.45, 0.43), BASE, None, [], generation=8, cfg=cfg)  # top-1 −10,5 points
    assert not d["accept"]


def test_guards_jump_and_value_alerts():
    cfg = SelfPlayTrainConfig()
    d = decide(m(1.43, 0.505), BASE, m(1.37, 0.53), [], generation=6, cfg=cfg)
    assert d["accept"] and any("chute brutale" in a for a in d["alerts"])
    d = decide(m(1.41, 0.53, value=0.79), BASE, None, [], generation=1, cfg=cfg)
    assert d["accept"] and any("G3 (alerte)" in a for a in d["alerts"])


def test_guards_overfitting_on_heldout_selfplay_rejects():
    cfg = SelfPlayTrainConfig()
    hist = [m(1.40, 0.53, heldout=0.50, train_value=0.60), m(1.40, 0.53, heldout=0.52, train_value=0.55)]
    d = decide(m(1.40, 0.53, heldout=0.55, train_value=0.50), BASE, None, hist, generation=7, cfg=cfg)
    assert not d["accept"] and d["actions"] == ["reduce_reuse"]
    hist2 = [m(1.40, 0.53, heldout=0.50, train_value=0.60), m(1.40, 0.53, heldout=0.49, train_value=0.55)]
    assert decide(m(1.40, 0.53, heldout=0.55, train_value=0.50), BASE, None, hist2, generation=7, cfg=cfg)["accept"]


# --------------------------------------------------------------------------- tours d'entraînement


def test_round_trains_with_exact_human_share_and_loss_decreases(world):
    t = trainer(world, "run")
    res = t.run_round([world["buffer"]], world["human"], world["val"], steps_override=60)
    assert res["status"] in ("candidate", "rejected")
    log = [json.loads(l) for l in (world["root"] / "run" / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["human_share"] == pytest.approx(4 / 16) for row in log)
    assert log[-1]["sp_policy"] < log[0]["sp_policy"] and log[-1]["loss"] < log[0]["loss"]
    rounds = (world["root"] / "run" / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rounds) == 1 and json.loads(rounds[0])["steps"] == 60


def test_resume_after_interruption_is_identical(world):
    full = trainer(world, "full")
    full.run_round([world["buffer"]], world["human"], world["val"], steps_override=23)
    cut = trainer(world, "cut")
    r = cut.run_round([world["buffer"]], world["human"], world["val"], steps_override=23, stop_after=11)
    assert r["status"] == "interrupted" and r["done"] == 11
    resumed = trainer(world, "cut")  # nouveau processus : tout est relu depuis train_state.pt
    resumed.run_round([world["buffer"]], world["human"], world["val"])
    a, b = full.model.state_dict(), resumed.model.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert full.state["global_step"] == resumed.state["global_step"] == 23

    def losses(out):
        lines = (world["root"] / out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        return [(json.loads(l)["step"], json.loads(l)["loss"]) for l in lines]

    assert losses("full") == losses("cut")  # journal identique aussi (fenêtre de moyenne reprise)


def test_candidate_is_a_loadable_checkpoint_and_not_yet_a_generation(world):
    t = trainer(world, "gen", cfg=replace(CFG, g1_policy_loss=10, g2_top1_points=100))
    res = t.run_round([world["buffer"]], world["human"], world["val"], steps_override=10)
    assert res["status"] == "candidate" and t.state["generation"] == 0  # c'est l'arène qui accepte (étape 7b)
    model, meta = load_for_inference(world["root"] / "gen" / "candidate.pt")
    assert meta["step"] == 10


def test_rejection_restores_last_accepted_network_and_reacts(world):
    cfg = replace(CFG, g1_policy_loss=-10.0)  # toute valeur passe le seuil : rejet assuré (G1, premières générations)
    t = trainer(world, "rej", cfg=cfg)
    before = {k: v.clone() for k, v in t.model.state_dict().items()}
    res = t.run_round([world["buffer"]], world["human"], world["val"], steps_override=10)
    assert res["status"] == "rejected" and "halve_lr" in res["actions"]
    assert all(torch.equal(before[k], v) for k, v in t.model.state_dict().items())  # retour au réseau accepté
    assert t.state["lr_scale"] == 0.5 and t.state["generation"] == 0 and t.state["trained_shards"] == []


def test_g4_stops_after_three_rejections(world):
    t = trainer(world, "g4", cfg=replace(CFG, g1_policy_loss=-10.0))
    for _ in range(3):
        res = t.run_round([world["buffer"]], world["human"], world["val"], steps_override=3)
    assert res["loop_status"] == "stop" and any("G4" in a for a in res["alerts"])
    assert t.state["stop_reason"] == "G4" and t.state["arena_rejections"] == 0
    with pytest.raises(RuntimeError):
        t.run_round([world["buffer"]], world["human"], world["val"], steps_override=3)


def test_waits_when_buffer_too_small(world):
    t = trainer(world, "wait", cfg=replace(CFG, min_buffer_positions=10**6))
    res = t.run_round([world["buffer"]], world["human"], world["val"])
    assert res["status"] == "waiting" and t.state["round"] is None


def test_bootstrap_by_game_detects_clear_difference_and_not_noise():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from eval_value_target import bootstrap_diff

    rng = np.random.default_rng(0)
    game = np.repeat(np.arange(200), 10)
    a = rng.random(2000)
    d, lo, hi = bootstrap_diff(a, a - 0.05, game, 500, 0)
    assert d == pytest.approx(-0.05) and hi < 0
    noise = a + rng.normal(0, 0.01, 2000)
    d, lo, hi = bootstrap_diff(a, noise, game, 500, 0)
    assert lo < 0 < hi


def test_round_time_breakdown_is_recorded(world):
    """Rapport de session : décomposition du temps d'un tour (chargement, lots Lichess, tampon, masques, préparation,
    calcul, évaluation), cumulée sur les reprises."""
    from chesszero.selfplay_train import timing_shares

    t = trainer(world, "timing")
    t.load_seconds = 0.5
    t.run_round([world["buffer"]], world["human"], world["val"], steps_override=8, stop_after=3)
    t = trainer(world, "timing")  # reprise
    res = t.run_round([world["buffer"]], world["human"], world["val"])
    tm = res["timing"]
    for k in ("load", "human_read", "buffer_read", "human_masks", "prepare", "compute", "eval", "wall"):
        assert tm[k] >= 0, k
    assert tm["load"] == pytest.approx(0.5) and tm["compute"] > 0 and not any(k.startswith("_") for k in tm)
    sh = timing_shares(tm)
    assert 0 <= sh["reading"] <= 1 and sh["other"] >= 0
    assert sum(sh[k] for k in ("load", "human_read", "buffer_read", "human_masks", "prepare", "compute", "eval",
                               "other")) == pytest.approx(1.0, abs=1e-6)


def test_pruned_targets_on_old_shards_compute_priors_once(world):
    """Cibles débruitées sur des shards d'avant le 2026-10-24 (sans prior ni Q) : priors recalculés avec le réseau de
    départ (génération 0), gardés dans policy_priors/ et réutilisés ; le tour enregistre le résumé du débruitage."""
    from chesszero.selfplay import load_shard

    old = world["root"] / "old_buffer"
    old.mkdir()
    for p in sorted(world["buffer"].glob("*.npz")):
        d = load_shard(p)
        assert "policy_prior" in d and "policy_q" in d  # les shards d'aujourd'hui les ont
        np.savez(old / p.name, **{k: v for k, v in d.items() if k not in ("policy_prior", "policy_q")})
        (old / p.with_suffix(".json").name).write_text(p.with_suffix(".json").read_text(encoding="utf-8"),
                                                       encoding="utf-8")
    cfg = replace(CFG, policy_target="pruned", g1_policy_loss=10, g2_top1_points=100)
    t = trainer(world, "pruned_old", cfg)
    res = t.run_round([old], world["human"], world["val"], steps_override=4)
    assert res["status"] == "candidate"
    cache = sorted((world["root"] / "pruned_old" / "policy_priors").glob("*.npz"))
    assert [p.name for p in cache] == sorted(p.name for p in old.glob("*.npz"))
    row = json.loads((world["root"] / "pruned_old" / "rounds.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["policy_target"] == "pruned" and row["policy_target_stats"]["exact_q_share"] == 0
    assert row["timing"]["priors"] > 0
    # même prior que celui que le réseau de départ donne réellement
    from chesszero.policy_target import network_priors

    model, _ = load_for_inference(world["init"], "cpu")
    d = load_shard(sorted(old.glob("*.npz"))[0])
    with np.load(cache[0]) as z:
        assert np.allclose(z["prior"].astype(float), network_priors(model, d["boards"], d["policy_offsets"],
                                                                    d["policy_index"], "cpu").astype(float))
    mtimes = [p.stat().st_mtime_ns for p in cache]
    t.commit_arena(0, {"games": 2, "score": 0.0, "points": 0.0, "elo": -100.0, "elo_low": -200.0, "elo_high": 0.0,
                       "los": 0.0, "elo_se": 50.0}, None, __import__("chesszero.arena", fromlist=["ArenaConfig"]).ArenaConfig())
    t2 = trainer(world, "pruned_old", cfg)
    t2.run_round([old], world["human"], world["val"], steps_override=2)
    assert [p.stat().st_mtime_ns for p in cache] == mtimes and t2.prior_seconds == 0  # cache réutilisé


def test_pruned_targets_refuse_another_start_network(world, tmp_path):
    from chesszero.selfplay import load_shard

    old = tmp_path / "old"
    old.mkdir()
    p = sorted(world["buffer"].glob("*.npz"))[0]
    np.savez(old / p.name, **{k: v for k, v in load_shard(p).items() if k not in ("policy_prior", "policy_q")})
    (old / p.with_suffix(".json").name).write_text(p.with_suffix(".json").read_text(encoding="utf-8"), encoding="utf-8")
    t = trainer(world, "pruned_other", replace(CFG, policy_target="pruned"))
    torch.manual_seed(1)
    other = tmp_path / "other.pt"
    model_cfg = ModelConfig.named("tiny")
    torch.save({"step": 0, "model": build_model(model_cfg).state_dict(), "config": {"model": asdict(model_cfg)}}, other)
    t.init_checkpoint = other
    with pytest.raises(RuntimeError, match="SHA-256"):
        t.run_round([old], world["human"], world["val"], steps_override=2)


def test_reset_line_returns_to_accepted_network_and_keeps_buffer(world):
    """Remise de la ligne (décision humaine, 2026-10-24) : poids du réseau accepté (ici le réseau de départ), AdamW
    neuf et nouvelle montée du taux, compteurs à zéro, tampon entièrement « nouveau » ; refusée si un candidat attend."""
    from chesszero.arena import ArenaConfig
    from chesszero.selfplay_train import lr_at, read_train_state

    cfg = replace(CFG, g1_policy_loss=10, g2_top1_points=100)
    t = trainer(world, "reset", cfg)
    t.run_round([world["buffer"]], world["human"], world["val"], steps_override=12)
    with pytest.raises(ValueError, match="candidat 0"):
        t.reset_line("trop tôt")
    lost = {"games": 2, "score": 0.0, "points": 0.0, "elo": -100.0, "elo_low": -200.0, "elo_high": 0.0, "los": 0.0,
            "elo_se": 50.0}
    t.commit_arena(0, lost, None, ArenaConfig())
    t.state["lr_scale"], t.state["consecutive_rejections"] = 0.5, 2
    t._save_state()
    assert read_train_state(world["root"] / "reset")["trained_shards"]
    row = t.reset_line("politique aplatie")
    assert row["before"]["arena_rejections"] == 1 and row["before"]["lr_scale"] == 0.5
    t2 = trainer(world, "reset", cfg)  # relu depuis train_state.pt
    init = torch.load(world["init"], weights_only=False)["model"]
    assert all(torch.equal(t2.model.state_dict()[k], init[k]) for k in init)
    st = t2.state
    assert (st["arena_rejections"], st["consecutive_rejections"], st["lr_scale"], st["trained_shards"]) == (0, 0, 1.0, [])
    assert st["generation"] == 0 and st["global_step"] == 12 and st["warmup_from"] == 12
    assert not t2.optimizer.state  # AdamW neuf
    line_ok = torch.load(world["root"] / "reset" / "line_ok.pt", weights_only=False)
    assert line_ok["round"] is None and all(torch.equal(line_ok["model"][k], init[k]) for k in init)
    view = read_train_state(world["root"] / "reset")
    assert len(view["resets"]) == 1 and view["resets"][0]["note"] == "politique aplatie"
    rows = [json.loads(l) for l in (world["root"] / "reset" / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["type"] == "reset"
    # le tour suivant : nouvelle montée du taux, tout le tampon compte comme nouveau
    t2.run_round([world["buffer"]], world["human"], world["val"], steps_override=3)
    log = [json.loads(l) for l in (world["root"] / "reset" / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[-1]["lr"] == pytest.approx(lr_at(2, cfg, 1.0))
    last = [json.loads(l) for l in (world["root"] / "reset" / "rounds.jsonl").read_text(encoding="utf-8").splitlines()][-1]
    first = [json.loads(l) for l in (world["root"] / "reset" / "rounds.jsonl").read_text(encoding="utf-8").splitlines()][0]
    assert last["new_positions"] == first["new_positions"] > 0


def test_reset_line_lifts_a_stop_due_to_reset_counters_only(world):
    cfg = replace(CFG, g1_policy_loss=10, g2_top1_points=100)
    t = trainer(world, "reset_stop", cfg)
    for reason, lifted in (("G4", True), ("stagnation", True), ("ancre", False)):
        t.state["status"], t.state["stop_reason"] = "stop", reason
        row = t.reset_line(f"arrêt {reason}")
        assert (row["cleared_stop"] == reason) is lifted and (row["kept_stop"] == reason) is not lifted
        assert (t.state["status"] == "ok") is lifted
