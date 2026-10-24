"""Étape 7d : orchestrateur de la boucle (phases, reprise après coupure, budget, contrôles, nouvel essai, rapport)."""

import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

import chesszero.selfplay_train as spt
from chesszero.arena import ArenaConfig, read_jsonl, sha256_file, summarize
from chesszero.loop import (CrashForTest, LoopConfig, LoopStop, Orchestrator, actions, classify_output,
                            check_expected_session, measured_v2_generations, next_phase, read_loop_state,
                            selfplay_chunk)
from chesszero.mcts import MCTSConfig
from chesszero.model import ModelConfig, build_model
from chesszero.selfplay import (SelfPlayConfig, ShardWriter, list_shards, load_shard, repetition_stats,
                                run_selfplay, sync_buffer)
from chesszero.selfplay_train import SelfPlayTrainConfig, SelfPlayTrainer, human_val_set, read_train_state

ROOT = Path(__file__).resolve().parents[1]
TRAIN = replace(SelfPlayTrainConfig.named("smoke"), min_buffer_positions=200, lichess_val_positions=0,
                buffer_positions=10**6)
LOOP = replace(LoopConfig.named("smoke"), new_positions_per_generation=200)
ARENA = ArenaConfig.named("smoke")
SP = replace(SelfPlayConfig.named("smoke"), simulations=8, games_per_worker=4, max_plies=60)
CONFIGS = {"loop": LOOP, "train": TRAIN, "arena": ARENA, "selfplay": SP}


class Stub:
    def evaluate(self, boards, legal_indices):
        return [np.full(len(i), 1.0 / len(i)) for i in legal_indices], np.zeros(len(boards))


def summ(pair_points):
    recs = []
    for o, p in enumerate(pair_points):
        a = min(p, 1.0)
        for w, s in ((True, a), (False, p - a)):
            recs.append({"key": f"{o}{w}", "opening": o, "a_white": w, "score_a": s, "result_white": 0,
                         "termination": "x", "plies": 80})
    return summarize(recs)


WIN, NEUTRAL, LOSE = summ([2, 1.5, 1] * 2), summ([1] * 6), summ([0, 0.5, 1] * 2)
MIRROR = summ([1] * 2)  # contrôle (a) parfait : chaque paire 1 point sur 2


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Sortie d'une session de self-play SEUL au format exact de l'étape 6 (vrai scripts/selfplay.py), réseau tiny
    de départ, données « humaines » tirées de ces positions."""
    root = tmp_path_factory.mktemp("loop")
    torch.manual_seed(0)
    mc = ModelConfig.named("tiny")
    init = root / "kaggle_sup.pt"
    torch.save({"step": 0, "model": build_model(mc).state_dict(), "config": {"model": asdict(mc)}}, init)
    s0 = root / "s0"
    cmd = [sys.executable, str(ROOT / "scripts" / "selfplay.py"), "--out", str(s0 / "selfplay"), "--checkpoint",
           str(init), "--config", "smoke", "--workers", "1", "--threads", "2", "--games-per-worker", "4",
           "--max-games", "6", "--simulations", "8", "--syzygy", "", "--device", "cpu", "--session", "s6e_test"]
    subprocess.run(cmd, check=True, capture_output=True)
    shards = [load_shard(s["npz"]) for s in list_shards([s0 / "selfplay"])]
    boards = np.concatenate([d["boards"] for d in shards])
    human = {"boards": boards, "moves": np.concatenate([d["played"] for d in shards]).astype(np.int16),
             "results": np.concatenate([d["result"] for d in shards]),
             "value_valid": np.ones(len(boards), dtype=np.uint8)}
    return {"root": root, "s0": s0, "init": init, "human": human, "val": human_val_set(human, 0, 0, None),
            "positions": len(boards)}


@pytest.fixture(autouse=True)
def accept_guardrails(monkeypatch):
    """Garde-fous toujours passés (sauf script contraire) : les tests portent sur l'orchestrateur."""
    script = {"outcomes": []}

    def fake(metrics, baseline, previous, history, generation, cfg):
        ok = script["outcomes"].pop(0) if script["outcomes"] else True
        return {"accept": ok, "reasons": [] if ok else ["G1 : test"], "alerts": [], "actions": []}
    monkeypatch.setattr(spt, "decide", fake)
    return script


class FakeExecutor:
    """Phases jouées en mémoire, horloge simulée (secondes) ; mêmes codes de sortie que les vrais scripts."""

    def __init__(self, world, work):
        self.world, self.work = world, Path(work)
        self.clock = [0.0]
        self.calls = []
        self.fail = {}
        self.scores = {"control_a": MIRROR, "control_b": WIN}
        self.gate, self.anchor = WIN, WIN  # NEUTRAL contre l'ancre après une génération 1 à 80 % = veto
        self.minutes = {"selfplay": None, "train": 3.0, "arena": 5.0, "commit": 0.1}
        self.rate = 120.0  # positions par minute de self-play
        self.last_tail = ""
        self.n = 0

    def _fail(self, kind):
        if self.fail.get(kind, 0) > 0:
            self.fail[kind] -= 1
            self.last_tail = f"Traceback (most recent call last):\nRuntimeError: boom ({kind})"
            self.clock[0] += 30
            return True
        return False

    def selfplay(self, minutes, checkpoint, generation, session, soft, hard, target_positions=None, min_minutes=None):
        self.calls.append(("selfplay", round(minutes, 3), Path(checkpoint).name, generation))
        self.chunks = getattr(self, "chunks", []) + [(minutes, target_positions, min_minutes)]
        if self._fail("selfplay"):
            return 1
        out = self.work / "selfplay"
        writer = ShardWriter(out, f"{session}_w0", {"generation": generation}, 1000, 1e9)
        target, produced, stats = max(int(self.rate * minutes), 1), 0, None
        if target_positions is not None:  # tranche sur objectif : le manque, au moins `min_minutes`, au plus `minutes`
            target = min(target, max(target_positions, int(self.rate * (min_minutes or 0)), 1))
            minutes = target / self.rate
        rng = np.random.default_rng(self.n)
        self.n += 1
        while produced < target:
            stats = run_selfplay(Stub(), SP, SP.mcts_config(MCTSConfig()), writer, rng, f"{session}_{produced}",
                                 max_games=2)
            produced += stats["positions"]
        own = [load_shard(p) for p in writer.written]
        (out / "sessions").mkdir(parents=True, exist_ok=True)
        (out / "sessions" / f"{session}.json").write_text(json.dumps({
            "session": session, "games": sum(len(d["game_uid"]) for d in own), "positions": produced,
            "play_minutes": minutes, "simulations_per_second": 3900.0, "unfinished_dropped": 0,
            "terminations": {"mat": 1}, "repetitions": repetition_stats(own), "false_resignations": 0,
            "resign_triggered_in_control": 0, "control_games": 0}), encoding="utf-8")
        sync_buffer([], out, SP.buffer_positions)
        self.clock[0] += 60 * minutes
        return 0

    def trainer(self):
        return SelfPlayTrainer(self.work / "sp_train", TRAIN, "cpu", self.world["init"], log=lambda m: None)

    def train(self, max_minutes):
        self.calls.append(("train",))
        if self._fail("train"):
            return 1
        self.clock[0] += 60 * self.minutes["train"]
        try:
            res = self.trainer().run_round([self.work / "selfplay"], self.world["human"], self.world["val"],
                                           steps_override=4)
        except RuntimeError:
            return 1
        if res["status"] == "interrupted":
            return 3
        if res["status"] == "rejected":
            return 2 if res["loop_status"] == "stop" else 1
        return 0

    def arena(self, kind, seed, a, a_name, a_sims, b, b_name, b_sims, out, pairs, max_minutes):
        self.calls.append(("arena", out.name, kind, Path(b).name))
        if self._fail("arena"):
            return 1
        name = out.name
        s = self.scores.get(name) or (self.anchor if name.endswith("anchor") else self.gate)
        if callable(s):
            s = s(name)
        out.mkdir(parents=True, exist_ok=True)
        (out / "arena.json").write_text(json.dumps({"players": {"a": {"sha256": sha256_file(a)},
                                                                "b": {"sha256": sha256_file(b)}}}), encoding="utf-8")
        (out / "result.json").write_text(json.dumps({"summary": s}), encoding="utf-8")
        self.clock[0] += 60 * self.minutes["arena"]
        return 0

    def commit(self, cid, gate, anchor):
        self.calls.append(("commit", cid))
        if self._fail("commit"):
            return 1
        load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))["summary"] if p else None
        c = self.trainer().commit_arena(cid, load(gate), load(anchor), ARENA)
        self.clock[0] += 6
        return 2 if c["loop_status"] == "stop" else (0 if c["accepted"] else 1)


def orch(world, ex, work, minutes, expect, resume_from=None, crash_at=None, configs=CONFIGS, root=None):
    return Orchestrator(work, "smoke", world["init"], ex, minutes, expect, resume_from=resume_from,
                        crash_at=crash_at, clock=lambda: ex.clock[0], log=lambda m: None,
                        root=root or ROOT, configs=configs)


def boot(world, tmp_path, minutes=60, name="s1", ex=None, **kw):
    work = tmp_path / name
    ex = ex or FakeExecutor(world, work)
    ex.work = work
    o = orch(world, ex, work, minutes, 0, resume_from=world["s0"], **kw)
    return o, ex, work


def arena_calls(ex):
    return [c for c in ex.calls if c[0] == "arena"]


# --------------------------------------------------------------------------- fonctions pures


VIEW = {"status": "ok", "candidate": None, "round_in_progress": None, "accepted": {"is_init": False}}


def test_next_phase_follows_durable_state():
    np_ = lambda v, g=None, a=None, new=0, wait=False: next_phase({**VIEW, **v}, g, a, new, LOOP, 0.52, wait)
    cand = {"candidate": {"id": 3}}
    assert np_({"status": "stop"}) == "stop"
    assert np_(cand) == "arena_gate"
    assert np_(cand, g=WIN) == "arena_anchor"
    assert np_(cand, g=WIN, a=NEUTRAL) == "verdict"
    assert np_(cand, g=LOSE) == "verdict"  # arène perdue : pas d'ancre
    assert np_({**cand, "accepted": {"is_init": True}}, g=WIN) == "verdict"  # génération 1 : pas d'arène ancre
    assert np_({"round_in_progress": {"id": 2}}) == "train"
    assert np_({}, new=LOOP.new_positions_per_generation) == "train"
    ninety = math.ceil(0.9 * LOOP.new_positions_per_generation)  # un tour dès 90 % de l'objectif
    assert np_({}, new=ninety) == "train" and np_({}, new=ninety - 1) == "selfplay"
    assert np_({}, new=LOOP.new_positions_per_generation, wait=True) == "selfplay"
    assert np_({}, new=10) == "selfplay"


def test_selfplay_chunk_stops_on_a_positions_target_or_plays_to_the_end():
    """Règle des tranches (2026-10-24) : une tranche s'arrête sur l'objectif de positions (le manque), sa durée
    maximale est le temps qui laisse la place au tour ; sinon self-play jusqu'à la fin de la session."""
    cfg = LoopConfig()  # marges 15 / 10 min, tranche minimale 25 min
    est = {"positions_per_minute": 1000.0}
    # 60 min de self-play + 40 de tour + 15 de marge <= 290 : objectif = le manque, durée maximale 290 - 40 - 15
    assert selfplay_chunk(300, 60_000, est, 40, cfg) == (235, False, 60_000)
    assert selfplay_chunk(300, 1_000, est, 40, cfg) == (235, False, 1_000)  # la durée minimale est dans selfplay.py
    assert selfplay_chunk(300, -5, est, 40, cfg)[2] == 0
    minutes, final, target = selfplay_chunk(100, 60_000, est, 40, cfg)  # pas le temps de tout faire : jusqu'à la fin
    assert final and minutes == 90 and target is None
    assert selfplay_chunk(30, 60_000, est, 40, cfg) == (0.0, True, None)  # moins que la tranche minimale


def test_selfplay_chunks_of_a_session_use_the_target_and_the_minimum_duration(world, tmp_path):
    o, ex, work = boot(world, tmp_path, minutes=60)
    o.run()
    chunks = [c for c in ex.chunks if c[1] is not None]
    assert chunks and all(c[2] == CONFIGS["loop"].min_chunk_minutes for c in chunks)
    events = [e for e in read_loop_state(work / "loop" / "loop_state.json")["events"] if e["phase"] == "selfplay"]
    assert any(e.get("target_positions") for e in events)


def test_expected_session_messages():
    check_expected_session(0, 0, "selfplay_only")
    with pytest.raises(LoopStop, match="obligatoire"):
        check_expected_session(0, None, "selfplay_only")
    with pytest.raises(LoopStop, match="trouvé la sortie de la session de boucle n°1, attendu la session n°2"):
        check_expected_session(1, 2, "loop")
    with pytest.raises(LoopStop, match="self-play seul"):
        check_expected_session(0, 1, "selfplay_only")


def test_actions_recommend_v2_and_flag_problems():
    hist = [{"generation": g, "elo": e} for g, e in ((1, 20), (2, 40), (3, 70), (4, 80), (5, 90))]
    view = {"status": "ok", "generation": 5, "anchor_history": hist}
    base = {"view": view, "end_reason": "budget", "notes": [], "v2": [], "selfplay": [], "alerts": [],
            "gen2_bias": False, "prev_sps": None, "sps": None, "session": 4}
    acts = actions(base, LOOP)
    assert any(a.startswith("MESURE v2 recommandée sur gen_0005") for a in acts)  # lissé +80 > +50
    assert not any("MESURE v2" in a for a in actions({**base, "v2": [4]}, LOOP))  # 80 - 80 : rien
    assert acts[-1].endswith("EXPECTED_SESSION = 4")
    sp = [{"resign_triggered_in_control": 10, "false_resignations": 2,
           "repetitions": {"games": 100, "by_ply": {"< 60": 9}}}]
    acts = actions({**base, "v2": [5], "selfplay": sp, "gen2_bias": True, "prev_sps": 4000, "sps": 3000,
                    "view": {**view, "status": "stop", "stop_reason": "stagnation"}}, LOOP)
    text = "\n".join(acts)
    for word in ("ARRÊT (stagnation)", "FAUX ABANDONS", "RÉPÉTITIONS PRÉCOCES", "GÉNÉRATION 2", "DÉBIT"):
        assert word in text


def test_measured_v2_generations_are_read_from_the_repository(tmp_path):
    d = tmp_path / "docs" / "stockfish" / "v2"
    d.mkdir(parents=True)
    for name in ("kaggle_sup.json", "sp_gen0005.json", "sp_gen0012.json"):
        (d / name).write_text("{}", encoding="utf-8")
    assert measured_v2_generations(tmp_path) == [5, 12]


# --------------------------------------------------------------------------- entrée et démarrage


def test_bootstrap_from_a_selfplay_only_output(world, tmp_path):
    info = classify_output(world["s0"])
    assert info["kind"] == "selfplay_only" and info["positions"] == world["positions"]
    o, ex, work = boot(world, tmp_path, minutes=1)  # budget minuscule : démarrage seulement
    logs = []
    o.log = logs.append
    o.run()
    assert any(l.startswith("Démarrage de la boucle à partir d'une sortie de self-play seul") and
               f"{world['positions']:,} positions" in l for l in logs)
    st = read_train_state(work / "sp_train")
    assert st["accepted"]["is_init"] and st["init_sha256"] == sha256_file(world["init"])
    loop = read_loop_state(work / "loop" / "loop_state.json")
    assert [s["n"] for s in loop["sessions"]] == [1] and loop["sessions"][0]["input"] == "selfplay_only"
    assert (work / "selfplay" / "sessions" / "s6e_test.json").exists()  # statistiques de l'étape 6 recopiées
    with pytest.raises(LoopStop, match="attendu la session n°1"):
        orch(world, FakeExecutor(world, tmp_path / "x"), tmp_path / "x", 10, 1, resume_from=world["s0"]).run()


def test_bootstrap_refused_under_the_minimum_buffer(world, tmp_path):
    configs = {**CONFIGS, "train": replace(TRAIN, min_buffer_positions=250_000)}
    o, ex, work = boot(world, tmp_path, configs=configs)
    with pytest.raises(LoopStop, match=r"Tampon trop petit .* \d[\d,]* positions d'entraînement.*minimum 250,000"):
        o.run()
    assert not (work / "sp_train").exists()


def test_inconsistent_inputs_are_refused(world, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(world["s0"], bad)
    (bad / "loop").mkdir()
    (bad / "loop" / "loop_state.json").write_text(json.dumps({"sessions": [{"n": 3}]}), encoding="utf-8")
    assert classify_output(bad)["kind"] == "inconsistent"  # tampon + état de boucle, sans état d'entraînement
    with pytest.raises(LoopStop, match="Entrée incohérente"):
        orch(world, FakeExecutor(world, tmp_path / "w"), tmp_path / "w", 10, 0, resume_from=bad).run()
    bad2 = tmp_path / "bad2"
    shutil.copytree(world["s0"], bad2)
    (bad2 / "sp_train").mkdir()  # tampon + dossier d'entraînement (boucle provisoire de 6e), sans état de boucle
    assert classify_output(bad2)["kind"] == "inconsistent"


def test_an_older_but_valid_output_is_refused(world, tmp_path):
    o, ex, a = boot(world, tmp_path, minutes=30, name="A")
    o.run()
    ex.work = tmp_path / "B"
    orch(world, ex, tmp_path / "B", 30, 1, resume_from=a).run()
    assert read_loop_state(tmp_path / "B" / "loop" / "loop_state.json")["sessions"][-1]["n"] == 2
    with pytest.raises(LoopStop) as e:  # sortie de la session 1 alors qu'on attend celle de la session 2
        orch(world, FakeExecutor(world, tmp_path / "C"), tmp_path / "C", 30, 2, resume_from=a).run()
    assert "trouvé la sortie de la session de boucle n°1, attendu la session n°2" in str(e.value)


# --------------------------------------------------------------------------- une session complète


def test_full_session_generations_controls_report(world, tmp_path):
    o, ex, work = boot(world, tmp_path, minutes=90)
    res = o.run()
    st = read_train_state(work / "sp_train")
    assert st["generation"] >= 3, ex.calls
    gates = [c for c in arena_calls(ex) if c[1].endswith("_gate")]
    assert gates[0][2] == "anchor" and gates[0][3] == "kaggle_sup.pt"  # génération 1 : ouvertures fixes, ancre
    assert all(c[2] == "gate" and c[3].startswith("gen_") for c in gates[1:])
    anchors = [c for c in arena_calls(ex) if c[1].endswith("_anchor")]
    assert len(anchors) == len(gates) - 1 and all(c[3] == "kaggle_sup.pt" for c in anchors)
    assert [c[1] for c in arena_calls(ex)[:2]] == ["control_a", "control_b"]
    sp = [c for c in ex.calls if c[0] == "selfplay"]
    # le self-play joue toujours avec le réseau ACCEPTÉ (jamais un candidat), le dernier à la fin
    assert all(c[2] == ("kaggle_sup.pt" if c[3] == 0 else f"gen_{c[3]:04d}.pt") for c in sp)
    assert sp[-1][3] >= st["generation"] - 1
    text = res["text"]
    assert "**À mettre dans EXPECTED_SESSION pour la session suivante : 1**" in text
    assert "## Actions pour toi" in text and "GÉNÉRATION 2" in text and "Contrôle (a)" in text
    assert (work / "loop" / "progress.png").exists()
    hist = (work / "loop" / "history.csv").read_text(encoding="utf-8").splitlines()
    assert hist[0].startswith("candidate,accepted") and len(hist) == 1 + st["commits"]
    loop = read_loop_state(work / "loop" / "loop_state.json")
    assert loop["controls"]["done"] and loop["estimates"]["arena_per_game"]
    assert loop["sessions"][0]["finished"] is not None and loop["sessions"][0]["report"].endswith("0001.md")


def test_control_a_only_warns_and_control_b_stops(world, tmp_path):
    o, ex, work = boot(world, tmp_path, minutes=40, name="ctl_a")
    ex.scores["control_a"] = summ([2, 1, 1])  # 66,7 %, paires asymétriques
    res = o.run()
    assert "AVERTISSEMENT" in res["text"] and any(c[0] == "train" for c in ex.calls)  # la session continue
    o, ex, work = boot(world, tmp_path, minutes=40, name="ctl_b")
    ex.scores["control_b"] = LOSE
    res = o.run()
    assert res["end_reason"].startswith("contrôle (b) échoué") and not any(c[0] == "train" for c in ex.calls)
    assert "CONTRÔLE (b) ÉCHOUÉ" in res["text"]


def test_a_failed_phase_is_retried_once_then_stops(world, tmp_path):
    o, ex, work = boot(world, tmp_path, minutes=40, name="r1")
    ex.fail = {"selfplay": 1}
    res = o.run()
    assert not res["end_reason"].startswith("erreur") and "ÉCHEC n°1" in res["text"] and "boom (selfplay)" in res["text"]
    o, ex, work = boot(world, tmp_path, minutes=40, name="r2")
    ex.fail = {"arena": 2}
    res = o.run()
    assert res["end_reason"] == "erreur : la phase control_a a échoué 2 fois (code 1)"
    assert len(arena_calls(ex)) == 2 and "ERREUR" in res["text"]


def test_decisional_stop_is_immediate_without_retry(world, tmp_path, accept_guardrails):
    accept_guardrails["outcomes"] = [False, False, False]  # G4 au 3e rejet par les garde-fous
    o, ex, work = boot(world, tmp_path, minutes=60, name="g4")
    res = o.run()
    assert res["end_reason"] == "arrêt décisionnel : G4"
    assert sum(c[0] == "train" for c in ex.calls) == 3 and "ARRÊT (G4)" in res["text"]


def test_not_enough_time_for_the_arena_defers_it_to_the_next_session(world, tmp_path):
    # contrôles 10 min (5 + 5) -> estimation ~1 min par partie d'arène ; tour possible (8 min restantes >= ~7,1),
    # puis 5 min restantes < arène estimée (0,2 + 6 x 1,0) + marge : arène reportée, self-play jusqu'à la fin
    o, ex, work = boot(world, tmp_path, minutes=18, name="b1")
    res = o.run()
    st = read_train_state(work / "sp_train")
    assert st["candidate"] is not None and "reportée" in res["end_reason"], res["end_reason"]
    assert ex.calls[-1][0] == "selfplay"  # le reste du temps a servi au self-play
    ex.calls.clear()
    orch(world, ex, work, 30, 1).run()  # session suivante, reprise sur place
    assert ex.calls[0][0] == "arena" and ex.calls[0][1] == f"cand_{st['candidate']['id']:04d}_gate"


# --------------------------------------------------------------------------- coupures et fichiers orphelins

CRASH_POINTS = ["after_prepare", "before_selfplay", "after_selfplay", "before_train", "after_train",
                "before_arena_gate", "after_arena_gate", "before_arena_anchor", "after_arena_anchor", "before_verdict",
                "after_verdict_before_save", "after_verdict", "before_report"]


@pytest.mark.parametrize("point", CRASH_POINTS)
def test_cut_at_each_phase_resumes_consistently(world, tmp_path, point):
    o, ex, work = boot(world, tmp_path, minutes=70, name="cut", crash_at=point)
    if point == "before_arena_anchor":  # il faut une génération 1 avant la première arène ancre
        pass
    with pytest.raises(CrashForTest):
        o.run()
    resumed = orch(world, ex, work, 70, 1)  # même dossier, la session coupée est la n°1
    resumed.run()
    st = read_train_state(work / "sp_train")
    commits = [r for r in read_jsonl(work / "sp_train" / "rounds.jsonl") if r.get("type") == "arena"]
    assert len({c["candidate"] for c in commits}) == len(commits) == st["commits"]  # un verdict par candidat
    assert st["generation"] == sum(c["accepted"] for c in commits) >= 2
    names = [c[1] for c in arena_calls(ex)]
    assert len(names) == len(set(names)), names  # aucune arène rejouée
    for s in list_shards([work / "selfplay"]):
        load_shard(s["npz"])  # tampon intact
    loop = read_loop_state(work / "loop" / "loop_state.json")
    assert [s["n"] for s in loop["sessions"]] == [1, 2] and loop["sessions"][-1]["report"]


def test_stale_arena_dir_of_another_candidate_is_set_aside_and_replayed(world, tmp_path):
    o, ex, work = boot(world, tmp_path, minutes=30, name="stale")
    o.crash_at = "before_arena_gate"
    with pytest.raises(CrashForTest):
        o.run()
    cid = read_train_state(work / "sp_train")["candidate"]["id"]
    d = work / "loop" / "arena" / f"cand_{cid:04d}_gate"
    d.mkdir(parents=True)
    (d / "arena.json").write_text(json.dumps({"players": {"a": {"sha256": "0" * 64}}}), encoding="utf-8")
    (d / "result.json").write_text(json.dumps({"summary": LOSE}), encoding="utf-8")  # résultat d'un autre réseau
    ex.calls.clear()
    orch(world, ex, work, 30, 1).run()
    assert ex.calls[0] == ("arena", d.name, "anchor", "kaggle_sup.pt")  # rejouée
    assert list((work / "loop" / "arena").glob(f"{d.name}.stale*"))
    assert read_train_state(work / "sp_train")["generation"] >= 1  # décidé avec le bon résultat (WIN)


def test_orphan_arena_dirs_of_committed_candidates_are_ignored(world, tmp_path):
    o, ex, work = boot(world, tmp_path, minutes=45, name="orph")
    o.run()
    done = {c[1] for c in arena_calls(ex)}
    ex.calls.clear()
    orch(world, ex, work, 30, 1).run()
    assert not {c[1] for c in arena_calls(ex)} & done  # aucune arène d'un candidat déjà décidé n'est relancée


def test_mid_game_repetition_alert_against_the_reference_session():
    """Référence (self-play seul, 2026-10-22) : 916 répétitions sur 3 803 parties dont 364 avec < 20 points de
    matériel -> 552 / 3 803 = 14,5 % ; alerte au-delà de 19,5 %."""
    def ctx(games, reps, endgame):
        sp = [{"repetitions": {"games": games, "repetition_draws": reps, "by_ply": {"< 60": 0},
                               "by_material": {">= 60 (presque complet)": 0, "40-59": 0, "20-39": reps - endgame,
                                               "< 20 (finale)": endgame}}}]
        return {"view": {"status": "ok", "generation": 0, "anchor_history": []}, "end_reason": "budget", "notes": [],
                "v2": [], "selfplay": sp, "alerts": [], "gen2_bias": False, "prev_sps": None, "sps": None,
                "session": 1}
    alert = lambda c: any("RÉPÉTITIONS EN MILIEU DE PARTIE" in a for a in actions(c, LoopConfig.named("kaggle")))
    assert not alert(ctx(3803, 916, 364))  # la session de référence elle-même
    assert not alert(ctx(1000, 290, 100))  # 19,0 %
    assert alert(ctx(1000, 300, 100))  # 20,0 % > 14,5 % + 5 points


def test_first_loop_session_consumes_the_starting_buffer_as_planned():
    """Tampon de la session de self-play seul : 370 840 positions (334 393 d'entraînement), aucune déjà utilisée,
    objectif 100 000 par génération."""
    from chesszero.loop import estimates, new_loop_state, round_minutes
    from chesszero.selfplay_train import plan_steps

    lc, tc = LoopConfig.named("kaggle"), SelfPlayTrainConfig.named("kaggle")
    view = {**VIEW, "accepted": {"is_init": True}}
    assert next_phase(view, None, None, 370_840, lc, 0.52) == "train"  # entraînement tout de suite, pas de self-play
    # premier tour plafonné à un passage moyen par position du tampon : 435 pas (et non 4 x 334 393 / 768 = 1 742)
    assert plan_steps(334_393, 334_393, tc, tc.reuse, tc.human_fraction) == 435
    # ensuite tout le tampon est marqué comme utilisé : self-play de 100 000 positions avec le réseau accepté
    assert next_phase(view, None, None, 0, lc, 0.52) == "selfplay"
    est = estimates(new_loop_state(), lc)
    after = round_minutes(est, 469, 200, 200)
    minutes, final, target = selfplay_chunk(200, 100_000, est, after, lc)
    assert not final and target == 100_000 and 77 < 100_000 / est["positions_per_minute"] < 78  # ~77,5 min attendues
    assert minutes == pytest.approx(200 - lc.end_margin_minutes - after - lc.start_margin_minutes)  # durée maximale
    # deuxième tour : 90 000 nouvelles positions d'entraînement sur ~424 000 : 4 passages, sans plafond
    assert plan_steps(90_000, 334_393 + 90_000, tc, tc.reuse, tc.human_fraction) == 469
    # premier tour estimé : 5 + 435 x 0,006 min + arène de la génération 1 (200 parties, 17,6 min)
    assert 25 < round_minutes(est, 435, 200, 0) < 26


def test_slow_data_reading_is_flagged_above_30_percent():
    from chesszero.selfplay_train import timing_shares

    def ctx(timing):
        return {"view": {"status": "ok", "generation": 0, "anchor_history": []}, "end_reason": "budget", "notes": [],
                "v2": [], "selfplay": [], "alerts": [], "gen2_bias": False, "prev_sps": None, "sps": None,
                "session": 1, "timings": [(7, timing_shares(timing))]}
    flagged = lambda t: any("LECTURE DES DONNÉES LENTE (tour 7)" in a for a in actions(ctx(t), LoopConfig()))
    base = {"load": 10, "human_read": 5, "buffer_read": 5, "human_masks": 5, "prepare": 5, "compute": 60, "eval": 10}
    assert not flagged({**base, "wall": 100})  # lecture des lots 10 %
    assert flagged({**base, "human_read": 20, "buffer_read": 16, "wall": 116})  # 36/116 = 31 %
    # le chargement initial des données Lichess ne compte plus dans la lecture : alerte à part au-delà de 8 min
    loaded = lambda t: [a for a in actions(ctx(t), LoopConfig()) if "CHARGEMENT INITIAL" in a]
    assert not flagged({**base, "load": 260, "wall": 327}) and not loaded({**base, "load": 260, "wall": 327})
    assert loaded({**base, "load": 8.5 * 60, "wall": 600}) and not flagged({**base, "load": 8.5 * 60, "wall": 600})


def test_reset_line_at_the_start_of_a_session(world, tmp_path):
    """Option 1 (décision du 2026-10-24) : `--reset-line` remet la ligne sur le réseau accepté au début de la session
    suivante ; le tampon est gardé et redevient « nouveau » : la session commence par un tour."""
    ex = FakeExecutor(world, tmp_path / "A")
    ex.gate = LOSE  # deux candidats rejetés, comme la session Kaggle 1
    o, ex, a = boot(world, tmp_path, minutes=40, name="A", ex=ex)
    o.run()
    before = read_train_state(a / "sp_train")
    assert before["generation"] == 0 and before["arena_rejections"] >= 1 and before["trained_shards"]
    b = tmp_path / "B"
    ex.work, ex.calls = b, []
    lines = Orchestrator(b, "smoke", world["init"], ex, 30, 1, resume_from=a, clock=lambda: ex.clock[0],
                         log=lambda m: None, configs=CONFIGS, reset_line="politique aplatie").check_inputs()
    assert any("Remise de la ligne demandée (« politique aplatie »)" in x for x in lines) and not b.exists()
    o2 = Orchestrator(b, "smoke", world["init"], ex, 30, 1, resume_from=a, clock=lambda: ex.clock[0],
                      log=lambda m: None, configs=CONFIGS, reset_line="politique aplatie")
    res = o2.run()
    after_reset = read_train_state(b / "sp_train")["resets"]
    assert len(after_reset) == 1 and after_reset[0]["before"]["arena_rejections"] == before["arena_rejections"]
    assert ex.calls[0] == ("train",)  # tout le tampon est redevenu nouveau pour la ligne remise à zéro
    assert "Ligne d'entraînement REMISE" in res["text"]
    loop = read_loop_state(b / "loop" / "loop_state.json")
    assert loop["sessions"][-1]["reset_line"]["note"] == "politique aplatie"
    assert read_train_state(a / "sp_train")["resets"] == []  # l'entrée (lecture seule sur Kaggle) n'est pas touchée
