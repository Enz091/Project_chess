"""Étape 7b : candidat puis arène, continuation de la ligne, compteurs en générations acceptées, arrêts, coupures."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

import chesszero.selfplay_train as spt
from chesszero.arena import ArenaConfig, read_jsonl, summarize
from chesszero.checkpoint import load_for_inference
from chesszero.selfplay_train import SelfPlayTrainer, read_train_state

from test_selfplay_train import CFG, world  # noqa: F401  (fixture du module voisin : tampon, Lichess, réseau tiny)

ACFG = ArenaConfig()
NO_VETO = ArenaConfig(anchor_veto_se=100.0)


def summ(pair_points):
    """Résumé d'arène (score du candidat) à partir des points de chaque paire (0 à 2 par demi-points)."""
    recs = []
    for o, p in enumerate(pair_points):
        a = min(p, 1.0)
        for w, s in ((True, a), (False, p - a)):
            recs.append({"key": f"{o}{w}", "opening": o, "a_white": w, "score_a": s, "result_white": 0,
                         "termination": "x", "plies": 80})
    return summarize(recs)


WIN = summ([2, 1.5, 1, 2, 1.5] * 4)  # 80 %
NEUTRAL = summ([1] * 10 + [0.5, 1.5] * 5)  # 50 %
BELOW = summ([0, 0.5, 1, 0.5, 0] * 4)  # 20 % : borne haute de l'Elo < 0
LOSE = summ([1, 0.5, 1, 0.5] * 5)  # 37,5 % : rejet
assert BELOW["elo_high"] < 0 and NEUTRAL["elo_high"] > 0 and WIN["score"] >= 0.52 > LOSE["score"]


def make(world, out, cfg=CFG):
    torch.manual_seed(0)
    return SelfPlayTrainer(world["root"] / out, cfg, "cpu", world["init"], log=lambda m: None)


def run(t, world, steps=3):
    return t.run_round([world["buffer"]], world["human"], world["val"], steps_override=steps)


@pytest.fixture
def scripted(monkeypatch):
    """Garde-fous scriptés : `scripted(True, False, ...)` ; renvoie la liste des `generation` reçues par decide."""
    seen = []

    def install(*outcomes):
        it = iter(outcomes)

        def fake(metrics, baseline, previous, history, generation, cfg):
            seen.append(generation)
            ok = next(it)
            return {"accept": ok, "reasons": [] if ok else ["G1 : test"], "alerts": [], "actions": []}
        monkeypatch.setattr(spt, "decide", fake)
        return seen
    return install


def weights(model_or_sd):
    sd = model_or_sd.state_dict() if hasattr(model_or_sd, "state_dict") else model_or_sd
    return {k: v.clone() for k, v in sd.items()}


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def cand_weights(t):
    return torch.load(t.out / "candidate.pt", map_location="cpu", weights_only=False)["model"]


# --------------------------------------------------------------------------- candidat et verdict


def test_guardrail_pass_makes_a_pending_candidate(world, scripted):
    scripted(True)
    t = make(world, "c1")
    res = run(t, world)
    assert res["status"] == "candidate" and t.state["generation"] == 0
    st = read_train_state(t.out)
    assert st["candidate"]["id"] == res["candidate"] == 0 and st["accepted"]["is_init"]
    assert st["candidate"]["opponent_sha256"] == st["accepted"]["sha256"]  # SHA-256 du réseau de départ
    assert t.state["trained_shards"] and t.state["line_ok_round"] == 0
    assert torch.load(t.out / "line_ok.pt", weights_only=False)["round"] == 0
    with pytest.raises(RuntimeError, match="en attente"):
        run(t, world)  # pas de nouveau tour avant le verdict


def test_first_acceptance_writes_gen_0001_and_uses_the_gate_as_anchor_measure(world, scripted):
    scripted(True)
    t = make(world, "c2")
    run(t, world)
    c = t.commit_arena(0, WIN, None, ACFG)
    assert c["accepted"] and t.state["generation"] == 1 and t.state["candidate"] is None
    model, meta = load_for_inference(t.out / "generations" / "gen_0001.pt")
    payload = torch.load(t.out / "generations" / "gen_0001.pt", weights_only=False)
    assert payload["generation"] == 1 and "biaisée par la sélection" in payload["arena"]["anchor_source"]
    st = read_train_state(t.out)
    assert st["accepted"]["path"].endswith("gen_0001.pt") and st["anchor_history"][0]["elo"] == WIN["elo"]
    assert t.commit_arena(0, WIN, None, ACFG) == c  # verdict rejoué : aucun effet
    assert t.state["generation"] == 1 and len(t.state["commits"]) == 1
    assert sum(r.get("type") == "arena" for r in read_jsonl(t.out / "rounds.jsonl")) == 1
    with pytest.raises(ValueError, match="pas en attente"):
        t.commit_arena(1, WIN, None, ACFG)


def test_arena_rejection_continues_the_line_and_next_candidate_faces_the_same_accepted(world, scripted):
    scripted(True, True, True, True)
    t = make(world, "c3")
    run(t, world)
    t.commit_arena(0, WIN, None, ACFG)  # génération 1
    gen1 = read_train_state(t.out)["accepted"]["sha256"]
    run(t, world)  # candidat 1
    c1 = weights(cand_weights(t))
    step = t.state["global_step"]
    t.commit_arena(1, LOSE, None, ACFG)  # rejeté en arène
    assert same(weights(t.model), c1)  # rien n'est défait : la ligne continue depuis le candidat rejeté
    assert t.state["arena_rejections"] == 1 and t.state["generation"] == 1
    run(t, world)  # candidat 2
    assert t.state["global_step"] == step + 3
    st = read_train_state(t.out)
    assert st["candidate"]["opponent_sha256"] == gen1 == st["accepted"]["sha256"]  # toujours gen 1, pas le candidat 1
    t.commit_arena(2, WIN, NEUTRAL, NO_VETO)
    assert t.state["generation"] == 2 and t.state["arena_rejections"] == 0  # une acceptation remet à zéro


def test_guardrail_rejection_after_arena_rejection_returns_to_that_candidate(world, scripted):
    """Poids comparés au bit près (torch.equal sur chaque tenseur), dans les deux cas : réseau accepté = réseau de
    départ (génération 0), puis réseau accepté = gen_0001 ; l'optimiseur revient lui aussi à l'état du candidat."""
    scripted(True, False, True, True, False)
    init = torch.load(world["init"], weights_only=False)["model"]
    t = make(world, "c4")
    run(t, world)
    c0 = weights(cand_weights(t))
    t.commit_arena(0, LOSE, None, ACFG)  # rejeté en arène : la ligne continue
    assert run(t, world)["status"] == "rejected"  # rejeté par les garde-fous
    assert same(weights(t.model), c0) and not same(weights(t.model), init)  # le candidat, pas l'accepté (départ)
    # même chose quand l'accepté est une vraie génération
    run(t, world)
    t.commit_arena(2, WIN, None, ACFG)  # génération 1
    gen1 = torch.load(t.out / "generations" / "gen_0001.pt", weights_only=False)["model"]
    run(t, world)
    c3 = weights(cand_weights(t))
    opt3 = torch.load(t.out / "line_ok.pt", weights_only=False)["optimizer"]
    t.commit_arena(3, LOSE, None, ACFG)  # rejeté en arène
    assert run(t, world)["status"] == "rejected"
    assert same(weights(t.model), c3) and not same(weights(t.model), gen1)  # le candidat 3, pas gen_0001
    exp = opt3["state"]
    got = t.optimizer.state_dict()["state"]
    assert all(torch.equal(exp[i]["exp_avg"], got[i]["exp_avg"]) for i in exp)


def test_veto_and_missing_anchor_through_commit(world, scripted):
    scripted(True, True, True)
    t = make(world, "c5")
    run(t, world)
    t.commit_arena(0, WIN, None, ACFG)  # gen 1 : mesure contre l'ancre = WIN (80 %)
    run(t, world)
    with pytest.raises(ValueError, match="arène ancre manquante"):
        t.commit_arena(1, WIN, None, ACFG)
    c = t.commit_arena(1, WIN, BELOW, ACFG)  # bat l'accepté, mais bien plus faible que lui contre l'ancre
    assert not c["accepted"] and "veto de l'ancre" in c["reasons"][0] and t.state["arena_rejections"] == 1
    run(t, world)
    assert t.commit_arena(2, LOSE, None, ACFG)["accepted"] is False  # arène perdue : ancre inutile


# --------------------------------------------------------------------------- compteurs et arrêts


def test_strict_guards_and_human_share_follow_arena_accepted_generations(world, scripted):
    seen = scripted(*[True] * 9)
    t = make(world, "c6")
    for cid in range(4):  # 4 candidats rejetés en arène : toujours génération 0
        run(t, world)
        t.commit_arena(cid, LOSE, None, ACFG)
    assert t.state["generation"] == 0 and t.human_fraction() == CFG.human_fraction and seen == [0, 0, 0, 0]
    for cid in range(4, 9):  # 5 acceptations
        run(t, world)
        t.commit_arena(cid, WIN, None if cid == 4 else NEUTRAL, NO_VETO)
    assert seen == [0, 0, 0, 0, 0, 1, 2, 3, 4]  # decide reçoit le nombre de générations ACCEPTÉES EN ARÈNE
    assert t.state["generation"] == 5 and t.human_fraction() == CFG.human_fraction_late  # 10 % dès la 6e


def test_stop_after_five_arena_rejections_ignores_guardrail_rejections(world, scripted):
    scripted(True, True, True, True, False, False, True, True)
    t = make(world, "c7")
    for cid in range(4):
        run(t, world)
        t.commit_arena(cid, LOSE, None, ACFG)
    for _ in range(2):  # deux rejets par les garde-fous : ni comptés, ni remis à zéro (et pas de G4 : < 3)
        assert run(t, world)["status"] == "rejected"
    assert t.state["arena_rejections"] == 4 and t.state["consecutive_rejections"] == 2 and t.state["status"] == "ok"
    run(t, world)
    c = t.commit_arena(6, LOSE, None, ACFG)
    assert c["loop_status"] == "stop" and t.state["stop_reason"] == "stagnation" and "5 rejets" in c["alerts"][0]
    with pytest.raises(RuntimeError, match="stagnation"):
        run(t, world)
    t.clear_stop("essai")
    assert t.state["status"] == "ok" and t.state["arena_rejections"] == 0
    run(t, world)  # repart


def test_anchor_stop_counts_only_generations_from_3(world, scripted):
    """Lecture A : sous l'ancre aux générations 2 et 3 -> pas d'arrêt (la 2 ne compte pas) ; 4 aussi -> arrêt."""
    scripted(*[True] * 4)
    t = make(world, "c8")
    anchors = [None, BELOW, BELOW, BELOW]  # gen 1 : mesure = l'arène contre l'accepté (WIN)
    for cid, anchor in enumerate(anchors):
        run(t, world)
        c = t.commit_arena(cid, WIN, anchor, NO_VETO)
        assert c["accepted"]
        assert c["loop_status"] == ("stop" if cid == 3 else "ok"), cid
    assert t.state["stop_reason"] == "ancre" and "3 à 4" in c["alerts"][0]
    t.clear_stop("vu")
    assert t.state["anchor_stop_ack"] == 4  # il faudra deux nouvelles générations sous l'ancre


def test_isolated_generation_below_the_anchor_does_not_stop(world, scripted):
    scripted(*[True] * 5)
    t = make(world, "c9")
    for cid, anchor in enumerate([None, NEUTRAL, BELOW, NEUTRAL, BELOW]):
        run(t, world)
        assert t.commit_arena(cid, WIN, anchor, NO_VETO)["loop_status"] == "ok"
    assert t.state["generation"] == 5


# --------------------------------------------------------------------------- coupures et fichiers orphelins


class Crash(Exception):
    pass


def crash_after(monkeypatch, name, after=None):
    """Fait échouer le processus juste après l'écriture atomique de `name` (si `after` : seulement une fois que
    le fichier `after` a été écrit)."""
    real = spt._atomic_torch_save
    armed = {"on": after is None}

    def boom(obj, path):
        real(obj, path)
        if Path(path).name == after:
            armed["on"] = True
        elif armed["on"] and Path(path).name == name:
            monkeypatch.setattr(spt, "_atomic_torch_save", real)
            raise Crash(name)
    monkeypatch.setattr(spt, "_atomic_torch_save", boom)


def test_cut_after_candidate_written_resumes_to_the_same_candidate(world, scripted, monkeypatch):
    scripted(True, True, True)  # tour de référence, tour coupé, tour repris
    ref = make(world, "k1ref")
    run(ref, world, steps=7)
    crash_after(monkeypatch, "candidate.pt")
    t = make(world, "k1")
    with pytest.raises(Crash):
        run(t, world, steps=7)
    t = make(world, "k1")  # nouveau processus : le tour n'est pas validé, il reprend (pas 6 et 7 refaits)
    assert t.state["candidate"] is None and t.state["round"] is not None
    res = run(t, world)
    assert res["status"] == "candidate" and same(weights(cand_weights(t)), weights(cand_weights(ref)))
    assert sum(r.get("type", "round") == "round" for r in read_jsonl(t.out / "rounds.jsonl")) == 1


def test_cut_before_line_ok_is_repaired_from_train_state(world, scripted, monkeypatch):
    scripted(True, False)
    crash_after(monkeypatch, "train_state.pt", after="candidate.pt")
    t = make(world, "k2")
    with pytest.raises(Crash):
        run(t, world)
    assert torch.load(t.out / "line_ok.pt", weights_only=False)["round"] is None  # encore l'ancien point
    t = make(world, "k2")  # line_ok.pt régénéré depuis train_state.pt
    assert torch.load(t.out / "line_ok.pt", weights_only=False)["round"] == 0
    c0 = weights(cand_weights(t))
    t.commit_arena(0, LOSE, None, ACFG)
    run(t, world)  # rejet par les garde-fous : retour au candidat 0, grâce au line_ok.pt réparé
    assert same(weights(t.model), c0)


def test_cut_after_generation_file_leaves_an_ignored_orphan(world, scripted, monkeypatch):
    scripted(True)
    t = make(world, "k3")
    run(t, world)
    crash_after(monkeypatch, "gen_0001.pt")
    with pytest.raises(Crash):
        t.commit_arena(0, WIN, None, ACFG)
    assert (t.out / "generations" / "gen_0001.pt").exists()  # orphelin : écrit, non référencé
    st = read_train_state(t.out)
    assert st["generation"] == 0 and st["accepted"]["is_init"] and st["candidate"]["id"] == 0
    t = make(world, "k3")
    c = t.commit_arena(0, WIN, None, ACFG)  # verdict appliqué à nouveau : le fichier est réécrit et référencé
    assert c["accepted"] and t.state["generation"] == 1
    from chesszero.arena import sha256_file
    assert read_train_state(t.out)["accepted"]["sha256"] == sha256_file(t.out / "generations" / "gen_0001.pt")


def test_orphan_files_are_ignored_and_a_tampered_candidate_is_refused(world, scripted):
    scripted(True, True)
    t = make(world, "k4")
    (t.out / "generations").mkdir(exist_ok=True)
    (t.out / "generations" / "gen_0007.pt").write_bytes(b"orphelin")
    (t.out / "candidate.pt").write_bytes(b"orphelin")
    st = read_train_state(t.out)
    assert st["accepted"]["is_init"] and st["candidate"] is None  # rien n'est pris dans le dossier
    run(t, world)  # le candidat écrase l'orphelin
    load_for_inference(t.out / "candidate.pt")
    (t.out / "candidate.pt").write_bytes(b"autre")
    with pytest.raises(RuntimeError, match="SHA-256"):
        t.commit_arena(0, WIN, None, ACFG)


def test_cut_between_commit_and_journal_does_not_duplicate(world, scripted, monkeypatch):
    scripted(True)
    t = make(world, "k5")
    run(t, world)
    real = SelfPlayTrainer._append_once
    calls = {"n": 0}

    def fail_once(self, row, key):
        if key[0] == "arena" and calls["n"] == 0:
            calls["n"] += 1
            raise Crash("journal")
        return real(self, row, key)
    monkeypatch.setattr(SelfPlayTrainer, "_append_once", fail_once)
    with pytest.raises(Crash):
        t.commit_arena(0, WIN, None, ACFG)
    t = make(world, "k5")
    assert t.state["generation"] == 1  # le verdict était validé
    t.commit_arena(0, WIN, None, ACFG)  # rejoué : complète le journal, une seule fois
    t.commit_arena(0, WIN, None, ACFG)
    assert sum(r.get("type") == "arena" for r in read_jsonl(t.out / "rounds.jsonl")) == 1


# --------------------------------------------------------------------------- compatibilité avec l'étape 6


def _make_v1(t):
    st = torch.load(t.out / "train_state.pt", weights_only=False)
    for k in spt._new_state_fields(None):
        st.pop(k, None)
    torch.save(st, t.out / "train_state.pt")
    lo = torch.load(t.out / "line_ok.pt", weights_only=False)
    lo.pop("round")
    torch.save(lo, t.out / "accepted.pt")
    (t.out / "line_ok.pt").unlink()
    return st


def test_step6_state_without_generation_is_migrated(world):
    t = make(world, "m1")
    t.state["round_id"] = 2
    t._save_state()
    _make_v1(t)
    assert read_train_state(t.out)["needs_migration"]
    t = make(world, "m1")
    assert t.state["format"] == 2 and t.state["round_id"] == 2 and t.state["accepted"]["generation"] == 0
    assert (t.out / "line_ok.pt").exists() and not (t.out / "accepted.pt").exists()
    kept = torch.load(t.out / "accepted.pt.migre", weights_only=False)  # renommé, pas supprimé
    lo = torch.load(t.out / "line_ok.pt", weights_only=False)
    assert same(kept["model"], lo["model"]) and lo["round"] is None
    assert read_train_state(t.out)["format"] == 2


def test_step6_state_with_generations_is_refused(world):
    t = make(world, "m2")
    st = _make_v1(t)
    st["generation"] = 1
    torch.save(st, t.out / "train_state.pt")
    with pytest.raises(RuntimeError, match="sans arène"):
        make(world, "m2")


def test_read_train_state_without_state():
    assert read_train_state(Path("dossier_absent_xyz")) == {"exists": False}
