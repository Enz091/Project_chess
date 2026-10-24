"""Analyse du tampon de self-play d'une génération (parties rejouables, PGN, accord politique brute / cible MCTS).

Commandes, sur un dossier de shards (`<sortie>/selfplay/`), sans modifier le code de la boucle (seule `policy` lit
la validation Lichess) :

  analyse_generation.py check    --buffer <dossier>                       les parties se rejouent-elles ? (coups, indices joués)
  analyse_generation.py pgn      --buffer <dossier> --out docs/analyse_gen0   PGN commentés : répétitions >= 40 points de
                                 matériel (5 avant le 60e demi-coup + 5 après), 5 mats les plus courts, 5 abandons (graine fixe)
  analyse_generation.py agreement --buffer <dossier> --checkpoint <réseau> [<autre réseau> ...] --out <json>
                                 accord entre la politique brute et la cible MCTS, par phase (ouverture / milieu / finale)
  analyse_generation.py noise    --buffer <dossier> --checkpoint <réseau qui a joué> [--compare <réseau> ...]
                                 recherche à 200 simulations avec et sans bruit de Dirichlet sur 500 positions, et
                                 variantes du débruitage des cibles (résultat : docs/analyse_gen0/bruit_dirichlet.json)
  analyse_generation.py policy   --buffer <dossier> --checkpoint <A> <B> ...   netteté de la politique (parties mises de
                                 côté : entropie, probabilité du coup préféré) et validation Lichess de plusieurs réseaux

Les shards contiennent tout ce qu'il faut : coups UCI depuis la position initiale (ouverture du livre comprise),
résultat, raison de fin, Q racine de chaque position cherchée (les coups forcés ne sont pas enregistrés). Les PGN
donnent `Q(Blancs)` après chaque coup cherché et, au dernier coup, la raison de fin, le matériel restant et le dernier Q.

L'accord se calcule comme dans le rapport de session (chesszero/agreement.py) : biais du bruit de Dirichlet à la racine
rappelé dans le texte. Référence de la génération 0 (kaggle_sup, session de self-play seul) : voir docs/analyse_gen0/.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_shards(buffer: Path) -> list[tuple[str, dict]]:
    from chesszero.selfplay import load_shard

    return [(p.stem, load_shard(p)) for p in sorted(Path(buffer).glob("*.npz")) if p.with_suffix(".json").exists()]


def replay(shards, every: int = 10) -> dict:
    """Rejoue une partie sur `every` : chaque coup enregistré doit avoir l'indice `played`, et le nombre de demi-coups
    doit correspondre."""
    from chesszero.encoding import move_to_index

    games = positions = checked = bad = 0
    for _, d in shards:
        n = len(d["game_uid"])
        games += n
        positions += len(d["played"])
        for g in range(0, n, every):
            idx = np.nonzero(d["game"] == g)[0]
            recorded = {int(d["ply"][i]): i for i in idx}
            board = chess.Board()
            for ply, uci in enumerate(d["game_moves"][g].split()):
                if ply in recorded:
                    checked += 1
                    bad += move_to_index(chess.Move.from_uci(uci), board) != int(d["played"][recorded[ply]])
                board.push_uci(uci)
            bad += len(board.move_stack) != int(d["plies"][g])
    return {"games": games, "positions": positions, "positions_checked": checked, "inconsistencies": bad}


def cmd_check(args) -> int:
    r = replay(load_shards(args.buffer))
    print(f"{r['games']} parties, {r['positions']} positions ; {r['positions_checked']} positions rejouées ; "
          f"{r['inconsistencies']} incohérence(s)")
    return 1 if r["inconsistencies"] else 0


def cmd_pgn(args) -> int:
    from chesszero.selfplay import TERMINATIONS, material_points

    T = {name: i for i, name in enumerate(TERMINATIONS)}
    games = []
    for name, d in load_shards(args.buffer):
        for g in range(len(d["game_uid"])):
            board = chess.Board()
            for uci in d["game_moves"][g].split():
                board.push_uci(uci)
            games.append({"shard": name, "d": d, "g": g, "mat": material_points(board), "term": int(d["termination"][g]),
                          "plies": int(d["plies"][g]), "uid": str(d["game_uid"][g])})
    print(len(games), "parties")
    rep = [x for x in games if x["term"] == T["triple répétition"]]
    rep_mat = [x for x in rep if x["mat"] >= args.min_material]
    early = [x for x in rep_mat if x["plies"] < 60]
    late = [x for x in rep_mat if x["plies"] >= 60]
    print(f"répétitions {len(rep)} ; avec >= {args.min_material} points {len(rep_mat)} (avant le 60e demi-coup {len(early)}, "
          f"après {len(late)})")
    rng = random.Random(args.seed)
    sel_rep = sorted(rng.sample(early, min(5, len(early))), key=lambda x: x["plies"]) + \
        sorted(rng.sample(late, min(5, len(late))), key=lambda x: x["plies"])
    mates = sorted([x for x in games if x["term"] == T["mat"]], key=lambda x: (x["plies"], x["uid"]))[:5]
    resign = [x for x in games if x["term"] == T["abandon"]]
    sel_res = random.Random(args.seed).sample(resign, min(5, len(resign)))

    def q_white(d, i):  # camp au trait = Blancs aux demi-coups pairs (départ standard)
        q = float(d["root_q"][i])
        return q if int(d["ply"][i]) % 2 == 0 else -q

    def to_pgn(x, label):
        d, g = x["d"], x["g"]
        rec = {int(d["ply"][i]): i for i in np.nonzero(d["game"] == g)[0]}
        moves = d["game_moves"][g].split()
        rw = int(d["result_white"][g])
        game = chess.pgn.Game()
        game.headers.update({"Event": f"ChessZero self-play {args.name} ({label})".replace("  ", " "), "Site": "Kaggle",
                             "Date": args.date, "Round": "-", "White": args.player, "Black": args.player,
                             "Result": {1: "1-0", 0: "1/2-1/2", -1: "0-1"}[rw], "Termination": TERMINATIONS[x["term"]],
                             "PlyCount": str(x["plies"]), "GameUID": x["uid"], "Shard": x["shard"],
                             "BookIndex": str(int(d["start_index"][g])), "BookPlies": str(int(d["start_plies"][g])),
                             "Control": str(bool(d["control"][g])), "FinalMaterial": str(x["mat"])})
        last_q, node = None, game
        for ply, uci in enumerate(moves):
            node = node.add_variation(chess.Move.from_uci(uci))
            bits = []
            if ply == int(d["start_plies"][g]) - 1 and ply >= 0:
                bits.append("fin de l'ouverture du livre (coups suivants joués par le réseau)")
            if ply in rec:
                last_q = (ply, q_white(d, rec[ply]))
                bits.append(f"Q(Blancs)={last_q[1]:+.2f}")
            if ply == len(moves) - 1:
                side = "Blancs" if ply % 2 == 0 else "Noirs"
                tail = (f"FIN : {TERMINATIONS[x['term']]} après {len(moves)} demi-coups (dernier coup par les {side}) ; "
                        f"matériel restant {x['mat']} points")
                if last_q:
                    lag = len(moves) - 1 - last_q[0]
                    tail += (f" ; dernier Q enregistré = {last_q[1]:+.2f} (vue des Blancs) au demi-coup {last_q[0] + 1}"
                             + ("" if lag == 0 else f", {lag} demi-coup(s) avant la fin (coups forcés ou arbitrage non "
                                                    "enregistrés)"))
                if x["term"] == T["abandon"]:
                    tail += f" ; abandon des {'Noirs' if rw == 1 else 'Blancs'}"
                bits.append(tail)
            if bits:
                node.comment = " ; ".join(bits)
        return str(game)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, sel, label in (("repetitions_40pts.pgn", sel_rep, f"répétition, >= {args.min_material} points"),
                             ("mats_les_plus_courts.pgn", mates, "mat le plus court"),
                             ("abandons.pgn", sel_res, f"abandon, tirage graine {args.seed}")):
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            for x in sel:
                f.write(to_pgn(x, label) + "\n\n")
        print(name, len(sel))
    lines = [f"# Analyse du tampon (source : {args.buffer})", "",
             "Fichiers générés par `scripts/analyse_generation.py pgn` (non versionnables à la légère : à regénérer).",
             "Coups en ordre depuis la position initiale (ouverture du livre comprise, `BookPlies` demi-coups) ; les commentaires",
             "donnent `Q(Blancs)` aux positions cherchées (les coups forcés ne sont pas enregistrés).", "",
             f"Répétitions : {len(rep)} ; avec >= {args.min_material} points de matériel : {len(rep_mat)} (avant le 60e "
             f"demi-coup : {len(early)} ; après : {len(late)}). Tirage graine {args.seed}, 5 + 5.", ""]
    for title, sel in (("repetitions_40pts.pgn", sel_rep), ("mats_les_plus_courts.pgn", mates), ("abandons.pgn", sel_res)):
        lines += [f"## {title}", "", "| # | UID | demi-coups | fin | matériel | résultat | départ livre |",
                  "|---|---|---|---|---|---|---|"]
        for k, x in enumerate(sel, 1):
            d, g = x["d"], x["g"]
            lines.append(f"| {k} | {x['uid']} | {x['plies']} | {TERMINATIONS[x['term']]} | {x['mat']} | "
                         f"{int(d['result_white'][g])} | {int(d['start_plies'][g])} |")
        lines.append("")
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return 0


def cmd_agreement(args) -> int:
    import torch

    from chesszero.agreement import (DIRICHLET_NOTE, PHASE_LABELS, PHASES, gather, policy_agreement, sample_positions,
                                     summarize)
    from chesszero.checkpoint import load_for_inference
    from chesszero.device import get_device

    shards = [d for _, d in load_shards(args.buffer)]
    picks = sample_positions(shards, args.positions, args.seed, exclude_heldout=None if args.all_positions else
                             args.heldout_fraction)
    batch = gather(shards, picks)
    device = get_device(args.device)
    torch.set_num_threads(args.threads)
    result = {"buffer": str(args.buffer), "positions": len(picks), "seed": args.seed,
              "sample": "toutes les positions" if args.all_positions else
              f"positions d'entraînement (parties mises de côté exclues, fraction {args.heldout_fraction})",
              "definitions": "accord = coup préféré du réseau (softmax sur les coups légaux) = coup le plus visité ; "
                             "KL = KL(cible MCTS || politique brute)", "note": DIRICHLET_NOTE, "networks": {}}
    for path in args.checkpoint:
        model, meta = load_for_inference(path, device)
        agree, kl = policy_agreement(model, batch, device)
        s = summarize(batch, agree, kl)
        result["networks"][str(path)] = {"step": meta.get("step"), "phases": s}
        print(f"\n{path} (pas {meta.get('step')}) — {len(picks)} positions")
        for ph in ("all",) + PHASES:
            c = s[ph]
            if c["n"]:
                print(f"  {PHASE_LABELS[ph]:<10} n={c['n']:>6}  accord {c['agree']:.1%}  KL {c['kl']:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("->", args.out)
    return 0


def _dist_stats(counts: list[np.ndarray], clean: list[np.ndarray], priors: list[np.ndarray], barely: float,
                tail: float) -> dict:
    """Moyennes sur les positions : entropie (nats), part du coup le plus visité, part des visites sur les coups que
    la recherche sans bruit visite à peine (< `barely` de ses visites), part sur les coups de prior réseau < `tail`
    (« queue »), distance en variation totale à la recherche sans bruit, même coup le plus visité que la recherche
    sans bruit. Les parts sont linéaires : leur moyenne sur un tirage du bruit par position estime sans biais celle
    de la cible MOYENNE (ce qu'apprend le réseau), contrairement à l'entropie d'un tirage."""
    ent, top, low, tl, tv, same = [], [], [], [], [], []
    for c, ref, pr in zip(counts, clean, priors):
        p = c / c.sum()
        r = ref / ref.sum()
        nz = p > 0
        ent.append(float(-(p[nz] * np.log(p[nz])).sum()))
        top.append(float(p.max()))
        low.append(float(p[r < barely].sum()))
        tl.append(float(p[pr < tail].sum()))
        tv.append(float(0.5 * np.abs(p - r).sum()))
        same.append(float(np.argmax(c) == np.argmax(ref)))
    return {"entropy": float(np.mean(ent)), "top_share": float(np.mean(top)), "barely_share": float(np.mean(low)),
            "tail_share": float(np.mean(tl)), "tv_to_clean": float(np.mean(tv)),
            "same_best_as_clean": float(np.mean(same))}


def cmd_noise(args) -> int:
    """(a) Le bruit de Dirichlet explique-t-il l'étalement des cibles ? Même position, recherche à 200 simulations avec
    et sans bruit (arbre neuf), avec le réseau qui a joué ; plus les visites enregistrées (arbre réutilisé) et les
    variantes du débruitage (chesszero/policy_target.py) appliquées à la recherche avec bruit et aux visites
    enregistrées."""
    import time
    from dataclasses import replace

    import torch

    from chesszero.agreement import sample_positions
    from chesszero.checkpoint import load_for_inference
    from chesszero.encoding import pack_board
    from chesszero.mcts import MCTS, MCTSConfig, NetEvaluator, legal_moves_and_indices, search_batched, terminal_value
    from chesszero.policy_target import prune_one
    from chesszero.selfplay import SelfPlayConfig
    from chesszero.syzygy import Tablebase

    torch.set_num_threads(args.threads)
    t0 = time.time()
    shards = [d for _, d in load_shards(args.buffer)]
    picks = sample_positions(shards, args.positions, args.seed)
    boards, recorded = [], []
    for k, i in picks:
        d = shards[k]
        board = chess.Board()
        for uci in d["game_moves"][d["game"][i]].split()[:int(d["ply"][i])]:
            board.push_uci(uci)
        if not np.array_equal(pack_board(board), d["boards"][i]):
            raise SystemExit(f"rejeu incohérent : shard {k}, ligne {i}")
        a, b = int(d["policy_offsets"][i]), int(d["policy_offsets"][i + 1])
        boards.append(board)
        recorded.append((d["policy_index"][a:b].astype(np.int64), d["policy_visits"][a:b].astype(np.float64)))
    model, meta = load_for_inference(args.checkpoint, "cpu")
    evaluator = NetEvaluator(model, "cpu")
    sp = SelfPlayConfig.named(args.config)
    noisy_cfg = sp.mcts_config(MCTSConfig.named(args.config))
    clean_cfg = replace(noisy_cfg, dirichlet_epsilon=0.0)
    tb = Tablebase.open(args.syzygy, wdl_only=True) if args.syzygy else None
    print(f"{len(boards)} positions ({args.buffer}), réseau {args.checkpoint} (pas {meta.get('step')}), "
          f"{noisy_cfg.num_simulations} simulations, lots de {noisy_cfg.batch_size} feuilles par arbre, "
          f"epsilon {noisy_cfg.dirichlet_epsilon}, c_puct {noisy_cfg.c_puct}, tables : {tb.describe() if tb else 'non'}",
          flush=True)

    def run(cfg, seed):
        rng = np.random.default_rng(seed)
        out = []
        for s in range(0, len(boards), args.trees):
            trees = [MCTS(evaluator, cfg, b, rng=rng, tablebase=tb) for b in boards[s:s + args.trees]]
            search_batched(trees, evaluator, cfg.num_simulations, cfg.batch_size)
            for t in trees:
                r = t.root
                child_v = np.full(len(r.moves), np.nan)
                for a, ch in enumerate(r.children):
                    if ch is not None:
                        child_v[a] = -(ch.terminal if ch.terminal is not None else ch.value)
                q = np.where(r.N > 0, r.W / np.maximum(r.N, 1), np.nan)
                out.append({"idx": r.idx.astype(np.int64), "N": r.N.astype(np.float64), "Q": q, "Qnet": child_v,
                            "P": r.P_raw.copy(), "Pnoisy": r.P.copy(), "root_q": r.q()})
            print(f"  {min(s + args.trees, len(boards))}/{len(boards)} ({time.time() - t0:.0f} s)", flush=True)
        return out

    clean = run(clean_cfg, 0)
    noisy = run(noisy_cfg, args.seed + 1)
    c_puct, thr = noisy_cfg.c_puct, args.barely
    ref = [c["N"] for c in clean]

    # visites enregistrées (arbre réutilisé), remises dans l'ordre des coups de l'arbre neuf
    rec, rec_net = [], []
    need = []  # (position, coup) des fils visités dans l'enregistrement : valeur réseau pour l'approximation
    for j, (idx, vis) in enumerate(recorded):
        order = {int(x): n for n, x in enumerate(idx)}
        if set(order) != set(clean[j]["idx"].tolist()):
            raise SystemExit(f"coups légaux différents pour la position {j}")
        v = np.array([vis[order[int(x)]] for x in clean[j]["idx"]])
        rec.append(v)
        need += [(j, a) for a in np.nonzero(v > 0)[0]]
    qnet_rec = [np.full(len(c["N"]), np.nan) for c in clean]
    for s in range(0, len(need), 256):
        part = need[s:s + 256]
        bs, li, exact = [], [], []
        for j, a in part:
            b = boards[j].copy()
            moves, _ = legal_moves_and_indices(b)
            b.push(moves[a])
            tv = terminal_value(b)
            if tv is None and tb is not None:
                tv = tb.value(b)
            exact.append(tv)
            bs.append(b)
            li.append(legal_moves_and_indices(b)[1] if tv is None else None)
        todo = [n for n, e in enumerate(exact) if e is None]
        if todo:
            _, values = evaluator.evaluate([bs[n] for n in todo], [li[n] for n in todo])
            for n, val in zip(todo, values):
                exact[n] = float(val)
        for (j, a), e in zip(part, exact):
            qnet_rec[j][a] = -e
    for j, c in enumerate(clean):
        rec_net.append(prune_one(rec[j], c["P"], qnet_rec[j], c_puct))

    def net_priors(path):
        m, _ = load_for_inference(path, "cpu")
        ev = NetEvaluator(m, "cpu")
        out = []
        for s in range(0, len(boards), 256):
            pr, _ = ev.evaluate(boards[s:s + 256], [c["idx"] for c in clean[s:s + 256]])
            out += pr
        return out

    variants = {f"politique de {Path(path).parent.name}/{Path(path).name}": net_priors(path)
                for path in [args.checkpoint] + (args.compare or [])}
    variants.update({
        "recherche sans bruit (référence)": ref,
        "recherche avec bruit (arbre neuf)": [n["N"] for n in noisy],
        "visites enregistrées (arbre réutilisé, bruit)": rec,
        "avec bruit, débruitée : Q exact": [prune_one(n["N"], n["P"], n["Q"], c_puct) for n in noisy],
        "avec bruit, débruitée : Q exact + 1 visite -> 0": [prune_one(n["N"], n["P"], n["Q"], c_puct, True)
                                                           for n in noisy],
        "avec bruit, débruitée : Q = valeur réseau du fils": [prune_one(n["N"], n["P"], n["Qnet"], c_puct)
                                                             for n in noisy],
        "avec bruit, débruitée : sans Q (Q = Q du meilleur)": [prune_one(n["N"], n["P"], np.where(n["N"] > 0, 0.0,
                                                                                                    np.nan), c_puct)
                                                              for n in noisy],
        "enregistrées, débruitées : Q = valeur réseau du fils": rec_net,
        "enregistrées, débruitées : sans Q (Q = Q du meilleur)": [prune_one(v, c["P"], np.where(v > 0, 0.0, np.nan),
                                                                         c_puct) for v, c in zip(rec, clean)],
        "sans bruit, « débruitée » : Q exact (doit peu changer)": [prune_one(c["N"], c["P"], c["Q"], c_puct)
                                                                    for c in clean],
    })
    visited = {"sans bruit": float(np.mean([(c["N"] > 0).sum() for c in clean])),
               "avec bruit": float(np.mean([(n["N"] > 0).sum() for n in noisy])),
               "enregistrées": float(np.mean([(v > 0).sum() for v in rec])),
               "coups légaux": float(np.mean([len(c["N"]) for c in clean]))}
    noise_gain = []  # prior avec bruit / prior sans bruit, et visites gagnées, pour les coups que le bruit a gonflés
    for n, c in zip(noisy, clean):
        boosted = n["Pnoisy"] > 2 * n["P"] + 0.02
        noise_gain.append((boosted.sum(), (n["N"][boosted] - c["N"][boosted]).sum() / n["N"].sum()))
    result = {"buffer": str(args.buffer), "checkpoint": str(args.checkpoint), "positions": len(boards),
              "seed": args.seed, "simulations": noisy_cfg.num_simulations, "epsilon": noisy_cfg.dirichlet_epsilon,
              "barely_threshold": thr, "visited_moves": visited,
              "boosted_moves_per_position": float(np.mean([g[0] for g in noise_gain])),
              "visit_share_gained_by_boosted_moves": float(np.mean([g[1] for g in noise_gain])),
              "seconds": time.time() - t0,
              "tail_threshold": args.tail,
              "variants": {name: _dist_stats(v, ref, [c["P"] for c in clean], thr, args.tail)
                           for name, v in variants.items()}}
    print(f"\nCoups visités en moyenne : {visited}")
    print(f"Coups gonflés par le bruit (prior x2 + 0,02) : {result['boosted_moves_per_position']:.2f} par position, "
          f"part des visites qu'ils gagnent : {result['visit_share_gained_by_boosted_moves']:.1%}")
    print(f"\n{'':<56}{'entropie':>9}{'top':>8}{'à peine':>9}{'queue':>8}{'VT réf.':>9}{'même top':>10}")
    for name, s in result["variants"].items():
        print(f"{name:<56}{s['entropy']:>9.3f}{s['top_share']:>8.3f}{s['barely_share']:>9.3f}{s['tail_share']:>8.3f}"
              f"{s['tv_to_clean']:>9.3f}{s['same_best_as_clean']:>10.1%}")
    print(f"(« à peine » = coups à moins de {thr:.0%} des visites de la recherche sans bruit ; « queue » = coups de "
          f"prior < {args.tail:.0%} pour {Path(args.checkpoint).name} ; VT = distance en variation totale à la "
          f"recherche sans bruit) — {result['seconds'] / 60:.1f} min")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("->", args.out)
    return 0


def cmd_policy(args) -> int:
    """Netteté de la politique de plusieurs réseaux sur les mêmes positions : parties MISES DE CÔTÉ du tampon (jamais
    entraînées ; entropie, probabilité du coup préféré, masse sur la « queue » = coups de prior < 2 % pour le réseau de
    référence, accord avec le coup le plus visité, même coup préféré que la référence) et validation Lichess (perte
    de politique, top-1, perte de valeur, même coup préféré que la référence ; même sous-ensemble que l'entraîneur)."""
    import torch

    from chesszero.checkpoint import load_for_inference
    from chesszero.data import load_split
    from chesszero.encoding import unpack_batch_torch
    from chesszero.selfplay_train import eval_human, human_val_set, is_heldout

    torch.set_num_threads(args.threads)
    rows = []
    for _, d in load_shards(args.buffer):
        held = {u: is_heldout(u, args.heldout_fraction) for u in set(d["game_uid"].tolist())}
        off = d["policy_offsets"]
        for i in np.nonzero([held[u] for u in d["game_uid"][d["game"]].tolist()])[0]:
            rows.append((d["boards"][i], d["policy_index"][off[i]:off[i + 1]].astype(np.int64),
                         d["policy_visits"][off[i]:off[i + 1]].astype(np.float64)))
    if len(rows) > args.positions:
        keep = np.sort(np.random.default_rng(args.seed).choice(len(rows), args.positions, replace=False))
        rows = [rows[int(k)] for k in keep]
    human = human_val_set(load_split(args.human, "val"), args.lichess_val, 0, None) if args.human else None
    out = {"buffer": str(args.buffer), "heldout_positions": len(rows), "lichess_positions":
           len(human["boards"]) if human else 0, "networks": {}}
    print(f"{len(rows)} positions des parties mises de côté ({args.buffer}) ; "
          f"{out['lichess_positions']} positions de validation Lichess\n")
    def heldout_probs(model):
        """Politique (softmax sur les coups légaux enregistrés) de chaque position mise de côté."""
        probs = []
        with torch.inference_mode():
            for s in range(0, len(rows), 1024):
                part = rows[s:s + 1024]
                logits, _ = model(unpack_batch_torch(torch.from_numpy(np.stack([r[0] for r in part]))))
                for (_, idx, _), z in zip(part, logits.double().numpy()):
                    q = np.exp(z[idx] - z[idx].max())
                    probs.append(q / q.sum())
        return probs

    def lichess_preferred(model):
        """Coup préféré (parmi les coups légaux) de chaque position de la validation Lichess."""
        best = []
        with torch.inference_mode():
            for s in range(0, len(human["boards"]), 1024):
                logits, _ = model(unpack_batch_torch(torch.from_numpy(human["boards"][s:s + 1024])))
                z = logits.float().masked_fill(~torch.from_numpy(human["legal"][s:s + 1024]), float("-inf"))
                best.append(torch.argmax(z, dim=1).numpy())
        return np.concatenate(best)

    ref_model, _ = load_for_inference(args.reference, "cpu")
    ref_probs = heldout_probs(ref_model)
    ref_lichess = lichess_preferred(ref_model) if human is not None else None
    out["reference"] = str(args.reference)
    print(f"référence de la queue et de l'accord : {args.reference} (queue = coups de prior < {args.tail:.0%})\n")
    print(f"{'réseau':<44}{'entropie':>9}{'p(préféré)':>11}{'queue':>7}{'accord MCTS':>12}{'= réf.':>8}"
          f"{'= réf. Li.':>11}{'pol. Lichess':>13}{'top-1':>8}{'val. Lichess':>13}")
    for path in args.checkpoint:
        model, meta = load_for_inference(path, "cpu")
        probs = heldout_probs(model)
        ent = [float(-(q * np.log(np.maximum(q, 1e-300))).sum()) for q in probs]
        top = [float(q.max()) for q in probs]
        tail = [float(q[r < args.tail].sum()) for q, r in zip(probs, ref_probs)]
        agree = [float(np.argmax(q) == np.argmax(row[2])) for q, row in zip(probs, rows)]
        same = [float(np.argmax(q) == np.argmax(r)) for q, r in zip(probs, ref_probs)]
        res = {"step": meta.get("step"), "entropy": float(np.mean(ent)), "preferred_prob": float(np.mean(top)),
               "tail_mass": float(np.mean(tail)), "agreement_with_most_visited": float(np.mean(agree)),
               "same_preferred_as_reference": float(np.mean(same))}
        if human is not None:
            # avant eval_human, qui remet le réseau en mode entraînement (BatchNorm) en sortant
            res["lichess_same_preferred_as_reference"] = float((lichess_preferred(model) == ref_lichess).mean())
            res["lichess"] = eval_human(model, human, "cpu")
        out["networks"][str(path)] = res
        lic = res.get("lichess") or {}
        name = f"{Path(path).parent.name}/{Path(path).name}"
        print(f"{name:<44}{res['entropy']:>9.3f}{res['preferred_prob']:>11.3f}{res['tail_mass']:>7.1%}"
              f"{res['agreement_with_most_visited']:>12.1%}{res['same_preferred_as_reference']:>8.1%}"
              f"{res.get('lichess_same_preferred_as_reference', float('nan')):>11.1%}"
              f"{lic.get('policy_loss', float('nan')):>13.4f}{lic.get('top1', float('nan')):>8.2%}"
              f"{lic.get('value_loss', float('nan')):>13.4f}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("->", args.out)
    return 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("check", "pgn", "agreement"):
        p = sub.add_parser(name)
        p.add_argument("--buffer", required=True, type=Path, help="dossier de shards (<sortie>/selfplay)")
        if name == "pgn":
            p.add_argument("--out", required=True)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--min-material", type=int, default=40, help="matériel minimal des répétitions (points)")
            p.add_argument("--name", default="", help="nom de la génération dans l'en-tête Event (ex. « génération 0 »)")
            p.add_argument("--date", default="????.??.??", help="en-tête Date (AAAA.MM.JJ)")
            p.add_argument("--player", default="réseau (200 sim.)", help="noms des joueurs dans les en-têtes")
        if name == "agreement":
            p.add_argument("--checkpoint", nargs="+", required=True, help="un ou plusieurs réseaux (même échantillon)")
            p.add_argument("--positions", type=int, default=20000)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--all-positions", action="store_true",
                           help="toutes les positions (défaut : positions d'entraînement, parties mises de côté exclues)")
            p.add_argument("--heldout-fraction", type=float, default=0.10)
            p.add_argument("--device", default="cpu")
            p.add_argument("--threads", type=int, default=8)
            p.add_argument("--out", default=None, help="fichier JSON des résultats")
    p = sub.add_parser("noise")
    p.add_argument("--buffer", required=True, type=Path, help="dossier de shards (<sortie>/selfplay)")
    p.add_argument("--checkpoint", required=True, help="réseau qui a joué ces parties")
    p.add_argument("--config", default="kaggle", help="réglages [selfplay] et [mcts]")
    p.add_argument("--positions", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trees", type=int, default=64, help="arbres cherchés ensemble (lots de trees x 4 feuilles)")
    p.add_argument("--barely", type=float, default=0.02, help="« à peine visité » : moins de cette part des visites")
    p.add_argument("--tail", type=float, default=0.02, help="« queue » : coups de prior réseau sous ce seuil")
    p.add_argument("--compare", nargs="*", default=None, help="autres réseaux dont on mesure la politique")
    p.add_argument("--syzygy", default=str(ROOT / "tools" / "syzygy"), help="tables (WDL seul, comme le self-play)")
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--out", default=None, help="fichier JSON des résultats")
    p = sub.add_parser("policy")
    p.add_argument("--buffer", required=True, type=Path, help="dossier de shards (<sortie>/selfplay)")
    p.add_argument("--checkpoint", nargs="+", required=True, help="réseaux à comparer (mêmes positions)")
    p.add_argument("--human", default=str(ROOT / "data" / "lichess_24m"), help="données Lichess ('' = sans)")
    p.add_argument("--lichess-val", type=int, default=50000, help="sous-ensemble fixe de la validation (comme [selfplay_train])")
    p.add_argument("--reference", default=str(ROOT / "checkpoints" / "kaggle_sup" / "best.pt"),
                   help="réseau de référence : queue (ses coups de prior < --tail) et accord du coup préféré")
    p.add_argument("--tail", type=float, default=0.02)
    p.add_argument("--positions", type=int, default=50000, help="positions des parties mises de côté (au plus)")
    p.add_argument("--heldout-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--out", default=None, help="fichier JSON des résultats")
    args = ap.parse_args()
    sys.exit({"check": cmd_check, "pgn": cmd_pgn, "agreement": cmd_agreement, "noise": cmd_noise,
              "policy": cmd_policy}[args.command](args))


if __name__ == "__main__":
    main()
