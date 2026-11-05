"""Diagnostic des candidats rejetés d'une session de boucle (analyse seule : ne modifie ni le tampon ni l'état).

  diagnose_session.py prune  --buffer <sortie>/selfplay --prefix L0002 --out prune.json
        (a) cible débruitée EXACTE (prior et Q enregistrés) contre l'approximation SANS Q des anciens shards
  diagnose_session.py sfeval --arenas <arène> [<arène> ...] --out sf.json --depth 14 --workers 10 [--limit N]
        Stockfish à profondeur fixe sur toutes les positions jouées des arènes (reprise : relancer = compléter)
  diagnose_session.py games  --arenas <arène> ... --sf sf.json --out games.json [--qnet cand=<pt> --qnet sup=<pt>]
        (b) ACPL et gaffes par camp et par phase, bascules, répétitions avec avantage, conversions ratées
  diagnose_session.py nets   --buffer <sortie>/selfplay --checkpoint <A.pt> <B.pt> --eval-value <eval_value.npz> --out nets.json
        (c) échelle et calibrage de la valeur (résultats du self-play, Stockfish) ; la netteté de la politique est
        donnée par analyse_generation.py policy
  diagnose_session.py conversions --arenas <arène> ... --sf sf.json --out conv.json
        conversions ratées (> +300 cp sans gain) : fins, gaffe du camp avantagé, phase
  diagnose_session.py gates  --arenas <arène> ...  --out gates.json
        ce qu'auraient décidé plusieurs règles d'arène (score 52 %, 50 %, P(meilleur) 0,10 / 0,20, sans arène)
  diagnose_session.py hybrid --arena <arène> --sf sf.json --policy-a <pt> --policy-b <pt> --out hybrid.json
        politique de l'un + valeur de l'autre : perte en centipions (Stockfish) du coup choisi par MCTS-200

Conventions : centipions vus des Blancs dans le cache, bornés à ±1000 pour les pertes (un mat vaut ±1000) ;
phase d'un coup = ouverture si demi-coup < 30, finale si matériel < 20 points (P 1, C/F 3, T 5, D 9 ; 78 au départ),
milieu sinon ; seuls les coups joués par les réseaux comptent (pas ceux du livre).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import chess
import chess.engine
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STOCKFISH = ROOT / "tools" / "stockfish" / "stockfish" / "stockfish-windows-x86-64-universal.exe"
MATE_CP = 10_000
CLAMP = 1000
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board: chess.Board) -> int:
    return sum(VALUES.get(p.piece_type, 0) for p in board.piece_map().values())


def phase_of(ply: int, mat: int) -> str:
    """Phase d'un coup joué au demi-coup `ply` (0 = premier coup de la partie) avec `mat` points sur l'échiquier."""
    if ply < 30:
        return "ouverture"
    return "finale" if mat < 20 else "milieu"


def clamp(cp: float) -> float:
    return max(-CLAMP, min(CLAMP, cp))


def move_loss(cp_before_white: float, cp_after_white: float, white_moved: bool) -> float:
    """Perte en centipions (≥ 0) du coup, vue du camp qui l'a joué, évaluations bornées à ±1000."""
    sign = 1 if white_moved else -1
    return max(0.0, sign * (clamp(cp_before_white) - clamp(cp_after_white)))


# --------------------------------------------------------------------------- (a) débruitage exact / sans Q


def cmd_prune(args) -> int:
    from chesszero.policy_target import best_moves, prune_visits, segments

    files = sorted(p for p in Path(args.buffer).glob(f"{args.prefix}*.npz"))
    if not files:
        print(f"aucun shard {args.prefix}* dans {args.buffer}")
        return 1
    acc = defaultdict(float)
    n_pos = n_moves = 0
    hist_removed = []  # masse retirée en trop par l'approximation, par position concernée
    hist_q_gap = []  # Q(c) - Q(b) des coups « découverts » plafonnés à tort
    for f in files:
        d = np.load(f)
        off = d["policy_offsets"].astype(np.int64)
        v = d["policy_visits"].astype(np.float64)
        prior = d["policy_prior"].astype(np.float64)
        q = d["policy_q"].astype(np.float64)
        if len(off) < 2:
            continue
        exact = prune_visits(off, v, prior, q, args.c_puct)
        approx = prune_visits(off, v, prior, np.where(v > 0, 0.0, np.nan), args.c_puct)
        starts, seg = segments(off)
        tot = np.add.reduceat(v, starts)
        te, ta = np.add.reduceat(exact, starts), np.add.reduceat(approx, starts)
        pe, pa, pr = exact / te[seg], approx / ta[seg], v / tot[seg]
        best = best_moves(off, v, prior)
        is_best = np.zeros(len(v), bool)
        is_best[best] = True
        # « découverte supprimée » : l'approximation garde MOINS de visites que la règle exacte (coup que son Q justifie)
        sup = (approx < exact - 0.5) & ~is_best
        over = (approx > exact + 0.5) & ~is_best  # l'inverse : l'approximation garde un coup que le Q condamne
        sup_pos = np.add.reduceat(sup.astype(np.int64), starts) > 0
        over_pos = np.add.reduceat(over.astype(np.int64), starts) > 0
        removed_raw = np.add.reduceat(np.where(sup, (exact - approx) / tot[seg], 0.0), starts)  # en part des visites brutes
        lost_mass = np.add.reduceat(np.where(sup, pe - pa, 0.0), starts)  # masse de la cible exacte perdue
        acc["positions_with_suppressed"] += sup_pos.sum()
        acc["positions_with_overkept"] += over_pos.sum()
        acc["suppressed_moves"] += sup.sum()
        acc["overkept_moves"] += over.sum()
        acc["suppressed_raw_share_sum"] += removed_raw.sum()
        acc["suppressed_target_mass_sum"] += lost_mass.sum()
        hist_removed.extend(lost_mass[sup_pos].tolist())
        hist_q_gap.extend((q[sup] - q[best][seg[sup]]).tolist())
        acc["tv_sum"] += (0.5 * np.add.reduceat(np.abs(pe - pa), starts)).sum()
        acc["tv_raw_exact_sum"] += (0.5 * np.add.reduceat(np.abs(pr - pe), starts)).sum()
        acc["tv_raw_approx_sum"] += (0.5 * np.add.reduceat(np.abs(pr - pa), starts)).sum()
        acc["removed_exact"] += (v - exact).sum()
        acc["removed_approx"] += (v - approx).sum()
        acc["visits"] += v.sum()
        acc["best_share_exact_sum"] += (pe[best]).sum()
        acc["best_share_approx_sum"] += (pa[best]).sum()
        # coup le plus visité de chaque cible (égalité : plus grand prior) ; 2e coup
        be, ba = best_moves(off, exact, prior), best_moves(off, approx, prior)
        acc["best_changed"] += (be != ba).sum()
        e2, a2 = exact.copy(), approx.copy()
        e2[be], a2[ba] = -1, -1
        acc["second_changed"] += (best_moves(off, e2, prior) != best_moves(off, a2, prior)).sum()
        # coups visités dont le Q dépasse celui du meilleur (vraies « découvertes » de la recherche)
        better = (v > 0) & ~is_best & (q > q[best][seg])
        acc["moves_q_above_best"] += better.sum()
        acc["moves_q_above_best_suppressed"] += (better & sup).sum()
        n_pos += len(starts)
        n_moves += int((v > 0).sum())
    out = {"buffer": str(args.buffer), "prefix": args.prefix, "shards": len(files), "positions": n_pos,
           "visited_moves": n_moves, "c_puct": args.c_puct,
           "positions_with_suppressed_discovery": acc["positions_with_suppressed"] / n_pos,
           "positions_with_overkept_move": acc["positions_with_overkept"] / n_pos,
           "suppressed_moves_per_position": acc["suppressed_moves"] / n_pos,
           "suppressed_share_of_raw_visits_mean": acc["suppressed_raw_share_sum"] / n_pos,
           "suppressed_target_mass_mean_all": acc["suppressed_target_mass_sum"] / n_pos,
           "suppressed_target_mass_mean_affected": float(np.mean(hist_removed)) if hist_removed else 0.0,
           "suppressed_target_mass_quantiles_affected": (np.quantile(hist_removed, [0.5, 0.9, 0.99]).tolist()
                                                        if hist_removed else []),
           "suppressed_q_gap_quantiles": np.quantile(hist_q_gap, [0.1, 0.5, 0.9]).tolist() if hist_q_gap else [],
           "tv_exact_vs_approx": acc["tv_sum"] / n_pos,
           "tv_raw_vs_exact": acc["tv_raw_exact_sum"] / n_pos,
           "tv_raw_vs_approx": acc["tv_raw_approx_sum"] / n_pos,
           "removed_visits_exact": acc["removed_exact"] / acc["visits"],
           "removed_visits_approx": acc["removed_approx"] / acc["visits"],
           "best_share_exact": acc["best_share_exact_sum"] / n_pos,
           "best_share_approx": acc["best_share_approx_sum"] / n_pos,
           "most_visited_changed": acc["best_changed"] / n_pos,
           "second_move_changed": acc["second_changed"] / n_pos,
           "moves_q_above_best_per_position": acc["moves_q_above_best"] / n_pos,
           "moves_q_above_best_suppressed_share": (acc["moves_q_above_best_suppressed"]
                                                   / max(acc["moves_q_above_best"], 1))}
    for k, v in out.items():
        print(f"{k:<46} {v:.4f}" if isinstance(v, float) else f"{k:<46} {v}")
    write_json(args.out, out)
    return 0


# --------------------------------------------------------------------------- arènes


def read_arena(arena_dir: Path) -> tuple[dict, list[dict]]:
    res = json.loads((arena_dir / "result.json").read_text(encoding="utf-8"))
    games = [json.loads(l) for l in (arena_dir / "games.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return res, games


def game_positions(game: dict) -> list[tuple[int, str]]:
    """(demi-coup, FEN) de chaque position où un réseau joue, plus la position finale."""
    board = chess.Board()
    moves = game["moves"].split()
    out = []
    for ply, uci in enumerate(moves):
        if ply >= game["start_plies"]:
            out.append((ply, board.fen()))
        board.push_uci(uci)
    out.append((len(moves), board.fen()))
    return out


def fen_key(fen: str) -> str:
    return " ".join(fen.split()[:4])  # sans les compteurs : même évaluation de Stockfish


_E: dict = {}


def _init_sf(stockfish: str, depth: int, hash_mb: int) -> None:
    from chesszero import cpu_guard

    cpu_guard.disable_ecoqos()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    engine.configure({"Threads": 1, "Hash": hash_mb})
    try:
        cpu_guard.disable_ecoqos(engine.transport.get_pid())
    except Exception:  # noqa: BLE001
        pass
    _E.update(engine=engine, depth=depth)


def _eval_game(item: tuple[str, list[str]]) -> tuple[str, dict]:
    """Évalue les positions d'une partie dans l'ordre (le hachage de Stockfish sert d'une position à la suivante)."""
    gid, fens = item
    out = {}
    for fen in fens:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            if board.is_checkmate():
                out[fen_key(fen)] = -MATE_CP if board.turn == chess.WHITE else MATE_CP
            else:
                out[fen_key(fen)] = 0
            continue
        info = _E["engine"].analyse(board, chess.engine.Limit(depth=_E["depth"]))
        out[fen_key(fen)] = int(info["score"].white().score(mate_score=MATE_CP))
    return gid, out


def cmd_sfeval(args) -> int:
    cache = json.loads(Path(args.out).read_text(encoding="utf-8")) if Path(args.out).exists() else {}
    todo, seen = [], set(cache)
    for a in args.arenas:
        _, games = read_arena(Path(a))
        for g in games:
            fens = [fen for _, fen in game_positions(g) if fen_key(fen) not in seen]
            seen.update(fen_key(f) for f in fens)
            if fens:
                todo.append((f"{Path(a).name}:{g['key']}", fens))
    if args.limit:
        todo = todo[:args.limit]
    n = sum(len(f) for _, f in todo)
    print(f"{n} positions nouvelles dans {len(todo)} parties ({len(cache)} en cache), profondeur {args.depth}, "
          f"{args.workers} processus", flush=True)
    start, done = time.perf_counter(), 0
    with mp.get_context("spawn").Pool(args.workers, _init_sf, (str(args.stockfish), args.depth, args.hash)) as pool:
        for k, (_, res) in enumerate(pool.imap_unordered(_eval_game, todo), 1):
            cache.update(res)
            done += len(res)
            if k % 50 == 0 or k == len(todo):
                el = time.perf_counter() - start
                print(f"  {k}/{len(todo)} parties, {done}/{n} positions, {el / 60:.1f} min "
                      f"(reste ~{el / done * (n - done) / 60:.1f} min)", flush=True)
                if not args.limit:
                    Path(args.out).write_text(json.dumps(cache) + "\n", encoding="utf-8")
    wall = time.perf_counter() - start
    if args.limit:
        print(f"Essai : {done} positions en {wall:.0f} s = {wall / max(done, 1) * 1000:.1f} ms par position en parallèle")
    else:
        Path(args.out).write_text(json.dumps(cache) + "\n", encoding="utf-8")
        print(f"{len(cache)} positions en cache -> {args.out} ({wall / 60:.1f} min)")
    return 0


# --------------------------------------------------------------------------- (b) analyse des parties


def analyse_game(game: dict, sf: dict, a_name: str = "A") -> dict:
    """Coups des réseaux d'une partie d'arène : perte en centipions, phase ; bascule, répétition, conversion.
    Joueur A = candidat (`a_white` dit sa couleur), B = adversaire."""
    pos = game_positions(game)
    board = chess.Board()
    moves = game["moves"].split()
    for uci in moves[:game["start_plies"]]:
        board.push_uci(uci)
    evals = [sf[fen_key(fen)] for _, fen in pos]
    rows = []
    for i, (ply, _) in enumerate(pos[:-1]):
        white_moved = board.turn == chess.WHITE
        player = "A" if white_moved == game["a_white"] else "B"
        mat = material(board)
        loss = move_loss(evals[i], evals[i + 1], white_moved)
        rows.append({"ply": ply, "player": player, "phase": phase_of(ply, mat), "loss": loss, "material": mat,
                     "before": evals[i], "after": evals[i + 1], "white": white_moved})
        board.push_uci(moves[ply])
    a_sign = 1 if game["a_white"] else -1
    a_eval = [clamp(a_sign * e) for e in evals]  # vu du candidat
    out = {"rows": rows, "score_a": game["score_a"], "termination": game["termination"], "key": game["key"]}
    # bascule d'une partie perdue par X : coup après la dernière position où X était à >= -100 cp
    for loser, sign in (("A", 1), ("B", -1)):
        if (loser == "A" and game["score_a"] == 0) or (loser == "B" and game["score_a"] == 1):
            vals = [sign * e for e in a_eval]  # vu du perdant
            last_ok = max((i for i, v in enumerate(vals[:-1]) if v >= -100), default=None)
            if last_ok is None:
                out["swing"] = {"loser": loser, "never_ok": True}
            else:
                r = rows[last_ok]
                out["swing"] = {"loser": loser, "never_ok": False, "index": last_ok, "ply": r["ply"],
                                "phase": r["phase"], "material": r["material"], "by": r["player"],
                                "loser_cp_before": vals[last_ok], "loser_cp_after": vals[last_ok + 1],
                                "loss": r["loss"], "fen": pos[last_ok][1],
                                "fen_loser_turn": pos[last_ok][1] if r["player"] == loser else None}
            own = [r for r in rows if r["player"] == loser]
            if own:
                worst = max(own, key=lambda r: r["loss"])
                out["swing"]["worst_own"] = {"ply": worst["ply"], "phase": worst["phase"], "loss": worst["loss"],
                                             "material": worst["material"]}
    # répétition avec un avantage Stockfish > 150 cp : qui a renoncé (le camp favorisé)
    if game["termination"] == "triple répétition":
        tail = a_eval[-6:]
        adv = float(np.mean(tail))
        out["repetition"] = {"a_cp_mean_last6": adv, "final_a_cp": a_eval[-1],
                             "renouncer": ("A" if adv > 150 else "B" if adv < -150 else None)}
    # conversions ratées : > +300 cp sur deux positions consécutives sans gagner
    for side, sign in (("A", 1), ("B", -1)):
        vals = [sign * e for e in a_eval]
        reached = any(vals[i] > 300 and vals[i + 1] > 300 for i in range(len(vals) - 1))
        won = (game["score_a"] == 1) if side == "A" else (game["score_a"] == 0)
        out.setdefault("conversion", {})[side] = {"reached_300": reached, "won": won}
    return out


def summarize_rows(rows: list[dict]) -> dict:
    out = {}
    for player in ("A", "B"):
        for phase in ("ouverture", "milieu", "finale", "tout"):
            sel = [r for r in rows if r["player"] == player and (phase == "tout" or r["phase"] == phase)]
            if not sel:
                continue
            losses = np.array([r["loss"] for r in sel])
            out[f"{player}:{phase}"] = {"moves": len(sel), "acpl": float(losses.mean()),
                                        "blunders_200": int((losses > 200).sum()),
                                        "blunder_rate": float((losses > 200).mean()),
                                        "mistakes_100": int((losses > 100).sum())}
    return out


def cmd_games(args) -> int:
    sf = json.loads(Path(args.sf).read_text(encoding="utf-8"))
    qnets = dict(x.split("=", 1) for x in (args.qnet or []))
    report = {"depth_cache": args.sf, "arenas": {}}
    swings_for_q = []
    for a in args.arenas:
        res, games = read_arena(Path(a))
        name = res["players"]["a"]["name"]
        per = [analyse_game(g, sf) for g in games]
        rows = [r for p in per for r in p["rows"]]
        s = summarize_rows(rows)
        swings = [p["swing"] | {"key": p["key"]} for p in per if "swing" in p]
        reps = [p["repetition"] for p in per if "repetition" in p]
        conv = {side: {"reached": sum(p["conversion"][side]["reached_300"] for p in per),
                       "not_won": sum(p["conversion"][side]["reached_300"] and not p["conversion"][side]["won"]
                                      for p in per),
                       "lost": sum(p["conversion"][side]["reached_300"] and
                                   ((p["score_a"] == 0) if side == "A" else (p["score_a"] == 1)) for p in per)}
                for side in ("A", "B")}

        def swing_summary(loser):
            sw = [x for x in swings if x["loser"] == loser]
            ok = [x for x in sw if not x["never_ok"]]
            by_phase = defaultdict(int)
            for x in ok:
                by_phase[x["phase"]] += 1
            worst = defaultdict(int)
            for x in sw:
                w = x.get("worst_own")
                if w and w["loss"] > 200:
                    worst[w["phase"]] += 1
            return {"lost_games": len(sw), "never_ok": len(sw) - len(ok),
                    "worst_own_blunder_200_by_phase": dict(worst),
                    "own_move": sum(x["by"] == loser for x in ok),
                    "own_blunder_200": sum(x["by"] == loser and x["loss"] > 200 for x in ok),
                    "phase": dict(by_phase),
                    "ply_median": float(np.median([x["ply"] for x in ok])) if ok else None,
                    "material_median": float(np.median([x["material"] for x in ok])) if ok else None}

        report["arenas"][name] = {
            "score": res["summary"]["score"], "elo": res["summary"]["elo"], "games": len(games),
            "moves": s, "swing_A_lost": swing_summary("A"), "swing_B_lost": swing_summary("B"),
            "repetitions": {"total": len(reps), "adv_150": sum(r["renouncer"] is not None for r in reps),
                            "renounced_by_A": sum(r["renouncer"] == "A" for r in reps),
                            "renounced_by_B": sum(r["renouncer"] == "B" for r in reps)},
            "conversion_300": conv}
        for x in swings:
            if not x["never_ok"] and x["by"] == x["loser"]:
                swings_for_q.append((name, x))
    # Q (MCTS-200) du perdant juste avant son coup de bascule, pour les réseaux disponibles
    if qnets:
        report["swing_q"] = swing_q(swings_for_q, qnets, args)
    write_json(args.out, report)
    print_games(report)
    return 0


def swing_q(swings, qnets: dict, args) -> dict:
    """Q racine (MCTS-200, sans bruit, comme l'arène) du perdant à la position où il joue son coup de bascule."""
    import torch

    from chesszero.checkpoint import load_for_inference
    from chesszero.mcts import MCTS, MCTSConfig, NetEvaluator, search_batched

    torch.set_num_threads(args.threads)
    cfg = MCTSConfig.named("kaggle")
    cfg = dataclasses.replace(cfg, dirichlet_epsilon=0.0, temperature=0.0)
    out = {}
    for label, path in qnets.items():
        targets = [(name, x) for name, x in swings
                   if (x["loser"] == "A" and label != "sup" and name == label) or (x["loser"] == "B" and label == "sup")]
        if not targets:
            continue
        model, _ = load_for_inference(path, "cpu")
        ev = NetEvaluator(model, "cpu")
        trees = [MCTS(ev, cfg, chess.Board(x["fen"])) for _, x in targets]
        for i in range(0, len(trees), 64):
            search_batched(trees[i:i + 64], ev, 200, 4)
        qs = [float(t.root.q()) for t in trees]
        out[label] = {"positions": len(qs), "q_mean": float(np.mean(qs)), "q_median": float(np.median(qs)),
                      "q_above_0": float(np.mean(np.array(qs) > 0)), "q_below_-0.3": float(np.mean(np.array(qs) < -0.3)),
                      "cp_before_median": float(np.median([x["loser_cp_before"] for _, x in targets])),
                      "items": [{"arena": n, "key": x["key"], "ply": x["ply"], "q": q, "cp_before": x["loser_cp_before"],
                                 "cp_after": x["loser_cp_after"]} for (n, x), q in zip(targets, qs)]}
    return out


def print_games(report: dict) -> None:
    print(f"{'arène':<11}{'score':>7}{'Elo':>6} | {'ACPL A/B ouv.':>14}{'milieu':>13}{'finale':>13} | "
          f"{'gaffes A/B ouv.':>15}{'milieu':>9}{'finale':>9} | bascules A (propre, gaffe) | rép. adv. A/B | conv. ratées A/B")
    for name, r in report["arenas"].items():
        m = r["moves"]

        def g(p, ph, k):
            return m.get(f"{p}:{ph}", {}).get(k, float("nan"))
        print(f"{name:<11}{r['score']:>7.1%}{r['elo']:>6.0f} | "
              + "".join(f"{g('A', ph, 'acpl'):>6.1f}/{g('B', ph, 'acpl'):<6.1f}" for ph in ("ouverture", "milieu", "finale"))
              + " | " + "".join(f"{g('A', ph, 'blunders_200'):>4}/{g('B', ph, 'blunders_200'):<4}" for ph in ("ouverture", "milieu", "finale"))
              + f" | {r['swing_A_lost']['lost_games']} ({r['swing_A_lost']['own_move']}, {r['swing_A_lost']['own_blunder_200']})"
              + f" | {r['repetitions']['renounced_by_A']}/{r['repetitions']['renounced_by_B']}"
              + f" | {r['conversion_300']['A']['not_won']}/{r['conversion_300']['B']['not_won']}")


def cmd_conversions(args) -> int:
    """Conversions ratées (camp à > +300 cp sur deux positions consécutives, partie non gagnée) : fins, gaffe du camp
    avantagé ensuite, perdues, phase du premier +300 ; pour A (candidat) et B (kaggle_sup), toutes arènes ensemble."""
    from collections import Counter

    sf = json.loads(Path(args.sf).read_text(encoding="utf-8"))
    out = {}
    for side in "AB":
        term, phase, n, blunder, lost, reached, finals = Counter(), Counter(), 0, 0, 0, 0, []
        for a in args.arenas:
            _, games = read_arena(Path(a))
            for g in games:
                res = analyse_game(g, sf)
                conv = res["conversion"][side]
                reached += conv["reached_300"]
                if not (conv["reached_300"] and not conv["won"]):
                    continue
                n += 1
                sign = 1 if (side == "A") == g["a_white"] else -1
                ev = [clamp(sign * sf[fen_key(f)]) for _, f in game_positions(g)]
                first = next(k for k in range(len(ev) - 1) if ev[k] > 300 and ev[k + 1] > 300)
                rows = res["rows"]
                phase[rows[min(first, len(rows) - 1)]["phase"]] += 1
                blunder += max((r["loss"] for r in rows[first:] if r["player"] == side), default=0) > 200
                term[g["termination"]] += 1
                finals.append(ev[-1])
                lost += res["score_a"] == (0 if side == "A" else 1)
        out[side] = {"reached_300": reached, "not_won": n, "not_won_share": n / max(reached, 1), "lost": lost,
                     "terminations": dict(term), "own_blunder_200_after": blunder, "phase_of_first_300": dict(phase),
                     "final_cp_median": float(np.median(finals)) if finals else None}
        print(side, json.dumps(out[side], ensure_ascii=False))
    write_json(args.out, out)
    return 0


# --------------------------------------------------------------------------- (c) valeur


def net_values(path: str, boards: np.ndarray, threads: int) -> np.ndarray:
    import torch

    from chesszero.checkpoint import load_for_inference
    from chesszero.encoding import unpack_batch

    torch.set_num_threads(threads)
    model, _ = load_for_inference(path, "cpu")
    out = []
    with torch.inference_mode():
        for i in range(0, len(boards), 1024):
            out.append(model(torch.from_numpy(unpack_batch(boards[i:i + 1024])))[1].numpy())
    return np.concatenate(out).astype(np.float64)


def calibration(v: np.ndarray, target: np.ndarray, edges=(-1, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0001)) -> dict:
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo) & (v < hi)
        if m.sum():
            bins.append({"range": [lo, min(hi, 1.0)], "n": int(m.sum()), "mean_v": float(v[m].mean()),
                         "mean_target": float(target[m].mean())})
    slope = float((v * target).sum() / max((v * v).sum(), 1e-12))  # cible ≈ pente × v (sans constante)
    return {"mse": float(((v - target) ** 2).mean()), "mean_abs_v": float(np.abs(v).mean()),
            "mean_abs_target": float(np.abs(target).mean()), "slope_target_on_v": slope,
            "corr": float(np.corrcoef(v, target)[0, 1]), "bins": bins}


def cmd_nets(args) -> int:
    from chesszero.selfplay_train import is_heldout

    boards, results, plies = [], [], []
    for f in sorted(Path(args.buffer).glob(f"{args.prefix}*.npz")):
        d = np.load(f)
        held = np.array([is_heldout(u, 0.10) for u in d["game_uid"][d["game"]].tolist()], bool)
        boards.append(d["boards"][held])
        results.append(d["result"][held].astype(np.float64))
        plies.append(d["ply"][held])
    boards, results = np.concatenate(boards), np.concatenate(results)
    if len(boards) > args.positions:
        keep = np.sort(np.random.default_rng(0).choice(len(boards), args.positions, replace=False))
        boards, results = boards[keep], results[keep]
    ev = np.load(args.eval_value)
    out = {"buffer": str(args.buffer), "prefix": args.prefix, "heldout_positions": int(len(boards)),
           "eval_value": str(args.eval_value), "networks": {}}
    print(f"{len(boards)} positions des parties mises de côté ({args.prefix}*), {len(ev['boards'])} positions notées par Stockfish")
    for path in args.checkpoint:
        v = net_values(path, boards, args.threads)
        vs = net_values(path, ev["boards"], args.threads)
        res = {"selfplay_outcome": calibration(v, results)}
        for src in ("selfplay", "lichess"):
            m = ev["source"] == src
            res[f"stockfish_{src}"] = calibration(vs[m], ev["stockfish"][m].astype(np.float64))
            res[f"outcome_{src}"] = calibration(vs[m], ev["outcome"][m].astype(np.float64))
        out["networks"][path] = res
        print(f"\n{path}")
        for k, c in res.items():
            print(f"  {k:<20} MSE {c['mse']:.4f}  |v| {c['mean_abs_v']:.3f}  |cible| {c['mean_abs_target']:.3f}  "
                  f"pente {c['slope_target_on_v']:.3f}  corr {c['corr']:.3f}")
    write_json(args.out, out)
    return 0


# --------------------------------------------------------------------------- hybrides politique / valeur


class HybridEvaluator:
    """Prior d'un réseau, valeur d'un autre (deux appels par lot)."""

    def __init__(self, policy_ev, value_ev):
        self.p, self.v = policy_ev, value_ev

    def evaluate(self, boards, legal_indices):
        priors, _ = self.p.evaluate(boards, legal_indices)
        if self.v is self.p:
            return priors, _
        _, values = self.v.evaluate(boards, legal_indices)
        return priors, values


def cmd_hybrid(args) -> int:
    import torch

    from chesszero.checkpoint import load_for_inference
    from chesszero.mcts import MCTS, MCTSConfig, NetEvaluator, search_batched

    torch.set_num_threads(args.threads)
    sf = json.loads(Path(args.sf).read_text(encoding="utf-8"))
    _, games = read_arena(Path(args.arena))
    items = []
    for g in games:
        for ply, fen in game_positions(g)[:-1]:
            b = chess.Board(fen)
            if b.legal_moves.count() > 1:
                if args.min_mover_cp is not None:  # positions gagnantes pour le camp au trait (conversion)
                    cp = sf.get(fen_key(fen))
                    mover = None if cp is None else (cp if b.turn == chess.WHITE else -cp)
                    if mover is None or not (args.min_mover_cp <= mover <= args.max_mover_cp):
                        continue
                items.append((g["key"], ply, fen))
    rng = np.random.default_rng(args.seed)
    items = [items[i] for i in np.sort(rng.choice(len(items), min(args.positions, len(items)), replace=False))]
    evs = {k: NetEvaluator(load_for_inference(p, "cpu")[0], "cpu") for k, p in (("A", args.policy_a), ("B", args.policy_b))}
    cfg = MCTSConfig.named("kaggle")
    cfg = dataclasses.replace(cfg, dirichlet_epsilon=0.0, temperature=0.0)
    combos = {"PA+VA": ("A", "A"), "PB+VB": ("B", "B"), "PA+VB": ("A", "B"), "PB+VA": ("B", "A")}
    chosen = {}
    start = time.perf_counter()
    for name, (p, v) in combos.items():
        ev = HybridEvaluator(evs[p], evs[v])
        moves = []
        for i in range(0, len(items), 64):
            trees = [MCTS(ev, cfg, chess.Board(fen)) for _, _, fen in items[i:i + 64]]
            search_batched(trees, ev, 200, 4)
            moves.extend(t.best_move(0.0).uci() for t in trees)
        chosen[name] = moves
        print(f"  {name} : {len(moves)} positions, {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    # positions après chaque coup choisi, évaluées par Stockfish (cache commun)
    need = {}
    for name, moves in chosen.items():
        for (_, _, fen), uci in zip(items, moves):
            b = chess.Board(fen)
            b.push_uci(uci)
            if fen_key(b.fen()) not in sf:
                need[fen_key(b.fen())] = b.fen()
    for _, _, fen in items:
        if fen_key(fen) not in sf:
            need[fen_key(fen)] = fen
    if need:
        chunks = [(f"h{i}", list(need.values())[i::args.workers]) for i in range(args.workers)]
        with mp.get_context("spawn").Pool(args.workers, _init_sf, (str(STOCKFISH), args.depth, 64)) as pool:
            for _, res in pool.imap_unordered(_eval_game, chunks):
                sf.update(res)
        Path(args.sf).write_text(json.dumps(sf) + "\n", encoding="utf-8")
    out = {"arena": str(args.arena), "positions": len(items), "min_mover_cp": args.min_mover_cp, "A": args.policy_a, "B": args.policy_b, "combos": {}}
    losses = {}
    for name, moves in chosen.items():
        ls = []
        for (_, ply, fen), uci in zip(items, moves):
            b = chess.Board(fen)
            white = b.turn == chess.WHITE
            b.push_uci(uci)
            ls.append(move_loss(sf[fen_key(fen)], sf[fen_key(b.fen())], white))
        losses[name] = np.array(ls)
        out["combos"][name] = {"acpl": float(np.mean(ls)), "blunders_200": int((np.array(ls) > 200).sum()),
                               "mistakes_100": int((np.array(ls) > 100).sum())}
    base = losses["PB+VB"]
    for name in combos:
        d = losses[name] - base
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        out["combos"][name]["diff_vs_PB+VB"] = float(d.mean())
        out["combos"][name]["diff_ci95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        out["combos"][name]["same_move_as_PB+VB"] = float(np.mean([a == b for a, b in zip(chosen[name], chosen["PB+VB"])]))
    for name, c in out["combos"].items():
        print(f"{name}: ACPL {c['acpl']:.1f} (écart à PB+VB {c['diff_vs_PB+VB']:+.1f}, 95 % {c['diff_ci95'][0]:+.1f} à "
              f"{c['diff_ci95'][1]:+.1f}), gaffes {c['blunders_200']}, erreurs {c['mistakes_100']}, "
              f"même coup que PB+VB {c['same_move_as_PB+VB']:.1%}")
    write_json(args.out, out)
    return 0


# --------------------------------------------------------------------------- règles d'arène


def simulate_rules(cands: list[dict], rules: dict, veto_se: float = 2.0, stop_gens: int = 2,
                   stop_from: int = 3, counters_reset_after: tuple = ()) -> dict:
    """Ce que chaque règle aurait décidé sur une suite de candidats `cands` = [{"id", "elo", "elo_se", "score",
    "los"}] mesurés CONTRE kaggle_sup (l'ancre). Exact tant que l'accepté est kaggle_sup (l'arène jouée est la bonne) ;
    ensuite HYPOTHÉTIQUE : l'arène contre un accepté autre que kaggle_sup n'a pas été jouée, on l'estime par
    transitivité sur l'échelle de l'ancre (écart d'Elo, erreur type = racine de la somme des carrés), et l'on garde
    les mêmes candidats alors qu'en vrai le self-play (donc l'entraînement) aurait changé dès la 1re acceptation.
    Veto de l'ancre et arrêt « ancre » (borne haute < 0 sur `stop_gens` générations consécutives >= `stop_from`) comme
    `selfplay_train.commit_arena` ; arrêt « stagnation » après 5 rejets en arène d'affilée ; compteur de rejets remis à
    0 après les candidats de `counters_reset_after` (remise de la ligne entre les sessions 1 et 2, comme en vrai)."""
    from statistics import NormalDist

    from chesszero.rating import elo_from_score

    out = {}
    for name, rule in rules.items():
        acc = {"id": None, "elo": 0.0, "elo_se": 0.0}
        gen, rejections, hist, rows, status = 0, 0, [], [], "ok"
        for c in cands:
            if status != "ok":
                rows.append({"id": c["id"], "decision": f"pas joué (arrêt {status})"})
                continue
            exact = acc["id"] is None
            if exact:
                score, los, diff, se = c["score"], c["los"], c["elo"], c["elo_se"]
            else:
                diff = c["elo"] - acc["elo"]
                se = math.hypot(c["elo_se"], acc["elo_se"])
                score = 1 / (1 + 10 ** (-diff / 400))
                los = NormalDist().cdf(diff / se)
            passes = {"score": lambda: score >= rule[1] - 1e-9, "los": lambda: los >= rule[1] - 1e-12,
                      "all": lambda: True}[rule[0]]()
            veto = passes and not exact and diff < -veto_se * se
            ok = passes and not veto
            row = {"id": c["id"], "exact": exact, "vs": acc["id"], "score": score, "los": los, "diff": diff,
                   "accepted": ok, "veto": veto}
            if ok:
                gen += 1
                acc = {"id": c["id"], "elo": c["elo"], "elo_se": c["elo_se"]}
                hi = c["elo_high"] if c.get("elo_high") is not None else c["elo"] + 1.96 * c["elo_se"]
                hist.append({"generation": gen, "elo_high": hi})
                rejections = 0
                below = [h for h in hist if h["generation"] >= stop_from][-stop_gens:]
                if len(below) == stop_gens and all(h["elo_high"] < 0 for h in below) and \
                        [h["generation"] for h in below] == list(range(gen - stop_gens + 1, gen + 1)):
                    status = "ancre"
                row["generation"] = gen
            else:
                rejections += 1
                if rejections >= 5:
                    status = "stagnation"
            if c["id"] in counters_reset_after:
                rejections = 0
            rows.append(row)
        out[name] = {"rows": rows, "generations": gen, "final_accepted": acc["id"], "status": status,
                     "final_elo_vs_anchor": acc["elo"]}
    return out


def cmd_gates(args) -> int:
    cands = []
    for a in args.arenas:
        res = json.loads((Path(a) / "result.json").read_text(encoding="utf-8"))
        s = res["summary"]
        cands.append({"id": int(res["players"]["a"]["name"].split("_")[-1]), "elo": s["elo"], "elo_se": s["elo_se"],
                      "score": s["score"], "los": s["los"], "elo_low": s["elo_low"], "elo_high": s["elo_high"]})
    cands.sort(key=lambda c: c["id"])
    rules = {"score 52 %": ("score", 0.52), "score 50 %": ("score", 0.50), "los 0,10": ("los", 0.10),
             "los 0,20": ("los", 0.20), "aucune arène": ("all", None)}
    sims = simulate_rules(cands, rules, counters_reset_after=(1,))
    # tendance de l'Elo avec la part de Q exact (H1) : candidats 2 à 6, moindres carrés pondérés
    exact_q = dict(zip(range(2, 7), (0.0, 0.21, 0.41, 0.61, 0.82)))
    pts = [(exact_q[c["id"]], c["elo"], c["elo_se"]) for c in cands if c["id"] in exact_q]
    w = np.array([1 / s ** 2 for _, _, s in pts])
    x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    xm, ym = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
    slope = (w * (x - xm) * (y - ym)).sum() / (w * (x - xm) ** 2).sum()
    slope_se = math.sqrt(1 / (w * (x - xm) ** 2).sum())
    chi2 = float((w * (y - ym) ** 2).sum())
    pooled = {"elo": ym, "se": math.sqrt(1 / w.sum()), "chi2_constant": chi2, "dof": len(pts) - 1}
    out = {"candidates": cands, "rules": sims, "trend_exact_q": {"slope_per_100pct": slope, "slope_se": slope_se},
           "pooled_2_6": pooled}
    for c in cands:
        print(f"candidat {c['id']} : score {c['score']:.1%}, Elo {c['elo']:+.0f} ± {1.96 * c['elo_se']:.0f}, "
              f"P(meilleur) {c['los']:.0%}")
    print(f"\nH1 : pente de l'Elo avec la part de Q exact (candidats 2-6) : {slope:+.0f} Elo pour 100 % "
          f"(erreur type {slope_se:.0f}) ; Elo moyen pondéré {ym:+.0f} ± {1.96 * pooled['se']:.0f}, khi² d'un Elo "
          f"constant {chi2:.1f} pour {len(pts) - 1} degrés de liberté")
    for name, s in sims.items():
        dec = ", ".join(f"{r['id']}:{'A' if r.get('accepted') else ('V' if r.get('veto') else ('-' if 'accepted' in r else 'x'))}"
                        f"{'' if r.get('exact', True) else '*'}" for r in s["rows"])
        print(f"{name:<13} {dec}  -> {s['generations']} génération(s), accepté final {s['final_accepted']}, "
              f"état {s['status']}")
    print("(A accepté, - rejeté, V veto de l'ancre, x pas joué ; * = décision hypothétique, arène non jouée)")
    write_json(args.out, out)
    return 0


# --------------------------------------------------------------------------- commun


def write_json(path, obj) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print("->", path)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prune")
    p.add_argument("--buffer", required=True, type=Path)
    p.add_argument("--prefix", default="L0002")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--out", default=None)
    p = sub.add_parser("sfeval")
    p.add_argument("--arenas", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--depth", type=int, default=14)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--hash", type=int, default=64)
    p.add_argument("--stockfish", default=str(STOCKFISH))
    p.add_argument("--limit", type=int, default=0, help="essai sur N parties (rien n'est écrit)")
    p = sub.add_parser("games")
    p.add_argument("--arenas", nargs="+", required=True)
    p.add_argument("--sf", required=True)
    p.add_argument("--qnet", nargs="*", default=None,
                   help="label=checkpoint : Q du perdant à sa bascule ; label = nom du candidat (A) ou « sup » (B)")
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--out", default=None)
    p = sub.add_parser("nets")
    p.add_argument("--buffer", required=True, type=Path)
    p.add_argument("--prefix", default="L0002")
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--eval-value", required=True)
    p.add_argument("--positions", type=int, default=50000)
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--out", default=None)
    p = sub.add_parser("hybrid")
    p.add_argument("--arena", required=True)
    p.add_argument("--sf", required=True)
    p.add_argument("--policy-a", required=True, help="réseau A (candidat)")
    p.add_argument("--policy-b", required=True, help="réseau B (référence)")
    p.add_argument("--positions", type=int, default=800)
    p.add_argument("--min-mover-cp", type=int, default=None, help="seulement les positions où le camp au trait a au moins ça")
    p.add_argument("--max-mover-cp", type=int, default=999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth", type=int, default=14)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--out", default=None)
    p = sub.add_parser("conversions")
    p.add_argument("--arenas", nargs="+", required=True)
    p.add_argument("--sf", required=True)
    p.add_argument("--out", default=None)
    p = sub.add_parser("gates")
    p.add_argument("--arenas", nargs="+", required=True)
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.exit({"conversions": cmd_conversions, "gates": cmd_gates, "prune": cmd_prune, "sfeval": cmd_sfeval, "games": cmd_games, "nets": cmd_nets,
              "hybrid": cmd_hybrid}[args.command](args))


if __name__ == "__main__":
    main()
