"""Équilibre des ouvertures du self-play et de l'arène (analyse seule : ne modifie ni le livre ni la config).

  analyse_openings.py sets   --out docs/analyse_ouvertures/sets.json     # 2 000 positions du livre (graine 0), réserve d'arène, ancre
  analyse_openings.py eval   --sets <sets.json> --out <eval.json> --depth 14 --workers 6   # Stockfish à profondeur fixe, centipions (Blancs)
                             [--limit N]  = essai : mesure le temps par position et annonce la durée du calcul complet
  analyse_openings.py report --sets <sets.json> --eval <eval.json>      # part des positions au-delà de ±100 / ±200 cp par groupe
  analyse_openings.py buffer --buffer <selfplay/>                        # Q racine au premier coup cherché, lien avec le résultat
  analyse_openings.py pairs  --arenas <dossier d'arène> <dossier d'arène> --sets <sets.json> --eval <eval.json>
                                                                          # paires « mortes » (le camp décide, pas le joueur)

Une paire = les deux parties d'une même ouverture avec les couleurs échangées. Paire 1-1 : le MÊME camp gagne les deux parties
(Blancs deux fois ou Noirs deux fois) : l'ouverture décide, pas le réseau ; deux nulles : ouverture neutre ou trop sèche ;
même joueur vainqueur deux fois : vraie différence de force.
"""

import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "books" / "lichess_2000.tsv.gz"
RESERVE = ROOT / "books" / "lichess_2000_arena.tsv.gz"
STOCKFISH = ROOT / "tools" / "stockfish" / "stockfish" / "stockfish-windows-x86-64-universal.exe"
MATE_CP = 10_000


def board_of(line: str) -> chess.Board:
    board = chess.Board()
    for uci in line.split():
        board.push_uci(uci)
    return board


def epd_key(line: str) -> str:
    return board_of(line).epd()


# --------------------------------------------------------------------------- sets


def cmd_sets(args) -> int:
    from chesszero.arena import anchor_openings
    from chesszero.book import OpeningBook

    book, reserve = OpeningBook.load(BOOK), OpeningBook.load(RESERVE)
    book.exclude_from_sampling(reserve.lines)  # le self-play ne tire pas la réserve
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(book.allowed, size=args.book_size, replace=False)
    positions: dict[str, dict] = {}

    def add(line: str, value: float) -> str:
        key = epd_key(line)
        positions.setdefault(key, {"line": line, "net_value": float(value)})
        return key

    groups = {"book_random": [add(book.lines[i], book.values[i]) for i in picked],
              "reserve": [add(l, v) for l, v in zip(reserve.lines, reserve.values)]}
    reserve_keys = groups["reserve"]  # l'indice d'ouverture d'une arène est l'indice dans la réserve
    groups["anchor_100"] = [reserve_keys[i] for i in anchor_openings(len(reserve), args.anchor)]
    out = {"meta": {"seed": args.seed, "book": BOOK.name, "book_positions": len(book), "book_sampled_from": int(len(book.allowed)),
                    "reserve_positions": len(reserve), "anchor": args.anchor, "unique_positions": len(positions)},
           "positions": positions, "groups": groups, "reserve_keys": reserve_keys}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(positions)} positions distinctes : livre {len(set(groups['book_random']))}, réserve {len(set(reserve_keys))}, "
          f"ancre {len(set(groups['anchor_100']))} -> {args.out}")
    return 0


# --------------------------------------------------------------------------- évaluation Stockfish

_E: dict = {}


def init_eval(stockfish: str, depth: int, hash_mb: int) -> None:
    from chesszero import cpu_guard

    cpu_guard.disable_ecoqos()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    engine.configure({"Threads": 1, "Hash": hash_mb})
    try:
        cpu_guard.disable_ecoqos(engine.transport.get_pid())
    except Exception:  # noqa: BLE001
        pass
    _E.update(engine=engine, depth=depth)


def eval_one(item: tuple[str, str]) -> tuple[str, dict]:
    key, line = item
    board = board_of(line)
    t = time.perf_counter()
    info = _E["engine"].analyse(board, chess.engine.Limit(depth=_E["depth"]))
    score = info["score"].white()
    mate = score.mate()
    cp = score.score(mate_score=MATE_CP)
    return key, {"cp": int(cp), "mate": mate, "depth": info.get("depth"), "nodes": info.get("nodes"),
                 "seconds": round(time.perf_counter() - t, 3)}


def cmd_eval(args) -> int:
    sets = json.loads(Path(args.sets).read_text(encoding="utf-8"))
    done = json.loads(Path(args.out).read_text(encoding="utf-8")) if Path(args.out).exists() and not args.limit else {}
    todo = [(k, v["line"]) for k, v in sets["positions"].items() if k not in done]
    total_unique = len(sets["positions"])
    if args.limit:
        rng = np.random.default_rng(1)
        todo = [todo[i] for i in rng.choice(len(todo), size=min(args.limit, len(todo)), replace=False)]
    print(f"{len(todo)} positions à évaluer à la profondeur {args.depth} ({len(done)} déjà faites sur {total_unique}), "
          f"{args.workers} processus", flush=True)
    start = time.perf_counter()
    results = dict(done)
    with mp.get_context("spawn").Pool(args.workers, init_eval, (str(args.stockfish), args.depth, args.hash)) as pool:
        for n, (key, res) in enumerate(pool.imap_unordered(eval_one, todo, chunksize=4), 1):
            results[key] = res
            if not args.limit and n % 250 == 0:
                print(f"  {n}/{len(todo)} ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
                Path(args.out).write_text(json.dumps(results) + "\n", encoding="utf-8")
    wall = time.perf_counter() - start
    secs = [r["seconds"] for k, r in results.items() if k in dict(todo)]
    if args.limit:
        per_position = wall / len(todo)
        print(f"Essai : {len(todo)} positions en {wall:.0f} s, {per_position:.2f} s par position en parallèle "
              f"(médiane d'une position : {statistics.median(secs):.2f} s, profondeur atteinte {Counter(r['depth'] for r in results.values()).most_common(1)[0][0]})")
        remaining = total_unique - len(done)
        print(f"Durée estimée du calcul complet ({remaining} positions) : {remaining * per_position / 60:.1f} min")
        return 0
    Path(args.out).write_text(json.dumps(results) + "\n", encoding="utf-8")
    print(f"{len(results)} positions évaluées en {wall / 60:.1f} min -> {args.out}")
    return 0


# --------------------------------------------------------------------------- rapport sur les positions


def group_stats(cps: list[int]) -> dict:
    a = np.abs(np.array(cps, dtype=float))
    return {"n": len(cps), "mean_abs_cp": round(float(a.mean()), 1), "median_abs_cp": round(float(np.median(a)), 1),
            "p90_abs_cp": round(float(np.quantile(a, 0.9)), 1), "share_gt_100": round(float((a > 100).mean()), 4),
            "share_gt_200": round(float((a > 200).mean()), 4), "share_gt_300": round(float((a > 300).mean()), 4),
            "mean_cp_white": round(float(np.mean(cps)), 1), "share_white_better_100": round(float((np.array(cps) > 100).mean()), 4),
            "share_black_better_100": round(float((np.array(cps) < -100).mean()), 4)}


def cmd_report(args) -> int:
    sets = json.loads(Path(args.sets).read_text(encoding="utf-8"))
    ev = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    out = {"depth": Counter(r["depth"] for r in ev.values()).most_common(3), "groups": {}}
    for name, keys in sets["groups"].items():
        out["groups"][name] = group_stats([ev[k]["cp"] for k in keys])
    # lien avec la valeur du réseau de départ (filtre du livre : |valeur| <= 0,3, camp au trait)
    keys = list(sets["positions"])
    net = np.array([sets["positions"][k]["net_value"] for k in keys])
    white = np.array([sets["positions"][k]["line"].count(" ") % 2 == 1 for k in keys])  # nb de coups impair -> Noirs au trait
    cp_white = np.array([ev[k]["cp"] for k in keys], dtype=float)
    cp_stm = np.where(white, cp_white, -cp_white)  # point de vue du camp au trait
    out["net_value_vs_stockfish"] = {"pearson": round(float(np.corrcoef(net, cp_stm)[0, 1]), 3)}
    bins = [(-0.31, -0.2), (-0.2, -0.1), (-0.1, 0.1), (0.1, 0.2), (0.2, 0.31)]
    out["net_value_vs_stockfish"]["bins"] = [
        {"net_value": f"{lo:+.1f} à {hi:+.1f}", "n": int(m.sum()), "mean_cp_stm": round(float(cp_stm[m].mean()), 1),
         "share_abs_gt_200": round(float((np.abs(cp_stm[m]) > 200).mean()), 3)}
        for lo, hi in bins for m in [((net >= lo) & (net < hi))] if m.any()]
    Path(args.eval).with_name("rapport_positions.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"profondeurs atteintes : {out['depth']}")
    print(f"{'groupe':<14}{'n':>6}{'|cp| moy.':>10}{'médiane':>9}{'p90':>7}{'> 100':>8}{'> 200':>8}{'> 300':>8}{'cp moyen (B)':>14}")
    for name, s in out["groups"].items():
        print(f"{name:<14}{s['n']:>6}{s['mean_abs_cp']:>10}{s['median_abs_cp']:>9}{s['p90_abs_cp']:>7}"
              f"{s['share_gt_100']:>8.1%}{s['share_gt_200']:>8.1%}{s['share_gt_300']:>8.1%}{s['mean_cp_white']:>14}")
    print(f"valeur du réseau de départ contre Stockfish (camp au trait) : corrélation {out['net_value_vs_stockfish']['pearson']}")
    for b in out["net_value_vs_stockfish"]["bins"]:
        print(f"  valeur réseau {b['net_value']:>12} : n={b['n']:>5}, cp moyen {b['mean_cp_stm']:+7.1f}, |cp| > 200 : {b['share_abs_gt_200']:.1%}")
    return 0


# --------------------------------------------------------------------------- tampon de la génération 0


def cmd_buffer(args) -> int:
    from chesszero.selfplay import TERMINATIONS, load_shard

    rows = []
    for p in sorted(Path(args.buffer).glob("*.npz")):
        if not p.with_suffix(".json").exists():
            continue
        d = load_shard(p)
        for g in range(len(d["game_uid"])):
            idx = np.nonzero(d["game"] == g)[0]
            if not len(idx):
                continue
            i = idx[np.argmin(d["ply"][idx])]
            ply, q = int(d["ply"][i]), float(d["root_q"][i])
            qw = q if ply % 2 == 0 else -q
            rows.append({"q_white": qw, "ply": ply, "book": int(d["start_index"][g]) >= 0, "result_white": int(d["result_white"][g]),
                         "termination": TERMINATIONS[int(d["termination"][g])], "control": bool(d["control"][g]),
                         "start_plies": int(d["start_plies"][g])})
    q = np.array([r["q_white"] for r in rows])
    res = np.array([r["result_white"] for r in rows])
    book = np.array([r["book"] for r in rows])
    out = {"games": len(rows), "book_games": int(book.sum()), "initial_position_games": int((~book).sum())}

    def block(mask: np.ndarray) -> dict:
        a = np.abs(q[mask])
        favored = np.sign(q[mask])
        # résultat du point de vue du camp que le premier Q favorise : +1 gagné, 0 nul, -1 perdu
        mine = res[mask] * favored
        return {"n": int(mask.sum()), "mean_abs_q": round(float(a.mean()), 3), "median_abs_q": round(float(np.median(a)), 3),
                "share_gt_0.3": round(float((a > 0.3).mean()), 4), "share_gt_0.5": round(float((a > 0.5).mean()), 4),
                "favored_side": {"won": round(float((mine == 1).mean()), 3), "drawn": round(float((mine == 0).mean()), 3),
                                 "lost": round(float((mine == -1).mean()), 3)}}

    out["book_games_q"] = block(book)
    if (~book).any():
        out["initial_position_q"] = block(~book)
    bands = {"|Q| < 0.1": (0, 0.1), "0.1 à 0.3": (0.1, 0.3), "0.3 à 0.5": (0.3, 0.5), "> 0.5": (0.5, 9)}
    out["by_band_book_games"] = {}
    for name, (lo, hi) in bands.items():
        m = book & (np.abs(q) >= lo) & (np.abs(q) < hi)
        if m.any():
            mine = res[m] * np.sign(q[m])
            out["by_band_book_games"][name] = {"n": int(m.sum()), "share": round(float(m.sum() / book.sum()), 3),
                                                "decisive": round(float((res[m] != 0).mean()), 3),
                                                "favored_won": round(float((mine == 1).mean()), 3),
                                                "favored_lost": round(float((mine == -1).mean()), 3)}
    out["q_white_quantiles_book"] = {str(p): round(float(np.quantile(q[book], p)), 3) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
    # abandons par bande (l'abandon exige |Q| > 0,9 pendant 3 coups : lien avec un départ déséquilibré)
    resign = np.array([r["termination"] == "abandon" for r in rows])
    out["resign_share_by_band_book_games"] = {name: round(float(resign[book & (np.abs(q) >= lo) & (np.abs(q) < hi)].mean()), 3)
                                              for name, (lo, hi) in bands.items() if (book & (np.abs(q) >= lo) & (np.abs(q) < hi)).any()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------- paires mortes


def read_games(arena_dir: Path) -> dict[int, dict[bool, dict]]:
    by_opening: dict[int, dict[bool, dict]] = {}
    for line in (arena_dir / "games.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        by_opening.setdefault(int(g["opening"]), {})[bool(g["a_white"])] = g
    return by_opening


def pair_kind(games: dict[bool, dict]) -> str | None:
    if len(games) != 2:
        return None
    w, b = games[True]["result_white"], games[False]["result_white"]  # parties où A a les Blancs / les Noirs
    if w == 1 and b == 1:
        return "Blancs gagnent les deux (1-1)"
    if w == -1 and b == -1:
        return "Noirs gagnent les deux (1-1)"
    if w == 0 and b == 0:
        return "deux nulles"
    if (w == 1 and b == -1):
        return "A gagne les deux"
    if (w == -1 and b == 1):
        return "B gagne les deux"
    return "une nulle et une victoire"


def cmd_pairs(args) -> int:
    sets = json.loads(Path(args.sets).read_text(encoding="utf-8")) if args.sets else None
    ev = json.loads(Path(args.eval).read_text(encoding="utf-8")) if args.eval else None
    arenas = [read_games(Path(d)) for d in args.arenas]
    kinds = []
    for name, a in zip(args.arenas, arenas):
        k = Counter(pair_kind(g) for g in a.values())
        kinds.append(k)
        total = sum(k.values())
        print(f"\n{Path(name).name} : {total} paires")
        for label, n in k.most_common():
            print(f"  {label:<28}{n:>4}  ({n / total:.0%})")
    stats = []
    for name, a in zip(args.arenas, arenas):
        games = [g for pair in a.values() for g in pair.values()]
        n = len(games)
        pw = sum(g["result_white"] == 1 for g in games) / n
        pb = sum(g["result_white"] == -1 for g in games) / n
        pd = 1 - pw - pb
        pairs_n = len(a)
        exp_w, exp_b, exp_d = pairs_n * pw * pw, pairs_n * pb * pb, pairs_n * pd * pd  # si les deux parties d'une paire étaient indépendantes
        k = Counter(pair_kind(g) for g in a.values())
        stats.append({"arena": Path(name).name, "white_wins": round(pw, 3), "black_wins": round(pb, 3), "draws": round(pd, 3),
                      "white_both": [k["Blancs gagnent les deux (1-1)"], round(exp_w, 1)],
                      "black_both": [k["Noirs gagnent les deux (1-1)"], round(exp_b, 1)],
                      "two_draws": [k["deux nulles"], round(exp_d, 1)]})
        print(f"\n{Path(name).name} : Blancs gagnent {pw:.1%}, Noirs {pb:.1%}, nulles {pd:.1%} des {n} parties ; paires (observé / attendu si les "
              f"deux parties étaient indépendantes) : Blancs deux fois {k['Blancs gagnent les deux (1-1)']} / {exp_w:.1f}, Noirs deux fois "
              f"{k['Noirs gagnent les deux (1-1)']} / {exp_b:.1f}, deux nulles {k['deux nulles']} / {exp_d:.1f}")
    common = sorted(set(arenas[0]) & set(arenas[1]))
    print(f"\n{len(common)} ouvertures jouées dans les deux arènes")
    # même type de paire 1-1 dans les deux arènes : attendu si l'ouverture ne comptait pas (arènes indépendantes)
    f1 = {kk: v / len(arenas[0]) for kk, v in Counter(pair_kind(g) for g in arenas[0].values()).items()}
    f2 = {kk: v / len(arenas[1]) for kk, v in Counter(pair_kind(g) for g in arenas[1].values()).items()}
    p_same = sum(f1.get(kk, 0) * f2.get(kk, 0) for kk in ("Blancs gagnent les deux (1-1)", "Noirs gagnent les deux (1-1)"))
    from math import comb

    n_c = len(common)
    observed_dead = sum(1 for o in common if pair_kind(arenas[0][o]) == pair_kind(arenas[1][o]) and pair_kind(arenas[0][o]) in
                        ("Blancs gagnent les deux (1-1)", "Noirs gagnent les deux (1-1)"))
    p_tail = sum(comb(n_c, j) * p_same ** j * (1 - p_same) ** (n_c - j) for j in range(observed_dead, n_c + 1))
    print(f"paires 1-1 du même camp dans les deux arènes : observé {observed_dead}, attendu sans effet d'ouverture {n_c * p_same:.1f} "
          f"(probabilité d'en voir autant ou plus par hasard : {p_tail:.3f})")
    stats.append({"dead_both_observed": observed_dead, "dead_both_expected_if_no_opening_effect": round(n_c * p_same, 2),
                  "tail_probability": round(p_tail, 4)})
    dead = [o for o in common if pair_kind(arenas[0][o]) == pair_kind(arenas[1][o]) and pair_kind(arenas[0][o]) in
            ("Blancs gagnent les deux (1-1)", "Noirs gagnent les deux (1-1)")]
    both_draw = [o for o in common if pair_kind(arenas[0][o]) == pair_kind(arenas[1][o]) == "deux nulles"]
    print(f"paires 1-1 du même camp dans les DEUX arènes (ouverture très probablement décisive) : {len(dead)}")
    print(f"deux nulles dans les deux arènes : {len(both_draw)}")
    cross = None
    if sets and ev:
        keys = sets["reserve_keys"]
        cross = {"dead": [], "both_draw": [], "other": []}
        for o in common:
            cp = ev[keys[o]]["cp"]
            cross["dead" if o in dead else "both_draw" if o in both_draw else "other"].append((o, cp))
        for name, items in cross.items():
            if items:
                a = np.abs([cp for _, cp in items])
                print(f"  {name:<10} n={len(items):>3}  |cp| moyen {a.mean():6.1f}  médian {np.median(a):6.1f}  > 100 : {np.mean(a > 100):.0%}  > 200 : {np.mean(a > 200):.0%}")
        # coût du déséquilibre : part des paires « mortes » (même camp gagnant deux fois) par tranche d'évaluation, les deux arènes ensemble
        print("\npaires 1-1 du même camp par |évaluation| de l'ouverture (les deux arènes ensemble, 200 paires) :")
        bands = [("|cp| <= 50", 0, 50), ("50 < |cp| <= 100", 50, 100), ("100 < |cp| <= 200", 100, 200), ("|cp| > 200", 200, 10 ** 6)]
        table = []
        for label, lo, hi in bands:
            obs = [(pair_kind(a[o]) or "") for a in arenas for o in common if lo < abs(ev[keys[o]]["cp"]) <= hi or (lo == 0 and abs(ev[keys[o]]["cp"]) <= hi)]
            if lo == 0:
                obs = [(pair_kind(a[o]) or "") for a in arenas for o in common if abs(ev[keys[o]]["cp"]) <= hi]
            dead_n = sum(k in ("Blancs gagnent les deux (1-1)", "Noirs gagnent les deux (1-1)") for k in obs)
            draw_n = sum(k == "deux nulles" for k in obs)
            if obs:
                print(f"  {label:<20} paires {len(obs):>3} : 1-1 du même camp {dead_n / len(obs):>4.0%}, deux nulles {draw_n / len(obs):>4.0%}, "
                      f"paires informatives {1 - (dead_n + draw_n) / len(obs):>4.0%}")
                table.append({"band": label, "pairs": len(obs), "dead": dead_n, "two_draws": draw_n})
        cross["by_band"] = table
        print("\nouvertures mortes (1-1 du même camp dans les deux arènes) et évaluation Stockfish :")
        for o, cp in sorted(cross["dead"], key=lambda t: -abs(t[1])):
            print(f"  ouverture {o:>5} : {pair_kind(arenas[0][o]):<30} cp (Blancs) {cp:+5d}   {sets['positions'][keys[o]]['line']}")
        # sens : le camp qui gagne les deux parties est-il celui que Stockfish favorise ?
        agree = [(o, cp) for o, cp in cross["dead"] if (cp > 0) == pair_kind(arenas[0][o]).startswith("Blancs")]
        print(f"le camp gagnant est celui que Stockfish favorise : {len(agree)} sur {len(cross['dead'])}")
    out = {"arenas": [str(Path(a).name) for a in args.arenas], "pair_kinds": [dict(k) for k in kinds], "common_openings": len(common),
           "statistics": stats, "dead_both_arenas": dead, "both_draw_both_arenas": both_draw,
           "cross_stockfish": None if cross is None else {k: v for k, v in cross.items()}}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sets")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--book-size", type=int, default=2000)
    p.add_argument("--anchor", type=int, default=100)
    p = sub.add_parser("eval")
    p.add_argument("--sets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--depth", type=int, default=14)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--hash", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="essai : N positions, durée estimée du calcul complet")
    p.add_argument("--stockfish", default=str(STOCKFISH))
    p = sub.add_parser("report")
    p.add_argument("--sets", required=True)
    p.add_argument("--eval", required=True)
    p = sub.add_parser("buffer")
    p.add_argument("--buffer", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--arenas", nargs=2, required=True)
    p.add_argument("--sets", default=None)
    p.add_argument("--eval", default=None)
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.exit({"sets": cmd_sets, "eval": cmd_eval, "report": cmd_report, "buffer": cmd_buffer, "pairs": cmd_pairs}[args.command](args))


if __name__ == "__main__":
    main()
