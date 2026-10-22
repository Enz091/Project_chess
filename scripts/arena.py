"""Arène : réseau A (candidat) contre réseau B (accepté ou ancre), voir chesszero/arena.py. Relancer = reprendre.

  # contre le réseau accepté : 100 ouvertures de la réserve (graine = n° du candidat) x 2 couleurs
  arena.py --kind gate --seed 7 --a sp_train/candidate.pt --a-name cand_0007 \\
           --b sp_train/generations/gen_0003.pt --b-name gen_0003 --out loop/arena/cand_0007_gate
  # contre l'ancre : 100 ouvertures fixes
  arena.py --kind anchor --a ... --b /kaggle/input/chesszero-kaggle-sup/best.pt --b-name kaggle_sup --out ...
  # contrôles (version réduite en local, CPU) : contre lui-même (~50 % attendu), 200 contre 50 simulations
  arena.py --kind anchor --pairs 10 --a checkpoints/kaggle_sup/best.pt --b checkpoints/kaggle_sup/best.pt \\
           --out data/arena_controle_a --expect-min 0.45 --expect-max 0.55 --device cpu --workers 4 --threads 2
  arena.py --kind anchor --pairs 10 --a <best.pt> --a-sims 200 --b <best.pt> --b-sims 50 --expect-min 0.70 ...

Sorties dans --out : arena.json, games.jsonl, games.pgn, result.json (en dernier). Avec --history : une ligne
ajoutée (une seule fois) à l'historique, avec le hash git du code et un instantané de la configuration.
Code de sortie : 0 arène complète (et score dans --expect-min / --expect-max), 1 score hors de l'intervalle
attendu, 3 arène incomplète (budget de temps ou --stop-after : relancer pour reprendre).
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--a", required=True, help="checkpoint du joueur A (candidat)")
    ap.add_argument("--b", required=True, help="checkpoint du joueur B (accepté ou ancre)")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--a-sims", type=int, default=None, help="simulations par coup de A (défaut : [arena])")
    ap.add_argument("--b-sims", type=int, default=None, help="simulations par coup de B (défaut : [arena])")
    ap.add_argument("--kind", choices=["gate", "anchor"], required=True,
                    help="gate : ouvertures tirées (graine --seed) ; anchor : ouvertures fixes")
    ap.add_argument("--seed", type=int, default=0, help="gate : graine du tirage (numéro du candidat)")
    ap.add_argument("--pairs", type=int, default=None, help="ouvertures (x 2 couleurs) ; défaut : [arena]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None, help="identifiant de l'arène (défaut : nom de --out)")
    ap.add_argument("--book", default=str(ROOT / "books" / "lichess_2000.tsv.gz"), help="livre (réserve à côté)")
    ap.add_argument("--syzygy", default=str(ROOT / "tools" / "syzygy"), help="tables (WDL suffisent) ; '' = sans")
    ap.add_argument("--config", default="kaggle", help="configs/<nom>.toml, sections [mcts] et [arena]")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1, help="threads torch par processus")
    ap.add_argument("--games-per-worker", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-minutes", type=float, default=None, help="arrêt après ce temps (parties en cours perdues)")
    ap.add_argument("--stop-after", type=int, default=None, help="s'arrêter après N nouvelles parties (coupure simulée)")
    ap.add_argument("--history", default=None, help="historique JSONL des arènes (loop/arena/history.jsonl)")
    ap.add_argument("--expect-min", type=float, default=None, help="contrôle : score minimal attendu de A")
    ap.add_argument("--expect-max", type=float, default=None, help="contrôle : score maximal attendu de A")
    args = ap.parse_args()

    from dataclasses import replace

    from chesszero.arena import (ArenaConfig, Player, anchor_openings, append_history, describe, gate_openings,
                                 process_runner, run_arena)
    from chesszero.book import OpeningBook, arena_reserve_path
    from chesszero.mcts import MCTSConfig

    cfg = ArenaConfig.named(args.config)
    if args.games_per_worker:
        cfg = replace(cfg, games_per_worker=args.games_per_worker)
    base_mcts = MCTSConfig.named(args.config)
    reserve_path = arena_reserve_path(args.book)
    reserve = OpeningBook.load(reserve_path)
    if args.kind == "anchor":
        openings = anchor_openings(len(reserve), args.pairs or cfg.anchor_pairs)
    else:
        openings = gate_openings(len(reserve), args.pairs or cfg.pairs, args.seed, cfg.anchor_pairs)
    players = {"a": Player(args.a_name, str(Path(args.a)), args.a_sims or cfg.simulations),
               "b": Player(args.b_name, str(Path(args.b)), args.b_sims or cfg.simulations)}
    label = args.label or Path(args.out).name
    deadline = time.time() + 60 * args.max_minutes if args.max_minutes else None
    stats: list[dict] = []
    runner = process_runner(players, cfg, base_mcts, str(reserve_path), args.syzygy or None, args.workers,
                            args.threads, args.device, deadline, stats_out=stats)
    print(f"{players['a'].name} ({players['a'].simulations} sim) contre {players['b'].name} "
          f"({players['b'].simulations} sim) : {len(openings)} ouvertures x 2 couleurs, réserve {reserve_path.name} "
          f"({len(reserve)} positions), {args.workers} processus x {args.threads} thread(s)", flush=True)
    t0 = time.time()
    result = run_arena(args.out, players, openings, cfg, base_mcts, reserve, reserve_path.name, kind=args.kind,
                       label=label, run_games=runner, stop_after=args.stop_after)
    if stats:
        sims = sum(sum(s["simulations"].values()) for s in stats)
        play = max(s["seconds"] for s in stats)
        print(f"Ce lancement : {sum(s['games'] for s in stats)} parties, {sims:,} simulations en {play / 60:.1f} min "
              f"de jeu ({sims / max(play, 1e-9):,.0f} simulations/s), {(time.time() - t0) / 60:.1f} min au total")
    if result is None:
        sys.exit(3)
    s = result["summary"]
    print(describe(s, players["a"].name, players["b"].name))
    print(f"Fins : {s['terminations']} ; répétitions avant le demi-coup 60 : {s['repetitions_before_ply_60']} ; "
          f"longueur moyenne {s['mean_plies']:.0f} demi-coups ; score des Blancs {s['white_score']:.1%} ; "
          f"paires à exactement 1 point sur 2 : {s['pairs_scored_1']}/{s['complete_pairs']}")
    if args.history:
        added = append_history(args.history, {"id": label, **result})
        print(f"Historique : {'ligne ajoutée à' if added else 'déjà présente dans'} {args.history}")
    if args.expect_min is not None or args.expect_max is not None:
        lo = args.expect_min if args.expect_min is not None else 0.0
        hi = args.expect_max if args.expect_max is not None else 1.0
        ok = lo <= s["score"] <= hi
        print(f"CONTRÔLE {'RÉUSSI' if ok else 'ÉCHOUÉ'} : score de {players['a'].name} {s['score']:.1%}, "
              f"attendu entre {lo:.0%} et {hi:.0%}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
