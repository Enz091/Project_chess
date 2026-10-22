"""Self-play : génère des parties de l'IA contre elle-même et les enregistre (voir chesszero/selfplay.py).

  selfplay.py --out selfplay --max-games 4 --workers 2 --threads 2        # essai local (CPU)
  selfplay.py --out /kaggle/working/selfplay --inputs /kaggle/input/<sortie précédente>/selfplay \\
              --max-minutes 660 --workers 3                               # session Kaggle (GPU)

Déroulement :
  1. tampon : `sync_buffer(--inputs, --out)` copie dans --out les shards les plus récents des sessions
     précédentes, jusqu'à `buffer_positions` positions (500 000) ;
  2. `--workers` processus, chacun avec le réseau sur le device (GPU sur Kaggle), les tables Syzygy en mode WDL
     seul (arbitrage dès 5 pièces) et le livre d'ouvertures ; chacun mène `games_per_worker` parties ensemble ;
  3. budget de temps (--max-minutes) : plus de nouvelles parties `stop_new_games_minutes` avant la fin, arrêt
     `hard_stop_minutes` avant la fin (parties inachevées perdues) ; chaque processus écrit un shard tous les
     `shard_games` parties ou toutes les `shard_minutes` minutes ;
  4. tampon : nouveau `sync_buffer` : --out ne garde que la fenêtre (les nouveaux shards y entrent, les plus
     anciens en sortent). La sortie de la session contient donc le tampon complet pour la suivante.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def worker(task: dict) -> dict:
    import torch

    from chesszero.book import OpeningBook
    from chesszero.checkpoint import load_for_inference
    from chesszero.device import get_device
    from chesszero.mcts import MCTSConfig, NetEvaluator
    from chesszero.selfplay import SelfPlayConfig, ShardWriter, config_dict, run_selfplay
    from chesszero.syzygy import Tablebase

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # processus fils : même encodage que le parent
    torch.set_num_threads(task["threads"])
    cfg = SelfPlayConfig.named(task["config"])
    if task["overrides"]:
        from dataclasses import replace
        cfg = replace(cfg, **task["overrides"])
    mcts_cfg = cfg.mcts_config(MCTSConfig.named(task["config"]))
    device = get_device(task["device"])
    model, meta = load_for_inference(task["checkpoint"], device)
    evaluator = NetEvaluator(model)
    tb = Tablebase.open(task["syzygy"], max_pieces=cfg.tablebase_pieces, wdl_only=True) if task["syzygy"] else None
    book = OpeningBook.load(task["book"]) if task["book"] else None
    seed = zlib.crc32(f"{task['session']}/{task['worker']}/{task['seed']}".encode())
    rng = np.random.default_rng(seed)
    prefix = f"{task['session']}_w{task['worker']}"
    shard_meta = {"session": task["session"], "worker": task["worker"], "seed": seed,
                  "generation": task["generation"], "checkpoint": Path(task["checkpoint"]).name,
                  "checkpoint_step": meta.get("step"), "book": Path(task["book"]).name if task["book"] else None,
                  "tablebase": tb.describe() if tb else None, "device": str(device),
                  "config": config_dict(cfg, mcts_cfg)}
    writer = ShardWriter(task["out"], prefix, shard_meta, cfg.shard_games, cfg.shard_minutes)
    log = (lambda msg: print(f"[w{task['worker']}] {msg}", flush=True))
    log(f"{device}, {tb.describe() if tb else 'sans tables'}, livre {len(book) if book else 0} positions, "
        f"{cfg.games_per_worker} parties en parallèle, {cfg.simulations} simulations par coup")
    stats = run_selfplay(evaluator, cfg, mcts_cfg, writer, rng, prefix, book=book, tablebase=tb,
                         max_games=task["max_games"], soft_deadline=task["soft_deadline"],
                         hard_deadline=task["hard_deadline"], log=log)
    stats["shards"] = [p.name for p in writer.written]
    stats["eval_positions"], stats["eval_calls"], stats["eval_seconds"] = (
        evaluator.positions, evaluator.calls, evaluator.seconds)
    return stats


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "kaggle_sup" / "best.pt"))
    ap.add_argument("--config", default="kaggle", help="configs/<nom>.toml, sections [mcts] et [selfplay]")
    ap.add_argument("--book", default=str(ROOT / "books" / "lichess_2000.tsv.gz"), help="'' = position initiale seulement")
    ap.add_argument("--syzygy", default=str(ROOT / "tools" / "syzygy"), help="tables (WDL suffit) ; '' = sans")
    ap.add_argument("--out", required=True, help="dossier des shards (= tampon de cette session)")
    ap.add_argument("--inputs", nargs="*", default=[], help="dossiers de shards des sessions précédentes")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1, help="threads torch par processus")
    ap.add_argument("--max-games", type=int, default=None, help="parties commencées par processus (essai)")
    ap.add_argument("--max-minutes", type=float, default=None, help="budget de temps de la session")
    ap.add_argument("--games-per-worker", type=int, default=None)
    ap.add_argument("--simulations", type=int, default=None)
    ap.add_argument("--stop-new-games-minutes", type=float, default=None,
                    help="marge avant la fin : plus de nouvelles parties (défaut : [selfplay], 20)")
    ap.add_argument("--hard-stop-minutes", type=float, default=None,
                    help="marge avant la fin : arrêt, parties inachevées perdues (défaut : [selfplay], 5)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--session", default=None, help="identifiant (défaut : date et heure UTC)")
    ap.add_argument("--generation", type=int, default=0, help="numéro de génération du réseau (étape 7)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from chesszero.selfplay import SelfPlayConfig, list_shards, sync_buffer

    from dataclasses import replace

    cfg = SelfPlayConfig.named(args.config)
    overrides = {k: v for k, v in (("games_per_worker", args.games_per_worker), ("simulations", args.simulations),
                                   ("stop_new_games_minutes", args.stop_new_games_minutes),
                                   ("hard_stop_minutes", args.hard_stop_minutes)) if v is not None}
    cfg = replace(cfg, **overrides)
    session = args.session or time.strftime("s%Y%m%dT%H%M%S", time.gmtime())
    start = time.time()
    if args.inputs:
        s = sync_buffer(args.inputs, args.out, cfg.buffer_positions)
        print(f"Tampon repris : {s['shards']} shards, {s['positions']} positions ({s['copied']} copiés)", flush=True)
    soft = hard = None
    if args.max_minutes:
        soft = start + 60 * max(args.max_minutes - cfg.stop_new_games_minutes, 0)
        hard = start + 60 * max(args.max_minutes - cfg.hard_stop_minutes, 0)
    tasks = [{"worker": w, "threads": args.threads, "config": args.config, "overrides": overrides,
              "checkpoint": args.checkpoint, "book": args.book or None, "syzygy": args.syzygy or None,
              "out": args.out, "device": args.device, "session": session, "generation": args.generation,
              "seed": args.seed, "max_games": args.max_games, "soft_deadline": soft, "hard_deadline": hard}
             for w in range(args.workers)]
    print(f"Session {session} : {args.workers} processus x {args.threads} thread(s) -> {args.out}", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(worker, tasks)
    s = sync_buffer([], args.out, cfg.buffer_positions)
    seconds = time.time() - start
    games = sum(r["games"] for r in results)
    positions = sum(r["positions"] for r in results)
    sims = sum(r["simulations"] for r in results)
    from chesszero.selfplay import TERMINATIONS

    terms = np.sum([r["terminations"] for r in results], axis=0)
    print(f"\n{games} parties, {positions} positions enregistrées, {sims} simulations en {seconds / 60:.1f} min "
          f"({sims / seconds:.0f} simulations/s, {games / seconds * 3600:.0f} parties/h) ; "
          f"inachevées abandonnées : {sum(r['unfinished_dropped'] for r in results)}")
    print("Fins : " + ", ".join(f"{t} {n}" for t, n in zip(TERMINATIONS, terms) if n))
    calls = sum(r["eval_calls"] for r in results)
    print(f"Réseau : {sum(r['eval_positions'] for r in results) / max(calls, 1):.0f} positions par appel, "
          f"{sum(r['eval_seconds'] for r in results) / max(sum(r['seconds'] for r in results), 1e-9):.0%} du temps")
    print(f"Tampon : {s['shards']} shards, {s['positions']} positions ({s['removed']} shards sortis de la fenêtre) ; "
          f"{len(list_shards([args.out]))} shards dans {args.out}")

    # statistiques de la session (parties de CETTE session seulement), affichées et gardées dans sessions/
    from chesszero.selfplay import describe_repetitions, load_shard, repetition_stats

    own = [load_shard(Path(args.out) / name) for r in results for name in r["shards"]]
    reps = repetition_stats(own)
    print(describe_repetitions(reps))
    controls = sum(int(d["control"].sum()) for d in own)
    triggered = [(d, g, c) for d in own for g in np.nonzero(d["control"])[0] for c in (0, 1)
                 if int(d["would_resign"][g][c]) >= 0]
    false = sum(int(d["result_white"][g]) != (-1 if c == 0 else 1) for d, g, c in triggered)
    print(f"Parties de contrôle : {controls} ; abandons qui se seraient déclenchés : {len(triggered)} ; faux : {false}")
    work_seconds = max(r["seconds"] for r in results)  # temps de jeu des processus (hors chargement)
    stats = {"session": session, "generation": args.generation, "checkpoint": Path(args.checkpoint).name,
             "workers": args.workers, "games_per_worker": cfg.games_per_worker, "simulations_per_move": cfg.simulations,
             "minutes": seconds / 60, "play_minutes": work_seconds / 60, "games": games, "positions": positions,
             "simulations": sims, "simulations_per_second": sims / max(work_seconds, 1e-9),
             "games_per_hour": games / max(work_seconds, 1e-9) * 3600,
             "positions_per_eval_call": sum(r["eval_positions"] for r in results) / max(calls, 1),
             "eval_time_share": sum(r["eval_seconds"] for r in results) / max(sum(r["seconds"] for r in results), 1e-9),
             "plies": [int(x) for d in own for x in d["plies"]],  # longueur des parties (depuis la position initiale)
             "selfplay_plies": [int(x) - int(y) for d in own for x, y in zip(d["plies"], d["start_plies"])],
             "terminations": {t: int(n) for t, n in zip(TERMINATIONS, terms)},
             "unfinished_dropped": sum(r["unfinished_dropped"] for r in results),
             "repetitions": reps, "control_games": controls, "resign_triggered_in_control": len(triggered),
             "false_resignations": false, "shards": [n for r in results for n in r["shards"]]}
    out = Path(args.out) / "sessions"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{session}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
