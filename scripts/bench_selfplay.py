"""Débit du self-play : plusieurs processus, chacun avec beaucoup de parties dont les feuilles partent
ensemble au réseau (`search_batched`). Sert à chiffrer le self-play sur Kaggle avant de le lancer (6e).

  bench_selfplay.py --stub                       # coût CPU seul (faux réseau) : vitesse des cœurs
  bench_selfplay.py --processes 3 --games 64     # vrai réseau sur le device (GPU sur Kaggle)

Chaque processus joue `--games` parties en parallèle pendant `--moves` coups chacune (bruit de Dirichlet et
température 1 comme en self-play ; une partie finie repart de la position initiale + 8 coups au hasard).
Résultat : simulations/s au total, positions par appel au réseau, part du temps dans l'évaluation, et
parties par heure estimées pour plusieurs budgets de simulations par coup (dont la « playout cap
randomization » : une part des coups en recherche complète, le reste en recherche rapide).
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import replace

import chess
import numpy as np


class StubEvaluator:
    """Faux réseau : le même travail CPU que NetEvaluator côté Python (encodage compact), sans calcul."""

    def __init__(self):
        self.calls = self.positions = 0
        self.seconds = 0.0

    def evaluate(self, boards, legal_indices):
        from chesszero.encoding import pack_board

        start = time.perf_counter()
        np.stack([pack_board(b) for b in boards])
        out = [np.full(len(i), 1.0 / len(i)) for i in legal_indices], np.zeros(len(boards))
        self.calls += 1
        self.positions += len(boards)
        self.seconds += time.perf_counter() - start
        return out


def new_game(rng) -> chess.Board:
    board = chess.Board()
    for _ in range(8):
        board.push(list(board.legal_moves)[int(rng.integers(board.legal_moves.count()))])
    return board


def worker(args_dict: dict) -> dict:
    import torch

    from chesszero.mcts import MCTS, MCTSConfig, search_batched

    a = argparse.Namespace(**args_dict)
    torch.set_num_threads(1)
    rng = np.random.default_rng(a.seed)
    if a.stub:
        evaluator = StubEvaluator()
    else:
        from chesszero.checkpoint import load_for_inference
        from chesszero.device import get_device
        from chesszero.mcts import NetEvaluator

        model, _ = load_for_inference(a.checkpoint, get_device(a.device))
        evaluator = NetEvaluator(model)
    cfg = replace(MCTSConfig.named("kaggle"), batch_size=a.leaves, dirichlet_epsilon=0.25, temperature=1.0)
    trees = [MCTS(evaluator, cfg, new_game(rng), rng=rng) for _ in range(a.games)]
    search_batched(trees, evaluator, 16, a.leaves)  # échauffement (CUDA, allocations)
    evaluator.calls = evaluator.positions = 0
    evaluator.seconds = 0.0
    start, sims, finished = time.perf_counter(), 0, 0
    for _ in range(a.moves):
        search_batched(trees, evaluator, a.simulations, a.leaves)
        sims += a.simulations * len(trees)
        for i, tree in enumerate(trees):
            tree.advance(tree.best_move(1.0))
            if tree.root.terminal is not None or len(tree.board.move_stack) >= 300:
                trees[i] = MCTS(evaluator, cfg, new_game(rng), rng=rng)
                finished += 1
    seconds = time.perf_counter() - start
    return {"sims": sims, "seconds": seconds, "calls": evaluator.calls, "positions": evaluator.positions,
            "eval_seconds": evaluator.seconds, "finished": finished}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--processes", type=int, default=3)
    ap.add_argument("--games", type=int, default=64, help="parties en parallèle par processus")
    ap.add_argument("--leaves", type=int, default=4, help="feuilles par arbre et par appel au réseau")
    ap.add_argument("--simulations", type=int, default=200)
    ap.add_argument("--moves", type=int, default=10, help="coups joués par partie pendant la mesure")
    ap.add_argument("--plies", type=int, default=100, help="longueur supposée d'une partie (estimation)")
    ap.add_argument("--stub", action="store_true", help="faux réseau : coût CPU seul")
    ap.add_argument("--checkpoint", default="checkpoints/kaggle_sup/best.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    print(f"{args.processes} processus x {args.games} parties, {args.leaves} feuilles par arbre (lots d'environ "
          f"{args.games * args.leaves} positions), {args.simulations} simulations par coup, {args.moves} coups, "
          f"{'faux réseau' if args.stub else 'réseau ' + args.checkpoint} ; {os.cpu_count()} CPU logiques", flush=True)
    tasks = [{**vars(args), "seed": i} for i in range(args.processes)]
    wall = time.perf_counter()
    with mp.get_context("spawn").Pool(args.processes) as pool:
        results = pool.map(worker, tasks)
    wall = time.perf_counter() - wall
    sims = sum(r["sims"] for r in results)
    sps = sum(r["sims"] / r["seconds"] for r in results)  # débit pendant la mesure (hors démarrage)
    calls = sum(r["calls"] for r in results)
    share = sum(r["eval_seconds"] for r in results) / sum(r["seconds"] for r in results)
    print(f"\n{sims} simulations ; {sps:.0f} simulations/s au total ({sps / args.processes:.0f} par processus) ; "
          f"{sum(r['positions'] for r in results) / calls:.0f} positions par appel ; évaluation {share:.0%} du temps ; "
          f"durée totale {wall:.0f} s (démarrage compris)")
    print(f"\nParties par heure estimées ({args.plies} demi-coups par partie) :")
    budgets = [("200 simulations à chaque coup", 200, 1.0),
               ("PCR p = 0,5 : 240 complètes / 80 rapides", 0.5 * 240 + 0.5 * 80, 0.5),
               ("PCR p = 0,25 : 320 complètes / 64 rapides", 0.25 * 320 + 0.75 * 64, 0.25)]
    for name, per_move, share_full in budgets:
        games = sps * 3600 / (args.plies * per_move)
        print(f"  {name:<44} {per_move:>5.0f} sim/coup -> {games:>6.0f} parties/h, "
              f"{games * args.plies * share_full:>7.0f} cibles de politique/h")


if __name__ == "__main__":
    main()
