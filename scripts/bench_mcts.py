"""Vitesse du MCTS (simulations par seconde) selon la taille de lot.

  bench_mcts.py                                   # réseau kaggle, lots 1..64, 800 simulations
  bench_mcts.py --batch-sizes 1 8 16 --simulations 400 --repeats 3

Pour chaque position et taille de lot : nouvelle recherche depuis un arbre vide (meilleur de
`--repeats` essais). Colonnes : simulations/s, positions par appel au réseau, part du temps passée
dans l'évaluation (encodage + réseau + softmax ; le reste = Python et python-chess : descente,
coups légaux, remontée), collisions, et si le coup choisi est le même qu'avec des lots de 1.
"""

import argparse
import sys
import time
from dataclasses import replace

import chess
import torch

from chesszero.checkpoint import load_for_inference
from chesszero.device import describe_device, get_device
from chesszero.mcts import MCTS, MCTSConfig, NetEvaluator

POSITIONS = {
    "initiale": chess.STARTING_FEN,
    "milieu de jeu": "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8",
}


def run(evaluator: NetEvaluator, cfg: MCTSConfig, fen: str, sims: int) -> dict:
    evaluator.calls = evaluator.positions = 0
    evaluator.seconds = 0.0
    mcts = MCTS(evaluator, cfg, chess.Board(fen))
    start = time.perf_counter()
    mcts.search(sims)
    seconds = time.perf_counter() - start
    return {"sps": sims / seconds, "per_call": evaluator.positions / evaluator.calls,
            "eval_share": evaluator.seconds / seconds, "collisions": mcts.stats["collisions"],
            "move": mcts.board.san(mcts.best_move())}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/kaggle_sup/best.pt")
    ap.add_argument("--config", default="kaggle")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--simulations", type=int, default=800)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    model, meta = load_for_inference(args.checkpoint, device)
    evaluator = NetEvaluator(model)
    base = MCTSConfig.named(args.config)
    print(f"{args.checkpoint} (pas {meta['step']}) | {describe_device(device)} | torch {torch.__version__} | "
          f"{args.simulations} simulations, meilleur de {args.repeats}")
    run(evaluator, replace(base, batch_size=16), chess.STARTING_FEN, 64)  # échauffement

    for name, fen in POSITIONS.items():
        print(f"\n### {name}\n")
        print("| lot | simulations/s | positions/appel | part évaluation | collisions | coup | = lot 1 ? |")
        print("|---|---|---|---|---|---|---|")
        ref = None
        for bs in args.batch_sizes:
            cfg = replace(base, batch_size=bs)
            r = max((run(evaluator, cfg, fen, args.simulations) for _ in range(args.repeats)), key=lambda r: r["sps"])
            ref = ref or r["move"]
            print(f"| {bs} | {r['sps']:.0f} | {r['per_call']:.1f} | {r['eval_share']:.0%} | {r['collisions']} | "
                  f"{r['move']} | {'oui' if r['move'] == ref else 'non'} |", flush=True)


if __name__ == "__main__":
    main()
