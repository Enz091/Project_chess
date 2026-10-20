"""Montre ce que la recherche MCTS change par rapport à la politique seule, sur une position.

  show_search.py checkpoints/kaggle_sup/best.pt --fen "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
  show_search.py checkpoints/kaggle_sup/best.pt --moves "e2e4 e7e5" --simulations 800 --batch-size 8

Pour chaque coup : prior du réseau, visites, Q (valeur moyenne pour le camp au trait), puis la
variante principale (suite des coups les plus visités). Paramètres MCTS : section [mcts] de --config.
"""

import argparse
import sys
import time
from dataclasses import replace

import chess

from chesszero.checkpoint import load_for_inference
from chesszero.device import get_device
from chesszero.mcts import MCTS, MCTSConfig, NetEvaluator, policy_best_move


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    ap.add_argument("--moves", default="", help="coups UCI joués depuis --fen, séparés par des espaces")
    ap.add_argument("--config", default="kaggle", help="configs/<nom>.toml, section [mcts]")
    ap.add_argument("--simulations", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = MCTSConfig.named(args.config)
    if args.batch_size:
        cfg = replace(cfg, batch_size=args.batch_size)
    sims = args.simulations or cfg.num_simulations

    model, meta = load_for_inference(args.checkpoint, get_device(args.device))
    evaluator = NetEvaluator(model)
    board = chess.Board(args.fen)
    for uci in args.moves.split():
        board.push_uci(uci)
    if board.is_game_over(claim_draw=True):
        sys.exit(f"Position terminée ({board.result(claim_draw=True)}) : rien à chercher.")

    mcts = MCTS(evaluator, cfg, board)
    start = time.perf_counter()
    mcts.search(sims)
    seconds = time.perf_counter() - start

    print(board)
    print(f"\n{'Blancs' if board.turn == chess.WHITE else 'Noirs'} au trait | checkpoint : pas {meta['step']} | "
          f"{sims} simulations, lots de {cfg.batch_size} | {seconds:.2f} s ({sims / seconds:.0f} simulations/s)")
    print(f"Politique seule : {board.san(policy_best_move(evaluator, board))} | "
          f"valeur réseau {mcts.root.value:+.3f} -> après recherche {mcts.root.q():+.3f}")
    print(f"\n{'rang':>4}  {'coup':<7} {'prior':>7} {'visites':>8} {'part':>6} {'Q':>7}")
    for rank, s in enumerate(mcts.root_stats()[: args.top], 1):
        q = f"{s['q']:+.3f}" if s["q"] is not None else "-"
        print(f"{rank:>4}  {s['san']:<7} {s['prior']:>7.1%} {s['visits']:>8} {s['visits'] / sims:>6.1%} {q:>7}")
    print(f"\nCoup choisi : {board.san(mcts.best_move())}")
    print(f"Variante principale : {board.variation_san(mcts.principal_variation())}")
    st = mcts.stats
    print(f"Feuilles évaluées : {st['evaluated']}, terminales : {st['terminal']}, collisions : {st['collisions']}")


if __name__ == "__main__":
    main()
