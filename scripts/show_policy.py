"""Affiche, pour une position, les coups préférés du réseau (probabilités) et sa valeur.

  show_policy.py checkpoints/tiny_sup/best.pt                          # position initiale
  show_policy.py checkpoints/tiny_sup/best.pt --fen "<FEN>" --top 5
  show_policy.py checkpoints/tiny_sup/best.pt --moves "e2e4 e7e5 g1f3"   # position après ces coups

Le réseau est reconstruit à partir de la config enregistrée dans le checkpoint.
"""

import argparse
import sys

import chess
import numpy as np
import torch

from chesszero.checkpoint import load_checkpoint
from chesszero.encoding import encode_board, index_to_move, legal_move_mask
from chesszero.model import ModelConfig, build_model, masked_policy


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    ap.add_argument("--moves", default="", help="coups UCI joués depuis --fen, séparés par des espaces")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ModelConfig(**payload["config"]["model"]))
    meta = load_checkpoint(args.checkpoint, model, restore_rng=False)
    model.eval()

    board = chess.Board(args.fen)
    for uci in args.moves.split():
        board.push_uci(uci)
    if board.is_game_over():
        sys.exit(f"Position terminée ({board.result()}) : rien à jouer.")

    x = torch.from_numpy(encode_board(board))[None]
    mask = torch.from_numpy(legal_move_mask(board))[None]
    with torch.no_grad():
        logits, value = model(x)
    probs = masked_policy(logits, mask)[0].numpy()

    print(board)
    print(f"\n{'Blancs' if board.turn == chess.WHITE else 'Noirs'} au trait | checkpoint : pas {meta['step']} "
          f"| {int(mask.sum())} coups légaux")
    print(f"Valeur : {value.item():+.3f}  (+1 = le camp au trait gagne, −1 = il perd)")
    print(f"\n{'rang':>4}  {'coup':<7} {'UCI':<6} {'proba':>7}")
    for rank, idx in enumerate(np.argsort(-probs)[: args.top], 1):
        move = index_to_move(int(idx), board)
        print(f"{rank:>4}  {board.san(move):<7} {move.uci():<6} {probs[idx]:>7.1%}")
    print(f"{'':>4}  {'(autres)':<14} {1 - np.sort(probs)[::-1][: args.top].sum():>7.1%}")


if __name__ == "__main__":
    main()
