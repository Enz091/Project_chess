"""Match : MCTS (N simulations par coup) contre la politique seule (meilleur coup du réseau, sans recherche).

  match.py                                  # réseau kaggle, 200 simulations, 10 ouvertures x 2 couleurs
  match.py --simulations 100 --openings 3   # essai rapide (6 parties)

Les deux joueurs sont déterministes : depuis la même position, ils rejoueraient toujours la même
partie. Chaque partie part donc d'une ouverture fixe (4 à 6 demi-coups), jouée deux fois en échangeant
les couleurs. Fin de partie : règles officielles (mat, pat, matériel insuffisant, 50 coups, triple
répétition), ou nulle au-delà de --max-plies. Le MCTS réutilise son arbre d'un coup à l'autre.
Les parties sont écrites en PGN (--pgn).
"""

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import chess
import chess.pgn

from chesszero.checkpoint import load_for_inference
from chesszero.device import describe_device, get_device
from chesszero.mcts import MCTS, MCTSConfig, NetEvaluator, policy_best_move, terminal_value

OPENINGS = [
    ("Italienne", "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5"),
    ("Sicilienne", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4"),
    ("Française", "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6"),
    ("Caro-Kann", "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5"),
    ("Gambit dame refusé", "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6"),
    ("Est-indienne", "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7"),
    ("Anglaise", "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5"),
    ("Espagnole", "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6"),
    ("Scandinave", "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5"),
    ("London", "d2d4 d7d5 c1f4 g8f6 e2e3 c7c5"),
]


def termination(board: chess.Board, plies: int, max_plies: int) -> str:
    if plies >= max_plies and terminal_value(board) is None:
        return "limite de demi-coups"
    if board.is_checkmate():
        return "mat"
    if board.is_stalemate():
        return "pat"
    if board.is_insufficient_material():
        return "matériel insuffisant"
    if board.halfmove_clock >= 100:
        return "50 coups"
    return "triple répétition"


def play_game(evaluator, cfg, sims, opening, mcts_white, max_plies):
    board = chess.Board()
    for uci in opening.split():
        board.push_uci(uci)
    mcts = MCTS(evaluator, cfg, board)
    mcts_color = chess.WHITE if mcts_white else chess.BLACK
    plies, mcts_seconds, mcts_moves = len(board.move_stack), 0.0, 0
    while terminal_value(board) is None and plies < max_plies:
        if board.turn == mcts_color:
            start = time.perf_counter()
            mcts.search(sims)
            move = mcts.best_move()
            mcts_seconds += time.perf_counter() - start
            mcts_moves += 1
        else:
            move = policy_best_move(evaluator, board)
        mcts.advance(move)
        board.push(move)
        plies += 1
    value = terminal_value(board)
    if value == -1.0:  # le camp au trait est maté
        mcts_score = 0.0 if board.turn == mcts_color else 1.0
    else:
        mcts_score = 0.5
    return board, mcts_score, termination(board, plies, max_plies), mcts_seconds / max(mcts_moves, 1)


def elo_from_score(score: float) -> float:
    return -400 * math.log10(1 / score - 1)


def elo_summary(scores: list[float]) -> str:
    n, s = len(scores), sum(scores) / len(scores)
    if s in (0.0, 1.0):
        # aucune nulle ni défaite (ou l'inverse) : borne à 95 % exacte (binomiale) sur la probabilité de gain
        bound = elo_from_score(0.05 ** (1 / n))  # p >= 0,05^(1/n) si n gains sur n
        return f"{'≥' if s == 1 else '≤'} {bound if s == 1 else -bound:+.0f} (borne à 95 %)"
    sd = math.sqrt(sum((x - s) ** 2 for x in scores) / (n - 1)) / math.sqrt(n)
    lo, hi = max(s - 1.96 * sd, 1e-3), min(s + 1.96 * sd, 1 - 1e-3)
    return f"{elo_from_score(s):+.0f} (intervalle à 95 % : {elo_from_score(lo):+.0f} à {elo_from_score(hi):+.0f})"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/kaggle_sup/best.pt")
    ap.add_argument("--config", default="kaggle", help="configs/<nom>.toml, section [mcts]")
    ap.add_argument("--simulations", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--openings", type=int, default=len(OPENINGS), help="nombre d'ouvertures (x 2 parties)")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--pgn", default=None, help="défaut : docs/match_mcts<N>_vs_policy.pgn ; \"\" = pas de PGN")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = MCTSConfig.named(args.config)
    if args.batch_size:
        cfg = replace(cfg, batch_size=args.batch_size)
    sims = args.simulations or cfg.num_simulations
    if args.pgn is None:
        args.pgn = f"docs/match_mcts{sims}_vs_policy.pgn"
    device = get_device(args.device)
    model, meta = load_for_inference(args.checkpoint, device)
    evaluator = NetEvaluator(model)
    mcts_name, policy_name = f"MCTS-{sims}", "Politique seule"
    print(f"{args.checkpoint} (pas {meta['step']}) | {describe_device(device)} | {mcts_name} (lots de "
          f"{cfg.batch_size}, c_puct {cfg.c_puct}) contre {policy_name}\n")

    scores, games = [], []
    start = time.perf_counter()
    for name, opening in OPENINGS[: args.openings]:
        for mcts_white in (True, False):
            t = time.perf_counter()
            board, score, how, per_move = play_game(evaluator, cfg, sims, opening, mcts_white, args.max_plies)
            scores.append(score)
            game = chess.pgn.Game.from_board(board)
            game.headers.update({
                "Event": f"{mcts_name} contre {policy_name}", "Round": str(len(scores)), "Opening": name,
                "White": mcts_name if mcts_white else policy_name, "Black": policy_name if mcts_white else mcts_name,
                "Result": board.result(claim_draw=True) if how != "limite de demi-coups" else "1/2-1/2",
                "Termination": how, "Date": "????.??.??", "Site": "-",
            })
            games.append(game)
            print(f"{len(scores):>2}. {name:<19} MCTS {'Blancs' if mcts_white else 'Noirs '} | "
                  f"{game.headers['Result']:<7} | {how:<20} | {len(board.move_stack):>3} demi-coups | "
                  f"{per_move:.2f} s/coup MCTS | {time.perf_counter() - t:.0f} s", flush=True)

    wins, draws = scores.count(1.0), scores.count(0.5)
    losses = len(scores) - wins - draws
    print(f"\nMCTS : {wins} victoires, {draws} nulles, {losses} défaites ; score {sum(scores):g}/{len(scores)} "
          f"({sum(scores) / len(scores):.1%}) en {(time.perf_counter() - start) / 60:.1f} min")
    print(f"Écart Elo estimé (MCTS − politique) : {elo_summary(scores)}")

    if args.pgn:
        Path(args.pgn).parent.mkdir(parents=True, exist_ok=True)
        with open(args.pgn, "w", encoding="utf-8") as f:
            for game in games:
                print(game, file=f, end="\n\n")
        print(f"Parties : {args.pgn}")


if __name__ == "__main__":
    main()
