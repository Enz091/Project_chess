"""Match entre deux budgets de recherche du même réseau : MCTS-A contre MCTS-B (simulations par coup).

  match_sims.py --sims-a 64 --sims-b 200 --pgn docs/match_mcts64_vs_mcts200.pgn

Sert à chiffrer l'écart de force entre une recherche rapide et une recherche complète (« playout cap
randomization » du self-play). Les 40 ouvertures du protocole v2, chacune avec les deux couleurs (80 parties) ;
les deux camps : même réseau, arbre réutilisé, température 0, sans bruit, tables Syzygy si présentes ;
nulle au-delà de 400 demi-coups. Les deux joueurs sont déterministes : une ouverture + une couleur = une partie.
4 processus × 2 threads torch (8 threads au plus).
"""

import argparse
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder_openings import OPENINGS_V2  # noqa: E402
from match import termination  # noqa: E402

_W: dict = {}


def init_worker(checkpoint: str, syzygy: str | None, threads: int) -> None:
    import torch

    from chesszero.checkpoint import load_for_inference
    from chesszero.mcts import MCTSConfig, NetEvaluator
    from chesszero.syzygy import Tablebase

    torch.set_num_threads(threads)
    model, _ = load_for_inference(checkpoint, "cpu")
    _W.update(evaluator=NetEvaluator(model), cfg=MCTSConfig.named("kaggle"),
              tb=Tablebase.open(syzygy) if syzygy else None)


def play(task: dict) -> dict:
    from chesszero.mcts import MCTS, terminal_value

    name, opening = OPENINGS_V2[task["opening"]]
    board = chess.Board()
    for uci in opening.split():
        board.push_uci(uci)
    a_color = chess.WHITE if task["a_white"] else chess.BLACK
    trees = {c: MCTS(_W["evaluator"], replace(_W["cfg"], num_simulations=task["sims"][c == a_color]), board,
                     tablebase=_W["tb"]) for c in (chess.WHITE, chess.BLACK)}
    start = time.perf_counter()
    while terminal_value(board) is None and len(board.move_stack) < task["max_plies"]:
        tree = trees[board.turn]
        tree.search()
        move = tree.best_move(0)
        for t in trees.values():
            t.advance(move)
        board.push(move)
    value = terminal_value(board)
    score_a = 0.5 if value != -1.0 else (0.0 if board.turn == a_color else 1.0)
    return {**task, "name": name, "score_a": score_a, "plies": len(board.move_stack),
            "termination": termination(board, len(board.move_stack), task["max_plies"]),
            "moves": " ".join(m.uci() for m in board.move_stack), "seconds": time.perf_counter() - start}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sims-a", type=int, default=64)
    ap.add_argument("--sims-b", type=int, default=200)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "kaggle_sup" / "best.pt"))
    ap.add_argument("--syzygy", default=str(ROOT / "tools" / "syzygy"), help="'' pour jouer sans tables")
    ap.add_argument("--openings", type=int, default=len(OPENINGS_V2))
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pgn", default=None)
    args = ap.parse_args()
    if args.workers * args.threads > 8:
        sys.exit("8 threads au plus")

    from chesszero.rating import fit_elo

    sims = {True: args.sims_a, False: args.sims_b}  # clé : « est-ce le camp A ? »
    tasks = [{"opening": o, "a_white": w, "sims": sims, "max_plies": args.max_plies}
             for o in range(args.openings) for w in (True, False)]
    print(f"MCTS-{args.sims_a} (A) contre MCTS-{args.sims_b} (B) : {len(tasks)} parties, "
          f"tables {'oui' if args.syzygy else 'non'}", flush=True)
    games, start = [], time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers, init_worker, (args.checkpoint, args.syzygy or None, args.threads)) as pool:
        for g in pool.imap_unordered(play, tasks):
            games.append(g)
            res = {1.0: "A gagne", 0.5: "nulle", 0.0: "B gagne"}[g["score_a"]]
            print(f"{len(games):>3}/{len(tasks)} {g['name']:<28} A {'Blancs' if g['a_white'] else 'Noirs '} | "
                  f"{res:<7} | {g['termination']:<20} | {g['plies']:>3} demi-coups | {g['seconds']:.0f} s", flush=True)
    score = sum(g["score_a"] for g in games)
    wins = sum(g["score_a"] == 1.0 for g in games)
    draws = sum(g["score_a"] == 0.5 for g in games)
    # Elo de A avec B placé à 2000 (fit_elo cherche dans [0, 4500] : un adversaire à 0 tronquerait l'écart à 0)
    fit = fit_elo([(2000.0, g["score_a"]) for g in games])
    print(f"\nMCTS-{args.sims_a} contre MCTS-{args.sims_b} : {score:.1f} / {len(games)} ({score / len(games):.1%}) ; "
          f"A : {wins} gains, {draws} nulles, {len(games) - wins - draws} défaites")
    print(f"Écart d'Elo de A par rapport à B : {fit.rating - 2000:+.0f} (± {1.96 * fit.se:.0f} à 95 %)" if fit.se
          else f"Écart d'Elo de A par rapport à B : borne seulement ({fit.describe()})")
    print(f"Durée : {(time.perf_counter() - start) / 60:.1f} min")

    if args.pgn:
        with open(args.pgn, "w", encoding="utf-8") as f:
            for i, g in enumerate(sorted(games, key=lambda g: (g["opening"], not g["a_white"]))):
                pgn = chess.pgn.Game()
                a, b = f"ChessZero MCTS-{args.sims_a}", f"ChessZero MCTS-{args.sims_b}"
                pgn.headers.update({"Event": f"MCTS-{args.sims_a} contre MCTS-{args.sims_b}", "Site": "-",
                                    "Date": "????.??.??", "Round": str(i + 1), "Opening": g["name"],
                                    "White": a if g["a_white"] else b, "Black": b if g["a_white"] else a,
                                    "Result": {1.0: "1-0", 0.0: "0-1", 0.5: "1/2-1/2"}[
                                        g["score_a"] if g["a_white"] else 1 - g["score_a"]],
                                    "Termination": g["termination"]})
                node = pgn
                for uci in g["moves"].split():
                    node = node.add_variation(chess.Move.from_uci(uci))
                print(pgn, file=f, end="\n\n")


if __name__ == "__main__":
    main()
