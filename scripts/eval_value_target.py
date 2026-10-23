"""Évaluer la cible de valeur mélangée (résultat + Q racine) / 2 contre le résultat seul (étape 6d).

Principe (validé par l'auteur) : deux entraînements depuis le même réseau, sur les mêmes données et les mêmes lots
(train_selfplay.py --value-target result, puis blend, dans deux dossiers), puis comparaison de leurs valeurs sur un jeu
FIXE de positions, avec deux mesures :
  1. erreur quadratique contre l'évaluation de Stockfish plein (score attendu = gain − perte d'après son estimation
     WDL) : référence neutre, peu bruitée, sur des positions de self-play mises de côté ET de la validation Lichess ;
  2. erreur quadratique contre le RÉSULTAT RÉEL des parties de self-play mises de côté (jamais entraînées).
Intervalle à 95 % par bootstrap apparié en rééchantillonnant des PARTIES (les positions d'une même partie sont
corrélées). Verdict : adopter le mélange seulement si les deux mesures vont dans le même sens (mélange meilleur) et
que l'écart contre Stockfish est significatif ; sinon garder le résultat seul.

  eval_value_target.py label --buffer /kaggle/working/selfplay --human data/lichess_24m --out eval_value.npz
  eval_value_target.py compare eval_value.npz <résultat>/generations/gen_0001.pt <mélange>/generations/gen_0001.pt
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STOCKFISH = ROOT / "tools" / "stockfish" / "stockfish" / "stockfish-windows-x86-64-universal.exe"

_ENGINE = {}


def _init(stockfish: str) -> None:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    engine.configure({"Threads": 1, "Hash": 64, "UCI_ShowWDL": True})  # sans cette option, pas d'estimation WDL
    _ENGINE["e"] = engine


def _label(args) -> float:
    """Score attendu (camp au trait, dans [-1, 1]) d'après l'estimation gain / nulle / perte de Stockfish."""
    packed, movetime = args
    from chesszero.encoding import board_from_packed

    board = board_from_packed(packed)  # position vue du camp au trait : même évaluation que l'originale
    info = _ENGINE["e"].analyse(board, chess.engine.Limit(time=movetime))
    wdl = info["wdl"].pov(board.turn)
    return 2.0 * wdl.expectation() - 1.0


def label(args) -> None:
    from chesszero.data import load_split
    from chesszero.selfplay import buffer_window, list_shards
    from chesszero.selfplay_train import SelfPlayData

    rng = np.random.default_rng(args.seed)
    shards = buffer_window(list_shards(args.buffer), args.buffer_positions)
    sp = SelfPlayData(sorted(shards, key=lambda s: s["name"]), args.heldout_fraction)
    rows = sp.heldout_rows
    if len(rows) > args.positions:
        rows = np.sort(rng.choice(rows, args.positions, replace=False))
    val = load_split(args.human, "val")
    valid = np.nonzero(val["value_valid"] == 1)[0]
    hrows = np.sort(rng.choice(valid, min(args.positions, len(valid)), replace=False))
    boards = np.concatenate([sp.boards[rows], val["boards"][hrows]])
    outcome = np.concatenate([sp.result[rows], val["results"][hrows].astype(np.float32)])
    source = np.array(["selfplay"] * len(rows) + ["lichess"] * len(hrows))
    game = np.concatenate([sp.game_uid[rows].astype(str), np.array([f"lichess-{g}" for g in val["games"][hrows]])])
    print(f"{len(rows)} positions de self-play (parties mises de côté, {len(set(game[:len(rows)]))} parties) + "
          f"{len(hrows)} Lichess ; Stockfish {args.movetime * 1000:.0f} ms par position, {args.workers} processus", flush=True)
    start = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers, _init, (args.stockfish,)) as pool:
        sf = np.array(pool.map(_label, [(b, args.movetime) for b in boards], chunksize=16), dtype=np.float32)
    np.savez_compressed(args.out, boards=boards, stockfish=sf, outcome=outcome, source=source, game=game)
    meta = {"buffer": [str(b) for b in args.buffer], "human": args.human, "movetime": args.movetime,
            "heldout_fraction": args.heldout_fraction, "selfplay_positions": int(len(rows)),
            "lichess_positions": int(len(hrows)), "seconds": round(time.perf_counter() - start)}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"-> {args.out} ({meta['seconds']} s)")


def values(checkpoint: str, boards: np.ndarray) -> np.ndarray:
    import torch

    from chesszero.checkpoint import load_for_inference
    from chesszero.encoding import unpack_batch

    model, _ = load_for_inference(checkpoint, "cpu")
    out = []
    with torch.inference_mode():
        for i in range(0, len(boards), 512):
            out.append(model(torch.from_numpy(unpack_batch(boards[i:i + 512])))[1].numpy())
    return np.concatenate(out)


def bootstrap_diff(err_a: np.ndarray, err_b: np.ndarray, game: np.ndarray, reps: int, seed: int) -> tuple:
    """Différence moyenne (B − A) des erreurs et intervalle à 95 % par bootstrap apparié sur les parties."""
    keys, inv = np.unique(game, return_inverse=True)
    sum_d = np.bincount(inv, weights=err_b - err_a, minlength=len(keys))
    count = np.bincount(inv, minlength=len(keys))
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(keys), size=(reps, len(keys)))
    boot = sum_d[draws].sum(axis=1) / count[draws].sum(axis=1)
    return float((err_b - err_a).mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def compare(args) -> None:
    z = np.load(args.eval_set, allow_pickle=False)
    boards, sf, outcome, source, game = z["boards"], z["stockfish"], z["outcome"], z["source"], z["game"]
    va, vb = values(args.a, boards), values(args.b, boards)
    print(f"A = {args.a}\nB = {args.b}\n{len(boards)} positions ({(source == 'selfplay').sum()} self-play mises de "
          f"côté, {(source == 'lichess').sum()} Lichess)\n")
    results = {}
    tests = [("contre Stockfish, toutes positions", np.ones(len(sf), bool), sf),
             ("contre Stockfish, self-play", source == "selfplay", sf),
             ("contre Stockfish, Lichess", source == "lichess", sf),
             ("contre le résultat réel, self-play mis de côté", source == "selfplay", outcome)]
    for name, sel, target in tests:
        if not sel.any():
            continue
        ea, eb = (va[sel] - target[sel]) ** 2, (vb[sel] - target[sel]) ** 2
        d, lo, hi = bootstrap_diff(ea, eb, game[sel], args.reps, args.seed)
        results[name] = (d, lo, hi)
        print(f"{name:<48} A {ea.mean():.4f}  B {eb.mean():.4f}  B−A {d:+.4f} (95 % : {lo:+.4f} à {hi:+.4f})")
    sf_all = results["contre Stockfish, toutes positions"]
    real = results.get("contre le résultat réel, self-play mis de côté")
    same_direction = real is not None and np.sign(sf_all[0]) == np.sign(real[0])
    if sf_all[2] < 0 and same_direction and real[0] < 0:
        verdict = "B meilleur selon les deux mesures, écart contre Stockfish significatif : adopter B"
    elif sf_all[1] > 0 and same_direction and real[0] > 0:
        verdict = "A meilleur selon les deux mesures, écart contre Stockfish significatif : garder A"
    elif not same_direction:
        verdict = "les deux mesures se contredisent : garder la cible actuelle (résultat seul)"
    else:
        verdict = "écart non significatif contre Stockfish : garder la cible actuelle (résultat seul)"
    print(f"\nVerdict : {verdict}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    lab = sub.add_parser("label", help="construire le jeu d'évaluation (positions + Stockfish)")
    lab.add_argument("--buffer", nargs="+", required=True)
    lab.add_argument("--human", required=True)
    lab.add_argument("--out", required=True)
    lab.add_argument("--positions", type=int, default=3000, help="par source (self-play, Lichess)")
    lab.add_argument("--movetime", type=float, default=0.2, help="secondes de Stockfish par position")
    lab.add_argument("--workers", type=int, default=8)
    lab.add_argument("--stockfish", default=str(DEFAULT_STOCKFISH))
    lab.add_argument("--buffer-positions", type=int, default=500_000)
    lab.add_argument("--heldout-fraction", type=float, default=0.10, help="doit être celle de l'entraînement")
    lab.add_argument("--seed", type=int, default=0)
    cmp_ = sub.add_parser("compare", help="comparer deux réseaux (A = résultat, B = mélange)")
    cmp_.add_argument("eval_set")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--reps", type=int, default=2000)
    cmp_.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    label(args) if args.command == "label" else compare(args)


if __name__ == "__main__":
    main()
