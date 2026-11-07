"""Essai B2 : étiquettes Stockfish à NOMBRE DE NŒUDS FIXE sur des positions Lichess (chesszero/sf_value.py).

  label_stockfish.py --human data/lichess_24m --split train --positions 1000000 --out data/b2/labels_train
  label_stockfish.py --human data/lichess_24m --split val --positions 20000 --out data/b2/labels_val

Positions : tirées sans remise (graine fixe) dans la partie demandée, hors positions des lignes d'ouverture du
protocole v2 et de l'arène ancre (`sf_value.opening_keys`). Stockfish 19, 1 thread, hachage 16 Mo, `ucinewgame`
avant chaque position (le résultat ne dépend ni de l'ordre ni du processus : vérifié identique en relançant) ;
nœuds fixes : les étiquettes ne dépendent pas de la vitesse du PC. Écrit des morceaux `chunk_XXXX.npz` de
`--chunk` positions (boards, moves, results, value_valid, games, cp, mate, depth, wdl) puis `meta.json` ; relancer
= reprendre (morceaux déjà écrits gardés, mêmes positions vérifiées par leur empreinte).
"""

from __future__ import annotations

import argparse
import hashlib
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
_E: dict = {}


def _init(stockfish: str, nodes: int) -> None:
    from chesszero import cpu_guard

    cpu_guard.disable_ecoqos()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    engine.configure({"Threads": 1, "Hash": 16, "UCI_ShowWDL": True})
    try:
        cpu_guard.disable_ecoqos(engine.transport.get_pid())
    except Exception:  # noqa: BLE001
        pass
    _E.update(engine=engine, nodes=nodes)


def _label(packed: np.ndarray) -> tuple[int, int, int, float]:
    """(cp, mat, profondeur, espérance WDL) du camp au trait. mat : +1 si le camp au trait mate, −1 s'il est maté
    (cp vaut alors 0)."""
    from chesszero.encoding import board_from_packed

    board = board_from_packed(packed)  # vue du camp au trait (Blancs au trait) : même évaluation que l'originale
    info = _E["engine"].analyse(board, chess.engine.Limit(nodes=_E["nodes"]), game=object())  # ucinewgame à chaque fois
    score = info["score"].pov(board.turn)
    wdl = info["wdl"].pov(board.turn).expectation() * 2 - 1 if "wdl" in info else float("nan")
    if score.is_mate():
        return 0, (1 if score.mate() > 0 else -1), int(info.get("depth", 0)), float(wdl)
    return int(score.score()), 0, int(info.get("depth", 0)), float(wdl)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--human", default=str(ROOT / "data" / "lichess_24m"))
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--positions", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nodes", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--stockfish", default=str(DEFAULT_STOCKFISH))
    args = ap.parse_args()

    from chesszero.data import load_split
    from chesszero.sf_value import opening_keys, sample_rows

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    data = load_split(args.human, args.split)
    excluded = opening_keys()
    rows, dropped = sample_rows(data["boards"], args.positions, args.seed, excluded)
    digest = hashlib.sha256(rows.tobytes()).hexdigest()
    meta = {"human": str(args.human), "split": args.split, "positions": int(len(rows)), "seed": args.seed,
            "nodes": args.nodes, "stockfish": Path(args.stockfish).name, "threads": 1, "hash_mb": 16,
            "excluded_opening_positions": len(excluded), "dropped_excluded": dropped, "rows_sha256": digest,
            "chunk": args.chunk}
    plan = out / "plan.json"
    if plan.exists():
        old = json.loads(plan.read_text(encoding="utf-8"))
        if old["rows_sha256"] != digest or old["nodes"] != args.nodes or old["chunk"] != args.chunk:
            raise SystemExit(f"{out} contient des étiquettes d'un autre tirage ou d'autres réglages : {old}")
    else:
        plan.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rows):,} positions ({args.split}) tirées en {time.perf_counter() - t0:.0f} s ; {dropped} tirages "
          f"écartés (positions des {len(excluded)} positions d'ouverture protégées) ; Stockfish {args.nodes:,} nœuds, "
          f"{args.workers} processus", flush=True)
    chunks = [rows[i:i + args.chunk] for i in range(0, len(rows), args.chunk)]
    todo = [k for k in range(len(chunks)) if not (out / f"chunk_{k:04d}.npz").exists()]
    start, done = time.perf_counter(), 0
    with mp.get_context("spawn").Pool(args.workers, _init, (args.stockfish, args.nodes)) as pool:
        for k in todo:
            r = chunks[k]
            boards = data["boards"][r]
            res = pool.map(_label, list(boards), chunksize=16)
            cp, mate, depth, wdl = (np.array(x) for x in zip(*res))
            tmp = out / f"chunk_{k:04d}.tmp.npz"
            np.savez(tmp, rows=r, boards=boards, moves=data["moves"][r], results=data["results"][r],
                     value_valid=data["value_valid"][r], games=data["games"][r], cp=cp.astype(np.int32),
                     mate=mate.astype(np.int8), depth=depth.astype(np.int16), wdl=wdl.astype(np.float32))
            tmp.replace(out / f"chunk_{k:04d}.npz")
            done += len(r)
            el = time.perf_counter() - start
            left = sum(len(chunks[j]) for j in todo) - done
            print(f"  morceau {k + 1}/{len(chunks)} : {done:,} positions en {el / 60:.1f} min ({done / el:.0f}/s, reste "
                  f"~{left / (done / el) / 60:.0f} min) ; profondeur médiane {np.median(depth):.0f}, mats {int((mate != 0).sum())}",
                  flush=True)
    meta["seconds"] = round(time.perf_counter() - t0)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"-> {out} ({meta['seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
