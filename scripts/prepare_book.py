"""Construit le livre d'ouvertures du self-play (voir chesszero/book.py) à partir d'une base Lichess lue en flux.

  prepare_book.py                                     # janvier 2026, ~260 000 positions distinctes avant filtre
  prepare_book.py --months 2026-01 --target 5000 --out books/essai     # essai rapide

Étapes : parties filtrées comme les données supervisées (configs/data.toml), partie d'entraînement seulement ;
coupe à un demi-coup tiré dans [--min-ply, --max-ply] ; une entrée par position distincte ; puis valeur du
réseau (checkpoint de départ du self-play) sur chaque position, et rejet si |valeur| > --max-abs-value ;
enfin, retrait des entrées qui passent par une des 40 ouvertures du protocole de mesure v2 (transpositions
comprises) ou s'arrêtent sur leur ligne. Sortie : books/<nom>.tsv.gz + books/<nom>.json (paramètres, stats).

  prepare_book.py --refilter                          # applique seulement le retrait des ouvertures v2 au
                                                      # livre existant (même résultat, sans tout réextraire)
  prepare_book.py --arena-reserve                     # (re)calcule seulement la réserve d'arène du livre
                                                      # existant : books/<nom>_arena.tsv.gz (~40 s)
La réserve d'arène est aussi recalculée à la fin d'une extraction complète et après --refilter.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import chess
import numpy as np

from chesszero.book import (ARENA_RESERVE_PER_MILLE, OpeningBook, arena_reserve_path, book_entry,
                            compute_arena_reserve)
from chesszero.data import FilterConfig, iter_game_texts, open_pgn, parse_headers, reject_reason, split_of

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder_openings import OPENINGS_V2  # noqa: E402  (ouvertures de la mesure, exclues du livre)
MONTH_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"
CHUNK = 2000


def parse_chunk(args) -> list[tuple[str, str]]:
    items, min_ply, max_ply = args
    out = []
    for key, text in items:
        entry = book_entry(text, key, min_ply, max_ply)
        if entry is not None:
            out.append(entry)
    return out


def network_values(lines: list[str], checkpoint: str, threads: int, batch: int = 1024) -> np.ndarray:
    """Valeur du réseau (camp au trait) de la position au bout de chaque suite de coups."""
    import torch

    from chesszero.checkpoint import load_for_inference
    from chesszero.encoding import pack_board, unpack_batch

    torch.set_num_threads(threads)
    model, _ = load_for_inference(checkpoint, "cpu")
    out = np.empty(len(lines), dtype=np.float32)
    start = time.perf_counter()
    for i in range(0, len(lines), batch):
        packed = []
        for moves in lines[i:i + batch]:
            board = chess.Board()
            for uci in moves.split():
                board.push_uci(uci)
            packed.append(pack_board(board))
        with torch.inference_mode():
            _, values = model(torch.from_numpy(unpack_batch(np.stack(packed))))
        out[i:i + batch] = values.numpy()
        done = min(i + batch, len(lines))
        if (i // batch) % 20 == 0 or done == len(lines):
            rate = done / (time.perf_counter() - start)
            print(f"  valeurs : {done}/{len(lines)} ({rate:.0f} pos/s, reste ~{(len(lines) - done) / rate / 60:.1f} min)",
                  flush=True)
    return out


def save_arena_reserve(out: str) -> None:
    """Réserve d'arène du livre `<out>.tsv.gz` -> `<out>_arena.tsv.gz` (+ .json) : départs des arènes (étape 7),
    jamais tirés par le self-play."""
    import hashlib

    book_path = Path(out + ".tsv.gz")
    book = OpeningBook.load(book_path)
    start = time.perf_counter()
    reserve = compute_arena_reserve(book)
    path = arena_reserve_path(book_path)
    reserve.save(path, {"rule": "crc32('arena:' + EPD) % 1000 < per_mille", "per_mille": ARENA_RESERVE_PER_MILLE,
                        "positions": len(reserve), "book": book_path.name, "book_positions": len(book),
                        "book_sha256": hashlib.sha256(book_path.read_bytes()).hexdigest()})
    print(f"Réserve d'arène : {len(reserve)} positions sur {len(book)} ({len(reserve) / len(book):.2%}) -> {path} "
          f"({time.perf_counter() - start:.0f} s)", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--months", nargs="+", default=["2026-01"], metavar="AAAA-MM")
    ap.add_argument("--out", default=str(ROOT / "books" / "lichess_2000"), help="chemin sans extension")
    ap.add_argument("--target", type=int, default=260_000, help="positions distinctes avant le filtre de valeur")
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--max-ply", type=int, default=20)
    ap.add_argument("--max-abs-value", type=float, default=0.3)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "kaggle_sup" / "best.pt"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.toml"))
    ap.add_argument("--workers", type=int, default=7, help="processus de lecture des coups (8 threads au plus)")
    ap.add_argument("--refilter", action="store_true", help="seulement retirer les ouvertures v2 du livre --out existant")
    ap.add_argument("--arena-reserve", action="store_true", help="seulement (re)calculer la réserve d'arène")
    args = ap.parse_args()

    if args.arena_reserve:
        save_arena_reserve(args.out)
        return

    if args.refilter:
        book = OpeningBook.load(args.out + ".tsv.gz")
        meta_path = Path(args.out + ".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        save_without_measure_openings(book, meta, args.out)
        save_arena_reserve(args.out)
        return

    cfg = FilterConfig.from_toml(args.config)
    entries: dict[str, list] = {}  # EPD -> [coups, nombre de parties]
    stats: Counter = Counter()
    start = time.perf_counter()

    def merge(result):
        for epd, moves in result:
            stats["entries"] += 1
            if epd in entries:
                entries[epd][1] += 1
            else:
                entries[epd] = [moves, 1]

    with mp.get_context("spawn").Pool(args.workers) as pool:
        for month in args.months:
            if len(entries) >= args.target:
                break
            url = MONTH_URL.format(month)
            print(f"Lecture de {url}", flush=True)
            stream, counter = open_pgn(url)
            pending, chunk = [], []
            for game_no, text in enumerate(iter_game_texts(stream)):
                stats["read"] += 1
                headers = parse_headers(text)
                reason = reject_reason(headers, cfg)
                if reason is None and split_of(headers, game_no, cfg.val_fraction) == "val":
                    reason = "val"
                if reason is not None:
                    stats[f"rejet_{reason}"] += 1
                    continue
                stats["kept"] += 1
                chunk.append((headers.get("Site") or f"{month}-{game_no}", text))
                if len(chunk) >= CHUNK:
                    pending.append(pool.apply_async(parse_chunk, ((chunk, args.min_ply, args.max_ply),)))
                    chunk = []
                while pending and (pending[0].ready() or len(pending) > 4 * args.workers):
                    merge(pending.pop(0).get())
                if stats["read"] % 200_000 == 0:
                    print(f"  {stats['read']} parties lues, {stats['kept']} gardées, {len(entries)} positions distinctes, "
                          f"{counter.bytes_read / 1e6:.0f} Mo lus, {time.perf_counter() - start:.0f} s", flush=True)
                if len(entries) >= args.target:
                    break
            if chunk:
                pending.append(pool.apply_async(parse_chunk, ((chunk, args.min_ply, args.max_ply),)))
            for p in pending:
                merge(p.get())
            stream.close()

    lines = [moves for moves, _ in entries.values()]
    counts = np.array([count for _, count in entries.values()], dtype=np.int64)
    print(f"{stats['read']} parties lues, {stats['kept']} gardées, {stats['entries']} coupes valides, "
          f"{len(lines)} positions distinctes ({time.perf_counter() - start:.0f} s)", flush=True)

    values = network_values(lines, args.checkpoint, threads=8)
    keep = np.abs(values) <= args.max_abs_value
    book = OpeningBook([l for l, k in zip(lines, keep) if k], counts[keep], values[keep])
    meta = {
        "sources": [MONTH_URL.format(m) for m in args.months], "filter": asdict(cfg),
        "min_ply": args.min_ply, "max_ply": args.max_ply, "max_abs_value": args.max_abs_value,
        "checkpoint": str(Path(args.checkpoint).name), "games_read": stats["read"], "games_kept": stats["kept"],
        "valid_cuts": stats["entries"], "distinct_positions": len(lines), "after_value_filter": len(book),
        "rejected_by_value": int((~keep).sum()), "rejections": {k: v for k, v in stats.items() if k.startswith("rejet_")},
        "value_quantiles": {q: float(np.quantile(values, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "seconds": round(time.perf_counter() - start),
    }
    print(f"{int((~keep).sum())} positions écartées par la valeur (|v| > {args.max_abs_value})", flush=True)
    save_without_measure_openings(book, meta, args.out)
    save_arena_reserve(args.out)


def save_without_measure_openings(book: OpeningBook, meta: dict, out: str) -> None:
    """Retire les entrées qui touchent les 40 ouvertures du protocole v2, complète les statistiques, enregistre."""
    before = len(book)
    book, removed = book.excluding([tuple(o) for o in OPENINGS_V2])
    plies = Counter(len(l.split()) for l in book.lines)
    first = Counter(l.split()[0] for l in book.lines)
    meta.update({
        "excluded_openings": "stockfish-ladder-v2 (40 ouvertures de scripts/ladder_openings.py)",
        "excluded_through": removed["through"], "excluded_on_path": removed["on_path"],
        "positions_in_book": len(book), "plies": dict(sorted(plies.items())), "first_moves": dict(first.most_common(12)),
    })
    meta.setdefault("after_value_filter", before)
    book.save(out + ".tsv.gz", meta)
    print(f"Livre : {len(book)} positions ({removed['through']} retirées car elles passent par une ouverture v2, "
          f"{removed['on_path']} car elles s'arrêtent sur leur ligne) -> {out}.tsv.gz ; demi-coups "
          f"{dict(sorted(plies.items()))} ; premiers coups {dict(first.most_common(6))}")


if __name__ == "__main__":
    main()
