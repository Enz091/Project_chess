"""Extrait des positions d'entraînement d'une base Lichess (PGN, .pgn.zst, chemin ou URL).

Exemples :
  prepare_data.py --out data/essai --max-games 1000
  prepare_data.py --out data/lichess_2000 --target-positions 3000000
  prepare_data.py --source data/mon_fichier.pgn.zst --out data/local
"""

import argparse
import os
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

from chesszero.data import FilterConfig, extract

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = "https://database.lichess.org/standard/lichess_db_standard_rated_2026-09.pgn.zst"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="chemin ou URL (.pgn ou .pgn.zst)")
    ap.add_argument("--out", required=True, help="dossier de sortie (ex. data/lichess_2000)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.toml"))
    ap.add_argument("--target-positions", type=int, help="arrêt quand train+val atteint ce nombre")
    ap.add_argument("--max-games", type=int, help="arrêt après ce nombre de parties LUES")
    ap.add_argument("--min-elo", type=int)
    ap.add_argument("--min-plies", type=int)
    ap.add_argument("--val-fraction", type=float)
    ap.add_argument("--shard-size", type=int)
    ap.add_argument("--workers", type=int, default=max(1, min(14, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--overwrite", action="store_true", help="écraser un dossier déjà rempli")
    args = ap.parse_args()

    cfg = FilterConfig.from_toml(args.config)
    overrides = {k: getattr(args, k) for k in ("min_elo", "min_plies", "val_fraction") if getattr(args, k) is not None}
    cfg = replace(cfg, **overrides)
    with open(args.config, "rb") as f:
        shard_size = args.shard_size or tomllib.load(f).get("output", {}).get("shard_size", 250_000)

    m = extract(
        args.source,
        args.out,
        cfg,
        target_positions=args.target_positions,
        max_games=args.max_games,
        shard_size=shard_size,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    p = m["positions"]
    print(f"\nParties lues         : {m['games_read']:,}")
    print(f"Rejetées (en-têtes)  : {m['games_rejected_by_header']}")
    print(f"Rejetées (après lecture) : {m['games_rejected_after_parse']}")
    print(f"Parties gardées      : {m['games_kept']:,}")
    print(f"Positions            : {p['train'] + p['val']:,} (train {p['train']:,}, val {p['val']:,})")
    print(f"Flux lu (compressé)  : {m['bytes_read_compressed'] / 1e6:,.1f} Mo")
    print(f"Taille sur disque    : {m['bytes_on_disk'] / 1e6:,.1f} Mo")
    print(f"Durée                : {m['seconds']:.0f} s")


if __name__ == "__main__":
    main()
