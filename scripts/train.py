"""Pré-entraînement supervisé. Relancer la même commande reprend automatiquement depuis last.pt.

Exemples :
  train.py --config tiny --data data/lichess_2000 --out checkpoints/tiny_sup
  train.py --config tiny --steps 200 --out checkpoints/essai            # essai rapide
  train.py --config kaggle --data /kaggle/input/<dataset> --out /kaggle/working/kaggle_sup
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from chesszero.device import get_device
from chesszero.model import ModelConfig
from chesszero.train import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="tiny", help="nom d'un fichier de configs/ (tiny, kaggle)")
    ap.add_argument("--data", default=str(ROOT / "data" / "lichess_2000"))
    ap.add_argument("--out", required=True, help="dossier des checkpoints, de metrics.csv")
    ap.add_argument("--device", default=None, help="force cpu / cuda / xpu")
    ap.add_argument("--steps", type=int, help="remplace total_steps")
    ap.add_argument("--workers", type=int, help="remplace num_workers")
    ap.add_argument("--stop-after", type=int, help="simule une coupure à ce pas (test de reprise)")
    args = ap.parse_args()

    cfg = TrainConfig.named(args.config)
    overrides = {}
    if args.steps:
        overrides["total_steps"] = args.steps
    if args.workers is not None:
        overrides["num_workers"] = args.workers
    cfg = replace(cfg, **overrides)
    train(ModelConfig.named(args.config), cfg, args.data, args.out, get_device(args.device), stop_after=args.stop_after)


if __name__ == "__main__":
    main()
