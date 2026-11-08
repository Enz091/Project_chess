"""Pré-entraînement supervisé. Relancer la même commande reprend automatiquement depuis last.pt.

Exemples :
  train.py --config tiny --data data/lichess_2000 --out checkpoints/tiny_sup
  train.py --config tiny --steps 200 --out checkpoints/essai            # essai rapide
  train.py --config kaggle --data /kaggle/input/<dataset> --out /kaggle/working/kaggle_sup
  train.py --config kaggle --data ... --out ... --max-minutes 10        # essai de débit de 10 minutes
  train.py --config kaggle --data ... --out ... --no-mask-loss          # perte politique sans masque
  train.py --config kaggle_l --data ... --out ... --gpus auto           # B1 : essai 1 GPU / 2 GPU puis entraînement
  train.py --config kaggle_l --data ... --out ... --bench               # essai de débit seul (rien d'entraîné)

Codes de sortie : 0 terminé ou arrêté proprement (budget de temps, coupure simulée) ; 6 garde-fou de débit
déclenché (`min_pos_per_s` de la config, last.pt sauvegardé).
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from chesszero.device import get_device
from chesszero.model import ModelConfig
from chesszero.train import TrainConfig, bench_throughput, choose_gpus, train

ROOT = Path(__file__).resolve().parent.parent
EXIT_LOW_THROUGHPUT = 6


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="tiny", help="nom d'un fichier de configs/ (tiny, kaggle, kaggle_l)")
    ap.add_argument("--data", default=str(ROOT / "data" / "lichess_2000"))
    ap.add_argument("--out", required=True, help="dossier des checkpoints, de metrics.csv")
    ap.add_argument("--device", default=None, help="force cpu / cuda / xpu")
    ap.add_argument("--steps", type=int, help="remplace total_steps")
    ap.add_argument("--workers", type=int, help="remplace num_workers")
    ap.add_argument("--no-mask-loss", action="store_true", help="remplace mask_train_loss par false")
    ap.add_argument("--max-minutes", type=float, help="arrêt propre (last.pt sauvegardé) après ce temps d'entraînement")
    ap.add_argument("--stop-after", type=int, help="simule une coupure à ce pas (test de reprise)")
    ap.add_argument("--gpus", default="1", help="1, 2 (DataParallel) ou auto (essai de débit puis choix)")
    ap.add_argument("--bench", action="store_true", help="essai de débit 1 GPU / 2 GPU seulement, écrit <out>/bench.json")
    ap.add_argument("--bench-steps", type=int, default=300)
    ap.add_argument("--bench-shards", type=int, default=4)
    ap.add_argument("--min-pos-per-s", type=float, help="remplace min_pos_per_s (0 = sans garde-fou de débit)")
    args = ap.parse_args()

    cfg = TrainConfig.named(args.config)
    overrides = {}
    if args.steps:
        overrides["total_steps"] = args.steps
    if args.workers is not None:
        overrides["num_workers"] = args.workers
    if args.no_mask_loss:
        overrides["mask_train_loss"] = False
    if args.min_pos_per_s is not None:
        overrides["min_pos_per_s"] = args.min_pos_per_s
    cfg = replace(cfg, **overrides)
    model_cfg, device = ModelConfig.named(args.config), get_device(args.device)
    out = Path(args.out)
    if args.bench or args.gpus == "auto":
        out.mkdir(parents=True, exist_ok=True)
        bench = bench_throughput(model_cfg, cfg, args.data, device, steps=args.bench_steps, shards=args.bench_shards)
        gpus, why = choose_gpus(bench)
        print(f"Choix : {why}", flush=True)
        (out / "bench.json").write_text(json.dumps({"config": args.config, "results": bench, "choice": gpus,
                                                    "why": why}, indent=1) + "\n", encoding="utf-8")
        if args.bench:
            return 0
    else:
        gpus = int(args.gpus)
    result = train(model_cfg, cfg, args.data, out, device, stop_after=args.stop_after,
                   max_minutes=args.max_minutes, gpus=gpus)
    return EXIT_LOW_THROUGHPUT if result.get("stopped_low_throughput") else 0


if __name__ == "__main__":
    sys.exit(main())
