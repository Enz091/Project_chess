"""Résume le débit d'un entraînement (metrics.csv) : positions/s, utilisation du GPU, attente des données.

Verdict : le GPU « attend » si l'attente des données dépasse 15 % du temps d'un pas ou si l'utilisation
mesurée du GPU est sous 80 %. Dans ce cas, relancer avec `--no-mask-loss` (ou précalculer les masques).

Exemple : summarize_run.py /kaggle/working/bench
"""

import argparse
import statistics
import sys
from pathlib import Path

from chesszero.train import read_metrics

WAIT_LIMIT, GPU_LIMIT = 15.0, 80.0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--skip", type=int, default=2, help="fenêtres de démarrage ignorées (chauffe de cuDNN, files de lots)")
    args = ap.parse_args()

    rows = [r for r in read_metrics(Path(args.run_dir) / "metrics.csv") if r["split"] == "train"]
    if not rows:
        sys.exit("aucune ligne « train » dans metrics.csv")
    skip = args.skip if len(rows) > args.skip else 0  # run trop court : on garde tout
    kept = rows[skip:]

    def mean(key: str) -> float:
        vals = [r[key] for r in kept if r[key] == r[key]]  # sans les nan
        return statistics.fmean(vals) if vals else float("nan")

    pps, gpu, wait, mask = mean("pos_per_s"), mean("gpu_util"), mean("data_wait_pct"), mean("mask_wait_pct")
    print(f"Fenêtres analysées : {len(kept)} (pas {int(kept[0]['step'])} à {int(kept[-1]['step'])}, {skip} de chauffe ignorées)")
    print(f"Débit              : {pps:,.0f} positions/s (min {min(r['pos_per_s'] for r in kept):,.0f}, max {max(r['pos_per_s'] for r in kept):,.0f})")
    print(f"Utilisation GPU    : {'non mesurée' if gpu != gpu else f'{gpu:.0f} %'}")
    print(f"Attente données    : {wait:.1f} % du temps d'un pas, dont {mask:.1f} % pour les masques de coups légaux")
    print(f"Calcul             : {100 - wait:.1f} %")
    print(f"Une époque (23,6 M de positions) : ~{23.56e6 / pps / 60:,.0f} min à ce débit")
    starving = wait > WAIT_LIMIT or (gpu == gpu and gpu < GPU_LIMIT)
    if starving:
        print(f"\nVERDICT : le GPU ATTEND (attente > {WAIT_LIMIT:.0f} % ou GPU < {GPU_LIMIT:.0f} %).")
        if mask > wait / 2:
            print("  La cause principale est le calcul des masques : activer mask_train_loss = false (--no-mask-loss).")
        else:
            print("  Les masques ne sont pas la cause principale (préparation des lots, transferts, petit lot ?).")
    else:
        print("\nVERDICT : le GPU ne manque pas de données ; garder les masques à l'entraînement.")


if __name__ == "__main__":
    main()
