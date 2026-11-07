"""Essai B2 : affiner kaggle_sup avec une valeur entraînée sur des évaluations Stockfish (chesszero/sf_value.py).

  finetune_sf_value.py --labels data/b2/labels_train --val-labels data/b2/labels_val --out data/b2/ft

1. Échelle s de tanh(cp / s) ajustée sur les positions étiquetées d'entraînement terminées normalement (résultat
   valide) ; écrite dans `scale.json` avec l'erreur de la formule de Lichess et de l'espérance WDL de Stockfish.
2. Affinage (relancer = reprendre) : politique sur le coup humain, valeur contre la cible Stockfish ; évaluations
   aux pas 0, 250, 500, 750, 1000 (validation Lichess 50 000 positions = même sous-ensemble que les tours de
   self-play ; contrôle Stockfish = positions de validation étiquetées à part).
3. Verdict « politique intacte » (critère fixé à l'avance, journal du 2026-11-07) : top-1 Lichess >= 53,5 % ET perte
   de politique Lichess <= celle de kaggle_sup + 0,01 ; écrit dans `result.json`. Code de sortie 0 si le critère
   est rempli, 5 sinon (la chaîne ne lance alors pas l'arène).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOP1_MIN = 0.535
POLICY_MARGIN = 0.01


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    from chesszero.sf_value import FinetuneConfig

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--init", default=str(ROOT / "checkpoints" / "kaggle_sup" / "best.pt"))
    ap.add_argument("--labels", required=True)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--human", default=str(ROOT / "data" / "lichess_24m"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--workers", type=int, default=4, help="processus des masques de coups légaux")
    ap.add_argument("--stop-after", type=int, default=None, help="coupure simulée après N pas")
    for f in fields(FinetuneConfig):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = ap.parse_args()

    import torch

    from chesszero.data import load_split
    from chesszero.selfplay_train import human_val_set
    from chesszero.sf_value import fit_scale, finetune, load_labels

    torch.set_num_threads(args.threads)
    cfg = FinetuneConfig(**{f.name: getattr(args, f.name) for f in fields(FinetuneConfig)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels, val_labels = load_labels(args.labels), load_labels(args.val_labels)
    valid = labels["value_valid"] == 1
    fit = fit_scale(labels["cp"][valid], labels["mate"][valid], labels["results"][valid])
    wdl = labels["wdl"][valid & (labels["mate"] == 0)].astype(np.float64)
    res = labels["results"][valid & (labels["mate"] == 0)].astype(np.float64)
    fit["mse_stockfish_wdl"] = float(((wdl - res) ** 2).mean())
    fit["mates"] = int((labels["mate"] != 0).sum())
    fit["depth_median"] = float(np.median(labels["depth"]))
    (out / "scale.json").write_text(json.dumps(fit, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(labels['boards']):,} positions étiquetées ({fit['mates']} mats), contrôle {len(val_labels['boards']):,} ; "
          f"échelle ajustée s = {fit['scale']:.0f} cp (erreur contre le résultat {fit['mse']:.4f} ; formule de Lichess "
          f"s = 543 : {fit['mse_lichess_formula']:.4f} ; espérance WDL de Stockfish : {fit['mse_stockfish_wdl']:.4f} ; "
          f"toujours 0 : {fit['mse_zero']:.4f})", flush=True)

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(args.workers) as pool:
        human_val = human_val_set(load_split(args.human, "val"), cfg.lichess_val, 0, pool)
    last = finetune(args.init, labels, val_labels, human_val, out, cfg, fit["scale"], workers=args.workers,
                    stop_after=args.stop_after)
    if last.get("interrupted"):
        print(f"coupure simulée au pas {last['step']}")
        return 3
    metrics = [json.loads(l) for l in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    base = next(m for m in metrics if m["step"] == 0)["lichess"]
    final = last["lichess"]
    ok_top1 = final["top1"] >= TOP1_MIN
    ok_pol = final["policy_loss"] <= base["policy_loss"] + POLICY_MARGIN
    result = {"criterion": f"top-1 Lichess >= {TOP1_MIN:.1%} et perte de politique <= kaggle_sup + {POLICY_MARGIN}",
              "policy_intact": bool(ok_top1 and ok_pol), "top1_ok": bool(ok_top1), "policy_loss_ok": bool(ok_pol),
              "kaggle_sup": {"lichess": base, "value_sf_val": metrics[0]["value_sf_val"]},
              "final": {"lichess": final, "value_sf_val": last["value_sf_val"]}, "scale": fit,
              "config": asdict(cfg), "checkpoint": str(out / "final.pt")}
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nPOLITIQUE {'INTACTE' if result['policy_intact'] else 'ABÎMÉE : ÉCHEC'} : top-1 {final['top1']:.2%} "
          f"(kaggle_sup {base['top1']:.2%}, seuil {TOP1_MIN:.1%}) ; perte de politique {final['policy_loss']:.4f} "
          f"(kaggle_sup {base['policy_loss']:.4f}, seuil +{POLICY_MARGIN}) ; valeur contre Stockfish "
          f"{metrics[0]['value_sf_val']['mse_vs_stockfish']:.4f} -> {last['value_sf_val']['mse_vs_stockfish']:.4f}")
    return 0 if result["policy_intact"] else 5


if __name__ == "__main__":
    sys.exit(main())
