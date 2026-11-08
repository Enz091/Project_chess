"""Critère 2 de B1 (chesszero/b1_criteria.py) : politique du réseau B1 sur les 50 000 positions Lichess de B2.

  eval_b1_policy.py --checkpoint checkpoints/b1_12x160/last.pt --out checkpoints/b1_12x160/criterion2.json

Verdict sur les positions NON PROTÉGÉES (décision de l'auteur du 2026-11-08, prise avant le lancement) : top-1
>= 54,29 % ET perte de politique <= 1,3961. Chiffres sur les 50 000 positions complètes (seuils d'origine 54,2 % /
1,3949) et sur les positions protégées donnés pour information. `kaggle_sup` est évalué aussi, comme contrôle
(attendu : 53,79 % / 1,4061 sur les positions non protégées). Le réseau jugé est celui de la fin de l'entraînement
(pas 92 000, `last.pt`) : un autre pas est évalué mais signalé « hors critère ».
Code de sortie : 0 critère rempli, 5 non rempli, 4 réseau hors critère (pas différent de la fin de l'entraînement).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def line(name: str, part: str, m: dict | None) -> str:
    if m is None:
        return f"  {name:<12} {part:<15} —"
    return (f"  {name:<12} {part:<15} {m['positions']:>7,} positions : top-1 {m['top1']:.2%}, perte de politique "
            f"{m['policy_loss']:.4f}, perte de valeur {m['value_loss']:.4f}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "b1_12x160" / "last.pt"))
    ap.add_argument("--reference", default=str(ROOT / "checkpoints" / "kaggle_sup" / "best.pt"),
                    help="contrôle (kaggle_sup) ; « none » pour ne pas l'évaluer")
    ap.add_argument("--human", default=str(ROOT / "data" / "lichess_24m"))
    ap.add_argument("--positions", type=int, default=None, help="taille du sous-ensemble (défaut : 50 000, le critère)")
    ap.add_argument("--out", default=None, help="fichier JSON du verdict (défaut : criterion2.json à côté du réseau)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--workers", type=int, default=4, help="processus des masques de coups légaux")
    args = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor

    import torch

    from chesszero import b1_criteria as c
    from chesszero.checkpoint import load_for_inference
    from chesszero.data import load_split
    from chesszero.device import get_device
    from chesszero.model import ModelConfig
    from chesszero.selfplay_train import human_val_set
    from chesszero.train import TrainConfig

    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    n = c.LICHESS_VAL if args.positions is None else args.positions
    with ProcessPoolExecutor(args.workers) as pool:
        val = human_val_set(load_split(args.human, "val"), n, c.LICHESS_SEED, pool)
    mask = c.protected_mask(val["boards"])
    print(f"validation Lichess : {len(mask):,} positions (graine {c.LICHESS_SEED}), dont {int(mask.sum()):,} protégées "
          f"({mask.mean():.1%}) et {int((~mask).sum()):,} non protégées", flush=True)
    if n != c.LICHESS_VAL:
        print(f"ATTENTION : {n:,} positions au lieu de {c.LICHESS_VAL:,} : essai, hors critère")

    results = {}
    for name, path in (("kaggle_sup", args.reference), ("B1", args.checkpoint)):
        if name == "kaggle_sup" and str(path).lower() == "none":
            continue
        model, meta = load_for_inference(path, device)
        with torch.no_grad():
            m = c.evaluate(model, val, mask, device)
        m.update(checkpoint=str(path), step=meta["step"], model=meta["config"]["model"])
        results[name] = m
        for part, label in (("unprotected", "non protégées"), ("all", "toutes"), ("protected", "protégées")):
            print(line(name, label, m[part]))
        del model

    b1 = results["B1"]
    total = TrainConfig.named("kaggle_l").total_steps
    in_criterion = (b1["step"] == total and n == c.LICHESS_VAL
                    and b1["model"] == ModelConfig.named("kaggle_l").to_dict())
    verdict = c.judge(b1)
    verdict.update(step=b1["step"], total_steps=total, in_criterion=in_criterion, positions=n,
                   protected_positions=int(mask.sum()), results=results)
    out = Path(args.out) if args.out else Path(args.checkpoint).with_name("criterion2.json")
    out.write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8", newline="\n")
    u, a = b1["unprotected"], b1["all"]
    info = verdict["info_all_positions"]
    status = "REMPLI" if verdict["passed"] else "NON REMPLI : ÉCHEC, pas d'arène"
    print(f"\nCRITÈRE 2 {status} (positions non protégées) : "
          f"top-1 {u['top1']:.2%} (seuil {c.TOP1_MIN:.2%}) ; perte de politique {u['policy_loss']:.4f} "
          f"(seuil {c.POLICY_LOSS_MAX})")
    print(f"pour information, sur les {a['positions']:,} positions : top-1 {a['top1']:.2%} "
          f"({'≥' if info['top1_ok'] else '<'} {c.INFO_TOP1_MIN_ALL:.1%}) ; perte de politique {a['policy_loss']:.4f} "
          f"({'≤' if info['policy_loss_ok'] else '>'} {c.INFO_POLICY_LOSS_MAX_ALL}) : sans effet sur le verdict")
    if not in_criterion:
        why = [w for w, bad in ((f"réseau au pas {b1['step']:,} au lieu de {total:,}", b1["step"] != total),
                                ("réseau autre que 12 × 160 (kaggle_l)", b1["model"] != ModelConfig.named("kaggle_l").to_dict()),
                                (f"{n:,} positions au lieu de {c.LICHESS_VAL:,}", n != c.LICHESS_VAL)) if bad]
        print(f"HORS CRITÈRE ({' ; '.join(why)}) : verdict pour information seulement")
        return 4
    print(f"verdict écrit dans {out}")
    return 0 if verdict["passed"] else 5


if __name__ == "__main__":
    sys.exit(main())
