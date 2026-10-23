"""A/B de la cible de valeur (résultat seul contre (résultat + Q racine) / 2), d'une traite, sur le PC (une nuit).

  ab_value_target.py --buffer data/kaggle_out/selfplay_session1/selfplay --out data/ab_value

Étapes (une étape déjà faite est sautée : relancer = reprendre ; un tour coupé reprend au pas près) :
  1. étiquetage : `eval_value_target.py label` = positions des parties de self-play MISES DE CÔTÉ (jamais
     entraînées) + validation Lichess, notées par Stockfish plein (`--positions` par source, `--movetime` s) ;
  2. un tour `result` puis un tour `blend` depuis kaggle_sup, sur les mêmes données et les mêmes lots
     (`train_selfplay.py --value-target ... --export-trained` : le réseau du tour est gardé même si les
     garde-fous le rejettent) ;
  3. `eval_value_target.py compare` -> verdict (adopter `blend` seulement si les deux mesures vont dans le même sens
     et que l'écart contre Stockfish est significatif).
Règle de mesure : sur secteur, mode « meilleures performances » (avertissement sinon). Journal : <out>/ab_log.txt.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--buffer", required=True, help="tampon de self-play (dossier selfplay/ d'une sortie Kaggle)")
    ap.add_argument("--out", default=str(ROOT / "data" / "ab_value"))
    ap.add_argument("--human", default=str(ROOT / "data" / "lichess_24m"))
    ap.add_argument("--init", default=str(ROOT / "checkpoints" / "kaggle_sup" / "best.pt"))
    ap.add_argument("--positions", type=int, default=20000, help="positions étiquetées par source")
    ap.add_argument("--movetime", type=float, default=0.2, help="secondes de Stockfish par position")
    ap.add_argument("--sf-workers", type=int, default=8, help="processus Stockfish (1 thread chacun)")
    ap.add_argument("--threads", type=int, default=14, help="threads torch pour l'entraînement")
    ap.add_argument("--mask-workers", type=int, default=3, help="processus des masques Lichess")
    ap.add_argument("--steps", type=int, default=None, help="essai réduit : pas imposés par tour")
    ap.add_argument("--batch-size", type=int, default=None, help="essai réduit : taille des lots")
    ap.add_argument("--lichess-val", type=int, default=None, help="essai réduit : positions de validation Lichess")
    args = ap.parse_args()
    trial = [x for opt, v in (("--steps", args.steps), ("--batch-size", args.batch_size),
                              ("--lichess-val", args.lichess_val)) if v is not None for x in (opt, v)]

    sys.path.insert(0, str(ROOT / "scripts"))
    from match_stockfish import power_status, power_warning

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "ab_log.txt", "a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    power = power_status()
    say(f"Alimentation : {power}")
    if power_warning(power):
        say(power_warning(power))

    py = sys.executable
    steps = [
        ("étiquetage Stockfish", out / "eval_value.npz",
         [py, ROOT / "scripts" / "eval_value_target.py", "label", "--buffer", args.buffer, "--human", args.human,
          "--out", out / "eval_value.npz", "--positions", args.positions, "--movetime", args.movetime,
          "--workers", args.sf_workers], {0}),
    ]
    for target in ("result", "blend"):
        steps.append((f"tour « {target} »", out / f"{target}_trained.pt",
                      [py, ROOT / "scripts" / "train_selfplay.py", "--out", out / target, "--buffer", args.buffer,
                       "--human", args.human, "--init", args.init, "--value-target", target, "--device", "cpu",
                       "--threads", args.threads, "--workers", args.mask_workers,
                       "--export-trained", out / f"{target}_trained.pt", *trial], {0, 1}))  # 1 = rejeté : gardé
    start = time.time()
    for name, product, cmd, ok in steps:
        if product.exists():
            say(f"{name} : déjà fait ({product.name})")
            continue
        say(f"{name} : $ {' '.join(str(c) for c in cmd)}")
        t0 = time.time()
        code = subprocess.run([str(c) for c in cmd]).returncode
        if code not in ok or not product.exists():
            say(f"{name} : ÉCHEC (code {code}) ; relancer la même commande pour reprendre")
            sys.exit(1)
        say(f"{name} : terminé en {(time.time() - t0) / 60:.1f} min")
    say("comparaison")
    res = subprocess.run([py, str(ROOT / "scripts" / "eval_value_target.py"), "compare", str(out / "eval_value.npz"),
                          str(out / "result_trained.pt"), str(out / "blend_trained.pt")],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out / "compare.txt").write_text(res.stdout + res.stderr, encoding="utf-8", newline="\n")
    say("\n" + res.stdout + res.stderr)
    say(f"A/B terminé en {(time.time() - start) / 60:.1f} min (ce lancement) ; résultat : {out / 'compare.txt'}")


if __name__ == "__main__":
    main()
