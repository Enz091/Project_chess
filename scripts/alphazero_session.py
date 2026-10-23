"""Session de la boucle AlphaZero (étape 7d) : self-play -> entraînement -> arènes -> verdict, sous budget de temps.
Voir chesszero/loop.py.

  # Kaggle : première session de boucle, à partir de la sortie d'une session de self-play seul (étape 6)
  alphazero_session.py --work /kaggle/working --resume-from /kaggle/input/<sortie> --expect-session 0 \\
      --minutes 220 --init <kaggle_sup best.pt> --human /kaggle/input/lichess-24m --syzygy /tmp/syzygy
  # sessions suivantes : --resume-from <sortie de la session k> --expect-session k
  # essai complet local (réseau tiny, CPU, tout réduit)
  alphazero_session.py --smoke --work data/loop_smoke/s1 --resume-from data/loop_smoke/s0 --expect-session 0 \\
      --minutes 8

Sorties dans --work : selfplay/ (tampon), sp_train/ (entraînement : source de vérité), loop/ (loop_state.json,
arena/, reports/session_XXXX.md, history.csv, progress.png).
Code de sortie : 0 session terminée (budget), 2 arrêt (décisionnel ou erreur : voir le rapport), 4 refus de démarrer.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--work", required=True, help="dossier de travail (= sortie de la session sur Kaggle)")
    ap.add_argument("--resume-from", default=None, help="sortie de la session précédente (sinon : --work lui-même)")
    ap.add_argument("--expect-session", type=int, required=True,
                    help="numéro de la dernière session de boucle dans l'entrée (0 : sortie de self-play seul)")
    ap.add_argument("--minutes", type=float, required=True, help="budget de la session (660 au plus sur Kaggle)")
    ap.add_argument("--init", default=str(ROOT / "checkpoints" / "kaggle_sup" / "best.pt"),
                    help="réseau de départ = l'ancre kaggle_sup")
    ap.add_argument("--human", default=str(ROOT / "data" / "lichess_24m"), help="données Lichess")
    ap.add_argument("--config", default="kaggle")
    ap.add_argument("--syzygy", default=str(ROOT / "tools" / "syzygy"), help="tables WDL ; '' = sans")
    ap.add_argument("--book", default=None, help="livre d'ouvertures (défaut : celui du dépôt)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--games-per-worker", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="essai local : --config smoke, réseau tiny_sup, Lichess lichess_2000, CPU, sans tables")
    ap.add_argument("--expect-init-sha256", default=None,
                    help="SHA-256 attendu du réseau de départ (kaggle_sup) : refus de démarrer s'il diffère")
    ap.add_argument("--check-inputs", action="store_true",
                    help="vérifie l'entrée (type de sortie, session attendue, tampon, ancre, Lichess, tables) puis "
                         "s'arrête SANS RIEN ÉCRIRE ; code 0 = la session démarrerait, 4 = refus")
    ap.add_argument("--crash-at", default=None, help=argparse.SUPPRESS)  # tests : coupure simulée
    args = ap.parse_args()

    if args.smoke:
        args.config = "smoke"
        args.init = str(ROOT / "checkpoints" / "tiny_sup" / "best.pt")
        args.human = str(ROOT / "data" / "lichess_2000")
        args.device = args.device or "cpu"
        args.syzygy = ""
        args.workers = min(args.workers, 2)
        args.threads = max(args.threads, 2)
    if args.minutes > 660 and not args.smoke:
        sys.exit("--minutes : 660 au plus (coupure de Kaggle à 12 h, 1 h de marge)")

    from chesszero.loop import LoopStop, Orchestrator, SubprocessExecutor

    if args.expect_init_sha256:
        from chesszero.kaggle_inputs import sha256_of

        if not Path(args.init).exists() or sha256_of(args.init) != args.expect_init_sha256:
            print(f"\nREFUS DE DÉMARRER : le réseau de départ {args.init} n'a pas le SHA-256 attendu "
                  f"({args.expect_init_sha256[:12]}…)", flush=True)
            sys.exit(4)
    executor = SubprocessExecutor(Path(args.work), args.config, args.init, args.human, args.workers, args.threads,
                                  args.device, args.syzygy or None, args.book, args.games_per_worker)
    orch = Orchestrator(args.work, args.config, args.init, executor, args.minutes, args.expect_session,
                        resume_from=args.resume_from, crash_at=args.crash_at)
    if args.check_inputs:
        try:
            lines = orch.check_inputs(args.human, args.syzygy if args.syzygy is not None else None)
        except LoopStop as e:
            print(f"\nREFUS DE DÉMARRER : {e}", flush=True)
            sys.exit(4)
        print("\nVÉRIFICATION DES ENTRÉES : la session démarrerait.\n" + "\n".join(f"- {x}" for x in lines), flush=True)
        sys.exit(0)
    try:
        result = orch.run()
    except LoopStop as e:
        print(f"\nREFUS DE DÉMARRER : {e}", flush=True)
        sys.exit(4)
    print("\n" + result["text"], flush=True)
    stopped = result["end_reason"].startswith(("arrêt", "erreur", "contrôle"))
    sys.exit(2 if stopped else 0)


if __name__ == "__main__":
    main()
