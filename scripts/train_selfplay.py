"""Un tour d'entraînement sur le self-play (voir chesszero/selfplay_train.py) : relancer = reprendre le tour en cours.

  train_selfplay.py --out checkpoints/sp_train --buffer /kaggle/working/selfplay \\
                    --human /kaggle/input/lichess-24m --init checkpoints/kaggle_sup/best.pt
  # essai local : tampon minuscule, petits lots, nombre de pas imposé, coupure simulée
  train_selfplay.py --out data/sp_train_essai --buffer data/selfplay_essai --human data/lichess_2000 \\
                    --batch-size 64 --steps 200 --force --lichess-val 2000 --stop-after 100

  # verdict de l'arène pour le candidat en attente (résumés = result.json de scripts/arena.py)
  train_selfplay.py --out ... --commit-arena 7 --gate loop/arena/cand_0007_gate/result.json                     [--anchor loop/arena/cand_0007_anchor/result.json]
  train_selfplay.py --out ... --status          # état (source de vérité : train_state.pt)
  train_selfplay.py --out ... --clear-stop "raison"   # reprise après un arrêt, sur décision humaine

Sorties dans --out : train_state.pt (reprise et SOURCE DE VÉRITÉ), line_ok.pt (dernier état qui a passé les
garde-fous), candidate.pt (candidat en attente de l'arène), generations/gen_XXXX.pt (acceptés en arène),
rounds.jsonl (tours et verdicts d'arène), train_log.jsonl (pertes pendant les tours).
Code de sortie d'un tour : 0 candidat prêt ou en attente de données, 1 rejeté par les garde-fous, 2 arrêt,
3 interrompu (--stop-after). Verdict : 0 accepté, 1 rejeté, 2 arrêt.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--buffer", nargs="+", required=True, help="dossiers de shards de self-play")
    ap.add_argument("--human", required=True, help="dossier de données Lichess (prepare_data.py)")
    ap.add_argument("--init", default=str(ROOT / "checkpoints" / "kaggle_sup" / "best.pt"),
                    help="réseau de départ (premier lancement seulement)")
    ap.add_argument("--config", default="kaggle")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None, help="threads torch (CPU)")
    ap.add_argument("--steps", type=int, default=None, help="essai : nombre de pas imposé pour un nouveau tour")
    ap.add_argument("--force", action="store_true", help="essai : ignorer la taille minimale du tampon")
    ap.add_argument("--stop-after", type=int, default=None, help="s'arrêter après N pas (coupure simulée)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--value-target", choices=["result", "blend"], default=None)
    ap.add_argument("--lichess-val", type=int, default=None, help="positions de validation Lichess")
    ap.add_argument("--workers", type=int, default=None, help="processus pour les masques Lichess (0 = aucun)")
    ap.add_argument("--commit-arena", type=int, default=None, metavar="CANDIDAT",
                    help="applique le verdict de l'arène au candidat (avec --gate et, après la génération 1, --anchor)")
    ap.add_argument("--gate", default=None, help="result.json de l'arène contre le réseau accepté")
    ap.add_argument("--anchor", default=None, help="result.json de l'arène contre l'ancre kaggle_sup")
    ap.add_argument("--clear-stop", default=None, metavar="NOTE", help="reprise après un arrêt (décision humaine)")
    ap.add_argument("--status", action="store_true", help="affiche l'état et s'arrête")
    args = ap.parse_args()

    import torch

    from chesszero.data import load_split
    from chesszero.device import describe_device, get_device
    from chesszero.selfplay_train import SelfPlayTrainConfig, SelfPlayTrainer, human_val_set

    cfg = SelfPlayTrainConfig.named(args.config)
    overrides = {k: v for k, v in (("batch_size", args.batch_size), ("value_target", args.value_target),
                                   ("lichess_val_positions", args.lichess_val), ("num_workers", args.workers))
                 if v is not None}
    cfg = replace(cfg, **overrides)
    if args.status:
        from chesszero.selfplay_train import read_train_state

        print(json.dumps(read_train_state(args.out), ensure_ascii=False, indent=1, default=str))
        return
    if args.commit_arena is not None or args.clear_stop is not None:
        from chesszero.arena import ArenaConfig

        trainer = SelfPlayTrainer(args.out, cfg, "cpu", args.init)
        if args.clear_stop is not None:
            print(trainer.clear_stop(args.clear_stop))
            return
        load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))["summary"] if p else None
        if not args.gate:
            sys.exit("--commit-arena demande --gate")
        c = trainer.commit_arena(args.commit_arena, load(args.gate), load(args.anchor), ArenaConfig.named(args.config))
        print(f"Candidat {args.commit_arena} : {'ACCEPTÉ' if c['accepted'] else 'REJETÉ'} (génération acceptée : "
              f"{c['generation_after']}, rejets en arène d'affilée : {c['arena_rejections']})")
        for r in c["reasons"] + c["alerts"]:
            print(f"  {r}")
        sys.exit(2 if c["loop_status"] == "stop" else (0 if c["accepted"] else 1))
    if args.threads:
        torch.set_num_threads(args.threads)
    device = get_device(args.device)
    print(f"{describe_device(device)} ; réglages : {overrides or 'ceux de ' + args.config}", flush=True)

    human_train = load_split(args.human, "train")
    pool = ProcessPoolExecutor(cfg.num_workers) if cfg.num_workers > 0 else None
    try:
        human_val = human_val_set(load_split(args.human, "val"), cfg.lichess_val_positions, cfg.seed, pool)
        print(f"Lichess : {len(human_train['boards'])} positions d'entraînement, {len(human_val['boards'])} de "
              f"validation (sous-ensemble fixe)", flush=True)
        trainer = SelfPlayTrainer(args.out, cfg, device, args.init)
        result = trainer.run_round(args.buffer, human_train, human_val, pool, steps_override=args.steps,
                                   stop_after=args.stop_after, force=args.force)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)

    status = result["status"]
    if status == "waiting":
        print(f"Pas de tour : tampon de {result['buffer_train_positions']} positions d'entraînement "
              f"(minimum {result['min_buffer_positions']}), {result['new_positions']} nouvelles.")
        sys.exit(0)
    if status == "interrupted":
        print(f"Interrompu au pas {result['done']}/{result['steps']} du tour {result['round']} : relancer pour reprendre.")
        sys.exit(3)
    m = result["metrics"]
    print(f"\nTour {result['round']} : {status.upper()} (générations acceptées : {result['generation_after']})")
    print(f"  Lichess : politique {m['lichess']['policy_loss']:.4f} (départ {result['baseline']['lichess']['policy_loss']:.4f}), "
          f"top-1 {m['lichess']['top1']:.1%} (départ {result['baseline']['lichess']['top1']:.1%}), "
          f"valeur {m['lichess']['value_loss']:.4f} (départ {result['baseline']['lichess']['value_loss']:.4f})")
    h = m["heldout"]
    if h["positions"]:
        print(f"  Parties mises de côté ({h['positions']} positions) : politique {h['policy_loss']:.4f}, "
              f"erreur de valeur {h['value_mse']:.4f}")
    for r in result["reasons"]:
        print(f"  REJET : {r}")
    for a in result["alerts"]:
        print(f"  ALERTE : {a}")
    if status == "candidate":
        print(f"  Candidat {result['candidate']} : {Path(args.out) / 'candidate.pt'} (verdict : --commit-arena)")
    sys.exit(0 if status == "candidate" else (2 if result["loop_status"] == "stop" else 1))


if __name__ == "__main__":
    main()
