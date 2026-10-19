"""Démonstration de bout en bout : détection du device + checkpoints + reprise.

Entraîne un minuscule réseau sur des données aléatoires (rien à voir avec les échecs) :
    python scripts/smoke_resume.py --stop-after 60   # simule une coupure à l'étape 60
    python scripts/smoke_resume.py                   # reprend automatiquement depuis last.pt

C'est le squelette que suivront les vraies boucles d'entraînement :
détecter le device -> construire modèle/optimiseur -> reprendre si possible -> boucler en sauvegardant.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import nn

from chesszero.checkpoint import find_resume_checkpoint, load_checkpoint, save_checkpoint
from chesszero.device import describe_device, get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--ckpt-dir", default="checkpoints/smoke")
    parser.add_argument("--stop-after", type=int, default=None, help="simule une interruption")
    parser.add_argument("--device", default=None, help="force cpu / cuda / xpu")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device : {describe_device(device)}")

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    step = 0
    resume = find_resume_checkpoint(args.ckpt_dir)
    if resume is not None:
        meta = load_checkpoint(resume, model, optimizer, scheduler)
        step = meta["step"]
        print(f"Reprise depuis {resume} (étape {step})")

    while step < args.total_steps:
        x = torch.randn(64, 8, device=device)
        y = x.sum(dim=1, keepdim=True)
        loss = F.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.save_every == 0 or step == args.total_steps:
            save_checkpoint(args.ckpt_dir, model, optimizer, scheduler,
                            step=step, metrics={"loss": loss.item()}, config=vars(args))
            print(f"étape {step:4d}  loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.4g}  -> checkpoint")

        if args.stop_after is not None and step >= args.stop_after:
            print("Interruption simulée.")
            return

    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
