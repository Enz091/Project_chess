"""Vérifie qu'un entraînement interrompu puis repris donne exactement le même
résultat qu'un entraînement d'une traite."""

import torch
import torch.nn.functional as F
from torch import nn

from chesszero.checkpoint import find_resume_checkpoint, load_checkpoint, save_checkpoint
from chesszero.device import get_device


def build(seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    return model, optimizer, scheduler


def train(model, optimizer, scheduler, n_steps: int) -> None:
    for _ in range(n_steps):
        x = torch.randn(32, 8)  # dépend du générateur aléatoire : la reprise doit le restaurer
        loss = F.mse_loss(model(x), x.sum(dim=1, keepdim=True))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


def test_resume_matches_uninterrupted_run(tmp_path):
    # Référence : 10 étapes d'une traite.
    ref_model, ref_opt, ref_sched = build()
    train(ref_model, ref_opt, ref_sched, 10)

    # 5 étapes, sauvegarde, « crash »...
    model, opt, sched = build()
    train(model, opt, sched, 5)
    save_checkpoint(tmp_path, model, opt, sched, step=5)

    # ...puis nouveau processus : objets neufs, RNG différent, reprise.
    model, opt, sched = build(seed=999)
    meta = load_checkpoint(find_resume_checkpoint(tmp_path), model, opt, sched)
    assert meta["step"] == 5
    train(model, opt, sched, 5)

    for p_ref, p in zip(ref_model.parameters(), model.parameters()):
        assert torch.equal(p_ref, p)
    assert sched.get_last_lr() == ref_sched.get_last_lr()


def test_rotation_keeps_last_n_and_best(tmp_path):
    model, opt, sched = build()
    for step in (1, 2, 3, 4):
        save_checkpoint(tmp_path, model, opt, sched, step=step, keep_last=2, is_best=(step == 2))
    names = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert names == ["best.pt", "ckpt_00000003.pt", "ckpt_00000004.pt", "last.pt"]
    assert load_checkpoint(tmp_path / "best.pt", model)["step"] == 2


def test_get_device_returns_usable_device():
    device = get_device()
    assert torch.ones(2, device=device).sum().item() == 2
    assert get_device("cpu").type == "cpu"
