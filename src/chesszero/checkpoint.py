"""Sauvegarde et reprise de l'entraînement.

Un checkpoint contient tout ce qu'il faut pour reprendre *exactement* là où on
s'est arrêté : poids du modèle, état de l'optimiseur et du scheduler, compteurs
(étape, époque), états des générateurs aléatoires, config et métriques.

Organisation d'un dossier de checkpoints :
    ckpt_00001000.pt   <- un fichier par sauvegarde (seuls les `keep_last` derniers sont gardés)
    last.pt            <- copie du plus récent : c'est lui qu'on recharge pour reprendre
    best.pt            <- meilleur modèle selon une métrique choisie par l'appelant
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

LAST = "last.pt"
BEST = "best.pt"
_PREFIX = "ckpt_"


def _unwrap(model: nn.Module) -> nn.Module:
    """Retire les enveloppes DataParallel / torch.compile pour sauver des clés propres."""
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except (RuntimeError, IndexError):
            pass  # nombre de GPU différent entre la sauvegarde et la reprise : sans gravité


def _atomic_save(obj: Any, path: Path) -> None:
    """Écrit dans un fichier temporaire puis renomme : si le processus est tué
    pendant l'écriture (fin de session Kaggle), l'ancien fichier reste intact."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def save_checkpoint(
    ckpt_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    scaler: Any | None = None,
    step: int,
    epoch: int = 0,
    config: dict | None = None,
    metrics: dict | None = None,
    is_best: bool = False,
    keep_last: int = 3,
) -> Path:
    """Sauvegarde l'état complet de l'entraînement et met à jour last.pt (et best.pt)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "epoch": epoch,
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,  # précision mixte (GradScaler)
        "rng": _rng_state(),
        "config": config or {},
        "metrics": metrics or {},
        "torch_version": torch.__version__,
    }

    path = ckpt_dir / f"{_PREFIX}{step:08d}.pt"
    _atomic_save(payload, path)
    _atomic_copy(path, ckpt_dir / LAST)
    if is_best:
        _atomic_copy(path, ckpt_dir / BEST)

    # Rotation : on ne garde que les `keep_last` fichiers ckpt_*.pt les plus récents.
    if keep_last > 0:
        for old in sorted(ckpt_dir.glob(f"{_PREFIX}*.pt"))[:-keep_last]:
            old.unlink()
    return path


def find_resume_checkpoint(ckpt_dir: str | Path) -> Path | None:
    """Renvoie le checkpoint à partir duquel reprendre (last.pt), ou None s'il n'existe pas."""
    path = Path(ckpt_dir) / LAST
    return path if path.exists() else None


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    scaler: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Recharge un checkpoint dans le modèle (et l'optimiseur/scheduler s'ils sont fournis).

    Le fichier est toujours chargé sur CPU puis copié vers le device du modèle : un
    checkpoint créé sur le GPU de Kaggle se recharge donc sur un PC sans GPU, et l'inverse.
    Renvoie les métadonnées : step, epoch, config, metrics.
    """
    # weights_only=False : nos checkpoints contiennent aussi des états aléatoires Python/NumPy.
    # Ne charger ainsi que des fichiers produits par ce projet.
    payload = torch.load(path, map_location="cpu", weights_only=False)

    _unwrap(model).load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and "rng" in payload:
        _set_rng_state(payload["rng"])

    return {k: payload.get(k) for k in ("step", "epoch", "config", "metrics")}
