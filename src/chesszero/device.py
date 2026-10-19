"""Choix automatique du matériel de calcul.

Ordre de préférence : CUDA (GPU NVIDIA, ex. Kaggle) > XPU (GPU Intel) > CPU.
Le reste du code ne doit jamais écrire "cuda" en dur : il appelle get_device().
"""

from __future__ import annotations

import os

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Retourne le meilleur device disponible.

    `preferred` (ou la variable d'environnement CHESSZERO_DEVICE) force un choix,
    par ex. "cpu" pour déboguer sur Kaggle sans utiliser le GPU.
    """
    choice = preferred or os.environ.get("CHESSZERO_DEVICE")
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Description lisible, à afficher au démarrage d'un script."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)}, {torch.cuda.device_count()} GPU visible(s))"
    if device.type == "xpu":
        return f"xpu ({torch.xpu.get_device_name(device)})"
    return f"cpu ({torch.get_num_threads()} threads)"
