"""Vérification d'un dossier de shards (ou de l'archive d'un Dataset Kaggle) contre son `manifest.json`.

Le manifeste d'un jeu préparé par `scripts/prepare_b1_data.py` liste chaque shard avec son nombre de positions, sa
taille et son SHA-256. `check_shards` en déduit une liste d'erreurs (vide = cohérent) ; `find_dataset` retrouve le
dossier d'un jeu par le nom inscrit dans son manifeste (sous /kaggle/input, à n'importe quelle profondeur).
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Callable

import numpy as np


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_dataset(root: str | Path, name: str) -> list[Path]:
    """Dossiers sous `root` dont `manifest.json` porte `"dataset": name`."""
    found = []
    for m in sorted(Path(root).glob("**/manifest.json")):
        try:
            if json.loads(m.read_text(encoding="utf-8")).get("dataset") == name:
                found.append(m.parent)
        except (OSError, ValueError):
            continue
    return found


def check_shards(read: Callable[[str], bytes], names: list[str], manifest: dict, protected: set | None = None,
                 label: str = "dossier", progress=None) -> list[str]:
    """Erreurs (liste vide = cohérent) : shards attendus présents et seuls, SHA-256 et taille, nombres de positions
    par shard et par partie, et (si `protected`) aucune position protégée dans la partie entraînement.
    `read(nom)` renvoie les octets d'un fichier."""
    errors = []
    expected = manifest.get("shards")
    if not expected:
        return [f"{label} : le manifeste ne liste pas les shards (jeu non préparé par prepare_b1_data.py)"]
    present = sorted(n for n in names if n.endswith(".npz"))
    if present != sorted(expected):
        missing, extra = set(expected) - set(present), set(present) - set(expected)
        errors.append(f"{label} : shards différents du manifeste (manquants {sorted(missing)[:3]}, en trop {sorted(extra)[:3]})")
    counts = {"train": 0, "val": 0}
    for i, (name, info) in enumerate(sorted(expected.items()), 1):
        if name not in present:
            continue
        b = read(name)
        if len(b) != info["bytes"] or sha256_bytes(b) != info["sha256"]:
            errors.append(f"{label} : {name} : empreinte ou taille différente du manifeste")
            continue
        d = np.load(io.BytesIO(b))
        n = len(d["moves"])
        if n != info["positions"] or any(len(d[k]) != n for k in ("boards", "results", "value_valid", "games")):
            errors.append(f"{label} : {name} : {n} positions au lieu de {info['positions']}")
        split = "train" if name.startswith("train_") else "val"
        counts[split] += n
        if protected is not None and split == "train":
            from chesszero.sf_value import position_keys

            if any(k in protected for k in position_keys(d["boards"])):
                errors.append(f"{label} : {name} contient des positions protégées")
        if progress:
            progress(i, len(expected))
    if counts != manifest.get("positions"):
        errors.append(f"{label} : positions {counts} au lieu de {manifest.get('positions')}")
    return errors
