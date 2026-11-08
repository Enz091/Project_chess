"""Entrées d'une session Kaggle de pré-entraînement B1 (notebooks/kaggle_pretrain_b1.ipynb), vérifiées par leur contenu.

- Données : exactement un dossier sous /kaggle/input dont le manifeste porte `"dataset": "lichess_48m"`, chaque shard
  vérifié (SHA-256, taille, positions : `dataset_check.check_shards`).
- Reprise : la sortie d'une session précédente de ce notebook contient `<OUT_NAME>/last.pt`. Zéro sortie = départ de
  zéro (`expected_step` doit valoir 0) ; une seule = reprise, son pas doit valoir `expected_step` et son réseau doit
  être celui de la config ; plusieurs = refus. Un entraînement arrêté par le garde-fou de débit n'est pas repris sans
  `accept_low_throughput` (le critère approuvé dit : arrêt).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from chesszero.dataset_check import check_shards, find_dataset

DATASET = "lichess_48m"
OUT_NAME = "b1_12x160"
RESTORED = ("last.pt", "best.pt", "metrics.csv", "throughput_check.json", "bench.json")


def check_pretrain_inputs(input_root: str | Path, config: str = "kaggle_l", expected_step: int = 0,
                          accept_low_throughput: bool = False, verify_shards: bool = True, dataset: str = DATASET,
                          out_name: str = OUT_NAME, progress=None) -> dict:
    """Renvoie {"ok", "errors", "notes", "data", "previous", "previous_step", "total_steps"} ; rien n'est écrit."""
    import torch

    from chesszero.model import ModelConfig
    from chesszero.train import TrainConfig

    root = Path(input_root)
    errors, notes = [], []
    res = {"data": None, "previous": None, "previous_step": None}
    model_cfg, train_cfg = ModelConfig.named(config), TrainConfig.named(config)
    res["total_steps"] = train_cfg.total_steps
    found = find_dataset(root, dataset)
    if len(found) != 1:
        errors.append(f"données : {len(found)} dossier(s) « {dataset} » sous {root} (il en faut exactement un : "
                      f"Add Input -> le Dataset {dataset.replace('_', '-')}) : {[str(f) for f in found]}")
    else:
        data = found[0]
        res["data"] = data
        manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        pos = manifest["positions"]
        notes.append(f"données : {data} ; entraînement {pos['train']:,}, validation {pos['val']:,}, "
                     f"{len(manifest['shards'])} shards ; {manifest['exclusion']['train_removed']:,} positions protégées "
                     f"retirées de l'entraînement")
        if verify_shards:
            errs = check_shards(lambda n: (data / n).read_bytes(), [p.name for p in data.glob("*.npz")], manifest,
                                None, "données", progress)
            errors += errs
            if not errs:
                notes.append("données : chaque shard vérifié (SHA-256, taille, positions)")
    previous = sorted(root.glob(f"**/{out_name}/last.pt"))
    if len(previous) > 1:
        errors.append(f"reprise : {len(previous)} sorties précédentes ({[str(p.parent) for p in previous]}) : n'en "
                      f"garder qu'une en entrée")
    elif not previous:
        if expected_step != 0:
            errors.append(f"reprise : aucune sortie précédente ({out_name}/last.pt) alors que EXPECTED_STEP = "
                          f"{expected_step} (0 = départ de zéro)")
        else:
            notes.append("reprise : aucune sortie précédente, départ de zéro (EXPECTED_STEP = 0)")
    else:
        prev = previous[0].parent
        payload = torch.load(previous[0], map_location="cpu", weights_only=False)
        step = int(payload["step"])
        res.update(previous=prev, previous_step=step)
        if payload["config"]["model"] != model_cfg.to_dict():
            errors.append(f"reprise : {previous[0]} est un autre réseau ({payload['config']['model']})")
        if step != expected_step:
            errors.append(f"reprise : la sortie précédente est au pas {step}, EXPECTED_STEP = {expected_step}")
        check = prev / "throughput_check.json"
        if check.exists():
            c = json.loads(check.read_text(encoding="utf-8"))
            if not c["ok"] and not accept_low_throughput:
                errors.append(f"reprise : l'entraînement a été ARRÊTÉ par le garde-fou de débit ({c['pos_per_s']:,.0f} "
                              f"positions/s < {c['min_pos_per_s']:,.0f} au pas {c['step']}) : ne pas le reprendre sans "
                              f"décision (ACCEPT_LOW_THROUGHPUT)")
        notes.append(f"reprise : {prev} au pas {step:,}/{train_cfg.total_steps:,}"
                     + (" (TERMINÉ : rien à entraîner)" if step >= train_cfg.total_steps else ""))
    res.update(ok=not errors, errors=errors, notes=notes)
    return res


def restore_previous(previous: str | Path, out: str | Path) -> list[str]:
    """Copie dans `out` les fichiers de reprise d'une sortie précédente (sans écraser un last.pt déjà présent)."""
    previous, out = Path(previous), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists():
        return []
    copied = []
    for name in RESTORED:
        if (previous / name).exists():
            shutil.copy2(previous / name, out / name)
            copied.append(name)
    return copied
