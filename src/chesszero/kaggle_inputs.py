"""Découverte des entrées d'une session Kaggle (étape 7e) : réseau de départ, données Lichess, sortie précédente.

Sur Kaggle, les entrées ajoutées au notebook (Datasets, sortie d'un notebook) sont montées sous `/kaggle/input` à des
endroits qui varient (`/kaggle/input/<slug>/...`, `/kaggle/input/notebooks/<compte>/<slug>/...`) : on les trouve par
leur CONTENU, jamais par leur chemin.

  - réseau de départ : un `.pt` dont le SHA-256 est celui attendu (kaggle_sup, pas 92 000) ;
  - données Lichess : le dossier qui contient `train_00000.npz` (+ `val_*.npz`) ;
  - sortie précédente : un dossier qui contient `selfplay/` avec des shards complets (`.npz` + `.json`) ; son type
    (« self-play seul » ou « boucle, session N ») vient de `loop.classify_output`.

`discover` ne modifie rien. `select_previous` exige EXACTEMENT une sortie précédente : zéro (on oublie de l'ajouter) ou
deux (la boucle reprendrait peut-être la mauvaise) sont des refus avec un message qui dit quoi corriger.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MIN_NETWORK_BYTES = 10_000_000  # un checkpoint kaggle fait ~38 Mo


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_init(root: Path, expected_sha256: str) -> tuple[Path | None, list[str]]:
    """Premier `.pt` d'entrée au SHA-256 attendu. Les réseaux d'une sortie de boucle (`sp_train/`) ne sont pas examinés
    (candidats, générations : plusieurs dizaines de Mo chacun). Renvoie (chemin, fichiers .pt vus)."""
    seen = []
    for p in sorted(root.glob("**/*.pt")):
        if p.stat().st_size < MIN_NETWORK_BYTES or "sp_train" in p.parts:
            continue
        seen.append(str(p))
        if sha256_of(p) == expected_sha256:
            return p, seen
    return None, seen


def find_human(root: Path) -> Path | None:
    for p in sorted(root.glob("**/train_00000.npz")):
        return p.parent
    return None


def describe_human(folder: Path) -> dict:
    manifest = folder / "manifest.json"
    info = {"path": str(folder), "train_shards": len(list(folder.glob("train_*.npz"))),
            "val_shards": len(list(folder.glob("val_*.npz"))), "positions": None}
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
            pos = m.get("positions")  # {"train": n, "val": n} (data.extract)
            info["positions"] = sum(pos.values()) if isinstance(pos, dict) else pos
        except (OSError, ValueError):
            pass
    return info


def find_previous_outputs(root: Path) -> list[dict]:
    """Sorties de session trouvées : dossiers qui contiennent `selfplay/` avec au moins un shard complet."""
    from chesszero.loop import classify_output, last_session, read_loop_state

    found, seen = [], set()
    for npz in sorted(root.glob("**/selfplay/*.npz")):
        if not npz.with_suffix(".json").exists():
            continue
        out = npz.parent.parent
        if out in seen:
            continue
        seen.add(out)
        info = classify_output(out)
        session = None
        if info["kind"] == "loop":
            try:
                session = last_session(read_loop_state(out / "loop" / "loop_state.json"))
            except (OSError, ValueError):
                session = None
        found.append({"path": out, "kind": info["kind"], "session": session, "shards": info["shards"],
                      "positions": info["positions"]})
    return found


def describe_previous(p: dict) -> str:
    kind = {"selfplay_only": "self-play seul", "loop": f"boucle, session n°{p['session']}",
            "inconsistent": "INCOHÉRENTE", "empty": "vide"}.get(p["kind"], p["kind"])
    return f"{p['path']} ({kind}, {p['shards']} shards, {p['positions']:,} positions)"


def discover(input_root: str | Path, init_sha256: str, need_human: bool = True) -> dict:
    """{"init", "human", "previous", "problems", "notes"} ; `problems` non vide = ne pas continuer."""
    root = Path(input_root)
    problems, notes = [], []
    if not root.is_dir():
        return {"init": None, "human": None, "previous": [], "notes": notes,
                "problems": [f"{root} n'existe pas (pas une session Kaggle ?)"]}
    init, seen = find_init(root, init_sha256)
    if init is None:
        problems.append(f"Réseau de départ introuvable : ajouter le Dataset contenant best.pt de kaggle_sup (SHA-256 "
                        f"{init_sha256[:8]}…{init_sha256[-5:]}). Fichiers .pt vus : {seen or 'aucun'}")
    human = find_human(root)
    if human is None and need_human:
        problems.append("Données Lichess introuvables : ajouter le Dataset lichess-24m (dossier contenant "
                        "train_00000.npz)")
    elif human is not None:
        h = describe_human(human)
        positions = f", {h['positions']:,} positions" if h["positions"] else ""
        notes.append(f"Données Lichess : {h['path']} ({h['train_shards']} shards d'entraînement, {h['val_shards']} de "
                     f"validation{positions})")
        if h["val_shards"] == 0:
            problems.append(f"Données Lichess : aucun shard de validation (val_*.npz) dans {h['path']}")
    previous = find_previous_outputs(root)
    return {"init": init, "human": human, "previous": previous, "problems": problems, "notes": notes}


def select_previous(found: dict, expect_session: int) -> tuple[Path | None, list[str]]:
    """Exactement une sortie précédente ; cohérente avec EXPECTED_SESSION (le contrôle exact est refait par
    `alphazero_session.py` : ici on évite seulement de partir sur une mauvaise entrée)."""
    prev = found["previous"]
    if not prev:
        return None, ["Aucune sortie précédente trouvée : Add Input -> Your Work -> la version du notebook qui contient "
                      "selfplay/ (sortie de la session de self-play seul pour la première session de boucle ; la "
                      "sortie de la session précédente ensuite)"]
    if len(prev) > 1:
        lines = "\n  - ".join(describe_previous(p) for p in prev)
        return None, [f"{len(prev)} sorties précédentes trouvées, il en faut EXACTEMENT une (la boucle reprendrait "
                      f"peut-être la mauvaise) :\n  - {lines}\nRetirer les autres (Add Input -> croix à côté de l'entrée)."]
    p = prev[0]
    if p["kind"] == "inconsistent":
        return None, [f"Sortie précédente incohérente : {describe_previous(p)}"]
    if p["kind"] == "selfplay_only" and expect_session != 0:
        return None, [f"La seule entrée est une sortie de self-play seul alors que EXPECTED_SESSION = {expect_session} "
                      "(0 pour la première session de boucle)"]
    if p["kind"] == "loop" and p["session"] != expect_session:
        return None, [f"Mauvaise entrée : la sortie trouvée est celle de la session de boucle n°{p['session']}, "
                      f"EXPECTED_SESSION = {expect_session}. Si c'est voulu, mettre EXPECTED_SESSION = {p['session']}."]
    return p["path"], []
