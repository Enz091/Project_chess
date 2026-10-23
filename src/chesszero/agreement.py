"""Indicateur d'accord entre la politique brute d'un réseau et la cible du MCTS (suivi des générations).

Pour un échantillon fixe de positions du self-play (graine fixe), on compare, pour un réseau donné :
  - l'ACCORD = part des positions où le coup préféré du réseau (softmax sur les coups légaux, sans recherche) est le
    coup le plus visité par le MCTS de la génération (en cas d'égalité de visites : le premier dans l'ordre des
    coups, comme `np.argmax`) ;
  - la DIVERGENCE KL moyenne KL(cible MCTS || politique brute), la cible étant les visites normalisées.
Trois phases : ouverture (moins de 30 demi-coups), finale (au moins 30 demi-coups et moins de 20 points de
matériel : P 1, N/B 3, R 5, Q 9), milieu de partie (le reste).

Biais à connaître : les visites enregistrées viennent d'une recherche avec BRUIT DE DIRICHLET à la racine
(epsilon 0,25) ; ce bruit déplace un peu les visites vers des coups choisis au hasard et fait baisser l'accord
(et monter la KL) par rapport à une recherche sans bruit. L'indicateur se compare d'une génération à l'autre
(mêmes réglages), pas à une valeur absolue.

Calculé en fin de `scripts/selfplay.py` avec le réseau qui a joué les parties ET avec le réseau de référence
(kaggle_sup) sur les mêmes positions : l'écart entre les deux montre ce que le réseau a appris depuis le départ.
Référence de la génération 0 (kaggle_sup sur 20 000 positions de la session de self-play seul du 2026-10-22) :
accord 59,8 %, KL 0,535 (ouverture 66,2 % / 0,492 ; milieu 55,7 % / 0,582 ; finale 66,9 % / 0,428).
"""

from __future__ import annotations

import numpy as np

OPENING_PLIES = 30
ENDGAME_MATERIAL = 20
PHASES = ("opening", "middle", "endgame")
PHASE_LABELS = {"opening": "ouverture", "middle": "milieu", "endgame": "finale", "all": "toutes"}
GEN0_REFERENCE = {"all": (0.598, 0.535), "opening": (0.662, 0.492), "middle": (0.557, 0.582),
                  "endgame": (0.669, 0.428)}
DIRICHLET_NOTE = ("biais : les visites viennent d'une recherche avec bruit de Dirichlet à la racine (epsilon 0,25), "
                  "ce qui baisse un peu l'accord et monte la KL ; à comparer d'une génération à l'autre, pas à une "
                  "valeur absolue")


def phase_of(ply: int, material: int) -> str:
    if ply < OPENING_PLIES:
        return "opening"
    return "endgame" if material < ENDGAME_MATERIAL else "middle"


def sample_positions(shards: list[dict], n: int, seed: int, exclude_heldout: float | None = None
                     ) -> list[tuple[int, int]]:
    """`n` positions (indice du shard, ligne) tirées sans remise dans tous les shards, graine fixe (toutes si
    moins de `n`). Le tirage ne dépend que du contenu des shards. `exclude_heldout` (ex. 0,10) : écarte les parties
    mises de côté de l'entraînement (`selfplay_train.is_heldout`), pour n'échantillonner que des positions d'entraînement
    (analyse a posteriori d'une génération) ; l'indicateur de session garde toutes les positions."""
    rows = []
    for k, s in enumerate(shards):
        uids = s["game_uid"][s["game"]] if "game_uid" in s else None
        if exclude_heldout and uids is not None:
            from chesszero.selfplay_train import is_heldout

            cache = {u: is_heldout(u, exclude_heldout) for u in set(uids.tolist())}
            keep = [i for i, u in enumerate(uids.tolist()) if not cache[u]]
        else:
            keep = range(len(s["played"]))
        rows += [(k, i) for i in keep]
    if not rows:
        return []
    picks = np.sort(np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False))
    return [rows[int(p)] for p in picks]


def gather(shards: list[dict], picks: list[tuple[int, int]]) -> dict:
    """Positions compactes, demi-coup, matériel, coups légaux et visites de la cible pour les positions tirées."""
    from chesszero.encoding import board_from_packed
    from chesszero.selfplay import material_points

    boards = np.stack([shards[k]["boards"][i] for k, i in picks]) if picks else np.zeros((0, 67), np.uint8)
    ply = np.array([int(shards[k]["ply"][i]) for k, i in picks], dtype=np.int64)
    material = np.array([material_points(board_from_packed(boards[j])) for j in range(len(picks))], dtype=np.int64)
    legal, visits = [], []
    for k, i in picks:
        off = shards[k]["policy_offsets"]
        a, b = int(off[i]), int(off[i + 1])
        legal.append(shards[k]["policy_index"][a:b].astype(np.int64))
        visits.append(shards[k]["policy_visits"][a:b].astype(np.float64))
    phase = np.array([phase_of(int(p), int(m)) for p, m in zip(ply, material)])
    return {"boards": boards, "ply": ply, "material": material, "phase": phase, "legal": legal, "visits": visits}


def policy_agreement(model, batch: dict, device="cpu", batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """(accord 0/1, KL) de chaque position de `batch` pour `model` (politique brute, sans recherche)."""
    import torch

    from chesszero.encoding import unpack_batch_torch

    n = len(batch["legal"])
    agree, kl = np.zeros(n), np.zeros(n)
    model.eval()
    for s in range(0, n, batch_size):
        ids = range(s, min(s + batch_size, n))
        width = max(len(batch["legal"][i]) for i in ids)
        index = np.zeros((len(ids), width), dtype=np.int64)
        vis = np.zeros((len(ids), width), dtype=np.float64)
        for r, i in enumerate(ids):
            index[r, :len(batch["legal"][i])] = batch["legal"][i]
            vis[r, :len(batch["visits"][i])] = batch["visits"][i]
        lengths = torch.tensor([len(batch["legal"][i]) for i in ids])
        valid = torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
        with torch.inference_mode():
            packed = torch.from_numpy(batch["boards"][s:s + len(ids)]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.device(device).type == "cuda"):
                logits, _ = model(unpack_batch_torch(packed))
            z = logits.float().cpu().double().gather(1, torch.from_numpy(index)).masked_fill(~valid, float("-inf"))
        logp = torch.log_softmax(z, dim=1)
        v = torch.from_numpy(vis)
        t = v / v.sum(dim=1, keepdim=True)
        safe_t = torch.where(t > 0, t, torch.ones_like(t))
        term = torch.where(t > 0, t * (torch.log(safe_t) - logp), torch.zeros_like(t))
        kl[s:s + len(ids)] = term.sum(dim=1).numpy()
        agree[s:s + len(ids)] = (torch.argmax(z, dim=1) == torch.argmax(v, dim=1)).double().numpy()
    return agree, kl


def summarize(batch: dict, agree: np.ndarray, kl: np.ndarray) -> dict:
    """{phase: {"n", "agree", "kl"}} pour les trois phases et "all" (moyennes ; n = 0 -> None)."""
    out = {}
    for name, mask in (("all", np.ones(len(agree), bool)),) + tuple((p, batch["phase"] == p) for p in PHASES):
        n = int(mask.sum())
        out[name] = {"n": n, "agree": float(agree[mask].mean()) if n else None,
                     "kl": float(kl[mask].mean()) if n else None}
    return out


def session_agreement(shards: list[dict], networks: dict, device="cpu", n: int = 5000, seed: int = 0) -> dict:
    """Indicateur d'une session : `networks` = {"played": modèle, "reference": modèle} (le second peut manquer).
    Même échantillon pour tous les réseaux."""
    picks = sample_positions(shards, n, seed)
    batch = gather(shards, picks)
    result = {"sample": len(picks), "seed": seed, "networks": {}}
    for name, model in networks.items():
        agree, kl = policy_agreement(model, batch, device)
        result["networks"][name] = summarize(batch, agree, kl)
    return result


def merge_agreement(parts: list[dict]) -> dict | None:
    """Moyenne pondérée par n de plusieurs tranches d'une session (ou de sessions) : même forme que le résultat de
    `session_agreement`. Les tranches sans indicateur (ou en erreur) sont ignorées."""
    parts = [p for p in parts if p and p.get("networks")]
    if not parts:
        return None
    merged = {"sample": sum(p["sample"] for p in parts), "seed": parts[0].get("seed"), "networks": {}}
    for name in {k for p in parts for k in p["networks"]}:
        merged["networks"][name] = {}
        for ph in ("all",) + PHASES:
            cells = [p["networks"][name][ph] for p in parts if name in p["networks"]
                     and p["networks"][name][ph]["n"]]
            n = sum(c["n"] for c in cells)
            merged["networks"][name][ph] = {
                "n": n, "agree": sum(c["agree"] * c["n"] for c in cells) / n if n else None,
                "kl": sum(c["kl"] * c["n"] for c in cells) / n if n else None}
    return merged


def _cell(c: dict) -> str:
    return "—" if not c or not c["n"] else f"{c['agree']:.1%} (KL {c['kl']:.3f})"


def describe_agreement(a: dict | None, played_name: str = "réseau qui a joué") -> str:
    """Ligne(s) du rapport de session."""
    if not a or not a.get("networks"):
        return "accord politique brute / cible MCTS : non calculé"
    nets = a["networks"]
    lines = [f"accord politique brute / cible MCTS ({a['sample']:,} positions, graine {a.get('seed')}) :"]
    for key, label in (("played", played_name), ("reference", "kaggle_sup (départ)")):
        if key in nets:
            lines.append(f"  - {label} : {_cell(nets[key]['all'])} ; " + ", ".join(
                f"{PHASE_LABELS[p]} {_cell(nets[key][p])}" for p in PHASES))
    if "played" in nets and "reference" in nets and nets["played"]["all"]["n"]:
        d = nets["played"]["all"]["agree"] - nets["reference"]["all"]["agree"]
        dk = nets["played"]["all"]["kl"] - nets["reference"]["all"]["kl"]
        lines.append(f"  - écart (appris depuis le départ) : accord {d * 100:+.1f} points, KL {dk:+.3f}")
    ref = GEN0_REFERENCE["all"]
    lines.append(f"  - génération 0 (kaggle_sup, 20 000 positions) : {ref[0]:.1%} (KL {ref[1]:.3f}) ; {DIRICHLET_NOTE}")
    return "\n".join(lines)
