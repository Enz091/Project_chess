"""Cibles de politique débruitées (à la manière du « policy target pruning » de KataGo).

Problème : en self-play, la racine de chaque recherche reçoit un bruit de Dirichlet (P = 0,75 P_réseau + 0,25 Dir(0,3)).
Le bruit gonfle au hasard le prior de quelques coups, que la recherche visite alors plus qu'ils ne le méritent ; les
visites enregistrées (la cible de politique) sont donc plus étalées que celles d'une recherche sans bruit, et le
réseau entraîné sur ces cibles s'aplatit (journal, première session de boucle).

Remède : retirer à chaque coup les visites que ni sa valeur ni son prior SANS bruit ne justifient. À la fin de la
recherche, PUCT(c) = Q(c) + c_puct * sqrt(1 + N) * P(c) / (1 + N(c)). Pour le coup le plus visité b (jamais touché) on
calcule S_b = Q(b) + c_puct * sqrt(1 + N) * P(b) / (1 + N(b)) ; pour chaque autre coup visité c, on garde le plus petit
nombre de visites n tel que Q(c) + c_puct * sqrt(1 + N) * P(c) / (1 + n) <= S_b, avec le prior SANS bruit, borné par
ses visites réelles. Un coup dont la valeur dépasse S_b n'est jamais réduit ; un coup de faible prior et de faible
valeur qui n'a été visité que grâce au bruit retombe à (presque) zéro. Différence avec KataGo : là-bas, la réduction
est plafonnée par les « forced playouts » et le prior est celui avec bruit ; ici, c'est le prior sans bruit qui
fait l'essentiel du débruitage (pas de forced playouts dans ce projet).

Ingrédients par coup légal de la racine : visites, prior du réseau SANS bruit, Q du coup (camp au trait à la racine,
= W/N de l'arête). Les shards depuis le 2026-10-24 les enregistrent (`policy_prior`, `policy_q`). Pour les plus
anciens, l'entraîneur recalcule le prior avec le réseau qui a joué (`network_priors`, gardé en cache) et, faute de Q,
donne à chaque coup visité le Q du meilleur : le plafond devient (1 + N(b)) x P(c) / P(b) - 1 visites. Mesure sur
500 positions (journal, 2026-10-24) : c'est l'approximation la plus proche d'une recherche sans bruit ; elle a
une limite : un 2e coup bien meilleur que ne le croit le prior est plafonné lui aussi.
"""

from __future__ import annotations

import numpy as np


def segments(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(début de chaque position, numéro de position de chaque coup) pour des positions [offsets[i], offsets[i+1])."""
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    if (lengths <= 0).any():
        raise ValueError("chaque position doit avoir au moins un coup légal")
    return offsets[:-1], np.repeat(np.arange(len(lengths)), lengths)


def best_moves(offsets: np.ndarray, visits: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Indice (dans les tableaux à plat) du coup le plus visité de chaque position ; égalité : plus grand prior
    (comme `MCTS.best_move` à température 0)."""
    starts, seg = segments(offsets)
    order = np.lexsort((-np.asarray(prior, dtype=np.float64), -np.asarray(visits, dtype=np.float64), seg))
    return order[starts]


def prune_visits(offsets: np.ndarray, visits: np.ndarray, prior: np.ndarray, q: np.ndarray, c_puct: float,
                 drop_single: bool = False) -> np.ndarray:
    """Visites débruitées (float64, mêmes positions que `visits`) pour des positions à plat :
    `offsets` (P+1,), `visits` (M,), `prior` (M,) = prior SANS bruit, `q` (M,) = Q de chaque coup (NaN si non visité).
    `drop_single` : un coup réduit à une seule visite tombe à zéro (règle de KataGo)."""
    visits = np.asarray(visits, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    starts, seg = segments(offsets)
    best = best_moves(offsets, visits, prior)
    total = np.add.reduceat(visits, starts)
    scale = c_puct * np.sqrt(1.0 + total)  # c_puct * sqrt(1 + N) (perte virtuelle nulle en fin de recherche)
    s_best = q[best] + scale * prior[best] / (1.0 + visits[best])
    visited = visits > 0
    gap = s_best[seg] - q
    reducible = visited & (gap > 0) & np.isfinite(gap)
    out = visits.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        keep = np.ceil(scale[seg] * prior / np.where(reducible, gap, 1.0) - 1.0 - 1e-9)
    out[reducible] = np.clip(keep[reducible], 0.0, visits[reducible])
    if drop_single:
        out[reducible & (out <= 1) & (out < visits)] = 0.0
    out[best] = visits[best]
    return out


def target_stats(parts: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Résumé de cibles (liste de (offsets, visites)) : visites totales, entropie moyenne (nats) et part moyenne du
    coup le plus visité, par position."""
    total, ent, top, n = 0.0, 0.0, 0.0, 0
    for offsets, visits in parts:
        visits = np.asarray(visits, dtype=np.float64)
        starts, seg = segments(offsets)
        sums = np.add.reduceat(visits, starts)
        p = visits / sums[seg]
        with np.errstate(divide="ignore", invalid="ignore"):
            plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
        ent += float(-np.add.reduceat(plogp, starts).sum())
        top += float(np.maximum.reduceat(p, starts).sum())
        total += float(visits.sum())
        n += len(starts)
    return {"visits": total, "entropy": ent / max(n, 1), "top_share": top / max(n, 1)}


def network_priors(model, boards: np.ndarray, offsets: np.ndarray, index: np.ndarray, device,
                   batch_size: int = 1024) -> np.ndarray:
    """Prior SANS bruit du réseau sur les coups légaux enregistrés (softmax des logits restreint à ces coups), à plat
    comme `index` : sert aux shards d'avant le 2026-10-24, qui n'enregistraient pas le prior."""
    import torch

    from chesszero.encoding import unpack_batch_torch

    device = torch.device(device)
    offsets = np.asarray(offsets, dtype=np.int64)
    out = np.zeros(len(index), dtype=np.float16)
    model.eval()
    for i0 in range(0, len(boards), batch_size):
        i1 = min(i0 + batch_size, len(boards))
        a, b = offsets[i0], offsets[i1]
        starts, seg = segments(offsets[i0:i1 + 1] - a)
        with torch.inference_mode():
            packed = torch.from_numpy(np.ascontiguousarray(boards[i0:i1])).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(unpack_batch_torch(packed))
            z = logits.float()[torch.from_numpy(seg).to(device),
                               torch.from_numpy(np.asarray(index[a:b], dtype=np.int64)).to(device)]
        z = z.cpu().numpy().astype(np.float64)
        e = np.exp(z - np.maximum.reduceat(z, starts)[seg])
        out[a:b] = e / np.add.reduceat(e, starts)[seg]
    return out


def prune_one(visits, prior, q, c_puct: float, drop_single: bool = False) -> np.ndarray:
    """`prune_visits` pour une seule position (tableaux de ses coups légaux)."""
    return prune_visits(np.array([0, len(visits)]), visits, prior, q, c_puct, drop_single)
