"""Critère 2 de B1 (supervisé), fixé avant le lancement (journal, 2026-11-08) : jugé sur les positions NON PROTÉGÉES
du sous-ensemble fixe de 50 000 positions Lichess de B2 (validation de `lichess_24m`, graine 0).

Positions protégées = positions des 40 lignes v2 et des 100 lignes de l'arène ancre (`sf_value.opening_keys`) : le
réseau B1 ne les voit jamais à l'entraînement (retirées de `lichess_48m`), alors que `kaggle_sup` les a vues. Seuils
= chiffres de `kaggle_sup` sur les positions non protégées (top-1 53,79 %, perte de politique 1,4061) avec les marges
approuvées (+0,5 point, −0,01) : **top-1 ≥ 54,29 % ET perte de politique ≤ 1,3961**. Les chiffres sur les 50 000
positions complètes (seuils d'origine 54,2 % / 1,3949) et sur les positions protégées sont donnés pour information.
"""

from __future__ import annotations

import numpy as np

TOP1_MIN = 0.5429
POLICY_LOSS_MAX = 1.3961
INFO_TOP1_MIN_ALL = 0.542  # seuils d'origine sur les 50 000 positions : information seulement
INFO_POLICY_LOSS_MAX_ALL = 1.3949
LICHESS_VAL = 50000
LICHESS_SEED = 0


def protected_mask(boards: np.ndarray, keys: set | None = None) -> np.ndarray:
    """Vrai pour chaque position (compacte, (N, 67)) qui est une position protégée (compteur des 50 coups ignoré)."""
    from chesszero.sf_value import opening_keys, position_keys

    keys = opening_keys() if keys is None else keys
    return np.fromiter((k in keys for k in position_keys(boards)), bool, count=len(boards))


def subset(val: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in val.items()}


def evaluate(model, val: dict, mask: np.ndarray, device) -> dict:
    """Métriques Lichess (`selfplay_train.eval_human`) sur les positions non protégées, protégées et sur toutes."""
    from chesszero.selfplay_train import eval_human

    out = {"unprotected": eval_human(model, subset(val, ~mask), device),
           "protected": eval_human(model, subset(val, mask), device) if mask.any() else None}
    n_u, n_p = int((~mask).sum()), int(mask.sum())
    out["unprotected"]["positions"] = n_u
    if out["protected"] is not None:
        out["protected"]["positions"] = n_p
    # toutes les positions = moyenne pondérée (perte et top-1 sont des moyennes par position ; la valeur, par position
    # à résultat valide)
    vv = val["value_valid"]
    w_u, w_p = float(vv[~mask].sum()), float(vv[mask].sum())
    allm = {"positions": n_u + n_p}
    for k in ("policy_loss", "top1"):
        allm[k] = (out["unprotected"][k] * n_u + (out["protected"][k] * n_p if n_p else 0.0)) / max(n_u + n_p, 1)
    allm["value_loss"] = ((out["unprotected"]["value_loss"] * w_u + (out["protected"]["value_loss"] * w_p if n_p else 0.0))
                          / max(w_u + w_p, 1.0))
    out["all"] = allm
    return out


def judge(metrics: dict) -> dict:
    """Verdict du critère 2 sur les positions non protégées ; information sur les 50 000 (sans effet sur le verdict)."""
    u, a = metrics["unprotected"], metrics["all"]
    top1_ok, pol_ok = u["top1"] >= TOP1_MIN, u["policy_loss"] <= POLICY_LOSS_MAX
    return {"criterion": f"positions non protégées : top-1 >= {TOP1_MIN:.2%} ET perte de politique <= {POLICY_LOSS_MAX}",
            "passed": bool(top1_ok and pol_ok), "top1_ok": bool(top1_ok), "policy_loss_ok": bool(pol_ok),
            "info_all_positions": {"top1_ok": bool(a["top1"] >= INFO_TOP1_MIN_ALL),
                                   "policy_loss_ok": bool(a["policy_loss"] <= INFO_POLICY_LOSS_MAX_ALL),
                                   "thresholds": {"top1": INFO_TOP1_MIN_ALL, "policy_loss": INFO_POLICY_LOSS_MAX_ALL}}}
