"""Estimation d'Elo à partir de parties contre des adversaires de niveau connu (échelle de Stockfish).

Modèle : un joueur de niveau R contre un adversaire de niveau S marque en moyenne
  p(R, S) = 1 / (1 + 10^((S - R) / 400)).
Chaque partie donne un score s dans {0, 0,5, 1} (nulle = 0,5). On estime R en résolvant
  sum_i (s_i - p(R, S_i)) = 0     (quasi-vraisemblance de Bernoulli : même estimateur qu'une
                                   régression logistique avec des scores fractionnaires)
Un seul R explique toutes les parties, quel que soit le niveau de l'adversaire : c'est plus fiable
qu'un écart de score à un seul niveau, surtout quand le score est loin de 50 %.

Marge d'erreur : écart-type de R sous deux hypothèses, on garde le plus grand :
  - modèle (Bernoulli)  : 1 / (c * sqrt(sum p (1 - p))),             c = ln(10) / 400 ;
  - observé (sandwich)  : sqrt(sum (s - p)^2) / (c * sum p (1 - p)),
    qui tient compte des nulles (moins de variance) ou d'un écart au modèle (plus de variance).
Intervalle à 95 % = R ± 1,96 écart-type. Il suppose les parties indépendantes ; des parties rejouées
depuis la même ouverture ne le sont pas tout à fait, donc il est un peu optimiste.
Si tous les scores sont 1 (ou tous 0), R n'est pas mesurable : on renvoie une borne, sans intervalle.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

C = math.log(10) / 400
R_MIN, R_MAX = 0.0, 4500.0  # bornes de la recherche


def expected_score(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent - rating) / 400))


def elo_from_score(score: float) -> float:
    """Écart Elo (joueur - adversaire) pour un score moyen donné, dans ]0, 1[."""
    return -400 * math.log10(1 / score - 1)


@dataclass(frozen=True)
class EloFit:
    rating: float  # échelle de l'adversaire (ici : UCI_Elo de Stockfish)
    se: float | None  # écart-type ; None si R n'est pas mesurable
    low: float | None  # borne basse à 95 % ; None si inconnue
    high: float | None
    n: int
    bound: str | None = None  # "lower" (tous gagnés), "upper" (tous perdus) ou None

    def describe(self) -> str:
        if self.bound == "lower":
            return f"≥ {self.rating:.0f} (toutes les parties gagnées : borne seulement)"
        if self.bound == "upper":
            return f"≤ {self.rating:.0f} (toutes les parties perdues : borne seulement)"
        return f"{self.rating:.0f} ± {1.96 * self.se:.0f} (intervalle à 95 % : {self.low:.0f} à {self.high:.0f})"


def fit_elo(games: list[tuple[float, float]]) -> EloFit:
    """`games` : liste de (niveau de l'adversaire, score du joueur : 1, 0,5 ou 0)."""
    if not games:
        raise ValueError("aucune partie")
    n = len(games)
    levels, scores = [g[0] for g in games], [g[1] for g in games]
    if all(s == 1.0 for s in scores):
        return EloFit(max(levels), None, None, None, n, "lower")
    if all(s == 0.0 for s in scores):
        return EloFit(min(levels), None, None, None, n, "upper")

    def residual(r: float) -> float:
        return sum(s - expected_score(r, lv) for lv, s in games)

    lo, hi = R_MIN, R_MAX  # residual décroît avec r
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if residual(mid) > 0 else (lo, mid)
    r = (lo + hi) / 2
    p = [expected_score(r, lv) for lv in levels]
    info = sum(q * (1 - q) for q in p)
    se_model = 1 / (C * math.sqrt(info))
    se_observed = math.sqrt(sum((s - q) ** 2 for s, q in zip(scores, p))) / (C * info)
    se = max(se_model, se_observed)
    return EloFit(r, se, r - 1.96 * se, r + 1.96 * se, n)


def difference(a: EloFit, b: EloFit) -> tuple[float, float]:
    """(b - a, écart-type de la différence) pour deux ajustements indépendants mesurables."""
    if a.se is None or b.se is None:
        raise ValueError("différence non mesurable : l'un des deux ajustements n'est qu'une borne")
    return b.rating - a.rating, math.hypot(a.se, b.se)


def per_level(games: list[tuple[float, float]]) -> list[dict]:
    """Tableau par niveau : victoires, nulles, défaites, score, écart Elo implicite (avec marge à 95 %)."""
    by_level: dict[float, list[float]] = defaultdict(list)
    for level, score in games:
        by_level[level].append(score)
    rows = []
    for level in sorted(by_level):
        s = by_level[level]
        n, mean = len(s), sum(s) / len(s)
        row = {"level": level, "n": n, "wins": s.count(1.0), "draws": s.count(0.5), "losses": s.count(0.0),
               "score": mean, "elo": None, "low": None, "high": None}
        if 0.0 < mean < 1.0 and n > 1:
            sd = math.sqrt(sum((x - mean) ** 2 for x in s) / (n - 1)) / math.sqrt(n)
            lo, hi = max(mean - 1.96 * sd, 1e-3), min(mean + 1.96 * sd, 1 - 1e-3)
            row.update(elo=level + elo_from_score(mean), low=level + elo_from_score(lo), high=level + elo_from_score(hi))
        rows.append(row)
    return rows
