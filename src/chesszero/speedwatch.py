"""Surveillance de la vitesse de l'IA pendant une mesure contre Stockfish (`match_stockfish.py`).

Pourquoi : Stockfish joue AU TEMPS (300 ms par coup) alors que l'IA joue à simulations fixes. Si le processeur est
bridé en cours de mesure, Stockfish s'affaiblit (moins de profondeur) et l'IA non : l'Elo mesuré est gonflé. Vérifier
la vitesse au début ne suffit pas, le bridage peut commencer plus tard (nuit du 2026-10-22 : secteur, mode
« meilleures performances », mais temps par coup de l'IA à 2,48 s au lieu de 0,80 s du premier au dernier coup).

Principe : le temps par coup de l'IA (`ai_sec_per_move`, moyenne par partie) est suivi dans l'ordre où les parties
finissent. Moyenne glissante sur les `window` (20) dernières parties, comparée à la vitesse de RÉFÉRENCE (bench de
départ). Alerte quand l'écart relatif dépasse `tolerance` (25 %), dans un sens ou dans l'autre.

Limite connue : la mesure de référence v2 `kaggle_sup` a elle-même un épisode bref (parties 518-537, moyenne
glissante +42 %, quatre parties simultanées à 1,9-2,3 s par coup) : une alerte brève n'invalide pas une mesure, d'où
le verdict à trois états de `verdict()`.
"""

from __future__ import annotations

from collections import deque

NOMINAL_SECONDS_PER_MOVE = 0.80  # mesure v2 de référence (kaggle_sup, 2026-10-20), PC sur secteur
BRIEF_SHARE = 0.05  # alerte sur au plus 5 % des parties : épisode bref


def classify_power(ac_line: int, flag: int, percent: int) -> dict | None:
    """État d'alimentation d'après GetSystemPowerStatus : ACLineStatus (0 batterie, 1 secteur, 255 inconnu),
    BatteryFlag (8 = en charge, 128 = pas de batterie), BatteryLifePercent (255 = inconnu)."""
    pct = None if percent == 255 else int(percent)
    if ac_line == 255:
        return {"percent": pct, "state": "inconnu"}
    if ac_line == 0:
        return {"percent": pct, "state": "sur batterie"}
    if flag != 255 and flag & 128:
        return {"percent": None, "state": "secteur (pas de batterie)"}
    if flag != 255 and flag & 8:
        return {"percent": pct, "state": "en charge"}
    if pct is not None and pct >= 100:
        return {"percent": pct, "state": "chargée"}
    return {"percent": pct, "state": "secteur sans charge"}  # branchée mais ne charge pas (limite de charge...)


def read_power() -> dict | None:
    """{"percent", "state"} maintenant (Windows, appel système direct : instantané) ; None ailleurs ou en cas d'échec."""
    import sys

    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                        ("saver", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full_life", ctypes.c_ulong)]

        s = Status()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return None
        return classify_power(s.ac, s.flag, s.percent)
    except (OSError, AttributeError):
        return None


def _power_brief(power: dict | None) -> dict | None:
    return None if not power else {"percent": power.get("percent"), "state": power.get("state")}


def describe_power(state: dict) -> str:
    """Résumé de l'alimentation au fil de la mesure (pour le résumé final et le rapport)."""
    log = state.get("power")
    if not log or not log.get("first"):
        return "alimentation non relevée"
    f, last = log["first"], log["last"]
    text = (f"alimentation : départ {f['percent']} % ({f['state']}), fin {last['percent']} % ({last['state']}), "
            f"{len(log['changes'])} état(s) successif(s) : " + " -> ".join(
                f"{c['state']} {c['percent']} % (partie {c['game']})" for c in log["changes"][:8]))
    for e in state.get("episodes", [])[:4]:
        ps = e.get("power_start")
        if ps:
            text += f" ; épisode des parties {e['start']}-{e['end'] or state['games']} commencé {ps['state']} à {ps['percent']} %"
    return text


class SpeedWatch:
    """`reference` : secondes par coup du bench de départ. `state` : dictionnaire enregistré dans le fichier de
    résultats (clé « speed »), mis à jour à chaque partie ; une reprise de la mesure le continue."""

    def __init__(self, reference: float, window: int = 20, tolerance: float = 0.25, state: dict | None = None):
        if reference <= 0:
            raise ValueError("vitesse de référence invalide")
        self.window, self.tolerance = window, tolerance
        st = state or {}
        self.state = st
        st.setdefault("reference_s_per_move", reference)
        st.setdefault("nominal_s_per_move", NOMINAL_SECONDS_PER_MOVE)
        st.update({"window": window, "tolerance": tolerance})
        st.setdefault("benches", [])
        st.setdefault("games", 0)  # parties suivies (toutes reprises comprises)
        st.setdefault("alert_games", 0)
        st.setdefault("max_deviation", 0.0)
        st.setdefault("worst", None)  # {"game": n, "rolling": s, "deviation": d}
        st.setdefault("episodes", [])  # [{"start": n, "end": n, "max_deviation": d}] ; end None = en cours
        st.setdefault("series", [])  # [[n, moyenne glissante]] une valeur toutes les `window` parties
        self.recent = deque(st.get("recent", []), maxlen=window)
        st["recent"] = list(self.recent)

    @property
    def reference(self) -> float:
        return self.state["reference_s_per_move"]

    def deviation(self, rolling: float) -> float:
        return rolling / self.reference - 1.0

    def note_power(self, power: dict | None) -> None:
        """Mémorise l'alimentation de la partie qui vient de finir : premier relevé, dernier relevé et changements
        d'état (en charge / chargée / sur batterie...), pour confirmer ou écarter un bridage pendant la recharge."""
        if not power:
            return
        st, n = self.state, self.state["games"]
        rec = {"game": n, "percent": power.get("percent"), "state": power.get("state")}
        log = st.setdefault("power", {"first": None, "last": None, "changes": []})
        if log["first"] is None:
            log["first"] = rec
        if log["last"] is None or log["last"]["state"] != rec["state"]:
            log["changes"].append(rec)
        log["last"] = rec

    def add(self, seconds_per_move: float, power: dict | None = None) -> dict:
        """Enregistre une partie finie (`power` : alimentation relevée à cet instant, voir `read_power`). Renvoie
        {"rolling", "deviation", "alert", "event", "power"} où `event` vaut "start" (début d'un épisode), "end" (retour
        à la normale) ou None."""
        st = self.state
        st["games"] += 1
        self.note_power(power)
        self.recent.append(float(seconds_per_move))
        st["recent"] = list(self.recent)
        rolling = sum(self.recent) / len(self.recent)
        dev = self.deviation(rolling)
        # tant que la fenêtre n'est pas pleine, pas d'alerte : trop peu de parties pour une moyenne
        alert = len(self.recent) >= self.window and abs(dev) > self.tolerance
        event = None
        open_ep = st["episodes"][-1] if st["episodes"] and st["episodes"][-1]["end"] is None else None
        if alert:
            st["alert_games"] += 1
            if open_ep is None:
                st["episodes"].append({"start": st["games"], "end": None, "max_deviation": dev,
                                       "power_start": _power_brief(power)})
                event = "start"
            elif abs(dev) > abs(open_ep["max_deviation"]):
                open_ep["max_deviation"] = dev
            if abs(dev) > abs(st["max_deviation"]):
                st["max_deviation"] = dev
                st["worst"] = {"game": st["games"], "rolling": rolling, "deviation": dev}
        elif open_ep is not None:
            open_ep["end"] = st["games"]
            open_ep["power_end"] = _power_brief(power)
            event = "end"
        if st["games"] % self.window == 0:  # [partie, moyenne glissante, % batterie, état de charge]
            st["series"].append([st["games"], round(rolling, 4), (power or {}).get("percent"), (power or {}).get("state")])
        return {"rolling": rolling, "deviation": dev, "alert": alert, "event": event, "power": power}

    def bench(self, seconds_per_move: float, games: int) -> str | None:
        """Enregistre un bench de départ (un par lancement) ; avertit s'il s'écarte de la référence ou de la vitesse
        nominale de plus de la tolérance."""
        self.state["benches"].append({"s_per_move": round(seconds_per_move, 4), "games": games})
        problems = []
        if abs(seconds_per_move / NOMINAL_SECONDS_PER_MOVE - 1) > self.tolerance:
            problems.append(f"{seconds_per_move:.2f} s par coup contre {NOMINAL_SECONDS_PER_MOVE:.2f} s à la mesure "
                            f"de référence ({seconds_per_move / NOMINAL_SECONDS_PER_MOVE - 1:+.0%})")
        if abs(seconds_per_move / self.reference - 1) > self.tolerance:
            problems.append(f"{seconds_per_move:.2f} s par coup contre {self.reference:.2f} s au bench de départ de "
                            "cette mesure")
        return "; ".join(problems) or None


def verdict(state: dict | None) -> tuple[str, str]:
    """(« stable » | « perturbée brièvement » | « NON STABLE » | « NON COMPARABLE » | « non suivie », texte du résumé
    final). « NON COMPARABLE » : stable dans le temps mais d'une vitesse de départ très différente de la mesure de
    référence (nuit du 2026-10-22 : 2,59 s par coup contre 0,80 s, stable... mais Stockfish affaibli)."""
    state_name, text = _stability(state)
    if state and state.get("games"):
        ref, nominal, tol = state["reference_s_per_move"], state["nominal_s_per_move"], state["tolerance"]
        if abs(ref / nominal - 1) > tol:
            text += (f" ; ATTENTION : vitesse de départ {ref:.2f} s par coup, soit {ref / nominal - 1:+.0%} par rapport "
                     f"à la mesure de référence ({nominal:.2f} s) : le processeur n'allait pas à la même vitesse, "
                     "mesure NON COMPARABLE avec la référence")
            if state_name in ("stable", "perturbée brièvement"):
                state_name = "NON COMPARABLE"
        text += "\n" + describe_power(state)
    return state_name, text


def _stability(state: dict | None) -> tuple[str, str]:
    if not state or not state.get("games"):
        return "non suivie", "vitesse non suivie pendant cette mesure"
    n, bad = state["games"], state["alert_games"]
    ref = state["reference_s_per_move"]
    head = (f"vitesse de l'IA : référence {ref:.2f} s par coup (bench de départ, nominal "
            f"{state['nominal_s_per_move']:.2f}), moyenne glissante sur {state['window']} parties, tolérance "
            f"±{state['tolerance']:.0%}")
    eps = state["episodes"]
    ongoing = bool(eps) and eps[-1]["end"] is None
    if bad == 0:
        return "stable", f"{head} : mesure STABLE du début à la fin ({n} parties suivies, aucune alerte)"
    worst = state["worst"]
    detail = (f"{bad} parties sur {n} ({bad / n:.1%}) avec la moyenne glissante hors tolérance, {len(eps)} épisode(s) ; "
              f"pire écart {worst['deviation']:+.0%} ({worst['rolling']:.2f} s par coup) à la partie {worst['game']}")
    spans = ", ".join(f"parties {e['start']}-{e['end'] if e['end'] else n} ({e['max_deviation']:+.0%})" for e in eps[:6])
    if bad / n <= BRIEF_SHARE and not ongoing:
        return "perturbée brièvement", f"{head} : épisode(s) bref(s), mesure à regarder mais pas invalidée : {detail} [{spans}]"
    return "NON STABLE", (f"{head} : MESURE NON STABLE, le processeur a changé de vitesse en cours de mesure ; "
                          f"ne pas la combiner avec la référence : {detail} [{spans}]"
                          f"{' ; encore en cours à la fin' if ongoing else ''}")
