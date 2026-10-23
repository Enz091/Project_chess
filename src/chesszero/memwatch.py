"""Suivi de la mémoire (RAM) du système pendant une session de boucle (étape 7e).

Sur Kaggle (~31 Go libres), l'entraînement charge les données Lichess en RAM (1,8 Go) et le self-play lance 3
processus : mesuré sur le PC, ~3,5 Go (entraînement) et ~1,2 Go par processus de self-play (voir le journal). Cette
surveillance relève la mémoire DISPONIBLE du système toute entière (tous les processus, sous-processus compris) toutes
les 30 s, garde le minimum, le minimum par phase de la boucle, et note chaque relevé sous le seuil d'alerte (4 Go) :
le rapport de session le montre, pour confirmer la marge sur Kaggle.
"""

from __future__ import annotations

import sys
import threading
import time

GB = 1e9


def read_memory_gb() -> tuple[float, float] | None:
    """(mémoire disponible, mémoire totale) en Go : /proc/meminfo (Linux, Kaggle) ou GlobalMemoryStatusEx (Windows)."""
    try:
        if sys.platform.startswith("linux"):
            info = {}
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    key, _, rest = line.partition(":")
                    if key in ("MemTotal", "MemAvailable"):
                        info[key] = int(rest.split()[0]) * 1024
            return info["MemAvailable"] / GB, info["MemTotal"] / GB
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total_phys", ctypes.c_ulonglong),
                            ("avail_phys", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong),
                            ("avail_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                            ("avail_virtual", ctypes.c_ulonglong), ("avail_extended", ctypes.c_ulonglong)]

            s = Status()
            s.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return s.avail_phys / GB, s.total_phys / GB
    except (OSError, KeyError, ValueError, IndexError, AttributeError):
        pass
    return None


class MemoryWatch:
    """Relevés périodiques dans un thread (daemon). `phase` : fonction qui donne la phase en cours (nom court)."""

    MAX_ALERTS = 20

    def __init__(self, interval: float = 30.0, alert_gb: float = 4.0, reader=read_memory_gb, phase=lambda: None,
                 clock=time.time):
        self.interval, self.alert_gb, self.reader, self.phase, self.clock = interval, alert_gb, reader, phase, clock
        self.samples = 0
        self.total_gb = None
        self.first_gb = None
        self.min_gb = None
        self.min_phase = None
        self.min_time = None
        self.by_phase: dict[str, float] = {}
        self.alerts: list[dict] = []
        self.alert_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def sample(self) -> float | None:
        """Un relevé (appelable directement dans les tests)."""
        got = self.reader()
        if got is None:
            return None
        available, total = got
        phase = self.phase() or "—"
        with self._lock:
            self.samples += 1
            self.total_gb = total
            if self.first_gb is None:
                self.first_gb = available
            if self.min_gb is None or available < self.min_gb:
                self.min_gb, self.min_phase, self.min_time = available, phase, self.clock()
            if phase not in self.by_phase or available < self.by_phase[phase]:
                self.by_phase[phase] = available
            if available < self.alert_gb:
                self.alert_count += 1
                if len(self.alerts) < self.MAX_ALERTS:
                    self.alerts.append({"time": self.clock(), "phase": phase, "available_gb": round(available, 2)})
        return available

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sample()

        def loop():
            while not self._stop.wait(self.interval):
                self.sample()

        self._thread = threading.Thread(target=loop, name="memwatch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self.sample()  # relevé final

    def summary(self) -> dict:
        with self._lock:
            return {"interval_seconds": self.interval, "alert_gb": self.alert_gb, "samples": self.samples,
                    "total_gb": None if self.total_gb is None else round(self.total_gb, 2),
                    "first_available_gb": None if self.first_gb is None else round(self.first_gb, 2),
                    "min_available_gb": None if self.min_gb is None else round(self.min_gb, 2),
                    "min_phase": self.min_phase, "min_time": self.min_time,
                    "by_phase": {k: round(v, 2) for k, v in self.by_phase.items()},
                    "alert_samples": self.alert_count, "alerts": list(self.alerts)}


def describe_memory(m: dict | None) -> str:
    if not m or not m.get("samples") or m.get("min_available_gb") is None:
        return "mémoire : non relevée"
    phases = ", ".join(f"{k} {v:.1f}" for k, v in sorted(m["by_phase"].items()))
    text = (f"mémoire (RAM du système, relevée toutes les {m['interval_seconds']:.0f} s, {m['samples']} relevés) : "
            f"disponible au départ {m['first_available_gb']:.1f} Go sur {m['total_gb']:.1f} Go ; minimum "
            f"{m['min_available_gb']:.1f} Go pendant la phase « {m['min_phase']} » ; minimum par phase (Go) : {phases}")
    if m["alert_samples"]:
        text += f" ; {m['alert_samples']} relevé(s) sous {m['alert_gb']:g} Go"
    return text
