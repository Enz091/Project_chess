"""Protection des calculs locaux longs contre le ralentissement de Windows sur ce portable (Intel hybride, 6 cœurs P + 8 cœurs E).

Constat (2026-10-23, journal « Mesure de référence répétée ») : par moments Windows range les processus d'un calcul long sur
les cœurs E (logiques 12 à 19 à ~90 %, cœurs P endormis), environ 3 fois plus lent (2,4-2,5 s par coup au lieu de 0,80), sans
rapport avec la batterie ni le capot. Désactiver explicitement le mode « efficacité » (EcoQoS, `ProcessPowerThrottling`) sur chaque
processus de la mesure a tenu 143 min. Ce module :

  - `disable_ecoqos(pid)` : désactive EcoQoS sur un processus (soi-même par défaut) ;
  - `protect()` : le désactive sur le processus courant et lance un thread qui le fait pour tous ses descendants (workers,
    Stockfish...), y compris ceux créés plus tard ;
  - `core_layout()` : cœurs P et E (classe d'efficacité du système) ; `PlacementProbe` / `measure_placement` : part du travail
    qui tourne sur les cœurs E pendant une fenêtre ; `placement_verdict` : alerte si les cœurs E dominent ;
  - `busy_processes()` : processus (hors processus de la mesure) qui occupent plus de N cœurs, pour avertir avant un départ
    (onglet de navigateur emballé : 8 cœurs le 2026-10-23).

Windows seulement : ailleurs (Kaggle, Linux) tout est sans effet (`False` / `None` / liste vide).
"""

from __future__ import annotations

import os
import struct
import sys
import threading
import time

IS_WINDOWS = sys.platform == "win32"
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_SET_INFORMATION = 0x0200
PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS
EXECUTION_SPEED = 0x1
E_SHARE_ALERT = 0.60  # part du travail sur les cœurs E au-delà de laquelle on alerte
MIN_BUSY_CORES = 1.5  # en dessous, trop peu de charge pour juger du placement


# --------------------------------------------------------------------------- appels système (Windows)


def _k32():
    import ctypes

    return ctypes.WinDLL("kernel32", use_last_error=True)


class _ThrottlingState:  # PROCESS_POWER_THROTTLING_STATE
    def __new__(cls):
        import ctypes

        class State(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

        s = State()
        s.Version = 1
        return s


def disable_ecoqos(pid: int | None = None) -> bool:
    """Désactive EcoQoS sur `pid` (défaut : le processus courant). Vrai si Windows a accepté. Effet de bord sans danger :
    le processus n'est plus candidat au bridage d'efficacité."""
    if not IS_WINDOWS:
        return False
    try:
        import ctypes

        k32 = _k32()
        k32.OpenProcess.restype = ctypes.c_void_p
        pid = os.getpid() if pid is None else int(pid)
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SET_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            state = _ThrottlingState()
            state.ControlMask, state.StateMask = EXECUTION_SPEED, 0  # contrôle pris, bridage désactivé
            return bool(k32.SetProcessInformation(ctypes.c_void_p(handle), PROCESS_POWER_THROTTLING,
                                                  ctypes.byref(state), ctypes.sizeof(state)))
        finally:
            k32.CloseHandle(ctypes.c_void_p(handle))
    except (OSError, AttributeError):
        return False


def ecoqos_state(pid: int | None = None) -> tuple[int, int] | None:
    """(ControlMask, StateMask) lus pour `pid` ; (1, 0) = bridage désactivé explicitement ; None si illisible."""
    if not IS_WINDOWS:
        return None
    try:
        import ctypes

        k32 = _k32()
        k32.OpenProcess.restype = ctypes.c_void_p
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, os.getpid() if pid is None else int(pid))
        if not handle:
            return None
        try:
            state = _ThrottlingState()
            if not k32.GetProcessInformation(ctypes.c_void_p(handle), PROCESS_POWER_THROTTLING, ctypes.byref(state),
                                             ctypes.sizeof(state)):
                return None
            return int(state.ControlMask), int(state.StateMask)
        finally:
            k32.CloseHandle(ctypes.c_void_p(handle))
    except (OSError, AttributeError):
        return None


def process_table() -> list[tuple[int, int, str]]:
    """[(pid, pid du parent, nom)] de tous les processus (instantané Toolhelp)."""
    if not IS_WINDOWS:
        return []
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]

    k32 = _k32()
    k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    out = []
    if snap in (None, ctypes.c_void_p(-1).value):
        return out
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(Entry)
        ok = k32.Process32FirstW(ctypes.c_void_p(snap), ctypes.byref(entry))
        while ok:
            out.append((int(entry.th32ProcessID), int(entry.th32ParentProcessID), entry.szExeFile))
            ok = k32.Process32NextW(ctypes.c_void_p(snap), ctypes.byref(entry))
    finally:
        k32.CloseHandle(ctypes.c_void_p(snap))
    return out


def descendants(root: int | None = None, table: list | None = None) -> list[int]:
    """Processus descendants de `root` (défaut : le processus courant), enfants, petits-enfants..."""
    table = process_table() if table is None else table
    root = os.getpid() if root is None else root
    kids: dict[int, list[int]] = {}
    for pid, ppid, _ in table:
        kids.setdefault(ppid, []).append(pid)
    out, todo = [], [root]
    while todo:
        for child in kids.get(todo.pop(), []):
            if child != root and child not in out:
                out.append(child)
                todo.append(child)
    return out


# --------------------------------------------------------------------------- protection


class _Guard:
    def __init__(self, interval: float):
        self.interval, self.done, self.stop = interval, {}, threading.Event()
        self.thread = threading.Thread(target=self.run, name="cpu_guard", daemon=True)

    def tick(self) -> None:
        for pid in descendants():
            if pid not in self.done:
                self.done[pid] = disable_ecoqos(pid)

    def run(self) -> None:
        while not self.stop.is_set():
            self.tick()
            self.stop.wait(self.interval)


_GUARD: _Guard | None = None


def protect(interval: float = 3.0) -> dict:
    """À appeler au début d'un script de calcul long : EcoQoS désactivé sur ce processus, puis sur chaque descendant
    (thread daemon, toutes les `interval` s). Idempotent. Renvoie {"ecoqos_disabled": bool, "watching": bool}."""
    global _GUARD
    if not IS_WINDOWS:
        return {"ecoqos_disabled": False, "watching": False}
    ok = disable_ecoqos()
    if _GUARD is None:
        _GUARD = _Guard(interval)
        _GUARD.tick()
        _GUARD.thread.start()
    return {"ecoqos_disabled": ok, "watching": True}


def stop_protection() -> None:
    global _GUARD
    if _GUARD is not None:
        _GUARD.stop.set()
        _GUARD = None


def protected_children() -> dict[int, bool]:
    """pid -> EcoQoS désactivé avec succès, pour les descendants déjà traités (diagnostic, tests)."""
    return dict(_GUARD.done) if _GUARD else {}


# --------------------------------------------------------------------------- cœurs P / E


def core_layout() -> dict:
    """{"p": [logiques P], "e": [logiques E], "hybrid": bool} d'après la classe d'efficacité du système
    (GetSystemCpuSetInformation : la classe la plus élevée = cœurs P). Hors Windows ou en cas d'échec : listes vides."""
    empty = {"p": [], "e": [], "hybrid": False}
    if not IS_WINDOWS:
        return empty
    try:
        import ctypes

        k32 = _k32()
        needed = ctypes.c_ulong(0)
        k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(needed), None, 0)
        buf = ctypes.create_string_buffer(max(needed.value, 8))
        if not k32.GetSystemCpuSetInformation(buf, needed.value, ctypes.byref(needed), None, 0):
            return empty
        return layout_from_cpu_sets(buf.raw[:needed.value])
    except (OSError, AttributeError):
        return empty


def layout_from_cpu_sets(raw: bytes) -> dict:
    """Décode le tampon de GetSystemCpuSetInformation (fonction pure, testée)."""
    cpus, off = [], 0
    while off + 24 <= len(raw):
        size, typ = struct.unpack_from("<II", raw, off)
        if size < 24:
            break
        if typ == 0:  # CpuSetInformation : Id(4) Group(2) LogicalProcessorIndex(1) CoreIndex(1) LLC(1) Numa(1) EfficiencyClass(1)
            _id, group, logical, _core, _llc, _numa, eff = struct.unpack_from("<IHBBBBB", raw, off + 8)
            cpus.append((group * 64 + logical, eff))
        off += size
    classes = sorted({e for _, e in cpus})
    if len(classes) < 2:
        return {"p": sorted(c for c, _ in cpus), "e": [], "hybrid": False}
    top = classes[-1]
    return {"p": sorted(c for c, e in cpus if e == top), "e": sorted(c for c, e in cpus if e != top), "hybrid": True}


def _cpu_times() -> list[tuple[int, int]] | None:
    """[(occupé, total)] en unités de 100 ns, par processeur logique (NtQuerySystemInformation, classe 8)."""
    if not IS_WINDOWS:
        return None
    import ctypes

    n = os.cpu_count() or 1
    size = 48  # SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION : 5 x 8 octets + 4 + remplissage
    buf = ctypes.create_string_buffer(size * n)
    ret = ctypes.c_ulong(0)
    status = ctypes.windll.ntdll.NtQuerySystemInformation(8, buf, size * n, ctypes.byref(ret))
    if status != 0 or ret.value < size * n:
        return None
    out = []
    for i in range(n):
        idle, kernel, user = struct.unpack_from("<qqq", buf.raw, i * size)  # le temps « noyau » comprend le temps oisif
        total = kernel + user
        out.append((total - idle, total))
    return out


def measure_placement(seconds: float = 8.0, layout: dict | None = None, sleep=time.sleep) -> dict | None:
    """Occupation moyenne (en nombre de cœurs) des cœurs P et E sur `seconds` secondes."""
    layout = layout or core_layout()
    if not layout["hybrid"]:
        return None
    a = _cpu_times()
    if a is None:
        return None
    sleep(seconds)
    b = _cpu_times()
    if b is None:
        return None
    busy = []
    for (b0, t0), (b1, t1) in zip(a, b):
        busy.append((b1 - b0) / (t1 - t0) if t1 > t0 else 0.0)
    return placement_from_busy(busy, layout)


def placement_from_busy(busy: list[float], layout: dict) -> dict:
    p = sum(busy[i] for i in layout["p"] if i < len(busy))
    e = sum(busy[i] for i in layout["e"] if i < len(busy))
    total = p + e
    return {"p_cores_busy": round(p, 2), "e_cores_busy": round(e, 2), "e_share": round(e / total, 3) if total else 0.0,
            "total_cores_busy": round(total, 2)}


def placement_verdict(m: dict | None) -> tuple[str, str]:
    """(« ok » | « E_DOMINANT » | « peu chargé » | « inconnu », texte)."""
    if not m:
        return "inconnu", "placement P/E non mesuré (hors Windows ou processeur non hybride)"
    text = (f"cœurs P occupés : {m['p_cores_busy']:.1f}, cœurs E : {m['e_cores_busy']:.1f} "
            f"(part sur les cœurs E : {m['e_share']:.0%})")
    if m["total_cores_busy"] < MIN_BUSY_CORES:
        return "peu chargé", text + " : trop peu de charge pour juger"
    if m["e_share"] > E_SHARE_ALERT:
        return "E_DOMINANT", text + " : LE TRAVAIL TOURNE SURTOUT SUR LES CŒURS E (≈ 3 fois plus lent)"
    return "ok", text


class PlacementProbe:
    """Mesure le placement P/E dans un thread, `delay` s après le démarrage (le temps que le calcul monte en charge)."""

    def __init__(self, delay: float = 20.0, window: float = 8.0):
        self.delay, self.window, self.result = delay, window, None
        self.thread = threading.Thread(target=self._run, name="placement_probe", daemon=True)

    def _run(self) -> None:
        time.sleep(self.delay)
        self.result = measure_placement(self.window)

    def start(self) -> PlacementProbe:
        if IS_WINDOWS:
            self.thread.start()
        return self

    def finish(self, timeout: float = 30.0) -> dict | None:
        if self.thread.is_alive():
            self.thread.join(timeout)
        return self.result


# --------------------------------------------------------------------------- processus concurrents


def _process_cpu_seconds(pid: int) -> float | None:
    import ctypes
    from ctypes import wintypes

    k32 = _k32()
    k32.OpenProcess.restype = ctypes.c_void_p
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        c, e, kern, user = (wintypes.FILETIME() for _ in range(4))
        if not k32.GetProcessTimes(ctypes.c_void_p(h), ctypes.byref(c), ctypes.byref(e), ctypes.byref(kern),
                                   ctypes.byref(user)):
            return None
        ticks = lambda f: (f.dwHighDateTime << 32) | f.dwLowDateTime  # noqa: E731
        return (ticks(kern) + ticks(user)) / 1e7
    finally:
        k32.CloseHandle(ctypes.c_void_p(h))


def busy_processes(window: float = 3.0, threshold: float = 2.0, exclude_tree: bool = True, sleep=time.sleep) -> list[dict]:
    """Processus (hors le processus courant et ses descendants si `exclude_tree`) qui occupent plus de `threshold` cœurs en
    moyenne sur `window` s, et noms d'exécutable dont le total dépasse le seuil (navigateur à nombreux processus).
    [{"name", "cores", "pids"}] trié par charge décroissante."""
    if not IS_WINDOWS:
        return []
    table = process_table()
    names = {pid: name for pid, _, name in table}
    skip = {os.getpid(), 0, 4}
    if exclude_tree:
        skip |= set(descendants(table=table))
    before = {}
    for pid in names:
        if pid not in skip:
            t = _process_cpu_seconds(pid)
            if t is not None:
                before[pid] = t
    t0 = time.perf_counter()
    sleep(window)
    elapsed = time.perf_counter() - t0
    per_process = {}
    for pid, t in before.items():
        after = _process_cpu_seconds(pid)
        if after is not None and after >= t:
            per_process[pid] = (after - t) / elapsed
    return cores_by_name(per_process, names, threshold)


def cores_by_name(per_process: dict[int, float], names: dict[int, str], threshold: float) -> list[dict]:
    """Regroupe par nom d'exécutable (fonction pure, testée) : garde les noms dont la somme dépasse `threshold` cœurs ou
    dont un processus seul le dépasse."""
    groups: dict[str, dict] = {}
    for pid, cores in per_process.items():
        g = groups.setdefault(names.get(pid, "?"), {"name": names.get(pid, "?"), "cores": 0.0, "max_single": 0.0, "pids": []})
        g["cores"] += cores
        g["max_single"] = max(g["max_single"], cores)
        g["pids"].append(pid)
    out = [{"name": g["name"], "cores": round(g["cores"], 2), "max_single": round(g["max_single"], 2),
            "pids": sorted(g["pids"])} for g in groups.values() if g["cores"] > threshold or g["max_single"] > threshold]
    return sorted(out, key=lambda g: -g["cores"])


def describe_busy(found: list[dict], threshold: float = 2.0) -> str:
    if not found:
        return f"aucun autre programme n'occupe plus de {threshold:g} cœurs"
    return "; ".join(f"{g['name']} ≈ {g['cores']:.1f} cœurs ({len(g['pids'])} processus)" for g in found[:4])
