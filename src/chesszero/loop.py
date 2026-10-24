"""Boucle AlphaZero (étape 7d) : orchestrateur d'une session (Kaggle ou locale), budget de temps, rapport.

Une génération : self-play avec le réseau accepté jusqu'à `new_positions_per_generation` nouvelles positions ->
tour d'entraînement (garde-fous) -> candidat -> arène contre l'accepté -> arène contre l'ancre `kaggle_sup` (si
elle est gagnée et que l'accepté n'est plus le réseau de départ) -> verdict (`SelfPlayTrainer.commit_arena`).

Principes :
  - LA PHASE N'EST JAMAIS STOCKÉE COMME VÉRITÉ : à chaque pas, elle est déduite des fichiers durables
    (`selfplay_train.read_train_state` = source de vérité pour le candidat, l'accepté, les compteurs et l'arrêt ;
    `result.json` des arènes ; shards du tampon). Une coupure dans n'importe quelle phase se reprend donc d'elle-même.
  - `loop/loop_state.json` ne contient que ce qui appartient à l'orchestrateur : sessions (numéro, début, fin,
    budget, code), contrôles d'arène faits, durées mesurées (estimations), événements des phases, échecs.
  - chaque phase est un sous-processus (scripts/selfplay.py, train_selfplay.py, arena.py) ; un échec est rejoué une
    fois (la phase se reprend d'elle-même), l'arrêt n'a lieu qu'au deuxième échec de la même phase ; les arrêts
    décisionnels (G4, stagnation, ancre, contrôle (b)) sont immédiats ; tout arrêt : rapport puis fin de session.
  - budget : un tour ne démarre que s'il reste entraînement + arènes estimés + `start_margin_minutes` ; une arène,
    sa durée estimée + la marge ; sinon le temps restant sert au self-play (données toujours utiles) et le travail
    attend la session suivante ; `end_margin_minutes` à la fin pour le rapport et la sauvegarde.
  - entrée : la sortie de la session précédente (`--resume-from`), soit une sortie de SELF-PLAY SEUL (étape 6 :
    `selfplay/` seulement -> démarrage de la boucle), soit une sortie de la boucle (`selfplay/` + `sp_train/` +
    `loop/`) ; `--expect-session N` doit valoir le numéro de la dernière session de boucle (0 pour un démarrage).
"""

from __future__ import annotations

import csv
import io
import json
import math
import shutil
import statistics
import time
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from chesszero.mcts import CONFIG_DIR

ROOT = Path(__file__).resolve().parents[2]
LOOP_FORMAT = 1


@dataclass(frozen=True)
class LoopConfig:
    """Section [loop] de configs/<nom>.toml."""

    new_positions_per_generation: int = 100_000
    train_start_fraction: float = 0.9  # un tour démarre dès 90 % de l'objectif de positions nouvelles
    start_margin_minutes: float = 15.0  # marge avant un tour ou une arène
    end_margin_minutes: float = 10.0  # fin de session : rapport, sauvegarde de la sortie
    min_chunk_minutes: float = 25.0  # tranche de self-play minimale (même si l'objectif de positions est atteint)
    chunk_stop_new_games_minutes: float = 12.0  # tranche : plus de nouvelles parties avant sa durée maximale
    control_a_pairs: int = 10  # contrôle (a) : réseau de départ contre lui-même (avertissement seulement)
    control_a_low: float = 0.40
    control_a_high: float = 0.60
    control_b_pairs: int = 20  # contrôle (b) : 200 contre `control_b_weak_simulations` simulations (bloquant)
    control_b_weak_simulations: int = 50
    control_b_min_score: float = 0.70
    # valeurs de départ des estimations (remplacées par la moyenne des dernières mesures), d'après la session de
    # self-play seul sur Kaggle du 2026-10-22 (4 296 simulations/s, 97,5 coups cherchés par partie)
    est_positions_per_minute: float = 1290.0  # 4 296 / 200 x 60
    est_train_fixed_minutes: float = 5.0
    est_train_minutes_per_step: float = 0.006
    est_arena_fixed_minutes: float = 1.0
    est_simulations_per_second: float = 4296.0  # débit mesuré (3 processus x 64 parties)
    est_arena_searched_moves_per_game: float = 97.5  # coups cherchés par partie (self-play : 370 840 / 3 803)
    est_arena_slowdown: float = 1.10  # arène : deux lots de demi-taille par itération, fin des longues parties
    estimate_window: int = 3  # estimation = moyenne des 3 dernières mesures
    max_phase_failures: int = 2  # arrêt au 2e échec de la même phase
    v2_anchor_gain: float = 50.0  # « Actions » : mesure v2 si l'Elo contre l'ancre (lissé) a gagné plus que ça...
    v2_max_generations: int = 10  # ... ou au plus tard toutes les 10 générations acceptées
    false_resign_alert: float = 0.05
    early_repetition_alert: float = 0.05  # part des parties finies par répétition avant le demi-coup 60
    repetition_material_reference: float = 0.145  # parties finies par répétition avec >= 20 points de matériel
    repetition_material_alert_points: float = 5.0  # alerte au-delà de la référence + 5 points
    throughput_drop_alert: float = 0.15
    reading_alert: float = 0.30  # « Actions » : lecture des lots (Lichess + tampon) > 30 % du temps d'un tour
    load_alert_minutes: float = 8.0  # « Actions » : chargement initial des données Lichess plus long que ça
    ram_interval_seconds: float = 30.0  # relevé de la mémoire disponible du système (voir memwatch.py)
    ram_alert_gb: float = 4.0  # « Actions » : mémoire disponible tombée sous ce seuil

    @classmethod
    def from_toml(cls, path: str | Path) -> LoopConfig:
        with open(path, "rb") as f:
            section = tomllib.load(f).get("loop", {})
        unknown = set(section) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"clés inconnues dans [loop] : {sorted(unknown)}")
        return cls(**section)

    @classmethod
    def named(cls, name: str) -> LoopConfig:
        return cls.from_toml(CONFIG_DIR / f"{name}.toml")


class LoopStop(Exception):
    """Refus de démarrer (entrée incohérente, mauvaise session attendue, tampon trop petit)."""


# --------------------------------------------------------------------------- état de l'orchestrateur


def new_loop_state() -> dict:
    return {"format": LOOP_FORMAT, "sessions": [], "controls": {"done": False, "a": None, "b": None},
            "estimates": {"positions_per_minute": [], "train_fixed": [], "train_per_step": [], "arena_fixed": [],
                          "arena_per_game": []},
            "events": [], "failures": {}, "phase": None}


def read_loop_state(path: str | Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_loop_state(path: str | Path, state: dict) -> None:
    from chesszero.arena import write_json_atomic

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, state)


def last_session(state: dict | None) -> int:
    return max((s["n"] for s in state["sessions"]), default=0) if state else 0


def estimate(state: dict, key: str, default: float, window: int) -> float:
    values = state["estimates"].get(key, [])[-window:]
    return statistics.fmean(values) if values else default


def record_estimate(state: dict, key: str, value: float, keep: int = 20) -> None:
    if value is not None and math.isfinite(value) and value >= 0:
        state["estimates"].setdefault(key, []).append(float(value))
        state["estimates"][key] = state["estimates"][key][-keep:]


# --------------------------------------------------------------------------- entrée de la session


def classify_output(d: str | Path) -> dict:
    """Ce que contient une sortie de session : tampon (`selfplay/`), état d'entraînement, état de boucle."""
    from chesszero.selfplay import list_shards

    d = Path(d)
    shards = list_shards([d / "selfplay"]) if (d / "selfplay").is_dir() else []
    has_train = (d / "sp_train" / "train_state.pt").exists()
    has_loop = (d / "loop" / "loop_state.json").exists()
    if shards and not has_train and not has_loop and not (d / "sp_train").exists() and not (d / "loop").exists():
        kind = "selfplay_only"
    elif shards and has_train and has_loop:
        kind = "loop"
    elif not shards and not has_train and not has_loop:
        kind = "empty"
    else:
        kind = "inconsistent"
    return {"kind": kind, "shards": len(shards), "positions": sum(s["positions"] for s in shards),
            "has_train_state": has_train, "has_loop_state": has_loop,
            "has_sp_train_dir": (d / "sp_train").exists(), "has_loop_dir": (d / "loop").exists()}


def buffer_train_positions(buffer: str | Path, train_cfg) -> int:
    """Positions d'entraînement de la fenêtre du tampon (hors parties mises de côté), comme le compte l'entraîneur."""
    from chesszero.selfplay import buffer_window, list_shards
    from chesszero.selfplay_train import SelfPlayData

    shards = buffer_window(list_shards([buffer]), train_cfg.buffer_positions)
    return int(len(SelfPlayData(shards, train_cfg.heldout_fraction).train_rows)) if shards else 0


def check_expected_session(found: int, expected: int | None, kind: str) -> None:
    if expected is None:
        raise LoopStop("--expect-session est obligatoire (numéro de la dernière session de boucle ; 0 pour un "
                       "démarrage à partir d'une sortie de self-play seul)")
    if found != expected:
        what = ("une sortie de self-play seul (aucune session de boucle)" if kind == "selfplay_only"
                else f"la sortie de la session de boucle n°{found}")
        raise LoopStop(f"Mauvaise entrée : trouvé {what}, attendu la session n°{expected} (EXPECTED_SESSION = "
                       f"{expected}). Vérifier la version de sortie ajoutée en entrée ; si c'est voulu, mettre "
                       f"EXPECTED_SESSION = {found}.")


# --------------------------------------------------------------------------- phase et budget (fonctions pures)


def new_positions(buffer: str | Path, trained_shards) -> int:
    from chesszero.selfplay import list_shards

    done = set(trained_shards or [])
    return sum(s["positions"] for s in list_shards([buffer]) if s["name"] not in done)


def arena_result(path: str | Path) -> dict | None:
    p = Path(path) / "result.json"
    return json.loads(p.read_text(encoding="utf-8"))["summary"] if p.exists() else None


def next_phase(view: dict, gate: dict | None, anchor: dict | None, new: int, cfg: LoopConfig, accept_score: float,
               train_waiting: bool = False) -> str:
    """Phase suivante, déduite de l'état durable : "stop", "arena_gate", "arena_anchor", "verdict", "train",
    "selfplay". `gate` / `anchor` : résumés des arènes TERMINÉES du candidat en attente (None sinon)."""
    if view["status"] == "stop":
        return "stop"
    if view["candidate"] is not None:
        if gate is None:
            return "arena_gate"
        first = view["accepted"]["is_init"]  # génération 1 : l'arène contre l'accepté EST la mesure contre l'ancre
        if gate["score"] >= accept_score - 1e-9 and not first and anchor is None:
            return "arena_anchor"
        return "verdict"
    if view["round_in_progress"] is not None:
        return "train"
    if new >= train_threshold(cfg) and not train_waiting:
        return "train"
    return "selfplay"


def train_threshold(cfg: LoopConfig) -> int:
    """Positions nouvelles à partir desquelles un tour démarre : 90 % de l'objectif (une tranche qui s'arrête un peu
    avant l'objectif ne coûte pas une tranche de plus)."""
    return math.ceil(cfg.train_start_fraction * cfg.new_positions_per_generation)


def planned_steps(new: int, train_cfg, human_fraction: float) -> int:
    n_sp = train_cfg.batch_size - int(round(train_cfg.batch_size * human_fraction))
    return math.ceil(train_cfg.reuse * new / n_sp)


def default_arena_minutes_per_game(cfg: LoopConfig, simulations: int) -> float:
    """Durée d'une partie d'arène (en débit total) : coups cherchés x simulations / simulations par seconde."""
    return cfg.est_arena_searched_moves_per_game * simulations / cfg.est_simulations_per_second / 60 *         cfg.est_arena_slowdown


def estimates(state: dict, cfg: LoopConfig, arena_simulations: int = 200) -> dict:
    w = cfg.estimate_window
    per_game = default_arena_minutes_per_game(cfg, arena_simulations)
    return {"positions_per_minute": estimate(state, "positions_per_minute", cfg.est_positions_per_minute, w),
            "train_fixed": estimate(state, "train_fixed", cfg.est_train_fixed_minutes, w),
            "train_per_step": estimate(state, "train_per_step", cfg.est_train_minutes_per_step, w),
            "arena_fixed": estimate(state, "arena_fixed", cfg.est_arena_fixed_minutes, w),
            "arena_per_game": estimate(state, "arena_per_game", per_game, w)}


def arena_minutes(est: dict, games: int) -> float:
    return est["arena_fixed"] + est["arena_per_game"] * games


def round_minutes(est: dict, steps: int, gate_games: int, anchor_games: int) -> float:
    """Durée estimée d'une génération après le self-play : entraînement + arène A + arène ancre."""
    return est["train_fixed"] + est["train_per_step"] * steps + arena_minutes(est, gate_games) + \
        (arena_minutes(est, anchor_games) if anchor_games else 0.0)


def selfplay_chunk(remaining: float, missing_positions: int, est: dict, after_minutes: float,
                   cfg: LoopConfig) -> tuple[float, bool, int | None]:
    """(durée maximale de la tranche en minutes, dernière tranche ?, objectif de positions).

    Assez de temps pour produire les positions manquantes (au débit estimé) ET faire ensuite le tour et ses arènes
    (`after_minutes` + marge) : la tranche s'arrête sur l'OBJECTIF DE POSITIONS (le manque) : plus de nouvelles
    parties une fois le manque couvert et au moins `min_chunk_minutes` écoulées, les parties en cours se terminent ;
    sa durée maximale est le temps qui laisse encore la place au tour. Sinon : self-play jusqu'à la fin de la session
    (dernière tranche, sans objectif). Minutes <= 0 : plus rien à faire."""
    missing = max(int(missing_positions), 0)
    need = missing / max(est["positions_per_minute"], 1e-9)
    usable = remaining - cfg.end_margin_minutes
    if need + after_minutes + cfg.start_margin_minutes <= usable:
        cap = usable - after_minutes - cfg.start_margin_minutes
        return max(cap, cfg.min_chunk_minutes), False, missing
    return (usable if usable >= cfg.min_chunk_minutes else 0.0), True, None


# --------------------------------------------------------------------------- exécution des phases


class SubprocessExecutor:
    """Lance les scripts du dépôt ; garde la fin de la sortie de chaque lancement (journal des échecs)."""

    def __init__(self, work: Path, config: str, init: str, human: str | None, workers: int = 3, threads: int = 1,
                 device: str | None = None, syzygy: str | None = None, book: str | None = None,
                 games_per_worker: int | None = None, log=print):
        import sys

        self.python = sys.executable
        self.work, self.config, self.init, self.human = Path(work), config, init, human
        self.workers, self.threads, self.device, self.syzygy, self.book = workers, threads, device, syzygy, book
        self.games_per_worker, self.log = games_per_worker, log
        self.last_tail = ""

    def _run(self, script: str, args: list) -> int:
        import collections
        import subprocess

        cmd = [self.python, str(ROOT / "scripts" / script), *[str(a) for a in args]]
        self.log("$ " + " ".join(cmd))
        tail = collections.deque(maxlen=40)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line.rstrip("\n"))
        code = proc.wait()
        self.last_tail = "\n".join(tail)
        return code

    def _common(self) -> list:
        out = ["--config", self.config, "--workers", self.workers, "--threads", self.threads]
        if self.device:
            out += ["--device", self.device]
        return out

    def selfplay(self, minutes: float, checkpoint: str, generation: int, session: str, soft: float, hard: float,
                 target_positions: int | None = None, min_minutes: float | None = None) -> int:
        args = ["--out", self.work / "selfplay", "--checkpoint", checkpoint, "--max-minutes", f"{minutes:.3f}",
                "--generation", generation, "--session", session, "--stop-new-games-minutes", f"{soft:.3f}",
                "--hard-stop-minutes", f"{hard:.3f}", "--syzygy", self.syzygy or "",
                "--reference", self.init, *self._common()]  # --reference : indicateur d'accord (kaggle_sup)
        if target_positions is not None:  # tranche sur objectif de positions (voir selfplay_chunk)
            args += ["--target-positions", target_positions, "--min-minutes", f"{min_minutes or 0:.3f}"]
        if self.book is not None:
            args += ["--book", self.book]
        if self.games_per_worker:
            args += ["--games-per-worker", self.games_per_worker]
        return self._run("selfplay.py", args)

    def train(self, max_minutes: float) -> int:
        args = ["--out", self.work / "sp_train", "--buffer", self.work / "selfplay", "--human", self.human,
                "--init", self.init, "--config", self.config, "--max-minutes", f"{max_minutes:.3f}"]
        if self.device:
            args += ["--device", self.device]
        return self._run("train_selfplay.py", args)

    def arena(self, kind: str, seed: int, a: str, a_name: str, a_sims: int | None, b: str, b_name: str,
              b_sims: int | None, out: Path, pairs: int | None, max_minutes: float) -> int:
        args = ["--kind", kind, "--seed", seed, "--a", a, "--a-name", a_name, "--b", b, "--b-name", b_name,
                "--out", out, "--max-minutes", f"{max_minutes:.3f}", "--syzygy", self.syzygy or "", *self._common()]
        for opt, v in (("--a-sims", a_sims), ("--b-sims", b_sims), ("--pairs", pairs)):
            if v:
                args += [opt, v]
        if self.book is not None:
            args += ["--book", self.book]
        if self.games_per_worker:
            args += ["--games-per-worker", self.games_per_worker]
        return self._run("arena.py", args)

    def commit(self, cid: int, gate: Path, anchor: Path | None) -> int:
        args = ["--out", self.work / "sp_train", "--buffer", self.work / "selfplay", "--human", self.human or "x",
                "--init", self.init, "--config", self.config, "--commit-arena", cid, "--gate", gate]
        if anchor is not None:
            args += ["--anchor", anchor]
        return self._run("train_selfplay.py", args)


class CrashForTest(Exception):
    """Coupure simulée (tests) : levée à un point nommé de l'orchestrateur, comme un arrêt brutal de la session."""


class Orchestrator:
    """Une session de boucle. `executor` : `SubprocessExecutor` (réel) ou un faux exécuteur (tests).
    Codes attendus : selfplay 0 ; train 0 (candidat ou attente de données), 1 (rejet par les garde-fous),
    2 (arrêt), 3 (budget) ; arena 0 (complète), 3 (budget) ; commit 0 (accepté), 1 (rejeté), 2 (arrêt)."""

    OK_CODES = {"selfplay": {0}, "train": {0, 1, 2, 3}, "arena": {0, 3}, "commit": {0, 1, 2}}

    def __init__(self, work: str | Path, config: str, init: str, executor, minutes: float,
                 expect_session: int | None, resume_from: str | Path | None = None, crash_at: str | None = None,
                 clock=time.time, log=print, root: Path = ROOT, configs: dict | None = None, memwatch=None,
                 reset_line: str | None = None):
        from chesszero.arena import ArenaConfig
        from chesszero.selfplay import SelfPlayConfig
        from chesszero.selfplay_train import SelfPlayTrainConfig

        self.work, self.config, self.init = Path(work), config, str(init)
        configs = configs or {}  # tests : réglages remplacés ({"loop": LoopConfig(...), "train": ..., ...})
        self.loop_cfg = configs.get("loop") or LoopConfig.named(config)
        self.train_cfg = configs.get("train") or SelfPlayTrainConfig.named(config)
        self.arena_cfg = configs.get("arena") or ArenaConfig.named(config)
        self.sp_cfg = configs.get("selfplay") or SelfPlayConfig.named(config)
        self.executor, self.minutes, self.expect, self.resume_from = executor, minutes, expect_session, resume_from
        self.crash_at, self.clock, self.log, self.root = crash_at, clock, log, root
        self.reset_line = reset_line  # décision humaine : ligne d'entraînement remise sur le réseau accepté (note)
        self.start = clock()
        self.deadline = self.start + 60 * minutes
        self.state_path = self.work / "loop" / "loop_state.json"
        self.state: dict = {}
        self.session = 0
        self.train_waiting = False
        self.end_reason = None
        self.notes: list[str] = []  # informations de démarrage, avertissements (rapport)
        from chesszero.memwatch import MemoryWatch

        self.mem = memwatch or MemoryWatch(self.loop_cfg.ram_interval_seconds, self.loop_cfg.ram_alert_gb,
                                           phase=lambda: self.state.get("phase"))

    # ---------------------------------------------------------------- outils

    def remaining(self) -> float:
        return (self.deadline - self.clock()) / 60

    def crash_point(self, name: str) -> None:
        if self.crash_at == name:
            raise CrashForTest(name)

    def save(self) -> None:
        if self.state.get("sessions") and self.mem.samples:
            self.state["sessions"][-1]["ram"] = self.mem.summary()  # relevés de la session en cours
        write_loop_state(self.state_path, self.state)

    def event(self, phase: str, start: float, code: int | None, **detail) -> None:
        self.state["events"].append({"session": self.session, "phase": phase, "start": start, "end": self.clock(),
                                     "minutes": (self.clock() - start) / 60, "code": code, **detail})

    def view(self) -> dict:
        from chesszero.selfplay_train import read_train_state

        return read_train_state(self.work / "sp_train")

    def arena_dir(self, cid: int, which: str) -> Path:
        return self.work / "loop" / "arena" / f"cand_{cid:04d}_{which}"

    # ---------------------------------------------------------------- démarrage

    def validate_input(self) -> tuple[Path, dict, dict | None]:
        """Vérifie l'entrée SANS RIEN ÉCRIRE (mode `--check-inputs` et début de `prepare`) : type de sortie, session
        attendue, dossier de travail vide, taille du tampon (démarrage), cohérence de l'état d'entraînement et ancre
        (reprise). Renvoie (dossier source, infos, état de boucle) ; lève `LoopStop` avec le motif du refus."""
        from chesszero.arena import sha256_file
        from chesszero.selfplay_train import read_train_state

        src = Path(self.resume_from) if self.resume_from else self.work
        info = classify_output(src)
        if src != self.work and classify_output(self.work)["kind"] != "empty":
            raise LoopStop(f"{self.work} n'est pas vide : impossible d'y reprendre {src}")
        if info["kind"] == "inconsistent":
            raise LoopStop(f"Entrée incohérente dans {src} : tampon {info['shards']} shards, état d'entraînement "
                           f"{'présent' if info['has_train_state'] else 'absent'}, état de boucle "
                           f"{'présent' if info['has_loop_state'] else 'absent'} (attendu : self-play seul = "
                           "selfplay/ uniquement, ou boucle = selfplay/ + sp_train/ + loop/)")
        if info["kind"] == "empty":
            raise LoopStop(f"Aucun tampon de self-play dans {src} : lancer d'abord une session de self-play seul")
        loop_state = read_loop_state(src / "loop" / "loop_state.json") if info["kind"] == "loop" else None
        check_expected_session(last_session(loop_state), self.expect, info["kind"])
        if info["kind"] == "selfplay_only":
            n = buffer_train_positions(src / "selfplay", self.train_cfg)
            info["train_positions"] = n
            if n < self.train_cfg.min_buffer_positions:
                raise LoopStop(f"Tampon trop petit pour démarrer la boucle : {n:,} positions d'entraînement "
                               f"(hors parties mises de côté), minimum {self.train_cfg.min_buffer_positions:,}")
        else:
            view = read_train_state(src / "sp_train")
            if view.get("needs_migration") or view.get("format", 0) < 2:
                raise LoopStop("État d'entraînement d'un ancien format dans une sortie de boucle : incohérent")
            sha = sha256_file(self.init)
            if view.get("init_sha256") and view["init_sha256"] != sha:
                raise LoopStop(f"Le réseau de départ fourni ({sha[:12]}…) n'est pas celui de l'état "
                               f"({view['init_sha256'][:12]}…) : l'ancre doit rester la même")
            info["generation"], info["last_session"] = view.get("generation"), last_session(loop_state)
        return src, info, loop_state

    def check_inputs(self, human: str | None = None, syzygy: str | None = None) -> list[str]:
        """Mode `--check-inputs` (notebook : MODE = "check") : toutes les vérifications du démarrage, rien d'écrit.
        Renvoie les lignes à afficher ; lève `LoopStop` si la session serait refusée."""
        from chesszero.arena import sha256_file
        from chesszero.checkpoint import load_for_inference
        from chesszero.kaggle_inputs import describe_human
        from chesszero.memwatch import read_memory_gb

        src, info, _ = self.validate_input()
        what = ("sortie de SELF-PLAY SEUL : la boucle démarrera (génération 0, " f"{info['train_positions']:,} positions "
                "d'entraînement)" if info["kind"] == "selfplay_only" else
                f"sortie de la session de boucle n°{info['last_session']} (génération acceptée {info['generation']})")
        lines = [f"Entrée : {src} — {what}",
                 f"Tampon : {info['shards']} shards, {info['positions']:,} positions ; session attendue "
                 f"EXPECTED_SESSION = {self.expect} : conforme"]
        init = Path(self.init)
        if not init.exists():
            raise LoopStop(f"Réseau de départ absent : {init}")
        _, meta = load_for_inference(init, "cpu")
        lines.append(f"Réseau de départ : {init} (SHA-256 {sha256_file(init)[:12]}…, pas {meta.get('step')})")
        lines.append(f"Réglages : budget {self.minutes:g} min, {self.loop_cfg.new_positions_per_generation:,} positions "
                     f"par génération, arène {2 * self.arena_cfg.pairs} parties à {self.arena_cfg.simulations} "
                     "simulations")
        if human:
            if not Path(human).is_dir() or not any(Path(human).glob("train_*.npz")):
                raise LoopStop(f"Données Lichess introuvables dans {human}")
            h = describe_human(Path(human))
            lines.append(f"Lichess : {h['path']} ({h['train_shards']} shards d'entraînement, {h['val_shards']} de validation)")
        if syzygy is not None:
            n = len(list(Path(syzygy).glob("*.rtbw"))) if syzygy and Path(syzygy).is_dir() else 0
            lines.append(f"Tables Syzygy : {n} fichiers WDL dans {syzygy}" if n else
                         "Tables Syzygy : pas encore téléchargées (la cellule suivante le fait)")
        if self.reset_line is not None:
            lines.append(self._reset_line_check(src, info))
        mem = read_memory_gb()
        if mem:
            lines.append(f"RAM : {mem[0]:.1f} Go disponibles sur {mem[1]:.1f} Go")
        return lines

    def _reset_line_check(self, src: Path, info: dict) -> str:
        """Remise de la ligne demandée : refus si un candidat attend son verdict (l'arène doit décider d'abord)."""
        from chesszero.selfplay_train import read_train_state

        if info["kind"] == "selfplay_only":
            return "Remise de la ligne demandée : sans objet (la ligne démarre sur le réseau de départ)"
        view = read_train_state(src / "sp_train")
        if view["candidate"] is not None:
            raise LoopStop(f"Remise de la ligne demandée, mais le candidat {view['candidate']['id']} attend le verdict "
                           "de l'arène : lancer d'abord une session sans RESET_LINE")
        return (f"Remise de la ligne demandée (« {self.reset_line} ») : poids du réseau accepté (génération "
                f"{view['generation']}), AdamW neuf, compteurs de rejets à 0 ; tampon gardé "
                f"({view['arena_rejections']} rejet(s) en arène d'affilée avant la remise)")

    def prepare(self) -> None:
        """Entrée, vérification de la session attendue, démarrage éventuel, enregistrement de la session."""
        from chesszero.arena import git_info
        from chesszero.selfplay import sync_buffer
        from chesszero.selfplay_train import SelfPlayTrainer

        src, info, loop_state = self.validate_input()
        if src != self.work:  # Kaggle : l'entrée (lecture seule) est recopiée dans le dossier de travail
            sync_buffer([src / "selfplay"], self.work / "selfplay", self.sp_cfg.buffer_positions)
            if info["kind"] == "loop":
                for sub in ("sp_train", "loop"):
                    shutil.copytree(src / sub, self.work / sub)
            for sessions in (src / "selfplay" / "sessions",):
                if sessions.is_dir():
                    shutil.copytree(sessions, self.work / "selfplay" / "sessions", dirs_exist_ok=True)
        if info["kind"] == "selfplay_only":
            n = info["train_positions"]  # vérifié par validate_input (même fenêtre du tampon qu'après la copie)
            msg = (f"Démarrage de la boucle à partir d'une sortie de self-play seul : {info['positions']:,} positions "
                   f"dans le tampon ({n:,} d'entraînement)")
            self.log(msg)
            self.notes.append(msg)
            SelfPlayTrainer(self.work / "sp_train", self.train_cfg, "cpu", self.init, log=self.log)
            self.state = new_loop_state()
        else:
            self.state = read_loop_state(self.state_path)
        self.session = last_session(self.state) + 1
        self.state["sessions"].append({"n": self.session, "started": self.start, "finished": None,
                                       "budget_minutes": self.minutes, "git": git_info(), "input": info["kind"],
                                       "resumed_from_session": self.session - 1, "end_reason": None})
        if self.reset_line is not None:
            msg = self._reset_line_check(self.work, info)
            if info["kind"] != "selfplay_only":
                row = SelfPlayTrainer(self.work / "sp_train", self.train_cfg, "cpu", self.init,
                                      log=self.log).reset_line(self.reset_line)
                msg = (f"Ligne d'entraînement REMISE sur le réseau accepté (génération {row['network']['generation']}, "
                       f"SHA-256 {row['network']['sha256'][:12]}…) : « {self.reset_line} » ; AdamW neuf, compteurs de "
                       f"rejets à 0 (avant : {row['before']['arena_rejections']} en arène), les "
                       f"{row['before']['trained_shards']} shards déjà vus redeviennent nouveaux"
                       + (f", tour {row['discarded_round']} abandonné" if row["discarded_round"] is not None else ""))
                self.state["sessions"][-1]["reset_line"] = {"note": self.reset_line, "row": row["round"]}
            self.log(msg)
            self.notes.append(msg)
        self.save()

    # ---------------------------------------------------------------- phases

    def run_phase(self, key: str, kind: str, fn) -> int | None:
        """Lance une phase ; code inattendu = échec : rejoué une fois, arrêt au 2e échec de la même phase."""
        start = self.clock()
        code = fn()
        if code in self.OK_CODES[kind]:
            self.state["failures"].pop(key, None)
            return code
        n = self.state["failures"].get(key, 0) + 1
        self.state["failures"][key] = n
        tail = getattr(self.executor, "last_tail", "")[-3000:]
        self.event(key, start, code, failure=n, tail=tail)
        self.save()
        self.log(f"ÉCHEC de la phase {key} (code {code}, {n}e échec)")
        if n >= self.loop_cfg.max_phase_failures:
            self.end_reason = f"erreur : la phase {key} a échoué {n} fois (code {code})"
        return None

    def controls(self) -> bool:
        """Contrôles d'arène de la première session de boucle. (a) = avertissement seulement ; (b) = bloquant.
        Renvoie False si la session doit s'arrêter."""
        cfg, acfg = self.loop_cfg, self.arena_cfg
        for name, pairs, b_sims in (("a", cfg.control_a_pairs, None), ("b", cfg.control_b_pairs,
                                                                         cfg.control_b_weak_simulations)):
            out = self.work / "loop" / "arena" / f"control_{name}"
            while arena_result(out) is None:  # un échec est rejoué une fois (run_phase compte les échecs)
                est = arena_minutes(estimates(self.state, cfg, self.arena_cfg.simulations), 2 * pairs)
                if self.remaining() < est + cfg.end_margin_minutes:
                    self.end_reason = "budget : pas assez de temps pour les contrôles d'arène"
                    return False
                start = self.clock()
                code = self.run_phase(f"control_{name}", "arena", lambda: self.executor.arena(
                    "anchor", 0, self.init, "depart", acfg.simulations, self.init,
                    "depart_faible" if b_sims else "depart", b_sims or acfg.simulations, out, pairs,
                    self.remaining() - cfg.end_margin_minutes))
                if code is None:
                    if self.end_reason is not None:
                        return False
                    continue
                self.event(f"control_{name}", start, code)
                res = arena_result(out)
                if res is not None:  # les contrôles sont des arènes : ils alimentent l'estimation des arènes
                    est = estimates(self.state, cfg, self.arena_cfg.simulations)
                    record_estimate(self.state, "arena_per_game",
                                    max((self.clock() - start) / 60 - est["arena_fixed"], 0) / res["games"])
                self.save()
                if code == 3:
                    self.end_reason = "budget : contrôles d'arène inachevés"
                    return False
            s = arena_result(out)
            self.state["controls"][name] = {k: s[k] for k in ("games", "score", "elo", "elo_low", "elo_high",
                                                               "pairs_scored_1", "complete_pairs")}
            if name == "a":
                ok = cfg.control_a_low <= s["score"] <= cfg.control_a_high
                sym = s["pairs_scored_1"] == s["complete_pairs"]
                msg = (f"Contrôle (a) (réseau de départ contre lui-même, {s['games']} parties) : score {s['score']:.1%}, "
                       f"{s['pairs_scored_1']}/{s['complete_pairs']} paires symétriques")
                if not ok or not sym:
                    msg += (" -> AVERTISSEMENT (non bloquant : sur GPU les parties peuvent diverger et ce contrôle ne "
                            "détecte pas une inversion des réseaux)")
                self.notes.append(msg)
            else:
                msg = (f"Contrôle (b) (200 contre {b_sims} simulations, {s['games']} parties) : score du côté fort "
                       f"{s['score']:.1%} (seuil {cfg.control_b_min_score:.0%})")
                self.notes.append(msg)
                if s["score"] < cfg.control_b_min_score:
                    self.end_reason = f"contrôle (b) échoué : {s['score']:.1%} < {cfg.control_b_min_score:.0%}"
                    self.state["controls"]["failed"] = True
                    self.save()
                    return False
            self.save()
        self.state["controls"]["done"] = True
        self.save()
        return True

    def run(self) -> dict:
        try:
            self.prepare()
        except LoopStop as e:
            self.log(f"REFUS : {e}")
            raise
        self.crash_point("after_prepare")
        self.mem.start()  # relevés de la RAM du système pendant toute la session (rapport, alerte sous 4 Go)
        if not self.state["controls"]["done"] and not self.controls():
            return self.finish()
        while self.end_reason is None:
            if not self.step():
                break
        return self.finish()

    def step(self) -> bool:
        """Une phase. Renvoie False quand la session est finie (budget, arrêt)."""
        from chesszero.arena import sha256_file

        cfg, acfg = self.loop_cfg, self.arena_cfg
        view = self.view()
        cand = view["candidate"]
        gate = anchor = None
        if cand is not None:
            for which in ("gate", "anchor"):
                self.drop_stale_arena(cand, which, view)
            gate = arena_result(self.arena_dir(cand["id"], "gate"))
            anchor = arena_result(self.arena_dir(cand["id"], "anchor"))
        new = new_positions(self.work / "selfplay", view["trained_shards"])
        phase = next_phase(view, gate, anchor, new, cfg, acfg.accept_score, self.train_waiting)
        self.state["phase"] = phase
        est = estimates(self.state, cfg, self.arena_cfg.simulations)
        first = view["accepted"]["is_init"]
        gate_games = 2 * (acfg.anchor_pairs if first else acfg.pairs)
        anchor_games = 0 if first else 2 * acfg.anchor_pairs
        accepted = view["accepted"]["path"] or self.init
        self.crash_point(f"before_{phase}")
        start = self.clock()

        if phase == "stop":
            self.end_reason = f"arrêt décisionnel : {view['stop_reason']}"
            return False

        if phase in ("arena_gate", "arena_anchor"):
            games = gate_games if phase == "arena_gate" else 2 * acfg.anchor_pairs
            if self.remaining() < arena_minutes(est, games) + cfg.start_margin_minutes:
                return self.final_selfplay(view, est, f"budget : arène du candidat {cand['id']} reportée à la "
                                                      "session suivante")
            which = "gate" if phase == "arena_gate" else "anchor"
            out = self.arena_dir(cand["id"], which)
            kind = "anchor" if (which == "anchor" or first) else "gate"
            b, b_name = (self.init, "kaggle_sup") if kind == "anchor" else (accepted, f"gen_{view['generation']:04d}")
            code = self.run_phase(f"{phase}:{cand['id']}", "arena", lambda: self.executor.arena(
                kind, cand["id"], cand["path"], f"cand_{cand['id']:04d}", None, b, b_name, None, out, None,
                self.remaining() - cfg.end_margin_minutes))
            if code is not None:
                self.event(phase, start, code, candidate=cand["id"])
                res = arena_result(out)
                if res is not None:
                    record_estimate(self.state, "arena_per_game", max((self.clock() - start) / 60
                                                                       - est["arena_fixed"], 0) / res["games"])
                if code == 3:
                    self.end_reason = "budget : arène inachevée (reprise à la session suivante)"
            self.save()
            self.crash_point(f"after_{phase}")
            return self.end_reason is None

        if phase == "verdict":
            gate_path = self.arena_dir(cand["id"], "gate") / "result.json"
            anchor_path = self.arena_dir(cand["id"], "anchor") / "result.json"
            code = self.run_phase(f"verdict:{cand['id']}", "commit", lambda: self.executor.commit(
                cand["id"], gate_path, anchor_path if anchor_path.exists() else None))
            if code is not None:
                self.event(phase, start, code, candidate=cand["id"])
            self.crash_point("after_verdict_before_save")
            self.save()
            self.crash_point("after_verdict")
            return self.end_reason is None

        if phase == "train":
            if view["round_in_progress"]:
                steps = view["round_in_progress"]["steps"] - view["round_in_progress"]["done"]
            else:  # comme l'entraîneur : réutilisation, plafond d'un passage par position du tampon
                from dataclasses import replace

                from chesszero.selfplay_train import plan_steps

                bt = buffer_train_positions(self.work / "selfplay", self.train_cfg)
                steps = plan_steps(int(new * (1 - self.train_cfg.heldout_fraction)), bt,
                                   replace(self.train_cfg, min_buffer_positions=0), view["reuse"],
                                   self.train_cfg.human_fraction)
            if self.remaining() < round_minutes(est, steps, gate_games, anchor_games) + cfg.start_margin_minutes:
                return self.final_selfplay(view, est, "budget : pas assez de temps pour un tour et ses arènes")
            before = self.view()["round_id"]
            code = self.run_phase("train", "train", lambda: self.executor.train(
                self.remaining() - cfg.end_margin_minutes))
            if code is not None:
                after = self.view()
                self.event(phase, start, code, round=before)
                if code == 0 and after["candidate"] is None and after["round_in_progress"] is None \
                        and after["round_id"] == before:
                    self.train_waiting = True  # pas assez de données selon l'entraîneur : self-play d'abord
                elif code in (0, 1):
                    self.record_train_estimate(start, before)
                if code == 3:
                    self.end_reason = "budget : tour d'entraînement interrompu (reprise au pas près)"
            self.save()
            self.crash_point("after_train")
            return self.end_reason is None

        # self-play
        steps = planned_steps(cfg.new_positions_per_generation, self.train_cfg, self.train_cfg.human_fraction)
        minutes, final, target = selfplay_chunk(self.remaining(), cfg.new_positions_per_generation - new, est,
                                                round_minutes(est, steps, gate_games, anchor_games), cfg)
        if minutes <= 0:
            self.end_reason = "budget : fin de session"
            return False
        return self.selfplay(view, minutes, final, est, target)

    def selfplay(self, view: dict, minutes: float, final: bool, est: dict, target: int | None = None) -> bool:
        from chesszero.selfplay import list_shards

        cfg = self.loop_cfg
        before = {s["name"] for s in list_shards([self.work / "selfplay"])}
        start = self.clock()
        k = sum(1 for e in self.state["events"] if e["session"] == self.session and e["phase"] == "selfplay")
        name = f"L{self.session:04d}_{k:02d}"
        soft = self.sp_cfg.stop_new_games_minutes if final else min(cfg.chunk_stop_new_games_minutes, minutes / 2)
        hard = self.sp_cfg.hard_stop_minutes if final else 0.0
        checkpoint = view["accepted"]["path"] or self.init
        min_minutes = None if target is None else cfg.min_chunk_minutes
        code = self.run_phase("selfplay", "selfplay", lambda: self.executor.selfplay(
            minutes, checkpoint, view["generation"], name, soft, hard, target, min_minutes))
        if code is not None:
            produced = sum(s["positions"] for s in list_shards([self.work / "selfplay"]) if s["name"] not in before)
            elapsed = (self.clock() - start) / 60
            self.event("selfplay", start, code, session_name=name, positions=produced, final=final,
                       planned_minutes=minutes, target_positions=target)
            if not final and elapsed > 0 and produced > 0:
                record_estimate(self.state, "positions_per_minute", produced / elapsed)
            self.train_waiting = False
            if final:
                self.end_reason = self.end_reason or "budget : fin de session"
        self.save()
        self.crash_point("after_selfplay")
        return self.end_reason is None

    def final_selfplay(self, view: dict, est: dict, reason: str) -> bool:
        """Plus assez de temps pour la phase prévue : le reste de la session sert au self-play (données utiles)."""
        usable = self.remaining() - self.loop_cfg.end_margin_minutes
        self.notes.append(reason)
        self.end_reason = reason
        if usable >= self.loop_cfg.min_chunk_minutes:
            self.selfplay(view, usable, True, est)
        return False

    def record_train_estimate(self, start: float, round_id: int) -> None:
        from chesszero.arena import read_jsonl

        out = self.work / "sp_train"
        rows = [r for r in read_jsonl(out / "train_log.jsonl") if r.get("round") == round_id]
        rnd = [r for r in read_jsonl(out / "rounds.jsonl") if r.get("type", "round") == "round"
               and r.get("round") == round_id]
        if not rows or not rnd:
            return
        steps = rnd[-1]["steps"]
        batch = self.train_cfg.batch_size
        pps = statistics.fmean(r["positions_per_s"] for r in rows if r.get("positions_per_s"))
        per_step = batch / pps / 60 if pps else None
        total = (self.clock() - start) / 60
        if per_step is not None:
            record_estimate(self.state, "train_per_step", per_step)
            record_estimate(self.state, "train_fixed", max(total - steps * per_step, 0.0))

    def drop_stale_arena(self, cand: dict, which: str, view: dict) -> None:
        """Dossier d'arène d'un autre candidat portant le même numéro (entrée revenue en arrière) : mis de côté."""
        d = self.arena_dir(cand["id"], which)
        spec = d / "arena.json"
        if not spec.exists():
            return
        players = json.loads(spec.read_text(encoding="utf-8")).get("players", {})
        if players.get("a", {}).get("sha256") not in (None, cand["sha256"]):
            stale = d.with_name(d.name + f".stale{int(self.clock())}")
            d.replace(stale)
            self.notes.append(f"Dossier d'arène {d.name} d'un autre candidat mis de côté ({stale.name})")

    # ---------------------------------------------------------------- fin

    def finish(self) -> dict:
        self.crash_point("before_report")
        reason = self.end_reason or "fin"
        sess = self.state["sessions"][-1]
        self.mem.stop()
        sess["finished"], sess["end_reason"] = self.clock(), reason
        self.save()
        report = write_report(self.work, self.state, self.session, self.notes, self.loop_cfg, self.train_cfg,
                              self.arena_cfg, root=self.root)
        sess["report"] = report["path"]
        self.save()
        self.log(f"Session {self.session} terminée : {reason}. Rapport : {report['path']}")
        return {"session": self.session, "end_reason": reason, **report}


# --------------------------------------------------------------------------- rapport


def measured_v2_generations(root: Path = ROOT) -> list[int]:
    """Générations déjà mesurées contre Stockfish (protocole v2) : docs/stockfish/v2/sp_genXXXX.json (commités,
    donc présents dans le clone sur Kaggle)."""
    out = []
    for p in (root / "docs" / "stockfish" / "v2").glob("sp_gen*.json"):
        digits = "".join(ch for ch in p.stem.removeprefix("sp_gen") if ch.isdigit())
        if digits:
            out.append(int(digits))
    return sorted(out)


def history_rows(sp_train: Path) -> list[dict]:
    """Une ligne par verdict d'arène (rounds.jsonl), avec les métriques du tour du candidat."""
    from chesszero.arena import read_jsonl

    rows = read_jsonl(sp_train / "rounds.jsonl")
    rounds = {r["round"]: r for r in rows if r.get("type", "round") == "round"}
    out = []
    for c in (r for r in rows if r.get("type") == "arena"):
        rnd = rounds.get(c["candidate"], {})
        m = rnd.get("metrics", {})
        gate, anchor = c.get("gate") or {}, c.get("anchor") or {}
        out.append({"candidate": c["candidate"], "accepted": c["accepted"], "generation_after": c["generation_after"],
                    "gate_score": gate.get("score"), "gate_elo": gate.get("elo"), "gate_low": gate.get("elo_low"),
                    "gate_high": gate.get("elo_high"), "gate_los": gate.get("los"),
                    "anchor_elo": anchor.get("elo"), "anchor_low": anchor.get("elo_low"),
                    "anchor_high": anchor.get("elo_high"), "steps": rnd.get("steps"),
                    "new_positions": rnd.get("new_positions"), "human_fraction": rnd.get("human_fraction"),
                    "lichess_policy": m.get("lichess", {}).get("policy_loss"),
                    "lichess_top1": m.get("lichess", {}).get("top1"),
                    "lichess_value": m.get("lichess", {}).get("value_loss"),
                    "heldout_value_mse": m.get("heldout", {}).get("value_mse"),
                    "reasons": " ; ".join(c.get("reasons", [])), "time": c.get("time")})
    return out


def write_history_csv(rows: list[dict], path: Path) -> None:
    from chesszero.arena import atomic_write_text

    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    atomic_write_text(path, buf.getvalue())


def plot_progress(anchor_history: list[dict], rows: list[dict], v2: list[int], path: Path) -> str | None:
    """Elo contre l'ancre par génération (intervalles, mesures v2 marquées) ; top-1 Lichess et erreur de valeur des
    parties mises de côté par candidat. Renvoie None si matplotlib manque."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib absent : pas de courbe"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    gens = [0] + [h["generation"] for h in anchor_history]
    elo = [0.0] + [h["elo"] for h in anchor_history]
    lo = [0.0] + [h["elo_low"] if h["elo_low"] is not None else h["elo"] for h in anchor_history]
    hi = [0.0] + [h["elo_high"] if h["elo_high"] is not None else h["elo"] for h in anchor_history]
    ax1.errorbar(gens, elo, yerr=[[e - l for e, l in zip(elo, lo)], [h - e for e, h in zip(elo, hi)]], fmt="o-",
                 capsize=3, label="Elo contre kaggle_sup (intervalle à 95 %)")
    for g in v2:
        ax1.axvline(g, color="gray", ls=":", lw=1)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xlabel("génération acceptée (pointillés : mesures v2 contre Stockfish)")
    ax1.set_ylabel("Elo")
    ax1.legend(loc="best", fontsize=8)
    cands = [r["candidate"] for r in rows]
    if rows:
        ax2.plot(cands, [100 * (r["lichess_top1"] or 0) for r in rows], "o-", label="top-1 Lichess (%)")
        ax2.set_xlabel("candidat")
        ax2.set_ylabel("top-1 Lichess (%)")
        ax3 = ax2.twinx()
        ax3.plot(cands, [r["heldout_value_mse"] for r in rows], "s--", color="tab:red",
                 label="erreur de valeur, parties mises de côté")
        ax3.set_ylabel("erreur de valeur (MSE)")
        ax2.legend(loc="upper left", fontsize=8)
        ax3.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    tmp = path.with_name(path.name + ".tmp.png")
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    tmp.replace(path)
    return None


def session_selfplay_stats(work: Path, state: dict, session: int) -> list[dict]:
    names = [e["session_name"] for e in state["events"] if e["session"] == session and e["phase"] == "selfplay"
             and e.get("session_name")]
    out = []
    for n in names:
        p = work / "selfplay" / "sessions" / f"{n}.json"
        if p.exists():
            out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def actions(ctx: dict, cfg: LoopConfig) -> list[str]:
    """Lignes « Actions pour toi » (fonction pure, testée). `ctx` : voir `write_report`."""
    out = []
    view, end = ctx["view"], ctx["end_reason"]
    if view.get("status") == "stop":
        what = {"G4": "3 rejets d'affilée par les garde-fous : regarder les métriques Lichess des derniers tours "
                      "(taux, part humaine), puis `train_selfplay.py --clear-stop`",
                "stagnation": "5 rejets en arène d'affilée : l'entraînement ne progresse plus ; décider (taille des "
                              "générations, taux, cible de valeur), puis `--clear-stop`",
                "ancre": "le réseau accepté est sous kaggle_sup sur 2 générations : régression ; lancer la mesure v2 "
                         "et décider avant de continuer, puis `--clear-stop`"}.get(view.get("stop_reason"), "")
        out.append(f"ARRÊT ({view.get('stop_reason')}) : {what}")
    if end and end.startswith("erreur"):
        out.append(f"ERREUR : {end} ; voir la fin du message d'erreur dans la chronologie")
    if end and end.startswith("contrôle (b)"):
        out.append(f"CONTRÔLE (b) ÉCHOUÉ : {end}. Ne pas continuer : l'arène attribue peut-être mal les réseaux "
                   "ou les simulations ; examiner loop/arena/control_b/games.pgn")
    for n in ctx["notes"]:
        if "AVERTISSEMENT" in n:
            out.append(n)
    hist = view.get("anchor_history") or []
    gen = view.get("generation", 0)
    v2 = [g for g in ctx["v2"] if g <= gen]
    last_v2 = max(v2, default=0)
    ref = next((h["elo"] for h in hist if h["generation"] == last_v2), 0.0) if last_v2 else 0.0
    smooth = statistics.fmean(h["elo"] for h in hist[-3:]) if hist else None
    if gen > last_v2 and smooth is not None and (smooth - ref > cfg.v2_anchor_gain
                                                  or gen - last_v2 >= cfg.v2_max_generations):
        out.append(f"MESURE v2 recommandée sur gen_{gen:04d} (Elo contre l'ancre lissé {smooth:+.0f}, contre "
                   f"{ref:+.0f} à la dernière mesure, génération {last_v2}) : sur secteur, mode « meilleures "
                   f"performances » : match_stockfish.py run --tag sp_gen{gen:04d} --checkpoint gen_{gen:04d}.pt "
                   "--syzygy tools\\syzygy --like kaggle_sup")
    sp = ctx["selfplay"]
    trig = sum(s.get("resign_triggered_in_control", 0) for s in sp)
    false = sum(s.get("false_resignations", 0) for s in sp)
    if trig and false / trig > cfg.false_resign_alert:
        out.append(f"FAUX ABANDONS : {false} sur {trig} abandons déclenchés dans les parties de contrôle "
                   f"({false / trig:.0%} > {cfg.false_resign_alert:.0%}) : envisager un seuil plus bas (−0,95)")
    games = sum(s.get("repetitions", {}).get("games", 0) for s in sp)
    early = sum(s.get("repetitions", {}).get("by_ply", {}).get("< 60", 0) for s in sp)
    if games and early / games > cfg.early_repetition_alert:
        out.append(f"RÉPÉTITIONS PRÉCOCES : {early} parties sur {games} ({early / games:.1%}) finies par triple "
                   "répétition avant le demi-coup 60 : décision à prendre sur la double répétition dans l'arbre")
    reps_mid = sum(s.get("repetitions", {}).get("repetition_draws", 0)
                   - sum(v for k, v in s.get("repetitions", {}).get("by_material", {}).items() if k.startswith("< 20"))
                   for s in sp)
    if games and reps_mid / games > cfg.repetition_material_reference + cfg.repetition_material_alert_points / 100:
        out.append(f"RÉPÉTITIONS EN MILIEU DE PARTIE : {reps_mid / games:.1%} des parties finies par triple répétition "
                   f"avec au moins 20 points de matériel, contre {cfg.repetition_material_reference:.1%} à la session de "
                   f"référence (self-play seul, 2026-10-22) : plus de {cfg.repetition_material_alert_points:g} points "
                   "d'écart ; à regarder pour la décision sur la double répétition dans l'arbre")
    for rid, sh in ctx.get("timings", []):
        if sh["reading"] > cfg.reading_alert:
            out.append(f"LECTURE DES DONNÉES LENTE (tour {rid}) : {sh['reading']:.0%} du temps du tour "
                       f"({sh['wall_minutes']:.1f} min : lots Lichess {sh['human_read']:.0%}, tampon "
                       f"{sh['buffer_read']:.0%}) > {cfg.reading_alert:.0%} : regarder la lecture des lots (disque, "
                       "taille du tampon)")
        if sh["load_minutes"] > cfg.load_alert_minutes:
            out.append(f"CHARGEMENT INITIAL DES DONNÉES LICHESS LENT (tour {rid}) : {sh['load_minutes']:.1f} min > "
                       f"{cfg.load_alert_minutes:g} min : copier les données Lichess sur le disque local de la session "
                       "(/kaggle/tmp) ou en réduire la taille")
    ram = ctx.get("ram")
    if ram and ram.get("min_available_gb") is not None and ram["min_available_gb"] < cfg.ram_alert_gb:
        out.append(f"MÉMOIRE BASSE : {ram['min_available_gb']:.1f} Go disponibles au plus bas (< {cfg.ram_alert_gb:g} Go), "
                   f"pendant la phase « {ram['min_phase']} » ({ram['alert_samples']} relevé(s) sous le seuil) : risque "
                   "d'arrêt par manque de mémoire ; regarder les minimums par phase dans le rapport (chargement "
                   "complet des données Lichess en RAM : sous-ensemble ou projection en mémoire)")
    for a in ctx["alerts"]:
        out.append(f"ALERTE (tour {a[0]}) : {a[1]}")
    if ctx["gen2_bias"]:
        out.append("GÉNÉRATION 2 : la mesure contre l'ancre de la génération 1 vient de l'arène qui l'a acceptée "
                   "(biais de sélection, ~10 à 20 Elo) : un veto de l'ancre à la génération 2 est un peu plus "
                   "sévère que prévu ; le lire avec ce biais en tête")
    prev, cur = ctx["prev_sps"], ctx["sps"]
    if prev and cur and cur < (1 - cfg.throughput_drop_alert) * prev:
        out.append(f"DÉBIT en baisse : {cur:,.0f} simulations/s contre {prev:,.0f} à la session précédente")
    out.append(f"SESSION SUIVANTE : ajouter en entrée la sortie de CETTE version et mettre EXPECTED_SESSION = "
               f"{ctx['session']}")
    return out


def write_report(work: Path, state: dict, session: int, notes: list[str], cfg: LoopConfig, train_cfg, arena_cfg,
                 root: Path = ROOT) -> dict:
    """loop/reports/session_XXXX.md + .json ; loop/history.csv ; loop/progress.png."""
    from chesszero.agreement import describe_agreement, merge_agreement
    from chesszero.arena import atomic_write_text, read_jsonl, write_json_atomic
    from chesszero.memwatch import describe_memory
    from chesszero.selfplay import describe_repetitions
    from chesszero.selfplay_train import read_train_state

    sp_train = work / "sp_train"
    view = read_train_state(sp_train)
    sess = next(s for s in state["sessions"] if s["n"] == session)
    t0 = sess["started"]
    rows = read_jsonl(sp_train / "rounds.jsonl") if (sp_train / "rounds.jsonl").exists() else []
    rounds = [r for r in rows if r.get("type", "round") == "round" and (r.get("time") or 0) >= t0]
    verdicts = [r for r in rows if r.get("type") == "arena" and (r.get("time") or 0) >= t0]
    alerts = [(r["round"], a) for r in rounds for a in r.get("alerts", [])]
    selfplay = session_selfplay_stats(work, state, session)
    sps = statistics.fmean(s["simulations_per_second"] for s in selfplay) if selfplay else None
    prev = session_selfplay_stats(work, state, session - 1) if session > 1 else []
    prev_sps = statistics.fmean(s["simulations_per_second"] for s in prev) if prev else None
    gen2_bias = any(v.get("opponent", {}).get("generation") == 1 for v in verdicts)
    from chesszero.selfplay_train import timing_shares

    timings = [(r["round"], timing_shares(r["timing"])) for r in rounds if r.get("timing")]
    v2 = measured_v2_generations(root)
    ctx = {"view": view, "end_reason": sess["end_reason"], "notes": notes, "v2": v2, "selfplay": selfplay,
           "alerts": alerts, "gen2_bias": gen2_bias, "prev_sps": prev_sps, "sps": sps, "session": session,
           "timings": timings, "ram": sess.get("ram")}
    acts = actions(ctx, cfg)

    hist = history_rows(sp_train) if view.get("exists") else []
    write_history_csv(hist, work / "loop" / "history.csv")
    plot_note = plot_progress(view.get("anchor_history") or [], hist, v2, work / "loop" / "progress.png") \
        if view.get("exists") else "pas d'état d'entraînement"

    L = [f"# Session de boucle n°{session}", "",
         f"**À mettre dans EXPECTED_SESSION pour la session suivante : {session}**", "",
         f"Fin : {sess['end_reason']} ; durée {((sess['finished'] or time.time()) - t0) / 60:.0f} min sur "
         f"{sess['budget_minutes']:.0f} ; entrée : {sess['input']} ; code {sess['git'].get('commit')}"
         f"{' (modifié)' if sess['git'].get('dirty') else ''}", "", "## Actions pour toi", ""]
    L += [f"- {a}" for a in acts]
    L += ["", "## Notes de démarrage et contrôles", ""] + ([f"- {n}" for n in notes] or ["- (aucune)"])
    L += ["", "## Candidats de la session", "",
          "| candidat | pas | nouvelles positions | garde-fous | arène contre l'accepté | contre l'ancre | décision |",
          "|---|---|---|---|---|---|---|"]
    by_cand = {v["candidate"]: v for v in verdicts}
    for r in rounds:
        cid = r.get("candidate")
        g = "passés" if cid is not None else "REJET : " + " ; ".join(r.get("reasons", []))
        v = by_cand.get(cid) if cid is not None else None
        gate = anch = dec = "en attente" if cid is not None else "—"
        if v:
            gs, an = v["gate"], v.get("anchor")
            gate = (f"{gs['score']:.1%}, Elo {gs['elo']:+.0f} ({_fmt(gs['elo_low'])} à {_fmt(gs['elo_high'])}), "
                    f"P(meilleur) {gs['los']:.0%}")
            anch = f"Elo {an['elo']:+.0f} ({_fmt(an['elo_low'])} à {_fmt(an['elo_high'])})" if an else "—"
            dec = "ACCEPTÉ" if v["accepted"] else "rejeté : " + " ; ".join(v["reasons"])
        L.append(f"| {r['round']} | {r['steps']} | {r['new_positions']:,} | {g} | {gate} | {anch} | {dec} |")
    for v in verdicts:  # verdicts de candidats créés lors d'une session précédente
        if v["candidate"] not in {r["round"] for r in rounds}:
            L.append(f"| {v['candidate']} (session précédente) | | | | {v['gate']['score']:.1%} | | "
                     f"{'ACCEPTÉ' if v['accepted'] else 'rejeté'} |")
    if timings:
        L += ["", "Temps des tours d'entraînement (part du temps total du tour) :", ""]
        for rid, sh in timings:
            L.append(f"- tour {rid} : {sh['wall_minutes']:.1f} min ; lecture des lots {sh['reading']:.0%} (lots "
                     f"Lichess {sh['human_read']:.0%}, tampon {sh['buffer_read']:.0%}) ; masques Lichess "
                     f"{sh['human_masks']:.0%} ; préparation {sh['prepare']:.0%} ; calcul {sh['compute']:.0%} ; "
                     f"évaluation {sh['eval']:.0%} ; priors des anciens shards {sh['priors']:.0%} ; autre "
                     f"{sh['other']:.0%}")
            L.append(f"  - chargement initial des données Lichess : {sh['load_minutes']:.1f} min ({sh['load']:.0%} du "
                     f"tour ; alerte au-delà de {cfg.load_alert_minutes:g} min)")
    pruned = [(r["round"], r["policy_target_stats"]) for r in rounds if r.get("policy_target_stats")]
    if pruned:
        L += ["", "Cibles de politique débruitées (bruit de Dirichlet retiré, chesszero/policy_target.py) :", ""]
        for rid, ps in pruned:
            L.append(f"- tour {rid} : {ps['removed_visit_share']:.1%} des visites retirées ; entropie des cibles "
                     f"{ps['entropy_raw']:.3f} -> {ps['entropy_pruned']:.3f} ; part du coup le plus visité "
                     f"{ps['top_share_raw']:.3f} -> {ps['top_share_pruned']:.3f} ; Q exact pour "
                     f"{ps['exact_q_share']:.0%} des positions")
    acc = view.get("accepted") or {}
    L += ["", "## État en fin de session", "",
          f"- génération acceptée : {view.get('generation')} ({acc.get('path') or 'réseau de départ'}) ; "
          f"mesure contre l'ancre : {acc.get('anchor_source')}",
          f"- candidat en attente : {(view.get('candidate') or {}).get('id', 'aucun')} ; tour en cours : "
          f"{view.get('round_in_progress') or 'aucun'}",
          f"- rejets en arène d'affilée : {view.get('arena_rejections')} ; rejets par les garde-fous d'affilée : "
          f"{view.get('guard_rejections')} ; état : {view.get('status')} {view.get('stop_reason') or ''}",
          f"- taux × {view.get('lr_scale')} ; réutilisation {view.get('reuse')} ; part humaine maintenue à 25 % : "
          f"{view.get('human_sticky_25')}"]
    L += ["", "## Self-play", ""]
    for s in selfplay:
        L.append(f"- {s['session']} : {s['games']} parties, {s['positions']:,} positions, {s['play_minutes']:.0f} min, "
                 f"{s['simulations_per_second']:,.0f} simulations/s, inachevées {s['unfinished_dropped']} ; fins "
                 + ", ".join(f"{k} {n}" for k, n in s["terminations"].items() if n))
        L.append("  " + describe_repetitions(s["repetitions"]).replace("\n", "\n  "))
        L.append(f"  faux abandons : {s['false_resignations']} sur {s['resign_triggered_in_control']} déclenchés "
                 f"({s['control_games']} parties de contrôle)")
    if not selfplay:
        L.append("- (aucune tranche)")
    else:
        L.append("- " + describe_agreement(merge_agreement([s.get("agreement") for s in selfplay])).replace(
            "\n", "\n  "))
    L += ["", "## Mémoire", "", "- " + describe_memory(sess.get("ram"))]
    L += ["", "## Chronologie", ""]
    for e in (e for e in state["events"] if e["session"] == session):
        extra = f" — ÉCHEC n°{e['failure']} :\n```\n{e['tail'][-1500:]}\n```" if e.get("failure") else ""
        L.append(f"- {e['phase']} : {e['minutes']:.1f} min, code {e['code']}{extra}")
    L += ["", f"Courbes : loop/progress.png{' (' + plot_note + ')' if plot_note else ''} ; historique : "
              "loop/history.csv"]
    text = "\n".join(L) + "\n"
    rep_dir = work / "loop" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    path = rep_dir / f"session_{session:04d}.md"
    atomic_write_text(path, text)
    write_json_atomic(path.with_suffix(".json"), {"session": session, "actions": acts, "notes": notes,
                                                  "end_reason": sess["end_reason"], "view": view})
    return {"path": str(path.relative_to(work)).replace("\\", "/"), "actions": acts, "text": text}


def _fmt(x) -> str:
    return f"{x:+.0f}" if x is not None else "non bornée"
