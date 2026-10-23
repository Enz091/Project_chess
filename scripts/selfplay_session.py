"""Session de self-play (Kaggle ou local) : essai de débit ou session complète, avec rapport.

  selfplay_session.py bench --minutes 15 --work /kaggle/working --init <best.pt> --syzygy /tmp/syzygy
  selfplay_session.py full --minutes 660 --work /kaggle/working --init <best.pt> --syzygy /tmp/syzygy \\
                      --inputs <sorties précédentes>/selfplay [--with-training --human <lichess-24m>]

bench : `--minutes` au total, en deux phases, dans <work>/bench_selfplay (séparé du tampon ; parties en cours à la fin
        de chaque phase abandonnées : on mesure, on ne remplit pas le tampon) :
          1. DÉBIT (`--throughput-minutes`, 5) avec les réglages réels ([selfplay] : 64 parties par processus) :
             simulations/s, GPU, CPU, taille des lots envoyés au GPU ;
          2. PARTIES (le reste, 10 min) avec `--stats-games-per-worker` (16) parties par processus : toutes les parties
             d'un processus avancent ensemble, donc avec 64 parties une partie dure ~15 min sur Kaggle et presque
             aucune ne finirait ; avec 16, ~4 min : longueurs, fins, répétitions précoces, coups cherchés par partie.
        Parties/h = simulations/s (phase 1) / (simulations par coup x coups cherchés par partie (phase 2)).
        Rapport final : heures de self-play nécessaires pour atteindre 250 000 positions d'entraînement.
full  : remplit le tampon <work>/selfplay (repris depuis --inputs, fenêtre de 500 000 positions), budget --minutes.
        Par défaut self-play SEUL (aucun tour d'entraînement). Avec --with-training : boucle SIMPLE et PROVISOIRE,
        par tranches de --chunk-minutes : self-play avec le dernier réseau accepté, puis un tour de train_selfplay.py
        (rien tant que le tampon a moins de 250 000 positions) ; l'arène et les ancres fixes viendront à l'étape 7.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(cmd: list, check: bool = True) -> int:
    """Lance une commande, recopie sa sortie au fil de l'eau, renvoie le code de sortie."""
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
    code = proc.wait()
    if check and code:
        raise RuntimeError(f"échec ({code}) : {' '.join(str(c) for c in cmd)}")
    return code


class ResourceSampler:
    """Échantillonne l'utilisation du GPU (nvidia-smi) et du CPU (/proc/stat ou psutil) toutes les `interval` s."""

    def __init__(self, interval: float = 2.0):
        self.interval, self.gpu, self.gpu_mem, self.cpu = interval, [], [], []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @staticmethod
    def _cpu_times():
        try:
            fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            values = [int(x) for x in fields]
            return values[3] + values[4], sum(values)  # (inactif + attente E/S, total)
        except OSError:
            return None

    def _run(self) -> None:
        smi = shutil.which("nvidia-smi")
        prev = self._cpu_times()
        try:
            import psutil
            psutil.cpu_percent(None)
        except ImportError:
            psutil = None
        while not self._stop.wait(self.interval):
            if smi:
                try:
                    out = subprocess.run([smi, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
                    if out:  # premier GPU (celui qu'utilisent les processus)
                        util, mem = (float(x) for x in out[0].split(","))
                        self.gpu.append(util)
                        self.gpu_mem.append(mem)
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
            now = self._cpu_times()
            if now and prev and now[1] > prev[1]:
                self.cpu.append(100 * (1 - (now[0] - prev[0]) / (now[1] - prev[1])))
            elif psutil is not None:
                self.cpu.append(psutil.cpu_percent(None))
            prev = now

    def summary(self) -> dict:
        mean = lambda xs: statistics.fmean(xs) if xs else None
        return {"gpu_util": mean(self.gpu), "gpu_util_max": max(self.gpu) if self.gpu else None,
                "gpu_mem_mb": max(self.gpu_mem) if self.gpu_mem else None, "cpu_util": mean(self.cpu),
                "samples": len(self.cpu)}


def selfplay_cmd(args, out: Path, minutes: float, checkpoint: str, inputs: list, generation: int, extra: list) -> list:
    cmd = [sys.executable, SCRIPTS / "selfplay.py", "--out", out, "--checkpoint", checkpoint, "--max-minutes", minutes,
           "--workers", args.workers, "--threads", args.threads, "--generation", generation, "--config", args.config,
           "--syzygy", args.syzygy or "", "--book", args.book]
    if args.games_per_worker:
        cmd += ["--games-per-worker", args.games_per_worker]
    if args.device:
        cmd += ["--device", args.device]
    if inputs:
        cmd += ["--inputs", *inputs]
    return cmd + extra


def latest_session(out: Path) -> dict:
    files = sorted((out / "sessions").glob("*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else {}


def report(stats: dict, res: dict, buffer_positions: int, cfg) -> None:
    from chesszero.selfplay import describe_repetitions

    print("\n" + "=" * 30 + " RAPPORT " + "=" * 30)
    sps = stats["simulations_per_second"]
    pos_per_h = sps / stats["simulations_per_move"] * 3600  # une position enregistrée par coup cherché
    print(f"Parties terminées : {stats['games']} en {stats['play_minutes']:.1f} min de jeu "
          f"({stats['games_per_hour']:.0f} parties/h terminées pendant la mesure ; inachevées abandonnées à la fin : "
          f"{stats['unfinished_dropped']})")
    print(f"Simulations/s : {sps:,.0f} au total ({sps / stats['workers']:,.0f} par processus, {stats['workers']} processus "
          f"x {stats['games_per_worker']} parties)")
    print(f"Positions enregistrées : {stats['positions']} ; rythme de production ~ {pos_per_h:,.0f} positions/h "
          f"(= simulations/s / {stats['simulations_per_move']} x 3600)")
    print(f"Lots envoyés au GPU : {stats['positions_per_eval_call']:.0f} positions par appel en moyenne ; "
          f"réseau {stats['eval_time_share']:.0%} du temps des processus")
    gpu = f"{res['gpu_util']:.0f} % en moyenne (max {res['gpu_util_max']:.0f} %), mémoire max {res['gpu_mem_mb']:.0f} Mo" \
        if res.get("gpu_util") is not None else "non mesuré (pas de nvidia-smi)"
    cpu = f"{res['cpu_util']:.0f} % de tous les cœurs en moyenne" if res.get("cpu_util") is not None else "non mesuré"
    print(f"GPU : {gpu} ; CPU : {cpu} ({os.cpu_count()} cœurs logiques)")
    if stats["plies"]:
        plies, sp = stats["plies"], stats["selfplay_plies"]
        print(f"Longueur des parties : moyenne {statistics.fmean(plies):.0f} demi-coups (médiane {statistics.median(plies):.0f}, "
              f"min {min(plies)}, max {max(plies)}) ; dont joués en self-play : moyenne {statistics.fmean(sp):.0f}")
    print("Fins : " + (", ".join(f"{t} {n}" for t, n in stats["terminations"].items() if n) or "aucune partie terminée"))
    print(describe_repetitions(stats["repetitions"]))
    print(f"Parties de contrôle : {stats['control_games']} ; abandons qui se seraient déclenchés : "
          f"{stats['resign_triggered_in_control']} ; faux abandons : {stats['false_resignations']}")
    target = cfg.min_buffer_positions / (1 - cfg.heldout_fraction)  # 250 000 d'entraînement + 10 % mis de côté
    missing = max(0.0, target - buffer_positions)
    print(f"\nESTIMATION : pour {cfg.min_buffer_positions:,} positions d'entraînement (soit ~{target:,.0f} au total avec "
          f"les {cfg.heldout_fraction:.0%} de parties mises de côté), il manque {missing:,.0f} positions -> "
          f"~{missing / max(pos_per_h, 1e-9):.1f} h de self-play à ce rythme"
          f" (un peu plus en réalité : marges de début et de fin de session, parties inachevées).")


def report_bench(throughput: dict, games: dict, res: dict, cfg) -> None:
    """Rapport de l'essai de débit : débit de la phase 1 (réglages réels), parties de la phase 2."""
    from chesszero.selfplay import describe_repetitions

    print("\n" + "=" * 26 + " RAPPORT DE L'ESSAI DE DÉBIT " + "=" * 26)
    sps, sims = throughput["simulations_per_second"], throughput["simulations_per_move"]
    print(f"[phase 1, réglages réels : {throughput['workers']} processus x {throughput['games_per_worker']} parties, "
          f"{throughput['play_minutes']:.1f} min]")
    print(f"Simulations/s : {sps:,.0f} au total ({sps / throughput['workers']:,.0f} par processus)")
    print(f"Lots envoyés au GPU : {throughput['positions_per_eval_call']:.0f} positions par appel en moyenne ; réseau "
          f"{throughput['eval_time_share']:.0%} du temps des processus (le reste : recherche en Python, CPU)")
    gpu = (f"{res['gpu_util']:.0f} % en moyenne (max {res['gpu_util_max']:.0f} %), mémoire max {res['gpu_mem_mb']:.0f} Mo"
           if res.get("gpu_util") is not None else "non mesuré (pas de nvidia-smi)")
    cpu = f"{res['cpu_util']:.0f} % de tous les cœurs" if res.get("cpu_util") is not None else "non mesuré"
    print(f"GPU : {gpu} ; CPU : {cpu} ({os.cpu_count()} cœurs logiques)")
    pos_per_h = sps / sims * 3600  # une position enregistrée par coup cherché
    print(f"Production : ~{pos_per_h:,.0f} positions/h (= simulations/s / {sims} x 3600)")
    print(f"\n[phase 2 : {games['workers']} processus x {games['games_per_worker']} parties, {games['play_minutes']:.1f} min]")
    n = games["games"]
    if n:
        plies, sp = games["plies"], games["selfplay_plies"]
        per_game = games["positions"] / n  # coups cherchés (= positions enregistrées) par partie
        print(f"{n} parties terminées ; longueur moyenne {statistics.fmean(plies):.0f} demi-coups (médiane "
              f"{statistics.median(plies):.0f}, min {min(plies)}, max {max(plies)}), dont {statistics.fmean(sp):.0f} joués "
              f"en self-play et {per_game:.0f} cherchés (les coups forcés ne le sont pas)")
        print("Fins : " + ", ".join(f"{t} {k}" for t, k in games["terminations"].items() if k))
        print(describe_repetitions(games["repetitions"]))
        print(f"Parties de contrôle : {games['control_games']} ; abandons qui se seraient déclenchés : "
              f"{games['resign_triggered_in_control']} ; faux abandons : {games['false_resignations']}")
        print(f"\nPARTIES PAR HEURE (réglages réels) : ~{pos_per_h / per_game:,.0f} "
              f"(= {pos_per_h:,.0f} positions/h / {per_game:.0f} coups cherchés par partie)")
    else:
        print("Aucune partie terminée en phase 2 : allonger --minutes ou baisser --stats-games-per-worker.")
    target = cfg.min_buffer_positions / (1 - cfg.heldout_fraction)
    print(f"\nESTIMATION : {cfg.min_buffer_positions:,} positions d'entraînement = ~{target:,.0f} positions au total "
          f"(avec les {cfg.heldout_fraction:.0%} de parties mises de côté) -> ~{target / pos_per_h:.1f} h de self-play ; "
          f"compter un peu plus (démarrage, marges de fin de session de 20 + 5 min, parties inachevées).")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["bench", "full"])
    ap.add_argument("--minutes", type=float, required=True, help="budget (bench : durée du jeu ; full : toute la session)")
    ap.add_argument("--work", required=True, help="dossier de travail (= sortie de la session sur Kaggle)")
    ap.add_argument("--init", required=True, help="réseau de départ (best.pt de kaggle_sup)")
    ap.add_argument("--syzygy", default=str(ROOT / "tools" / "syzygy"), help="tables (WDL suffisent) ; '' = sans")
    ap.add_argument("--book", default=str(ROOT / "books" / "lichess_2000.tsv.gz"))
    ap.add_argument("--inputs", nargs="*", default=[], help="full : dossiers selfplay des sessions précédentes")
    ap.add_argument("--train-state", default=None, help="full + entraînement : dossier sp_train d'une session précédente")
    ap.add_argument("--with-training", action="store_true", help="full : tours d'entraînement (boucle simple provisoire)")
    ap.add_argument("--human", default=None, help="données Lichess (avec --with-training)")
    ap.add_argument("--chunk-minutes", type=float, default=120, help="full + entraînement : self-play entre deux tours")
    ap.add_argument("--train-reserve-minutes", type=float, default=25, help="temps gardé pour chaque tour d'entraînement")
    ap.add_argument("--throughput-minutes", type=float, default=5, help="bench : durée de la phase de débit")
    ap.add_argument("--stats-games-per-worker", type=int, default=16, help="bench : parties par processus en phase 2")
    ap.add_argument("--min-chunk-minutes", type=float, default=30, help="full : tranche de self-play minimale")
    ap.add_argument("--stop-new-games-minutes", type=float, default=None, help="full : marge de fin (essais locaux)")
    ap.add_argument("--hard-stop-minutes", type=float, default=None, help="full : marge d'arrêt (essais locaux)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--games-per-worker", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--config", default="kaggle")
    args = ap.parse_args()

    from chesszero.selfplay import list_shards
    from chesszero.selfplay_train import SelfPlayTrainConfig

    work = Path(args.work)
    tcfg = SelfPlayTrainConfig.named(args.config)
    start = time.time()

    if args.mode == "bench":
        out = work / "bench_selfplay"
        shutil.rmtree(out, ignore_errors=True)
        margins = ["--stop-new-games-minutes", 0, "--hard-stop-minutes", 0]
        print(f"\n### Phase 1 : débit, {args.throughput_minutes:.0f} min, réglages réels ###", flush=True)
        with ResourceSampler() as sampler:
            run(selfplay_cmd(args, out / "debit", args.throughput_minutes, args.init, [], 0,
                             margins + ["--session", "debit"]))
        throughput, resources = latest_session(out / "debit"), sampler.summary()
        stats_minutes = args.minutes - args.throughput_minutes
        print(f"\n### Phase 2 : parties, {stats_minutes:.0f} min, {args.stats_games_per_worker} parties par processus ###",
              flush=True)
        args.games_per_worker = args.stats_games_per_worker
        run(selfplay_cmd(args, out / "parties", stats_minutes, args.init, [], 0, margins + ["--session", "parties"]))
        games = latest_session(out / "parties")
        run([sys.executable, SCRIPTS / "check_selfplay.py", out / "parties", "--syzygy", args.syzygy or "",
             "--book", args.book], check=False)
        report_bench(throughput, games, resources, tcfg)
        (out / "bench_report.json").write_text(json.dumps({"throughput": throughput, "games": games,
                                                           "resources": resources}, indent=1) + "\n",
                                               encoding="utf-8", newline="\n")
        return

    # ---------------------------------------------------------------- session complète
    buffer = work / "selfplay"
    train_out = work / "sp_train"
    if args.train_state and not (train_out / "train_state.pt").exists():
        shutil.copytree(args.train_state, train_out, dirs_exist_ok=True)
        print(f"État d'entraînement repris depuis {args.train_state}")
    deadline = start + args.minutes * 60

    def current_network() -> tuple[str, int]:
        # réseau ACCEPTÉ selon train_state.pt (source de vérité), jamais « le dernier gen_*.pt » du dossier : un
        # fichier orphelin (coupure avant la validation) serait pris à tort. Depuis l'étape 7b, une génération
        # n'est acceptée qu'après une arène : cette boucle provisoire (sans arène) garde le réseau de départ ;
        # l'orchestrateur de l'étape 7d la remplacera.
        from chesszero.selfplay_train import read_train_state

        st = read_train_state(train_out)
        if st.get("exists") and st.get("accepted", {}).get("path"):
            return st["accepted"]["path"], st["generation"]
        return args.init, 0

    inputs = list(args.inputs)
    with ResourceSampler() as sampler:
        while True:
            remaining = (deadline - time.time()) / 60
            if args.with_training:
                chunk = min(args.chunk_minutes, remaining - args.train_reserve_minutes)
            else:
                chunk = remaining
            if chunk < args.min_chunk_minutes:  # trop court pour une tranche utile (marges de 20 + 5 min)
                break
            checkpoint, generation = current_network()
            print(f"\n### Self-play {chunk:.0f} min avec {checkpoint} (génération {generation}) ###", flush=True)
            margins = [x for opt, v in (("--stop-new-games-minutes", args.stop_new_games_minutes),
                                        ("--hard-stop-minutes", args.hard_stop_minutes)) if v is not None for x in (opt, v)]
            run(selfplay_cmd(args, buffer, chunk, checkpoint, inputs, generation, margins))
            inputs = []  # le tampon est maintenant dans <work>/selfplay
            if not args.with_training:
                break
            if not args.human:
                raise ValueError("--with-training demande --human (données Lichess)")
            code = run([sys.executable, SCRIPTS / "train_selfplay.py", "--out", train_out, "--buffer", buffer,
                        "--human", args.human, "--init", args.init, "--config", args.config], check=False)
            if code == 2:
                print("G4 : arrêt demandé par les garde-fous, décision humaine nécessaire.")
                break
            if code not in (0, 1):
                raise RuntimeError(f"train_selfplay.py a échoué ({code})")
    stats = latest_session(buffer)
    window = sum(s["positions"] for s in list_shards([buffer]))
    report(stats, sampler.summary(), window, tcfg)
    print(f"Tampon en fin de session : {window:,} positions dans {buffer} (sortie de la session)")


if __name__ == "__main__":
    main()
