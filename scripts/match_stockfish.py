"""Match contre Stockfish bridé (UCI_LimitStrength / UCI_Elo) : estimation d'Elo de l'IA.

Deux protocoles figés (voir docs/journal.md), choisis par --protocol (défaut : v2) :
  - IA : MCTS 200 simulations par coup, lots de 8, arbre réutilisé, température 0, sans bruit ;
  - Stockfish : `go movetime 300`, 1 thread, Hash 16 Mo, niveau fixé par UCI_Elo, sans tables de finales ;
  - ouvertures, chacune jouée avec les deux couleurs (un « tour » = 2 parties par ouverture et par niveau) :
      v1 : les 10 de match.py (6 demi-coups) -> 20 parties par tour ;
      v2 : les 40 de ladder_openings.py (6 à 10 demi-coups, dont 8 débuts rares) -> 80 parties par tour ;
    fin de partie : règles officielles, nulle au-delà de 400 demi-coups ;
  - phase 1 : niveaux 1320, 1600, 1900, 2200, 2500 (un tour chacun) ;
    tant que l'IA marque plus de 80 % au niveau le plus haut joué, un niveau 300 Elo au-dessus
    (2800, puis 3100, puis 3190 = plafond de UCI_Elo), un tour chacun ;
  - phase 2 : Elo ajusté sur la phase 1, arrondi à 50, puis 3 niveaux (−100, 0, +100) × des tours
    supplémentaires (v1 : 2 tours = 40 parties ; v2 : 1 tour = 80 parties). Elo final = ajustement sur tout.
  - 4 processus en parallèle, 2 threads torch chacun, Stockfish 1 thread : 8 threads au plus.

Options de l'IA qui ne font pas partie du protocole (enregistrées à part, dans « ai ») : --checkpoint et
--syzygy <dossier> (tables de finales, voir chesszero/syzygy.py). Deux résultats du même protocole se
comparent donc même si l'un utilise les tables et l'autre non : c'est ainsi qu'on mesure leur effet.

Commandes (à lancer avec .venv\\Scripts\\python.exe) :
  match_stockfish.py bench                        mesure la vitesse (rien n'est enregistré)
  match_stockfish.py run --tag kaggle_sup         protocole v2 complet ; reprend tout seul si interrompu
  match_stockfish.py run --tag essai_X --checkpoint checkpoints/.../best.pt --like kaggle_sup
                                                  rejoue EXACTEMENT les mêmes parties (niveaux, tours,
                                                  ouvertures, couleurs) sur un autre checkpoint, puis compare
  match_stockfish.py run --tag kaggle_sup_tb --syzygy tools/syzygy --like kaggle_sup    effet des tables
  match_stockfish.py report kaggle_sup            tableau par niveau + Elo estimé
  match_stockfish.py compare kaggle_sup essai_X   écart d'Elo entre deux résultats
  match_stockfish.py endgames kaggle_sup          finales : résultat exact à l'entrée dans les tables -> score
Un tag peut nommer un autre protocole : « v1/kaggle_sup » (sinon celui de --protocol).

Surveillance de la vitesse (règle de mesure) : `run` joue d'abord un bench de départ (une partie par processus,
rien n'est enregistré) puis suit la moyenne glissante (20 dernières parties) du temps par coup de l'IA ; un
avertissement visible apparaît quand elle s'écarte de plus de 25 % du bench de départ (processeur bridé en cours de
mesure : Stockfish, au temps, s'affaiblit et l'Elo est gonflé). Les valeurs sont dans le fichier de résultats (clé
« speed », et « ai_rolling_s_per_move » dans chaque partie) ; le résumé final dit si la mesure a été stable.
--no-speed-watch désactive le bench de départ et le suivi.

Résultats : docs/stockfish/<protocole>/<tag>.json (toutes les parties, protocole, IA, environnement), parties
en PGN dans docs/stockfish/<protocole>/<tag>.pgn. Stockfish n'est pas déterministe (le tirage des coups
faibles dépend de l'horloge) : deux lancements du même protocole ne rejouent pas exactement les mêmes coups.
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from chesszero.rating import difference, elo_from_score, fit_elo, per_level
from chesszero.speedwatch import SpeedWatch, read_power
from chesszero.speedwatch import verdict as speed_verdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder_openings import OPENINGS_V2  # noqa: E402
from match import OPENINGS, termination  # noqa: E402  (v1 : mêmes ouvertures que le match précédent)

DEFAULT_STOCKFISH = ROOT / "tools" / "stockfish" / "stockfish" / "stockfish-windows-x86-64-universal.exe"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "kaggle_sup" / "best.pt"
DEFAULT_SYZYGY = ROOT / "tools" / "syzygy"
OUT_DIR = ROOT / "docs" / "stockfish"
MAX_THREADS = 8

# Tout ce qui peut changer le résultat d'une partie, sauf l'IA elle-même. Comparer deux résultats n'a de
# sens que si ce bloc est identique (sauf --force). FIGÉS : ne jamais en modifier un, en créer un nouveau.
PROTOCOL_V1 = {
    "name": "stockfish-ladder-v1",
    "simulations": 200,
    "mcts_config": "kaggle",  # section [mcts] de configs/kaggle.toml, batch_size 8
    "stockfish_movetime_ms": 300,
    "stockfish_threads": 1,
    "stockfish_hash_mb": 16,
    "openings": [list(o) for o in OPENINGS],
    "max_plies": 400,
    "phase1_levels": [1320, 1600, 1900, 2200, 2500],
    "extension_step": 300,
    "extension_trigger": 0.80,
    "refine_offsets": [-100, 0, 100],
    "refine_rounding": 50,
    "refine_rounds": 2,
}
PROTOCOL_V2 = {
    **PROTOCOL_V1,
    "name": "stockfish-ladder-v2",
    "openings": [list(o) for o in OPENINGS_V2],
    "refine_rounds": 1,  # un tour = 80 parties : chaque ouverture et chaque couleur une fois par niveau
}
PROTOCOLS = {"v1": PROTOCOL_V1, "v2": PROTOCOL_V2}
ELO_MIN, ELO_MAX = 1320, 3190  # plage de UCI_Elo dans Stockfish


# --------------------------------------------------------------------------- parties (dans les workers)

_W: dict = {}


def init_worker(checkpoint: str, stockfish: str, protocol: dict, torch_threads: int, syzygy: str | None) -> None:
    import torch

    from chesszero.checkpoint import load_for_inference
    from chesszero.mcts import MCTSConfig, NetEvaluator
    from chesszero.syzygy import Tablebase

    torch.set_num_threads(torch_threads)
    _W["tb"] = Tablebase(syzygy) if syzygy else None
    model, meta = load_for_inference(checkpoint, "cpu")
    cfg = replace(MCTSConfig.named(protocol["mcts_config"]), num_simulations=protocol["simulations"])
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    engine.configure({"Threads": protocol["stockfish_threads"], "Hash": protocol["stockfish_hash_mb"]})
    _W.update(evaluator=NetEvaluator(model), cfg=cfg, engine=engine, protocol=protocol, step=meta["step"],
              sf_name=engine.id["name"])


def play_game(task: dict) -> dict:
    """Une partie : l'IA (MCTS) contre Stockfish au niveau `task['level']`. Renvoie le résultat."""
    from chesszero.mcts import MCTS, terminal_value

    p, engine = _W["protocol"], _W["engine"]
    level, ai_white = task["level"], task["ai_white"]
    name, opening = p["openings"][task["opening"]]
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": level})
    limit = chess.engine.Limit(time=p["stockfish_movetime_ms"] / 1000)
    ai_color = chess.WHITE if ai_white else chess.BLACK

    board = chess.Board()
    for uci in opening.split():
        board.push_uci(uci)
    tb = _W["tb"]
    mcts = MCTS(_W["evaluator"], _W["cfg"], board, tablebase=tb)
    game_key = f"{task['level']}/{task['round']}/{task['opening']}/{int(ai_white)}"
    start, ai_seconds, ai_moves, tb_moves = time.perf_counter(), 0.0, 0, 0
    while terminal_value(board) is None and len(board.move_stack) < p["max_plies"]:
        if board.turn == ai_color:
            t = time.perf_counter()
            mcts.search()
            move = mcts.best_move(0)
            ai_seconds += time.perf_counter() - t
            ai_moves += 1
            tb_moves += tb is not None and tb.covers(board)  # coups choisis par les tables
        else:
            move = engine.play(board, limit, game=game_key).move
        mcts.advance(move)
        board.push(move)

    value = terminal_value(board)
    if value == -1.0:  # le camp au trait est maté
        score = 0.0 if board.turn == ai_color else 1.0
    else:
        score = 0.5
    return {**task, "opening_name": name, "score": score, "plies": len(board.move_stack),
            "termination": termination(board, len(board.move_stack), p["max_plies"]),
            "moves": " ".join(m.uci() for m in board.move_stack), "ai_sec_per_move": ai_seconds / max(ai_moves, 1),
            "ai_tb_moves": int(tb_moves),
            "seconds": time.perf_counter() - start, "checkpoint_step": _W["step"], "stockfish": _W["sf_name"]}


# --------------------------------------------------------------------------- orchestration


# Mode d'alimentation de Windows 11 (Paramètres > Système > Alimentation), surcouche du plan actif
POWER_MODES = {"ded574b5-45a0-4f42-8737-46345c09c238": "meilleures performances",
               "00000000-0000-0000-0000-000000000000": "équilibré",
               "961cc777-2547-4f9d-8174-7d86181b8a7a": "meilleure efficacité énergétique"}

_POWER_PS = r"""
$b = Get-CimInstance Win32_Battery | Select-Object -First 1
$p = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes' -ErrorAction SilentlyContinue
$plan = (powercfg /getactivescheme) -join ' '
@{battery_status = $b.BatteryStatus; battery_percent = $b.EstimatedChargeRemaining; plan = $plan;
  overlay_ac = $p.ActiveOverlayAcPowerScheme; overlay_dc = $p.ActiveOverlayDcPowerScheme} | ConvertTo-Json -Compress
"""


def power_status() -> dict:
    """Alimentation au lancement d'une mesure (Windows) : secteur ou batterie, charge, plan et mode d'alimentation.
    Sur batterie le processeur est bridé : Stockfish (au temps) s'affaiblit, l'IA (simulations fixes) non, ce qui
    gonfle l'Elo (règle de CLAUDE.md : mesures sur secteur, mode « meilleures performances »)."""
    if os.name != "nt":
        return {"on_ac": None, "note": "non mesuré (hors Windows)"}
    try:
        raw = subprocess.run(["powershell", "-NoProfile", "-Command", _POWER_PS], capture_output=True, text=True,
                             timeout=30).stdout.strip()
        d = json.loads(raw)
    except Exception as e:  # pas de quoi bloquer une mesure, mais on le note
        return {"on_ac": None, "note": f"non mesuré ({type(e).__name__})"}
    status = d.get("battery_status")
    on_ac = None if status is None else status != 1  # Win32_Battery : 1 = en décharge, 2 = sur secteur, ...
    if status is None:  # pas de batterie : PC fixe, toujours sur secteur
        on_ac = True
    overlay = (d.get("overlay_ac") if on_ac else d.get("overlay_dc")) or "00000000-0000-0000-0000-000000000000"
    plan = d.get("plan") or ""
    return {"on_ac": on_ac, "battery_status": status, "battery_percent": d.get("battery_percent"),
            "power_mode": POWER_MODES.get(overlay.lower(), f"inconnu ({overlay})"), "power_mode_guid": overlay,
            "power_plan": plan[plan.find("(") + 1:plan.rfind(")")] if "(" in plan else plan}


def power_warning(power: dict) -> str | None:
    """Texte d'avertissement si la mesure ne se fait pas sur secteur en mode « meilleures performances »."""
    problems = []
    if power.get("on_ac") is False:
        problems.append(f"SUR BATTERIE ({power.get('battery_percent')} %)")
    elif power.get("on_ac") is None:
        problems.append(f"alimentation inconnue ({power.get('note')})")
    if power.get("on_ac") is not False and power.get("power_mode") not in (None, "meilleures performances"):
        problems.append(f"mode d'alimentation « {power.get('power_mode')} »")
    if not problems:
        return None
    line = "!" * 100
    return (f"{line}\nATTENTION : {', '.join(problems)}.\nRègle de mesure : sur secteur, mode « meilleures "
            f"performances ». Sinon le processeur est bridé : Stockfish (300 ms par coup) s'affaiblit, l'IA "
            f"(simulations fixes) non, et l'Elo mesuré est gonflé.\n{line}")


def environment(args) -> dict:
    import torch

    cpu = platform.processor()
    try:
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=20).stdout.strip() or cpu
    except Exception:
        pass
    checkpoint = Path(args.checkpoint).resolve()
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "workers": args.workers, "torch_threads_per_worker": args.torch_threads,
            "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
            "power": power_status()}  # au dernier lancement (une reprise le remplace)


def task_key(t: dict) -> tuple:
    return t["level"], t["round"], t["opening"], bool(t["ai_white"])


def round_tasks(protocol: dict, level: int, rnd: int, phase) -> list[dict]:
    """Un tour = chaque ouverture du protocole x 2 couleurs (v1 : 20 parties, v2 : 80)."""
    return [{"level": level, "round": rnd, "opening": o, "ai_white": c, "phase": phase}
            for o in range(len(protocol["openings"])) for c in (True, False)]


def parse_tag(spec: str, default_version: str) -> tuple[str, str]:
    """« v1/kaggle_sup » -> ("v1", "kaggle_sup") ; « kaggle_sup » -> (default_version, "kaggle_sup")."""
    version, _, tag = spec.rpartition("/")
    version = version or default_version
    if version not in PROTOCOLS:
        sys.exit(f"protocole inconnu : {version} (connus : {', '.join(PROTOCOLS)})")
    return version, tag


class Results:
    """Fichier JSON des résultats d'un tag ; chaque partie est enregistrée dès qu'elle est finie."""

    def __init__(self, spec: str, default_version: str = "v2"):
        self.version, self.tag = parse_tag(spec, default_version)
        self.protocol = PROTOCOLS[self.version]
        self.label = f"{self.version}/{self.tag}"
        self.path = OUT_DIR / self.version / f"{self.tag}.json"
        self.pgn_path = OUT_DIR / self.version / f"{self.tag}.pgn"
        self.data = {"tag": self.tag, "protocol": self.protocol, "ai": {"syzygy": None}, "environment": {},
                     "refine_levels": None, "games": []}
        self.watch: SpeedWatch | None = None  # surveillance de la vitesse (voir chesszero/speedwatch.py)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            self.data.setdefault("ai", {"syzygy": None})  # résultats v1 : jamais de tables

    @property
    def ai(self) -> str:
        ai = self.data["ai"]
        return f"Syzygy {ai.get('syzygy_pieces')} pièces" if ai.get("syzygy") else "sans tables"

    @property
    def games(self) -> list[dict]:
        return self.data["games"]

    def done_keys(self) -> set:
        return {task_key(g) for g in self.games}

    def add(self, game: dict) -> None:
        self.games.append(game)
        self.save()
        pgn = chess.pgn.Game()
        pgn.headers.update({
            "Event": f"ChessZero MCTS-{self.data['protocol']['simulations']} contre {game['stockfish']} (UCI_Elo {game['level']})",
            "Site": "-", "Date": "????.??.??", "Round": str(len(self.games)), "Opening": game["opening_name"],
            "White": "ChessZero" if game["ai_white"] else f"Stockfish {game['level']}",
            "Black": f"Stockfish {game['level']}" if game["ai_white"] else "ChessZero",
            "Result": {1.0: "1-0", 0.0: "0-1", 0.5: "1/2-1/2"}[game["score"] if game["ai_white"] else 1 - game["score"]],
            "Termination": game["termination"], "StockfishElo": str(game["level"]),
        })
        node = pgn
        for uci in game["moves"].split():
            node = node.add_variation(chess.Move.from_uci(uci))
        with open(self.pgn_path, "a", encoding="utf-8") as f:
            print(pgn, file=f, end="\n\n")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def pairs(self, phases=None) -> list[tuple[float, float]]:
        return [(g["level"], g["score"]) for g in self.games if phases is None or g["phase"] in phases]

    def level_score(self, level: int) -> float | None:
        s = [g["score"] for g in self.games if g["level"] == level]
        return sum(s) / len(s) if s else None


def play_tasks(pool, results: Results | None, tasks: list[dict], label: str) -> list[dict]:
    done = results.done_keys() if results else set()
    todo = [t for t in tasks if task_key(t) not in done]
    print(f"\n== {label} : {len(todo)} parties à jouer ({len(tasks) - len(todo)} déjà faites)", flush=True)
    out, start = [], time.perf_counter()
    watch = results.watch if results is not None else None
    for game in pool.imap_unordered(play_game, todo):
        out.append(game)
        speed = ""
        if watch is not None:
            power = read_power()  # appel système direct : instantané
            info = watch.add(game["ai_sec_per_move"], power)
            game["ai_rolling_s_per_move"] = round(info["rolling"], 4)
            if power:  # piste du bridage pendant la recharge : batterie et état de charge à chaque partie
                game["battery_percent"], game["power_state"] = power["percent"], power["state"]
            results.data["speed"] = watch.state
            speed = f" | moy.{watch.window} {info['rolling']:.2f} s ({info['deviation']:+.0%})"
            if power:
                speed += f" | {power['percent']} % {power['state']}"
            if info["event"] == "start":
                print(speed_banner(watch, info), flush=True)
            elif info["event"] == "end":
                print(f"--- vitesse revenue dans la tolérance (moyenne glissante {info['rolling']:.2f} s par coup) ---",
                      flush=True)
        if results:
            results.add(game)
        n = len(out)
        eta = (time.perf_counter() - start) / n * (len(todo) - n)
        res = {1.0: "gain", 0.5: "nulle", 0.0: "défaite"}[game["score"]]
        print(f"{n:>3}/{len(todo)} niveau {game['level']} tour {game['round']} {game['opening_name']:<19} "
              f"IA {'Blancs' if game['ai_white'] else 'Noirs '} | {res:<7} | {game['termination']:<20} | "
              f"{game['plies']:>3} demi-coups | {game['ai_sec_per_move']:.2f} s/coup IA | {game['seconds']:.0f} s | "
              f"reste ~{eta / 60:.0f} min{speed}", flush=True)
    return out


def speed_banner(watch: SpeedWatch, info: dict) -> str:
    line = "!" * 100
    return (f"{line}\nATTENTION VITESSE : temps par coup de l'IA (moyenne des {watch.window} dernières parties) = "
            f"{info['rolling']:.2f} s, soit {info['deviation']:+.0%} par rapport au bench de départ "
            f"({watch.reference:.2f} s ; tolérance ±{watch.tolerance:.0%}).\nLe processeur a changé de vitesse en cours "
            f"de mesure (bridage, autre charge, veille d'écran ?) : Stockfish joue au temps et change de force, "
            f"l'IA non.\nChercher la cause ; si cela dure, interrompre (relancer = reprendre) : la mesure ne sera pas "
            f"combinable avec la référence.\nAlimentation à cet instant : "
            f"{(info.get('power') or {}).get('percent')} % {(info.get('power') or {}).get('state', 'non relevée')} "
            f"(piste du bridage pendant la recharge : voir les changements d'état dans le résumé final).\n{line}")


def resolve_syzygy(path: str | None) -> dict:
    """--syzygy : bloc « ai » des résultats (dossier vérifié, nombre de pièces couvertes)."""
    if not path:
        return {"syzygy": None}
    from chesszero.syzygy import Tablebase

    tb = Tablebase.open(path)
    if tb is None:
        sys.exit(f"--syzygy : aucune table Syzygy dans {path}")
    pieces = tb.max_pieces
    tb.close()
    return {"syzygy": str(Path(path).resolve()), "syzygy_pieces": pieces}


def make_pool(args, protocol: dict, ai: dict):
    if args.workers * args.torch_threads > MAX_THREADS:
        sys.exit(f"{args.workers} workers x {args.torch_threads} threads > {MAX_THREADS} : limite de threads dépassée")
    ctx = mp.get_context("spawn")
    return ctx.Pool(args.workers, init_worker,
                    (str(args.checkpoint), str(args.stockfish), protocol, args.torch_threads, ai["syzygy"]))


def refine_levels(protocol: dict, fit_rating: float) -> list[int]:
    step = protocol["refine_rounding"]
    centre = round(fit_rating / step) * step
    levels = [min(max(centre + o, ELO_MIN), ELO_MAX) for o in protocol["refine_offsets"]]
    return sorted(set(levels))


def run_protocol(pool, results: Results) -> None:
    p = results.protocol
    phase1 = [t for lv in p["phase1_levels"] for t in round_tasks(p, lv, 0, 1)]
    play_tasks(pool, results, phase1, "phase 1 (grossière)")
    while True:  # extension : tant que l'IA écrase le niveau le plus haut joué, on monte
        top = max(g["level"] for g in results.games if g["phase"] in (1, "x"))
        score_top = results.level_score(top)
        print(f"\nScore de l'IA à {top} : {score_top:.1%}", flush=True)
        if score_top <= p["extension_trigger"] or top >= ELO_MAX:
            break
        nxt = min(top + p["extension_step"], ELO_MAX)
        print(f"Plus de {p['extension_trigger']:.0%} à {top} : niveau supplémentaire {nxt}", flush=True)
        play_tasks(pool, results, round_tasks(p, nxt, 0, "x"), f"niveau supplémentaire {nxt}")

    if results.data["refine_levels"] is None:
        fit = fit_elo(results.pairs(phases=(1, "x")))
        print(f"\nPhase 1 : Elo ajusté {fit.describe()}", flush=True)
        results.data["refine_levels"] = refine_levels(p, fit.rating)
        results.save()
    levels = results.data["refine_levels"]
    tasks = [t for lv in levels for r in range(1, 1 + p["refine_rounds"]) for t in round_tasks(p, lv, r, 2)]
    play_tasks(pool, results, tasks, f"phase 2 (affinage) : niveaux {levels}")


def replay_protocol(pool, results: Results, like: Results) -> None:
    tasks = [{"level": g["level"], "round": g["round"], "opening": g["opening"], "ai_white": g["ai_white"],
              "phase": g["phase"]} for g in like.games]
    results.data["refine_levels"] = like.data["refine_levels"]
    play_tasks(pool, results, tasks, f"rejeu exact de « {like.label} » ({len(tasks)} parties)")


# --------------------------------------------------------------------------- rapports


def report(results: Results) -> None:
    pairs = results.pairs()
    print(f"\n=== {results.label} ({results.protocol['name']}, {len(results.protocol['openings'])} ouvertures) : "
          f"{len(pairs)} parties, checkpoint pas {sorted({g['checkpoint_step'] for g in results.games})}, "
          f"IA {results.ai}, {results.games[0]['stockfish']} ===")
    print(f"{'niveau SF':>9} {'parties':>7} {'V':>3} {'N':>3} {'D':>3} {'score':>7}   Elo implicite (95 %)")
    for r in per_level(pairs):
        elo = f"{r['elo']:.0f} ({r['low']:.0f} à {r['high']:.0f})" if r["elo"] is not None else "non mesurable"
        print(f"{r['level']:>9.0f} {r['n']:>7} {r['wins']:>3} {r['draws']:>3} {r['losses']:>3} {r['score']:>7.1%}   {elo}")
    fit = fit_elo(pairs)
    print(f"\nElo estimé (échelle UCI_Elo de Stockfish, ajustement sur toutes les parties) : {fit.describe()}")
    top = max(r["level"] for r in per_level(pairs))
    if results.level_score(top) > 0.8 or results.level_score(min(r["level"] for r in per_level(pairs))) < 0.2:
        print("ATTENTION : l'IA est loin de 50 % au niveau le plus haut ou le plus bas testé : l'estimation "
              "extrapole, élargir la plage de niveaux.")
    secs = [g["seconds"] for g in results.games]
    print(f"Durée moyenne d'une partie : {statistics.mean(secs):.0f} s ; "
          f"{statistics.mean(g['ai_sec_per_move'] for g in results.games):.2f} s par coup de l'IA")
    state, text = speed_verdict(results.data.get("speed"))
    print(("ATTENTION : " if state == "NON STABLE" else "") + text)
    print("Limites : échelle CCRL Blitz de Stockfish (pas FIDE/Lichess), calibrée à 60 s + 0,6 s par coup alors que "
          f"Stockfish joue ici {results.protocol['stockfish_movetime_ms']} ms ; parties rejouées depuis les mêmes ouvertures "
          "non indépendantes ; intervalle donc un peu optimiste.")


def compare(a: Results, b: Results, force: bool = False) -> None:
    if a.data["protocol"] != b.data["protocol"] and not force:
        sys.exit("Protocoles différents : comparaison refusée (--force pour passer outre).")
    fa, fb = fit_elo(a.pairs()), fit_elo(b.pairs())
    print(f"\n=== Comparaison : {a.label} -> {b.label} ===")
    print(f"{a.label:>20} : {fa.describe()}  ({fa.n} parties, IA {a.ai})")
    print(f"{b.label:>20} : {fb.describe()}  ({fb.n} parties, IA {b.ai})")
    try:
        d, se = difference(fa, fb)
    except ValueError as exc:
        print(f"Différence : {exc}")
        return
    verdict = "significatif à 95 %" if abs(d) > 1.96 * se else "pas significatif à 95 %"
    print(f"Différence : {d:+.0f} Elo (± {1.96 * se:.0f} à 95 %) : {verdict}")
    sa, sb = {r["level"]: r for r in per_level(a.pairs())}, {r["level"]: r for r in per_level(b.pairs())}
    print(f"{'niveau':>7} {a.label:>20} {b.label:>20}")
    for lv in sorted(set(sa) | set(sb)):
        fmt = lambda r: f"{r['score']:.0%} ({r['n']})" if r else "-"
        print(f"{lv:>7.0f} {fmt(sa.get(lv)):>20} {fmt(sb.get(lv)):>20}")


def bench_tasks(protocol: dict, n: int) -> list[dict]:
    """`n` parties de bench (niveaux et ouvertures variés ; rien n'est enregistré)."""
    n_open = len(protocol["openings"])
    levels = [1320, 1600, 1900, 2200, 2500, 2300, 2400, 2500]
    return [{"level": levels[i % len(levels)], "round": 0, "opening": (i * 7) % n_open, "ai_white": i % 2 == 0,
             "phase": "bench"} for i in range(n)]


def start_speed_watch(pool, results: Results, games: int) -> SpeedWatch:
    """Bench de départ (une partie par processus, non enregistrées) = vitesse de référence de la mesure. Une
    reprise garde la référence du premier lancement et compare le nouveau bench à elle."""
    played = play_tasks(pool, None, bench_tasks(results.protocol, games), "bench de départ de la vitesse")
    s_move = statistics.mean(g["ai_sec_per_move"] for g in played)
    previous = results.data.get("speed")
    watch = SpeedWatch(s_move, state=previous if previous and previous.get("games") else None)
    problem = watch.bench(s_move, len(played))
    power = read_power()
    if power:
        print(f"Alimentation au départ : {power['percent']} % {power['state']}", flush=True)
        watch.note_power({**power, "game": 0})
        watch.state["benches"][-1].update({"battery_percent": power["percent"], "power_state": power["state"]})
    print(f"Vitesse de l'IA au bench de départ : {s_move:.2f} s par coup (référence de cette mesure : "
          f"{watch.reference:.2f} s ; mesure v2 de référence : {watch.state['nominal_s_per_move']:.2f} s)", flush=True)
    if problem:
        line = "!" * 100
        print(f"{line}\nATTENTION VITESSE AU DÉPART : {problem}.\nMême cause possible que la nuit du 2026-10-22 "
              f"(processeur ~3 fois plus lent sur secteur). Ne pas laisser tourner une mesure dans cet état.\n{line}",
              flush=True)
    results.watch = watch
    results.data["speed"] = watch.state
    return watch


def bench(args) -> None:
    """Parties en parallèle (niveaux et ouvertures variés) pour estimer la durée du protocole."""
    protocol = PROTOCOLS[args.protocol]
    n_open = len(protocol["openings"])
    tasks = bench_tasks(protocol, args.bench_games or args.workers)
    power = power_status()
    print(f"Alimentation : {'secteur' if power.get('on_ac') else 'batterie' if power.get('on_ac') is False else '?'}, "
          f"mode « {power.get('power_mode')} »", flush=True)
    if power_warning(power):
        print(power_warning(power), flush=True)
    with make_pool(args, protocol, resolve_syzygy(args.syzygy)) as pool:
        start = time.perf_counter()
        games = play_tasks(pool, None, tasks, f"benchmark ({len(tasks)} parties, rien n'est enregistré)")
        wall = time.perf_counter() - start
    mean_game = statistics.mean(g["seconds"] for g in games)
    ai = statistics.mean(g["ai_sec_per_move"] for g in games)
    plies = statistics.mean(g["plies"] for g in games)
    print(f"\nBenchmark : {wall:.0f} s pour {len(games)} parties ({args.workers} à la fois) ; partie moyenne "
          f"{mean_game:.0f} s ({plies:.0f} demi-coups, {ai:.2f} s par coup IA)")
    per_game = mean_game / args.workers
    per_round = 2 * n_open
    base = per_round * (len(protocol["phase1_levels"]) + len(protocol["refine_offsets"]) * protocol["refine_rounds"])
    for total, what in ((base, "protocole sans extension"), (base + per_round, "avec un niveau d'extension")):
        print(f"  {protocol['name']}, {what} ({total} parties) : ~{total * per_game / 60:.0f} min")


def endgames(results: Results, syzygy: str | None) -> None:
    """Finales : pour chaque partie, résultat exact (tables) à la première position couverte, puis score réel.

    Mesure ce que l'IA fait des finales qu'elle atteint, avec ou sans tables pendant la partie : un gain exact
    rendu en nulle ou en perte = des points perdus en finale ; une nulle ou une perte exacte transformée en
    mieux = des points pris sur les erreurs de Stockfish.
    """
    from chesszero.syzygy import Tablebase

    tb = Tablebase.open(syzygy)
    if tb is None:
        sys.exit(f"endgames : il faut les tables Syzygy (--syzygy, défaut {DEFAULT_SYZYGY})")
    names = {2: "gain", 1: "nulle", 0: "nulle", -1: "nulle", -2: "perte"}
    table: dict[tuple[str, float], int] = {}
    entered, plies_after = 0, []
    for g in results.games:
        board = chess.Board()
        moves = [chess.Move.from_uci(u) for u in g["moves"].split()]
        ai_color = chess.WHITE if g["ai_white"] else chess.BLACK
        for i, move in enumerate(moves + [None]):
            if tb.covers(board):
                w = tb.wdl(board)
                if w is not None:
                    w = w if board.turn == ai_color else -w
                    key = (names[w], g["score"])
                    table[key] = table.get(key, 0) + 1
                    entered += 1
                    plies_after.append(len(moves) - i)
                    break
            if move is not None:
                board.push(move)
    print(f"\n=== Finales de {results.label} (IA {results.ai}) : {entered} parties sur {len(results.games)} "
          f"entrent dans les tables ({tb.max_pieces} pièces) ===")
    if not entered:
        return
    print(f"{'à l entrée':>12} {'parties':>8} {'gagnées':>8} {'nulles':>7} {'perdues':>8}   points perdus / pris")
    lost = gained = 0.0
    for entry, exact in (("gain", 1.0), ("nulle", 0.5), ("perte", 0.0)):
        row = [table.get((entry, s), 0) for s in (1.0, 0.5, 0.0)]
        n = sum(row)
        if not n:
            continue
        delta = row[0] * 1.0 + row[1] * 0.5 - n * exact
        lost += max(0.0, -delta)
        gained += max(0.0, delta)
        print(f"{entry:>12} {n:>8} {row[0]:>8} {row[1]:>7} {row[2]:>8}   {delta:+.1f}")
    print(f"Points perdus par l'IA en finale (par rapport au jeu parfait) : {lost:.1f} ; pris sur les erreurs de "
          f"Stockfish : {gained:.1f} ; demi-coups joués après l'entrée : {statistics.mean(plies_after):.0f} en moyenne")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["bench", "run", "report", "compare", "endgames"])
    ap.add_argument("tags", nargs="*", help="report / endgames : <tag> ; compare : <tag_ancien> <tag_nouveau>")
    ap.add_argument("--protocol", default="v2", choices=sorted(PROTOCOLS), help="protocole (défaut v2)")
    ap.add_argument("--tag", default=None, help="run : nom du résultat (docs/stockfish/<protocole>/<tag>.json)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--stockfish", default=str(DEFAULT_STOCKFISH))
    ap.add_argument("--syzygy", default=None,
                    help=f"run / bench : tables de finales pour l'IA ; endgames : tables d'analyse (défaut {DEFAULT_SYZYGY})")
    ap.add_argument("--like", default=None, help="run : rejouer exactement les parties de ce tag")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--bench-games", type=int, default=0, help="bench : nombre de parties (défaut = --workers)")
    ap.add_argument("--force", action="store_true", help="compare / --like : ignorer les différences de protocole")
    ap.add_argument("--no-speed-watch", action="store_true",
                    help="run : pas de bench de départ ni de suivi de la vitesse de l'IA")
    args = ap.parse_args()

    if args.command == "report":
        report(Results(args.tags[0], args.protocol))
    elif args.command == "compare":
        compare(Results(args.tags[0], args.protocol), Results(args.tags[1], args.protocol), args.force)
    elif args.command == "endgames":
        for spec in args.tags:
            endgames(Results(spec, args.protocol), args.syzygy or str(DEFAULT_SYZYGY))
    elif args.command == "bench":
        bench(args)
    else:
        if not args.tag:
            sys.exit("run : --tag est obligatoire")
        results = Results(args.tag, args.protocol)
        ai = resolve_syzygy(args.syzygy)
        like = Results(args.like, results.version) if args.like else None
        if like is not None:
            if not like.games:
                sys.exit(f"« {like.label} » n'a aucun résultat")
            if like.version != results.version or (like.data["protocol"] != results.protocol and not args.force):
                sys.exit(f"Le protocole de « {like.label} » diffère de celui de « {results.label} » : "
                         "le rejeu ne serait pas identique (--force pour passer outre).")
        if results.games and results.data["protocol"] != results.protocol:
            sys.exit(f"« {results.label} » a été commencé avec un autre protocole : choisir un autre --tag.")
        if results.games and results.data["ai"] != ai:
            sys.exit(f"« {results.label} » a été commencé avec d'autres options d'IA ({results.data['ai']}) : "
                     "choisir un autre --tag.")
        results.data["protocol"], results.data["ai"] = results.protocol, ai
        results.data["environment"] = environment(args)
        warning = power_warning(results.data["environment"]["power"])
        if warning:
            print(warning, flush=True)
        print(f"{results.label} : {results.protocol['name']}, IA {results.ai}", flush=True)
        start = time.perf_counter()
        with make_pool(args, results.protocol, ai) as pool:
            if not args.no_speed_watch:
                start_speed_watch(pool, results, args.workers)
                results.save()
            if like is not None:
                replay_protocol(pool, results, like)
            else:
                run_protocol(pool, results)
        results.save()
        print(f"\nTerminé en {(time.perf_counter() - start) / 60:.1f} min")
        report(results)
        if like is not None:
            compare(like, results, args.force)


if __name__ == "__main__":
    main()
