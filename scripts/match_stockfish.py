"""Match contre Stockfish bridé (UCI_LimitStrength / UCI_Elo) : estimation d'Elo de l'IA.

Protocole de référence (le même pour tous les checkpoints, voir docs/journal.md) :
  - IA : MCTS 200 simulations par coup, lots de 8, arbre réutilisé, température 0, sans bruit ;
  - Stockfish : `go movetime 300`, 1 thread, Hash 16 Mo, niveau fixé par UCI_Elo ;
  - 10 ouvertures de match.py (6 demi-coups), chacune jouée avec les deux couleurs : 20 parties par
    « tour » et par niveau ; fin de partie : règles officielles, nulle au-delà de 400 demi-coups ;
  - phase 1 : niveaux 1320, 1600, 1900, 2200, 2500 (un tour = 20 parties chacun) ;
    tant que l'IA marque plus de 80 % au niveau le plus haut joué, un niveau 300 Elo au-dessus
    (2800, puis 3100, puis 3190 = plafond de UCI_Elo), 20 parties chacun ;
  - phase 2 : Elo ajusté sur la phase 1, arrondi à 50, puis 3 niveaux (−100, 0, +100) × 2 tours
    supplémentaires (40 parties chacun). Elo final = ajustement sur toutes les parties.
  - 4 processus en parallèle, 2 threads torch chacun, Stockfish 1 thread : 8 threads au plus.

Commandes (à lancer avec .venv\\Scripts\\python.exe) :
  match_stockfish.py bench                        mesure la vitesse sur 4 parties (rien n'est enregistré)
  match_stockfish.py run --tag kaggle_sup         protocole complet ; reprend tout seul si interrompu
  match_stockfish.py run --tag essai_X --checkpoint checkpoints/.../best.pt --like kaggle_sup
                                                  rejoue EXACTEMENT les mêmes parties (niveaux, tours,
                                                  ouvertures, couleurs) sur un autre checkpoint, puis compare
  match_stockfish.py report kaggle_sup            tableau par niveau + Elo estimé
  match_stockfish.py compare kaggle_sup essai_X   écart d'Elo entre deux résultats

Résultats : docs/stockfish/<tag>.json (toutes les parties, protocole, environnement), parties en PGN
dans docs/stockfish/<tag>.pgn. Stockfish n'est pas déterministe (le tirage des coups faibles dépend de
l'horloge) : deux lancements du même protocole ne rejouent pas exactement les mêmes coups.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from match import OPENINGS, termination  # noqa: E402  (mêmes ouvertures que le match précédent)

DEFAULT_STOCKFISH = ROOT / "tools" / "stockfish" / "stockfish" / "stockfish-windows-x86-64-universal.exe"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "kaggle_sup" / "best.pt"
OUT_DIR = ROOT / "docs" / "stockfish"
MAX_THREADS = 8

# Tout ce qui peut changer le résultat d'une partie. Comparer deux résultats n'a de sens que si ce
# bloc est identique (sauf --force).
PROTOCOL = {
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
ELO_MIN, ELO_MAX = 1320, 3190  # plage de UCI_Elo dans Stockfish


# --------------------------------------------------------------------------- parties (dans les workers)

_W: dict = {}


def init_worker(checkpoint: str, stockfish: str, protocol: dict, torch_threads: int) -> None:
    import torch

    from chesszero.checkpoint import load_for_inference
    from chesszero.mcts import MCTSConfig, NetEvaluator

    torch.set_num_threads(torch_threads)
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
    mcts = MCTS(_W["evaluator"], _W["cfg"], board)
    game_key = f"{task['level']}/{task['round']}/{task['opening']}/{int(ai_white)}"
    start, ai_seconds, ai_moves = time.perf_counter(), 0.0, 0
    while terminal_value(board) is None and len(board.move_stack) < p["max_plies"]:
        if board.turn == ai_color:
            t = time.perf_counter()
            mcts.search()
            move = mcts.best_move(0)
            ai_seconds += time.perf_counter() - t
            ai_moves += 1
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
            "seconds": time.perf_counter() - start, "checkpoint_step": _W["step"], "stockfish": _W["sf_name"]}


# --------------------------------------------------------------------------- orchestration


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
            "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint)}


def task_key(t: dict) -> tuple:
    return t["level"], t["round"], t["opening"], bool(t["ai_white"])


def round_tasks(level: int, rnd: int, phase) -> list[dict]:
    """Un tour = 10 ouvertures x 2 couleurs = 20 parties."""
    return [{"level": level, "round": rnd, "opening": o, "ai_white": c, "phase": phase}
            for o in range(len(OPENINGS)) for c in (True, False)]


class Results:
    """Fichier JSON des résultats d'un tag ; chaque partie est enregistrée dès qu'elle est finie."""

    def __init__(self, tag: str):
        self.tag = tag
        self.path = OUT_DIR / f"{tag}.json"
        self.pgn_path = OUT_DIR / f"{tag}.pgn"
        self.data = {"tag": tag, "protocol": PROTOCOL, "environment": {}, "refine_levels": None, "games": []}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

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
    for game in pool.imap_unordered(play_game, todo):
        out.append(game)
        if results:
            results.add(game)
        n = len(out)
        eta = (time.perf_counter() - start) / n * (len(todo) - n)
        res = {1.0: "gain", 0.5: "nulle", 0.0: "défaite"}[game["score"]]
        print(f"{n:>3}/{len(todo)} niveau {game['level']} tour {game['round']} {game['opening_name']:<19} "
              f"IA {'Blancs' if game['ai_white'] else 'Noirs '} | {res:<7} | {game['termination']:<20} | "
              f"{game['plies']:>3} demi-coups | {game['ai_sec_per_move']:.2f} s/coup IA | {game['seconds']:.0f} s | "
              f"reste ~{eta / 60:.0f} min", flush=True)
    return out


def make_pool(args):
    if args.workers * args.torch_threads > MAX_THREADS:
        sys.exit(f"{args.workers} workers x {args.torch_threads} threads > {MAX_THREADS} : limite de threads dépassée")
    ctx = mp.get_context("spawn")
    return ctx.Pool(args.workers, init_worker, (str(args.checkpoint), str(args.stockfish), PROTOCOL, args.torch_threads))


def refine_levels(fit_rating: float) -> list[int]:
    step = PROTOCOL["refine_rounding"]
    centre = round(fit_rating / step) * step
    levels = [min(max(centre + o, ELO_MIN), ELO_MAX) for o in PROTOCOL["refine_offsets"]]
    return sorted(set(levels))


def run_protocol(pool, results: Results) -> None:
    phase1 = [t for lv in PROTOCOL["phase1_levels"] for t in round_tasks(lv, 0, 1)]
    play_tasks(pool, results, phase1, "phase 1 (grossière)")
    while True:  # extension : tant que l'IA écrase le niveau le plus haut joué, on monte
        top = max(g["level"] for g in results.games if g["phase"] in (1, "x"))
        score_top = results.level_score(top)
        print(f"\nScore de l'IA à {top} : {score_top:.1%}", flush=True)
        if score_top <= PROTOCOL["extension_trigger"] or top >= ELO_MAX:
            break
        nxt = min(top + PROTOCOL["extension_step"], ELO_MAX)
        print(f"Plus de {PROTOCOL['extension_trigger']:.0%} à {top} : niveau supplémentaire {nxt}", flush=True)
        play_tasks(pool, results, round_tasks(nxt, 0, "x"), f"niveau supplémentaire {nxt}")

    if results.data["refine_levels"] is None:
        fit = fit_elo(results.pairs(phases=(1, "x")))
        print(f"\nPhase 1 : Elo ajusté {fit.describe()}", flush=True)
        results.data["refine_levels"] = refine_levels(fit.rating)
        results.save()
    levels = results.data["refine_levels"]
    tasks = [t for lv in levels for r in range(1, 1 + PROTOCOL["refine_rounds"]) for t in round_tasks(lv, r, 2)]
    play_tasks(pool, results, tasks, f"phase 2 (affinage) : niveaux {levels}")


def replay_protocol(pool, results: Results, like: Results) -> None:
    tasks = [{"level": g["level"], "round": g["round"], "opening": g["opening"], "ai_white": g["ai_white"],
              "phase": g["phase"]} for g in like.games]
    results.data["refine_levels"] = like.data["refine_levels"]
    play_tasks(pool, results, tasks, f"rejeu exact de « {like.tag} » ({len(tasks)} parties)")


# --------------------------------------------------------------------------- rapports


def report(results: Results) -> None:
    pairs = results.pairs()
    print(f"\n=== {results.tag} : {len(pairs)} parties, checkpoint pas "
          f"{sorted({g['checkpoint_step'] for g in results.games})}, {results.games[0]['stockfish']} ===")
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
    print("Limites : échelle CCRL Blitz de Stockfish (pas FIDE/Lichess), calibrée à 60 s + 0,6 s par coup alors que "
          f"Stockfish joue ici {PROTOCOL['stockfish_movetime_ms']} ms ; parties rejouées depuis les mêmes ouvertures "
          "non indépendantes ; intervalle donc un peu optimiste.")


def compare(a: Results, b: Results, force: bool = False) -> None:
    if a.data["protocol"] != b.data["protocol"] and not force:
        sys.exit("Protocoles différents : comparaison refusée (--force pour passer outre).")
    fa, fb = fit_elo(a.pairs()), fit_elo(b.pairs())
    print(f"\n=== Comparaison : {a.tag} -> {b.tag} ===")
    print(f"{a.tag:>20} : {fa.describe()}  ({fa.n} parties)")
    print(f"{b.tag:>20} : {fb.describe()}  ({fb.n} parties)")
    try:
        d, se = difference(fa, fb)
    except ValueError as exc:
        print(f"Différence : {exc}")
        return
    verdict = "significatif à 95 %" if abs(d) > 1.96 * se else "pas significatif à 95 %"
    print(f"Différence : {d:+.0f} Elo (± {1.96 * se:.0f} à 95 %) : {verdict}")
    sa, sb = {r["level"]: r for r in per_level(a.pairs())}, {r["level"]: r for r in per_level(b.pairs())}
    print(f"{'niveau':>7} {a.tag:>20} {b.tag:>20}")
    for lv in sorted(set(sa) | set(sb)):
        fmt = lambda r: f"{r['score']:.0%} ({r['n']})" if r else "-"
        print(f"{lv:>7.0f} {fmt(sa.get(lv)):>20} {fmt(sb.get(lv)):>20}")


def bench(args) -> None:
    """4 parties en parallèle (une par worker, niveaux variés) pour estimer la durée du protocole."""
    levels = [1320, 1900, 2200, 2500]
    tasks = [{"level": lv, "round": 0, "opening": i, "ai_white": i % 2 == 0, "phase": "bench"}
             for i, lv in enumerate(levels[: args.workers])]
    with make_pool(args) as pool:
        start = time.perf_counter()
        games = play_tasks(pool, None, tasks, f"benchmark ({len(tasks)} parties, rien n'est enregistré)")
        wall = time.perf_counter() - start
    mean_game = statistics.mean(g["seconds"] for g in games)
    ai = statistics.mean(g["ai_sec_per_move"] for g in games)
    plies = statistics.mean(g["plies"] for g in games)
    print(f"\nBenchmark : {wall:.0f} s pour {len(games)} parties simultanées ; partie moyenne {mean_game:.0f} s "
          f"({plies:.0f} demi-coups, {ai:.2f} s par coup IA)")
    per_game = mean_game / args.workers
    for total, what in ((220, "protocole sans extension"), (240, "avec un niveau d'extension")):
        print(f"  {what} ({total} parties) : ~{total * per_game / 60:.0f} min")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["bench", "run", "report", "compare"])
    ap.add_argument("tags", nargs="*", help="report : <tag> ; compare : <tag_ancien> <tag_nouveau>")
    ap.add_argument("--tag", default=None, help="run : nom du résultat (docs/stockfish/<tag>.json)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--stockfish", default=str(DEFAULT_STOCKFISH))
    ap.add_argument("--like", default=None, help="run : rejouer exactement les parties de ce tag")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="compare / --like : ignorer les différences de protocole")
    args = ap.parse_args()

    if args.command == "report":
        report(Results(args.tags[0]))
    elif args.command == "compare":
        compare(Results(args.tags[0]), Results(args.tags[1]), args.force)
    elif args.command == "bench":
        bench(args)
    else:
        if not args.tag:
            sys.exit("run : --tag est obligatoire")
        results = Results(args.tag)
        like = Results(args.like) if args.like else None
        if like is not None:
            if not like.games:
                sys.exit(f"« {args.like} » n'a aucun résultat")
            if like.data["protocol"] != PROTOCOL and not args.force:
                sys.exit(f"Le protocole de « {args.like} » diffère du protocole actuel du script : "
                         "le rejeu ne serait pas identique (--force pour passer outre).")
        if results.games and results.data["protocol"] != PROTOCOL:
            sys.exit(f"« {args.tag} » a été commencé avec un autre protocole : choisir un autre --tag.")
        results.data["protocol"], results.data["environment"] = PROTOCOL, environment(args)
        start = time.perf_counter()
        with make_pool(args) as pool:
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
