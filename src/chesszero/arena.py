"""Arène (étape 7) : un réseau A (candidat) contre un réseau B (accepté, ou ancre `kaggle_sup`).

Parties :
  - départs tirés de la RÉSERVE D'ARÈNE du livre (`books/<nom>_arena.tsv.gz`, jamais tirée par le self-play ; le
    livre exclut déjà les 40 ouvertures de la mesure v2), chaque ouverture jouée deux fois en inversant les
    couleurs (une « paire ») ;
  - ouvertures de l'arène ANCRE : les `anchor_pairs` premières d'un ordre fixe de la réserve, les mêmes à chaque
    génération (comparaisons appariées d'une génération à l'autre) ; arène CONTRE L'ACCEPTÉ (« gate ») : tirées
    parmi les autres avec une graine (numéro du candidat), nouvelles à chaque arène ;
  - `simulations` par coup (200 = self-play et mesure v2), température 0, sans bruit, un arbre par réseau
    réutilisé d'un coup à l'autre, tables WDL dans l'arbre et arbitrage dès `tablebase_pieces` pièces, nulle à
    `max_plies` demi-coups, coups forcés joués sans recherche ;
  - abandon seulement si LES DEUX CAMPS sont d'accord : le camp au trait a Q racine et Q de son meilleur coup
    < `resign_threshold` sur ses `resign_consecutive` derniers coups, et l'adversaire > −`resign_threshold` sur
    les siens, à partir du demi-coup `resign_min_ply` ;
  - aucune position n'est enregistrée pour l'entraînement.

Plusieurs parties par processus : à chaque itération, les arbres des parties où A est au trait partent au réseau A
dans un même appel (`search_batched`), ceux où B est au trait au réseau B.

Score, Elo, intervalle : score moyen de A (victoire 1, nulle 0,5) ; erreur type calculée PAR PAIRES d'ouvertures
(les deux parties d'une même ouverture ne sont pas indépendantes ; estimateur par grappes, `score_se`, = écart-type
des scores moyens des paires / racine du nombre de paires quand elles sont complètes ; plancher d'une demi-partie) ;
Elo = −400 log10(1/score − 1), intervalle à 95 % = image de score ± 1,96 erreur type ; probabilité que A soit
meilleur = Φ((score − 0,5) / erreur type). Toujours des nombres finis (cas limites : voir `summarize`).

Décision (`gate_decision`) : acceptation si score ≥ `accept_score` (52 % = 104 points sur 200) ; veto de l'ancre
si l'Elo du candidat contre l'ancre est inférieur à celui du réseau accepté de plus de `anchor_veto_se` erreurs
types de l'écart.

Fichiers d'une arène (dossier `out`) et ordre d'écriture :
  arena.json   description (joueurs avec SHA-256 de leur checkpoint, ouvertures, réglages, code git), écrite de
               façon atomique au premier lancement ; une reprise vérifie qu'elle décrit la même arène ;
  games.jsonl  une ligne par partie terminée, ajoutée et forcée sur disque (fsync) au fil de l'eau par le seul
               processus principal ; une dernière ligne tronquée (coupure pendant l'écriture) est ignorée et
               retirée à la reprise ; la reprise ne rejoue que les couples (ouverture, couleur) manquants ;
  games.pgn    toutes les parties, écrit (atomique) quand l'arène est complète ;
  result.json  résumé, écrit (atomique) EN DERNIER : sa présence signifie « arène terminée ».
Les fichiers temporaires `*.tmp` d'une écriture atomique interrompue sont ignorés puis écrasés.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import subprocess
import time
import tomllib
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from statistics import NormalDist
from typing import Callable

import chess
import chess.pgn
import numpy as np

from chesszero.mcts import CONFIG_DIR, MCTS, MCTSConfig, search_batched
from chesszero.rating import elo_from_score
from chesszero.selfplay import T_MAX_PLIES, T_REPETITION, T_RESIGN, TERMINATIONS, end_of_game

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ArenaConfig:
    """Section [arena] de configs/<nom>.toml."""

    pairs: int = 100  # ouvertures de l'arène contre l'accepté (x 2 couleurs = 200 parties)
    anchor_pairs: int = 100  # ouvertures fixes de l'arène contre l'ancre
    simulations: int = 200  # nouvelles simulations par coup (défaut des deux joueurs)
    leaves_per_tree: int = 4  # feuilles posées par arbre à chaque appel au réseau
    games_per_worker: int = 64  # parties menées ensemble par processus
    resign_threshold: float = -0.90
    resign_consecutive: int = 3
    resign_min_ply: int = 20
    max_plies: int = 400
    tablebase_pieces: int = 5
    accept_score: float = 0.52  # acceptation : score >= 52 % contre le réseau accepté
    anchor_veto_se: float = 2.0  # veto : Elo contre l'ancre < celui de l'accepté - 2 erreurs types de l'écart

    @classmethod
    def from_toml(cls, path: str | Path) -> ArenaConfig:
        with open(path, "rb") as f:
            section = tomllib.load(f).get("arena", {})
        unknown = set(section) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"clés inconnues dans [arena] : {sorted(unknown)}")
        return cls(**section)

    @classmethod
    def named(cls, name: str) -> ArenaConfig:
        return cls.from_toml(CONFIG_DIR / f"{name}.toml")

    def mcts_config(self, base: MCTSConfig, simulations: int) -> MCTSConfig:
        return replace(base, num_simulations=simulations, batch_size=self.leaves_per_tree, dirichlet_epsilon=0.0,
                       temperature=0.0)


@dataclass(frozen=True)
class Player:
    name: str  # nom affiché (PGN, rapport)
    checkpoint: str | None  # chemin du checkpoint (None : évaluateur fourni directement, tests)
    simulations: int


# --------------------------------------------------------------------------- ouvertures et programme


def _reserve_order(n_reserve: int) -> np.ndarray:
    return np.random.default_rng(zlib.crc32(b"arena-anchor")).permutation(n_reserve)


def anchor_openings(n_reserve: int, n: int) -> list[int]:
    """Ouvertures fixes de l'arène ancre : les `n` premières d'un ordre fixe de la réserve."""
    if n > n_reserve:
        raise ValueError(f"{n} ouvertures demandées, réserve de {n_reserve}")
    return [int(i) for i in _reserve_order(n_reserve)[:n]]


def gate_openings(n_reserve: int, n: int, seed: int, anchor_pairs: int) -> list[int]:
    """Ouvertures de l'arène contre l'accepté : `n` tirées parmi les positions de la réserve hors ancre."""
    rest = _reserve_order(n_reserve)[anchor_pairs:]
    if n > len(rest):
        raise ValueError(f"{n} ouvertures demandées, {len(rest)} disponibles hors ancre")
    rng = np.random.default_rng(zlib.crc32(f"arena-gate/{seed}".encode()))
    return sorted(int(i) for i in rng.choice(rest, size=n, replace=False))


def game_key(opening: int, a_white: bool) -> str:
    return f"{opening}:{'A-blancs' if a_white else 'A-noirs'}"


def schedule(openings: list[int]) -> list[dict]:
    """Deux parties par ouverture (A avec les Blancs, puis avec les Noirs)."""
    if len(set(openings)) != len(openings):
        raise ValueError("ouvertures en double")
    return [{"key": game_key(o, a_white), "opening": o, "a_white": a_white} for o in openings for a_white in (True, False)]


# --------------------------------------------------------------------------- une partie


class ArenaGame:
    """Une partie A contre B : un arbre par joueur, tous deux avancés de chaque coup joué."""

    def __init__(self, spec: dict, board: chess.Board, trees: dict[str, MCTS]):
        self.spec = spec
        self.trees = trees  # {"a": MCTS, "b": MCTS}, chacun avec sa copie de la position (historique compris)
        self.start_plies = len(board.move_stack)
        self.low = {"a": 0, "b": 0}  # coups d'affilée où le joueur se voit perdant (Q < seuil)
        self.high = {"a": 0, "b": 0}  # coups d'affilée où le joueur se voit gagnant (Q > -seuil)
        self.result_white: int | None = None
        self.termination: int | None = None
        self.reported = False

    @property
    def board(self) -> chess.Board:
        return self.trees["a"].board

    @property
    def done(self) -> bool:
        return self.result_white is not None

    def mover(self) -> str:
        """Joueur au trait : "a" si (Blancs au trait) == (A a les Blancs)."""
        return "a" if (self.board.turn == chess.WHITE) == self.spec["a_white"] else "b"

    def check_end(self, cfg: ArenaConfig, tablebase) -> None:
        end = end_of_game(self.board, tablebase)
        if end is not None:
            self.result_white, self.termination = end
        elif len(self.board.move_stack) >= cfg.max_plies:
            self.result_white, self.termination = 0, T_MAX_PLIES

    def play(self, move: chess.Move, cfg: ArenaConfig, tablebase) -> None:
        for tree in self.trees.values():
            tree.advance(move)
        self.check_end(cfg, tablebase)

    def play_forced_moves(self, cfg: ArenaConfig, tablebase) -> None:
        while not self.done and self.board.legal_moves.count() == 1:
            self.play(next(iter(self.board.legal_moves)), cfg, tablebase)

    def after_search(self, cfg: ArenaConfig, tablebase) -> None:
        """Le joueur au trait vient de chercher : abandon (si les deux camps sont d'accord) ou coup le plus visité."""
        p = self.mover()
        other = "b" if p == "a" else "a"
        tree = self.trees[p]
        root = tree.root
        best = int(np.lexsort((root.P_raw, root.N))[-1])
        q_root, q_best = root.q(), float(root.W[best] / root.N[best])
        thr = cfg.resign_threshold
        self.low[p] = self.low[p] + 1 if (q_root < thr and q_best < thr) else 0
        self.high[p] = self.high[p] + 1 if (q_root > -thr and q_best > -thr) else 0
        if (self.low[p] >= cfg.resign_consecutive and self.high[other] >= cfg.resign_consecutive
                and len(self.board.move_stack) >= cfg.resign_min_ply):
            self.result_white = -1 if self.board.turn == chess.WHITE else 1
            self.termination = T_RESIGN
            return
        self.play(tree.best_move(0.0), cfg, tablebase)

    def record(self) -> dict:
        rw = int(self.result_white)
        ra = rw if self.spec["a_white"] else -rw
        return {**self.spec, "score_a": (ra + 1) / 2, "result_white": rw, "termination": TERMINATIONS[self.termination],
                "plies": len(self.board.move_stack), "start_plies": self.start_plies,
                "moves": " ".join(m.uci() for m in self.board.move_stack)}


def play_games(specs: list[dict], reserve, evaluators: dict, players: dict[str, Player], cfg: ArenaConfig,
               base_mcts: MCTSConfig, emit: Callable[[dict], bool | None], tablebase=None,
               hard_deadline: float | None = None) -> dict:
    """Joue les parties `specs` (au plus `games_per_worker` à la fois) ; `emit(record)` à chaque partie terminée
    (s'il renvoie False : arrêt immédiat). À `hard_deadline` (temps `time.time()`), arrêt : les parties en cours
    sont perdues (rejouées à la reprise)."""
    mcts_cfg = {p: cfg.mcts_config(base_mcts, players[p].simulations) for p in ("a", "b")}
    pending, games = list(specs), []
    stats = {"games": 0, "unfinished_dropped": 0, "simulations": {"a": 0, "b": 0}, "seconds": 0.0,
             "stopped": False}
    rng = np.random.default_rng(0)  # sans bruit ni température : jamais utilisé pour choisir un coup
    start = time.time()

    def new_game(spec: dict) -> ArenaGame:
        board = reserve.board(spec["opening"])
        trees = {p: MCTS(evaluators[p], mcts_cfg[p], board, rng=rng, tablebase=tablebase) for p in ("a", "b")}
        game = ArenaGame(spec, board, trees)
        game.check_end(cfg, tablebase)
        return game

    while True:
        while len(games) < cfg.games_per_worker and pending:
            games.append(new_game(pending.pop(0)))
        if not games:
            break
        if hard_deadline is not None and time.time() >= hard_deadline:
            stats["unfinished_dropped"] += len(games) + len(pending)
            break
        for g in games:
            g.play_forced_moves(cfg, tablebase)
        active = [g for g in games if not g.done]
        movers = {id(g): g.mover() for g in active}
        for p in ("a", "b"):
            trees = [g.trees[p] for g in active if movers[id(g)] == p]
            if trees:
                search_batched(trees, evaluators[p], players[p].simulations, cfg.leaves_per_tree)
                stats["simulations"][p] += players[p].simulations * len(trees)
        for g in active:
            g.after_search(cfg, tablebase)
        for g in games:
            if g.done and not g.reported:
                g.reported = True
                stats["games"] += 1
                if emit(g.record()) is False:
                    stats["stopped"] = True
                    stats["seconds"] = time.time() - start
                    return stats
        games = [g for g in games if not g.done]
    stats["seconds"] = time.time() - start
    return stats


def _worker(task: dict, queue) -> None:
    """Processus fils : charge les réseaux, la réserve et les tables, joue ses parties, envoie chaque partie
    terminée au processus principal (seul à écrire games.jsonl)."""
    import sys
    import traceback

    from chesszero.cpu_guard import disable_ecoqos

    disable_ecoqos()  # pas de bridage d'efficacité de Windows (sans effet ailleurs)
    try:
        import torch

        from chesszero.book import OpeningBook
        from chesszero.checkpoint import load_for_inference
        from chesszero.device import get_device
        from chesszero.mcts import NetEvaluator
        from chesszero.syzygy import Tablebase

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        torch.set_num_threads(task["threads"])
        device = get_device(task["device"])
        cfg = ArenaConfig(**task["config"])
        players = {p: Player(**d) for p, d in task["players"].items()}
        models = {}
        for pl in players.values():  # un même checkpoint (contrôle « contre lui-même ») n'est chargé qu'une fois
            if pl.checkpoint not in models:
                models[pl.checkpoint] = load_for_inference(pl.checkpoint, device)[0]
        evaluators = {p: NetEvaluator(models[pl.checkpoint]) for p, pl in players.items()}
        tb = (Tablebase.open(task["syzygy"], max_pieces=cfg.tablebase_pieces, wdl_only=True)
              if task["syzygy"] else None)
        reserve = OpeningBook.load(task["reserve"])
        stats = play_games(task["specs"], reserve, evaluators, players, cfg, MCTSConfig(**task["mcts"]),
                           emit=lambda r: queue.put(("game", r)), tablebase=tb, hard_deadline=task["hard_deadline"])
        stats["eval_positions"] = {p: e.positions for p, e in evaluators.items()}
        stats["eval_calls"] = {p: e.calls for p, e in evaluators.items()}
        queue.put(("done", task["worker"], stats))
    except BaseException:
        queue.put(("error", task["worker"], traceback.format_exc()))
        raise


def process_runner(players: dict[str, Player], cfg: ArenaConfig, base_mcts: MCTSConfig, reserve_path: str,
                   syzygy: str | None, workers: int, threads: int = 1, device: str | None = None,
                   hard_deadline: float | None = None, stats_out: list | None = None) -> Callable:
    """`run_games` pour `run_arena` : `workers` processus (méthode spawn), parties réparties à tour de rôle.
    Les statistiques de chaque processus (simulations, temps, appels au réseau) sont ajoutées à `stats_out`."""
    import multiprocessing as mp
    import queue as queue_mod

    def run(todo: list[dict], emit) -> list[dict]:
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        chunks = [todo[w::workers] for w in range(workers) if todo[w::workers]]
        base = {"players": {p: asdict(pl) for p, pl in players.items()}, "config": asdict(cfg),
                "mcts": base_mcts.to_dict(), "reserve": str(reserve_path), "syzygy": syzygy, "threads": threads,
                "device": device, "hard_deadline": hard_deadline}
        procs = [ctx.Process(target=_worker, args=({**base, "worker": w, "specs": c}, q), daemon=True)
                 for w, c in enumerate(chunks)]
        for p in procs:
            p.start()
        finished, results = 0, []
        try:
            while finished < len(procs):
                try:
                    kind, *payload = q.get(timeout=30)
                except queue_mod.Empty:
                    dead = [p for p in procs if p.exitcode not in (None, 0)]
                    if dead:
                        raise RuntimeError(f"processus d'arène arrêté (code {dead[0].exitcode}) sans résultat")
                    continue
                if kind == "game":
                    if emit(payload[0]) is False:
                        break
                elif kind == "done":
                    finished += 1
                    results.append(payload[1])
                else:
                    raise RuntimeError(f"erreur dans le processus d'arène {payload[0]} :\n{payload[1]}")
        finally:
            for p in procs:
                if p.is_alive():
                    p.terminate()
                p.join()
        if stats_out is not None:
            stats_out.extend(results)
        return results

    return run


# --------------------------------------------------------------------------- résultats


def score_se(by_pair: dict[int, list[float]], score: float, n: int) -> float:
    """Erreur type du score moyen, les paires d'ouvertures étant des grappes (estimateur « par grappes » :
    (m / (m − 1)) × Σ_paires (Σ des écarts au score)² / n²). Pour des paires complètes, c'est exactement
    l'écart-type des scores moyens des paires / √m ; il reste juste quand certaines paires n'ont qu'une partie
    (arène reprise ou incomplète). Plancher = une demi-partie (0,5 / n) : le score n'est pas connu plus finement
    que l'effet d'un demi-point sur une partie ; sans lui, des paires toutes identiques (ou que des nulles, ou
    100 %) donneraient une erreur nulle, donc un intervalle vide et des décisions sur un écart de 1 Elo."""
    m = len(by_pair)
    cluster = 0.0
    if m >= 2:
        cluster = math.sqrt(m / (m - 1) * sum(sum(x - score for x in v) ** 2 for v in by_pair.values()) / n ** 2)
    return max(cluster, 0.5 / n)


def summarize(records: list[dict]) -> dict:
    """Score de A, Elo et intervalle (erreur type par paires d'ouvertures), probabilité que A soit meilleur, fins.
    Toujours des nombres finis (JSON strict). Score de 0 % ou 100 % : Elo calculé avec une demi-partie de moins
    (n − 0,5 sur n), noté comme une borne (`elo_bound` = "lower" ou "upper") ; une borne d'intervalle qui sortirait
    de ]0, 1[ en score est None (non bornée)."""
    if not records:
        raise ValueError("aucune partie")
    scores = [r["score_a"] for r in records]
    by_pair = defaultdict(list)
    for r in records:
        by_pair[r["opening"]].append(r["score_a"])
    n = len(scores)
    score = sum(scores) / n
    se = score_se(by_pair, score, n)
    half = 0.5 / n
    s = min(max(score, half), 1 - half)  # 0 % / 100 % : une demi-partie de moins
    elo = elo_from_score(s) + 0.0  # + 0.0 : pas de « −0 »
    elo_bound = "lower" if score >= 1 else "upper" if score <= 0 else None
    lo_s, hi_s = score - 1.96 * se, score + 1.96 * se
    low = elo_from_score(lo_s) + 0.0 if lo_s > 0 else None
    high = elo_from_score(hi_s) + 0.0 if hi_s < 1 else None
    elo_se = se * 400 / (math.log(10) * s * (1 - s))
    los = NormalDist().cdf((score - 0.5) / se)
    terms = Counter(r["termination"] for r in records)
    early_reps = sum(r["termination"] == TERMINATIONS[T_REPETITION] and r["plies"] < 60 for r in records)
    complete = sum(len(v) == 2 for v in by_pair.values())
    mirrored = sum(len(v) == 2 and sum(v) == 1.0 for v in by_pair.values())
    return {"games": n, "pairs": len(by_pair), "complete_pairs": complete, "points": sum(scores), "score": score,
            "wins": scores.count(1.0), "draws": scores.count(0.5), "losses": scores.count(0.0),
            "score_se": se, "elo": elo, "elo_bound": elo_bound, "elo_se": elo_se, "elo_low": low, "elo_high": high,
            "los": los,
            "pairs_scored_1": mirrored, "terminations": dict(terms),
            "repetitions_before_ply_60": early_reps,
            "mean_plies": sum(r["plies"] for r in records) / n,
            "white_score": sum((r["result_white"] + 1) / 2 for r in records) / n}


def describe(summary: dict, a: str = "A", b: str = "B") -> str:
    s = summary
    bound = {"lower": "≥ ", "upper": "≤ "}.get(s.get("elo_bound"), "")
    low = f"{s['elo_low']:+.0f}" if s["elo_low"] is not None else "non bornée"
    high = f"{s['elo_high']:+.0f}" if s["elo_high"] is not None else "non bornée"
    ci = (f"Elo {bound}{s['elo']:+.0f} (intervalle à 95 % : {low} à {high}), "
          f"probabilité que {a} soit meilleur {s['los']:.1%}")
    return (f"{a} contre {b} : {s['wins']} victoires, {s['draws']} nulles, {s['losses']} défaites en {s['games']} "
            f"parties ({s['complete_pairs']} paires complètes) ; score {s['score']:.1%} ({s['points']:g} points) ; {ci}")


def gate_decision(gate: dict, cfg: ArenaConfig, anchor: dict | None = None,
                  accepted_anchor: dict | None = None) -> dict:
    """Décision sur un candidat. `gate` : résumé contre l'accepté ; `anchor` / `accepted_anchor` : résumés du
    candidat et du réseau accepté contre l'ancre (veto seulement si les deux sont fournis). Les deux mesures
    contre l'ancre sont bruitées : l'erreur type de l'écart les combine (racine de la somme des carrés, comme si
    elles étaient indépendantes ; jouées sur les mêmes ouvertures, elles sont plutôt corrélées positivement, donc
    cette erreur est un peu surestimée : le veto est prudent). Une arène incomplète ne peut pas être décidée."""
    for name, summ in (("arène contre l'accepté", gate), ("arène ancre", anchor), ("ancre de l'accepté", accepted_anchor)):
        if summ is not None and "complete_pairs" in summ and (summ["complete_pairs"] != summ["pairs"]
                                                              or summ["games"] != 2 * summ["pairs"]):
            raise ValueError(f"{name} incomplète ({summ['games']} parties, {summ['complete_pairs']} paires "
                             f"complètes sur {summ['pairs']}) : pas de décision")
    out = {"accept": True, "reasons": [], "anchor_diff": None, "anchor_diff_se": None}
    if gate["score"] < cfg.accept_score - 1e-9:
        out["accept"] = False
        out["reasons"].append(f"score {gate['score']:.1%} ({gate['points']:g} points sur {gate['games']}) "
                              f"< {cfg.accept_score:.0%}")
    if anchor is not None and accepted_anchor is not None:
        diff = anchor["elo"] - accepted_anchor["elo"]
        if anchor.get("elo_se") is None or accepted_anchor.get("elo_se") is None:
            raise ValueError("veto de l'ancre : erreur type manquante (résumé d'une ancienne version ?)")
        se = math.hypot(anchor["elo_se"], accepted_anchor["elo_se"])
        out["anchor_diff"], out["anchor_diff_se"] = diff, se
        if diff < -cfg.anchor_veto_se * se:
            out["accept"] = False
            out["reasons"].append(f"veto de l'ancre : Elo contre l'ancre {diff:+.0f} par rapport au réseau accepté "
                                  f"(< -{cfg.anchor_veto_se:g} x {se:.0f})")
    return out


# --------------------------------------------------------------------------- fichiers


def atomic_write_text(path: str | Path, text: str) -> None:
    """Écrit dans `<path>.tmp`, force sur disque, puis renomme : `path` est soit l'ancien, soit le nouveau."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: str | Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def read_jsonl(path: str | Path, repair: bool = False) -> list[dict]:
    """Lignes JSON d'un fichier écrit par ajouts. Ce qui suit le dernier saut de ligne est une ligne tronquée
    (coupure pendant l'écriture) : ignorée, et retirée du fichier si `repair`. Une ligne complète illisible est
    une corruption : erreur."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1  # 0 s'il n'y a aucune ligne complète
    rows = []
    for i, raw in enumerate(data[:end].split(b"\n")[:-1]):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} : ligne {i + 1} illisible (fichier corrompu)") from e
    if repair and end < len(data):
        with open(path, "r+b") as f:
            f.truncate(end)
    return rows


def append_jsonl(f, row: dict) -> None:
    """Ajoute une ligne complète et la force sur disque (une coupure ne peut tronquer que cette ligne)."""
    f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    f.flush()
    os.fsync(f.fileno())


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(root: Path = ROOT) -> dict:
    """Commit du code et présence de modifications non commitées (None hors d'un dépôt git)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20, check=True).stdout
    try:
        return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(git("status", "--porcelain").strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def pgn_text(records: list[dict], players: dict[str, Player], event: str) -> str:
    out = io.StringIO()
    for r in sorted(records, key=lambda r: (r["opening"], not r["a_white"])):
        game = chess.pgn.Game()
        white, black = (players["a"], players["b"]) if r["a_white"] else (players["b"], players["a"])
        game.headers.update({"Event": event, "Round": r["key"], "White": f"{white.name} ({white.simulations} sim)",
                             "Black": f"{black.name} ({black.simulations} sim)",
                             "Result": {1: "1-0", 0: "1/2-1/2", -1: "0-1"}[r["result_white"]],
                             "Termination": r["termination"], "PlyCount": str(r["plies"]),
                             "Opening": f"réserve n° {r['opening']} ({r['start_plies']} demi-coups)"})
        node = game
        for uci in r["moves"].split():
            node = node.add_variation(chess.Move.from_uci(uci))
        print(game, file=out, end="\n\n")
    return out.getvalue()


def arena_spec(players: dict[str, Player], openings: list[int], cfg: ArenaConfig, base_mcts: MCTSConfig,
               reserve_name: str, kind: str, label: str) -> dict:
    """Ce qui définit une arène (comparé à la reprise) ; les chemins des checkpoints, eux, peuvent changer
    d'une session Kaggle à l'autre : ils sont identifiés par leur SHA-256."""
    return {"label": label, "kind": kind, "reserve": reserve_name, "openings": openings,
            "players": {p: {"name": pl.name, "simulations": pl.simulations,
                            "sha256": sha256_file(pl.checkpoint) if pl.checkpoint else None}
                        for p, pl in players.items()},
            "config": asdict(cfg), "mcts": base_mcts.to_dict()}


def run_arena(out_dir: str | Path, players: dict[str, Player], openings: list[int], cfg: ArenaConfig,
              base_mcts: MCTSConfig, reserve, reserve_name: str, kind: str = "list", label: str = "",
              run_games: Callable | None = None, stop_after: int | None = None, log=print) -> dict | None:
    """Joue (ou reprend) une arène dans `out_dir`. `run_games(todo, emit) -> stats` joue des parties (processus,
    ou en direct dans les tests). `stop_after` : s'arrête après ce nombre de nouvelles parties (coupure simulée).
    Renvoie le résumé si l'arène est complète, sinon None (relancer pour reprendre)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = arena_spec(players, openings, cfg, base_mcts, reserve_name, kind, label)
    spec_path = out / "arena.json"
    if spec_path.exists():
        old = json.loads(spec_path.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k in spec} != spec:
            raise ValueError(f"{spec_path} décrit une autre arène (joueurs, ouvertures ou réglages différents)")
        spec = old
    else:
        write_json_atomic(spec_path, {**spec, "git": git_info(), "created": time.time()})
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text(encoding="utf-8"))

    games_path = out / "games.jsonl"
    done = {r["key"]: r for r in read_jsonl(games_path, repair=True)}
    plan = schedule(openings)
    todo = [s for s in plan if s["key"] not in done]
    log(f"Arène {label or out.name} : {len(plan)} parties, {len(done)} déjà jouées, {len(todo)} à jouer")
    t0 = time.time()
    if todo:
        new = 0
        with open(games_path, "ab") as f:
            def emit(record: dict) -> bool:
                """Enregistre une partie ; renvoie False quand il faut s'arrêter (coupure simulée)."""
                nonlocal new
                if record["key"] in done:
                    return True
                append_jsonl(f, record)
                done[record["key"]] = record
                new += 1
                if len(done) % 10 == 0 or len(done) == len(plan):
                    s = summarize(list(done.values()))
                    log(f"  {len(done)}/{len(plan)} parties : score de {players['a'].name} {s['score']:.1%}, "
                        f"{(time.time() - t0) / 60:.1f} min")
                return stop_after is None or new < stop_after

            run_games(todo, emit)
    records = [done[s["key"]] for s in plan if s["key"] in done]
    if len(records) < len(plan):
        log(f"Arène incomplète : {len(records)}/{len(plan)} parties (relancer pour reprendre)")
        return None
    summary = summarize(records)
    atomic_write_text(out / "games.pgn", pgn_text(records, players, f"ChessZero arène {label or out.name}"))
    result = {"label": label, "kind": kind, "players": spec["players"], "summary": summary,
              "git": git_info(), "config": spec["config"], "mcts": spec["mcts"], "finished": time.time()}
    write_json_atomic(out / "result.json", result)  # en dernier : l'arène est terminée
    return result


def append_history(path: str | Path, entry: dict) -> bool:
    """Ajoute `entry` (avec un champ `id`) à un historique JSONL, sauf si cet `id` y est déjà (reprise après une
    coupure entre result.json et l'historique). Renvoie True si la ligne a été ajoutée."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(r.get("id") == entry["id"] for r in read_jsonl(path, repair=True)):
        return False
    with open(path, "ab") as f:
        append_jsonl(f, entry)
    return True
