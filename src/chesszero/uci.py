"""Interface UCI : fait jouer le MCTS guidé par le réseau dans n'importe quel logiciel d'échecs.

Protocole UCI = texte sur stdin / stdout. Le logiciel (Cute Chess, Arena...) lance ce programme, lui
envoie `position ...` puis `go ...`, et lit `info ...` puis `bestmove ...`. Ici :

Deux modes de force, choisis par l'option `UseClock` (décochée par défaut) :

  FORCE FIXE (`UseClock` = false) : l'IA fait exactement `Simulations` nouvelles simulations par coup, quelle
  que soit la pendule : `wtime/btime/winc/binc/movetime` du `go` sont ignorés. C'est le comportement de la
  mesure de référence contre Stockfish (200 simulations). Un `go nodes N` explicite remplace `Simulations`.
  PENDULE (`UseClock` = true) : le temps du coup se déduit de la pendule :
    - `go movetime X`                : X ms ;
    - `go wtime ... btime ... [winc binc movestogo]` : temps restant / 30 (ou / movestogo) + 80 % de
      l'incrément, jamais plus de la moitié du temps restant ; `Simulations` est alors ignoré ;
    - `go nodes N` avec un temps     : s'arrête au premier des deux.
Dans les deux modes : un `go` sans aucune limite utilise l'option `MoveTime` (ms) si elle est > 0, sinon
`Simulations`. `go infinite` : jusqu'à `stop` en mode pendule ; en force fixe, l'IA répond après `Simulations`
simulations (écart volontaire au protocole : la pendule « infinie » de Cute Chess envoie `go infinite` et
attend un coup, sinon la partie ne démarre jamais). Dans les deux modes `stop` et `quit` arrêtent la
recherche tout de suite.
La recherche tourne dans un thread, par petits paquets de simulations : `stop`, `isready` et `quit`
répondent aussitôt. Le coup renvoyé est toujours le plus visité (température 0, pas de bruit).

Réutilisation de l'arbre : si la nouvelle position prolonge celle de l'arbre (notre coup + la réponse
adverse), on joue ces coups dans l'arbre (`advance`) et on garde le sous-arbre. Sinon (nouvelle partie,
position sans rapport, option changée) on repart d'un arbre vide.

stdout ne contient QUE du protocole UCI : tout message de diagnostic va sur stderr.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

import chess
import torch

from chesszero.checkpoint import load_for_inference
from chesszero.device import get_device
from chesszero.mcts import MCTS, MCTSConfig, NetEvaluator, terminal_value

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "kaggle_sup" / "best.pt"
MAX_THREADS = 8  # plafond voulu pour ce PC (20 threads logiques, on en garde pour le reste)
MOVES_TO_GO = 30  # horizon supposé pour répartir le temps restant
OVERHEAD = 0.05  # secondes gardées pour la communication avec le logiciel
CHUNK = 16  # simulations entre deux contrôles du temps / de `stop`


@dataclass
class GoParams:
    movetime: int | None = None
    wtime: int | None = None
    btime: int | None = None
    winc: int = 0
    binc: int = 0
    movestogo: int | None = None
    nodes: int | None = None
    infinite: bool = False


def parse_go(tokens: list[str]) -> GoParams:
    """`go wtime 60000 btime 60000 winc 1000 ...` -> GoParams (jetons inconnus ignorés)."""
    params, i = GoParams(), 0
    int_keys = {"movetime", "wtime", "btime", "winc", "binc", "movestogo", "nodes"}
    while i < len(tokens):
        key = tokens[i]
        if key == "infinite":
            params.infinite = True
        elif key in int_keys and i + 1 < len(tokens):
            setattr(params, key, int(tokens[i + 1]))
            i += 1
        i += 1
    return params


def time_budget(go: GoParams, turn: chess.Color) -> float | None:
    """Secondes à consacrer au coup d'après les limites du `go`, ou None si aucune limite de temps."""
    if go.movetime is not None:
        return max(go.movetime / 1000 - OVERHEAD, 0.02)
    remaining = go.wtime if turn == chess.WHITE else go.btime
    if remaining is None:
        return None
    remaining /= 1000
    inc = (go.winc if turn == chess.WHITE else go.binc) / 1000
    budget = remaining / (go.movestogo or MOVES_TO_GO) + 0.8 * inc
    return max(min(budget, remaining * 0.5) - OVERHEAD, 0.02)


def search_limits(go: GoParams, turn: chess.Color, use_clock: bool, simulations: int,
                  move_time_ms: int = 0) -> tuple[float | None, int | None]:
    """(secondes, simulations) à ne pas dépasser pour ce `go` ; None = pas de limite de ce type.
    Priorité : infinite > pendule (si UseClock) > nodes > MoveTime (go sans limite) > Simulations."""
    if go.infinite:  # force fixe : on répond quand même (pendule « infinie » de Cute Chess), voir l'en-tête
        return (None, None) if use_clock else (None, simulations)
    nodes = max(go.nodes, 1) if go.nodes is not None else None
    budget = time_budget(go, turn) if use_clock else None
    if budget is not None:
        return budget, nodes
    if nodes is not None:
        return None, nodes
    has_time = go.movetime is not None or go.wtime is not None or go.btime is not None
    if not has_time and move_time_ms > 0:
        return max(move_time_ms / 1000 - OVERHEAD, 0.02), None
    return None, simulations


def score_cp(q: float) -> int:
    """Q (camp au trait, [-1, 1]) en centipions, par la courbe de Leela : 90 * tan(1,5638 * Q).
    Convention d'affichage, pas une mesure : ça ne sert qu'à remplir `score cp` pour le logiciel."""
    q = max(-0.99, min(0.99, q))
    return round(90 * math.tan(1.5637541897 * q))


class UCIEngine:
    def __init__(self, out: TextIO | None = None, evaluator=None, options: dict | None = None):
        self.out = out or sys.stdout
        self._write_lock = threading.Lock()
        self.options: dict[str, object] = {
            "Simulations": 200,
            "UseClock": False,
            "MoveTime": 0,
            "Threads": MAX_THREADS,
            "BatchSize": 8,
            "ReuseTree": True,
            "Checkpoint": str(DEFAULT_CHECKPOINT),
        }
        self.options.update(options or {})  # valeurs par défaut d'un profil (arguments de la ligne de commande)
        self.evaluator = evaluator  # injectable (tests) ; sinon chargé depuis `Checkpoint`
        self._injected = evaluator is not None
        self.board = chess.Board()
        self.mcts: MCTS | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    # ---------------------------------------------------------------- sortie

    def send(self, line: str) -> None:
        with self._write_lock:
            print(line, file=self.out, flush=True)

    # ---------------------------------------------------------------- boucle de commandes

    def run(self, lines) -> None:
        """Traite les commandes jusqu'à `quit` ou la fin de l'entrée."""
        log = os.environ.get("CHESSZERO_UCI_LOG")  # diagnostic : copie des commandes reçues du logiciel
        for line in lines:
            if log:
                with open(log, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} {line.strip()}\n")
            if not self.handle(line.strip()):
                break
        self._stop_search()

    def handle(self, line: str) -> bool:
        """Exécute une commande. Renvoie False pour `quit`."""
        tokens = line.split()
        if not tokens:
            return True
        cmd, args = tokens[0], tokens[1:]
        if cmd == "uci":
            self._cmd_uci()
        elif cmd == "isready":
            self._ensure_evaluator()
            self.send("readyok")
        elif cmd == "setoption":
            self._stop_search()
            self._cmd_setoption(args)
        elif cmd == "ucinewgame":
            self._stop_search()
            self.mcts, self.board = None, chess.Board()
        elif cmd == "position":
            self._stop_search()
            self._cmd_position(args)
        elif cmd == "go":
            self._stop_search()
            self._cmd_go(args)
        elif cmd == "stop":
            self._stop_search()
        elif cmd == "quit":
            self._stop_search()
            return False
        # autres commandes (debug, register, ponderhit...) : ignorées
        return True

    def _cmd_uci(self) -> None:
        self.send("id name ChessZero")
        self.send("id author Projet_chess")
        o = self.options
        self.send(f"option name Simulations type spin default {o['Simulations']} min 1 max 1000000")
        self.send(f"option name UseClock type check default {str(o['UseClock']).lower()}")
        self.send(f"option name MoveTime type spin default {o['MoveTime']} min 0 max 3600000")
        self.send(f"option name Threads type spin default {o['Threads']} min 1 max {MAX_THREADS}")
        self.send(f"option name BatchSize type spin default {o['BatchSize']} min 1 max 64")
        self.send(f"option name ReuseTree type check default {str(o['ReuseTree']).lower()}")
        self.send(f"option name Checkpoint type string default {o['Checkpoint']}")
        self.send("uciok")

    def _cmd_setoption(self, args: list[str]) -> None:
        # setoption name <nom> [value <valeur>] ; le nom et la valeur peuvent contenir des espaces
        if "name" not in args:
            return
        i = args.index("name")
        j = args.index("value") if "value" in args else len(args)
        name = " ".join(args[i + 1:j]).strip()
        value = " ".join(args[j + 1:]).strip()
        canonical = next((k for k in self.options if k.lower() == name.lower()), None)
        if canonical is None:
            self.send(f"info string option inconnue : {name}")
            return
        current = self.options[canonical]
        if isinstance(current, bool):
            self.options[canonical] = value.lower() == "true"
        elif isinstance(current, int):
            self.options[canonical] = int(value)
        else:
            self.options[canonical] = value
        if canonical == "Threads":
            self.options["Threads"] = max(1, min(int(self.options["Threads"]), MAX_THREADS))
            torch.set_num_threads(self.options["Threads"])
        if canonical == "Checkpoint" and not self._injected:
            self.evaluator = None  # rechargé à la prochaine demande
        self.mcts = None  # la config de recherche a pu changer

    def _cmd_position(self, args: list[str]) -> None:
        if not args:
            return
        if args[0] == "startpos":
            board, rest = chess.Board(), args[1:]
        elif args[0] == "fen":
            end = args.index("moves") if "moves" in args else len(args)
            board, rest = chess.Board(" ".join(args[1:end])), args[end:]
        else:
            return
        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                board.push_uci(uci)
        self.board = board

    # ---------------------------------------------------------------- recherche

    def _ensure_evaluator(self) -> None:
        if self.evaluator is not None:
            return
        torch.set_num_threads(int(self.options["Threads"]))
        path = Path(str(self.options["Checkpoint"]))
        model, meta = load_for_inference(path, get_device())
        self.evaluator = NetEvaluator(model)
        print(f"{path} (pas {meta['step']}) chargé", file=sys.stderr, flush=True)

    def _mcts_for(self, board: chess.Board) -> MCTS:
        """Arbre pour `board` : le sous-arbre existant si `board` prolonge sa position, sinon un neuf."""
        old = self.mcts
        if old is not None and self.options["ReuseTree"]:
            stack, old_stack = board.move_stack, old.board.move_stack
            if (stack[:len(old_stack)] == old_stack and len(stack) >= len(old_stack)
                    and board.root().fen() == old.board.root().fen()):
                try:
                    for move in stack[len(old_stack):]:
                        old.advance(move)
                    return old
                except ValueError:
                    pass  # coup inattendu : on repart de zéro
        cfg = replace(MCTSConfig.named("kaggle"), batch_size=int(self.options["BatchSize"]))
        return MCTS(self.evaluator, cfg, board)

    def _cmd_go(self, args: list[str]) -> None:
        go = parse_go(args)
        self._ensure_evaluator()
        board = self.board.copy()
        if terminal_value(board) is not None:
            self.send("bestmove 0000")
            return
        self.mcts = self._mcts_for(board)
        budget, max_sims = search_limits(go, board.turn, bool(self.options["UseClock"]),
                                         int(self.options["Simulations"]), int(self.options["MoveTime"]))
        self._stop.clear()
        self._worker = threading.Thread(target=self._search, args=(self.mcts, budget, max_sims), daemon=True)
        self._worker.start()

    def _search(self, mcts: MCTS, budget: float | None, max_sims: int | None) -> None:
        try:
            start = time.perf_counter()
            last_info, done = start, 0
            while True:
                n = CHUNK if max_sims is None else min(CHUNK, max_sims - done)
                mcts.search(n)
                done += n
                now = time.perf_counter()
                if (self._stop.is_set() or (max_sims is not None and done >= max_sims)
                        or (budget is not None and now - start >= budget)):
                    break
                if now - last_info >= 1.0:
                    self._send_info(mcts, now - start)
                    last_info = now
            self._send_info(mcts, time.perf_counter() - start)
            move = mcts.best_move(0)
        except Exception as exc:  # le logiciel attend toujours un bestmove, même si la recherche a planté
            print(f"erreur de recherche : {exc!r}", file=sys.stderr, flush=True)
            move = next(iter(mcts.board.legal_moves))
            self.mcts = None
        self.send(f"bestmove {move.uci()}")

    def _send_info(self, mcts: MCTS, seconds: float) -> None:
        root = mcts.root
        pv = mcts.principal_variation()
        best = int(root.N.argmax())
        q = float(root.W[best] / root.N[best]) if root.N[best] else 0.0
        nodes = int(root.n_sum)
        self.send(f"info depth {len(pv)} nodes {nodes} nps {int(nodes / max(seconds, 1e-3))} "
                  f"time {int(seconds * 1000)} score cp {score_cp(q)} pv {' '.join(m.uci() for m in pv)}")

    def _stop_search(self) -> None:
        """Arrête la recherche en cours (si elle existe) et attend son `bestmove`."""
        if self._worker is not None:
            self._stop.set()
            self._worker.join()
            self._worker = None


def main(argv: list[str] | None = None, stdin=None, stdout=None) -> None:
    ap = argparse.ArgumentParser(description="Moteur UCI ChessZero ; les options donnent les valeurs par défaut "
                                 "d'un profil (le logiciel peut ensuite les changer par setoption).")
    ap.add_argument("--simulations", type=int, help="force fixe : simulations par coup (défaut 200)")
    ap.add_argument("--use-clock", action="store_true", help="mode pendule au lieu de la force fixe")
    ap.add_argument("--move-time", type=int, help="ms par coup pour un `go` sans limite")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--checkpoint")
    args = ap.parse_args(argv)
    options = {"Simulations": args.simulations, "UseClock": args.use_clock or None, "MoveTime": args.move_time,
               "Threads": args.threads, "BatchSize": args.batch_size, "Checkpoint": args.checkpoint}
    options = {k: v for k, v in options.items() if v is not None}
    if "Threads" in options:
        options["Threads"] = max(1, min(options["Threads"], MAX_THREADS))
    engine = UCIEngine(out=stdout, options=options)
    engine.run(stdin or sys.stdin)


if __name__ == "__main__":
    main()
