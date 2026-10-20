"""Interface UCI : fait jouer le MCTS guidé par le réseau dans n'importe quel logiciel d'échecs.

Protocole UCI = texte sur stdin / stdout. Le logiciel (Cute Chess, Arena...) lance ce programme, lui
envoie `position ...` puis `go ...`, et lit `info ...` puis `bestmove ...`. Ici :

  - `go movetime X`                : recherche pendant X ms ;
  - `go wtime ... btime ... [winc binc movestogo]` : partie à la pendule, budget = temps restant / 30
    (ou / movestogo) + 80 % de l'incrément, jamais plus de la moitié du temps restant ;
  - `go nodes N`                   : N simulations ;
  - `go infinite`                  : jusqu'à `stop` ;
  - `go` sans limite               : option `MoveTime` si > 0, sinon option `Simulations`.
La recherche tourne dans un thread, par petits paquets de simulations : `stop`, `isready` et `quit`
répondent aussitôt. Le coup renvoyé est toujours le plus visité (température 0, pas de bruit).

Réutilisation de l'arbre : si la nouvelle position prolonge celle de l'arbre (notre coup + la réponse
adverse), on joue ces coups dans l'arbre (`advance`) et on garde le sous-arbre. Sinon (nouvelle partie,
position sans rapport, option changée) on repart d'un arbre vide.

stdout ne contient QUE du protocole UCI : tout message de diagnostic va sur stderr.
"""

from __future__ import annotations

import math
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


def score_cp(q: float) -> int:
    """Q (camp au trait, [-1, 1]) en centipions, par la courbe de Leela : 90 * tan(1,5638 * Q).
    Convention d'affichage, pas une mesure : ça ne sert qu'à remplir `score cp` pour le logiciel."""
    q = max(-0.99, min(0.99, q))
    return round(90 * math.tan(1.5637541897 * q))


class UCIEngine:
    def __init__(self, out: TextIO | None = None, evaluator=None):
        self.out = out or sys.stdout
        self._write_lock = threading.Lock()
        self.options: dict[str, object] = {
            "Simulations": 200,
            "MoveTime": 0,
            "Threads": MAX_THREADS,
            "BatchSize": 8,
            "ReuseTree": True,
            "Checkpoint": str(DEFAULT_CHECKPOINT),
        }
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
        for line in lines:
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
        self.send("option name Simulations type spin default 200 min 1 max 1000000")
        self.send("option name MoveTime type spin default 0 min 0 max 3600000")
        self.send(f"option name Threads type spin default {MAX_THREADS} min 1 max {MAX_THREADS}")
        self.send("option name BatchSize type spin default 8 min 1 max 64")
        self.send("option name ReuseTree type check default true")
        self.send(f"option name Checkpoint type string default {DEFAULT_CHECKPOINT}")
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
        budget = time_budget(go, board.turn)
        if go.nodes is not None:
            max_sims = max(go.nodes, 1)
        elif go.infinite or budget is not None:
            max_sims = None
        elif int(self.options["MoveTime"]) > 0:
            budget, max_sims = max(int(self.options["MoveTime"]) / 1000 - OVERHEAD, 0.02), None
        else:
            max_sims = int(self.options["Simulations"])
        if go.infinite:
            budget = None
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
                if self._stop.is_set() or (max_sims is not None and done >= max_sims)                         or (budget is not None and now - start >= budget):
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
    engine = UCIEngine(out=stdout)
    engine.run(stdin or sys.stdin)


if __name__ == "__main__":
    main()
