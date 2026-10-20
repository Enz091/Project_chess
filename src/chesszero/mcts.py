"""Recherche arborescente Monte-Carlo (MCTS) guidée par le réseau, façon AlphaZero.

Une simulation :
  1. sélection : depuis la racine, on descend en prenant à chaque nœud le coup qui maximise
     PUCT = Q + c_puct * P * sqrt(N_nœud) / (1 + N_coup) ;
  2. expansion / évaluation : la position atteinte est terminale (valeur exacte) ou évaluée par
     le réseau (priors sur les coups légaux + valeur) ;
  3. remontée : la valeur est ajoutée sur chaque arête du chemin, avec changement de signe à
     chaque niveau ;
le coup joué est le plus visité (ou tiré au sort selon les visites si température > 0).

Conventions de signe (le point le plus délicat) :
  - la valeur du réseau pour une position est vue par le camp au trait DANS cette position ;
  - `W[a]`, `Q[a]` d'une arête sont vus par le camp qui CHOISIT le coup `a` (celui au trait dans
    le nœud qui porte l'arête) : la sélection prend donc directement l'argmax ;
  - `Node.q()` = valeur du nœud pour le camp au trait dans ce nœud
    = (valeur réseau du nœud + somme des W de ses arêtes) / (1 + somme des N de ses arêtes)
    = l'opposé du Q de l'arête qui mène au nœud (même quantité, mais définie aussi à la racine).
  - FPU (coups jamais visités) : Q = `node.q() - fpu_reduction`, donc vue par le camp qui choisit.

Évaluation par lots : on enchaîne plusieurs descentes avant d'appeler le réseau une seule fois. Pour
qu'elles ne suivent pas toutes le même chemin, chaque descente pose une « perte virtuelle » sur ses
arêtes (compteur entier `VL`, compté comme des visites perdues), retirée à la remontée.

Les nœuds ne stockent pas de position : l'objet `MCTS` garde un seul `chess.Board` (avec tout
l'historique, nécessaire pour la triple répétition) et joue / déjoue les coups pendant la descente.

Tables de finales (optionnel, `tablebase=`) : une position de l'arbre couverte par les tables Syzygy devient
une feuille de valeur exacte (comme un mat), et à la racine couverte le coup vient des tables (`best_move`).
Voir `chesszero.syzygy`.
"""

from __future__ import annotations

import math
import time
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Protocol, Sequence

import chess
import numpy as np
import torch
from torch import nn

from chesszero.encoding import N_ACTIONS, move_to_index, pack_board, unpack_batch

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class MCTSConfig:
    """Section [mcts] de configs/<nom>.toml."""

    num_simulations: int = 200  # simulations ajoutées par appel à search()
    c_puct: float = 1.5  # poids de l'exploration (terme P * sqrt(N) / (1 + n))
    batch_size: int = 8  # feuilles évaluées par appel au réseau (perte virtuelle)
    virtual_loss: float = 1.0  # valeur d'une visite virtuelle (comptée comme une défaite)
    fpu_reduction: float = 0.2  # Q d'un coup jamais visité = Q(nœud) - fpu_reduction
    dirichlet_alpha: float = 0.3  # bruit à la racine (self-play) : concentration
    dirichlet_epsilon: float = 0.0  # part du bruit dans les priors de la racine (0 = désactivé)
    temperature: float = 0.0  # 0 = coup le plus visité ; > 0 = tirage selon N^(1/T)

    def __post_init__(self):
        if self.num_simulations < 1 or self.batch_size < 1:
            raise ValueError("num_simulations et batch_size doivent valoir au moins 1")
        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError("dirichlet_epsilon doit être dans [0, 1]")
        if self.temperature < 0:
            raise ValueError("temperature doit être >= 0")

    @classmethod
    def from_toml(cls, path: str | Path) -> MCTSConfig:
        with open(path, "rb") as f:
            section = tomllib.load(f).get("mcts", {})
        unknown = set(section) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"clés inconnues dans [mcts] : {sorted(unknown)}")
        return cls(**section)

    @classmethod
    def named(cls, name: str) -> MCTSConfig:
        return cls.from_toml(CONFIG_DIR / f"{name}.toml")

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- positions terminales


def terminal_value(board: chess.Board) -> float | None:
    """Valeur exacte d'une position terminale pour le camp au trait, ou None si la partie continue.

    Mat -> -1 (le camp au trait est maté) ; pat, matériel insuffisant, règle des 50 coups (nulle
    réclamable : 100 demi-coups sans prise ni coup de pion), triple répétition -> 0.
    Utilise l'historique de `board` (pile de coups) pour la répétition.
    """
    if not any(board.generate_legal_moves()):
        return -1.0 if board.is_check() else 0.0
    if board.is_insufficient_material() or board.halfmove_clock >= 100:
        return 0.0
    # Trois occurrences demandent au moins 8 demi-coups réversibles : on évite le test sinon (il est coûteux).
    if board.halfmove_clock >= 8 and board.is_repetition(3):
        return 0.0
    return None


def legal_moves_and_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    moves = list(board.legal_moves)
    return moves, np.fromiter((move_to_index(m, board) for m in moves), dtype=np.int64, count=len(moves))


# --------------------------------------------------------------------------- évaluateurs


class Evaluator(Protocol):
    def evaluate(
        self, boards: Sequence[chess.Board], legal_indices: Sequence[np.ndarray]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Pour chaque position : priors sur ses coups légaux (dans l'ordre de `legal_indices`, somme 1)
        et valeur dans [-1, 1] pour le camp au trait."""
        ...


class NetEvaluator:
    """Évalue un lot de positions avec le réseau : un seul appel à `model` par lot."""

    def __init__(self, model: nn.Module, device: torch.device | str | None = None):
        self.model = model.eval()
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self.calls = 0
        self.positions = 0
        self.seconds = 0.0  # temps passé dans evaluate (encodage + réseau + softmax)

    @torch.inference_mode()
    def evaluate(self, boards, legal_indices):
        start = time.perf_counter()
        x = torch.from_numpy(unpack_batch(np.stack([pack_board(b) for b in boards]))).to(self.device)
        logits, values = self.model(x)
        logits = logits.float().cpu().numpy()
        priors = []
        for row, idx in zip(logits, legal_indices):
            z = row[idx].astype(np.float64)
            p = np.exp(z - z.max())
            priors.append(p / p.sum())
        self.calls += 1
        self.positions += len(boards)
        self.seconds += time.perf_counter() - start
        return priors, values.float().cpu().numpy().astype(np.float64)


def policy_best_move(evaluator: Evaluator, board: chess.Board) -> chess.Move:
    """Meilleur coup selon la politique seule (sans recherche)."""
    moves, idx = legal_moves_and_indices(board)
    priors, _ = evaluator.evaluate([board], [idx])
    return moves[int(np.argmax(priors[0]))]


# --------------------------------------------------------------------------- arbre


class Node:
    """Une position de l'arbre. Les statistiques de ses arêtes (un coup légal chacune) sont des tableaux."""

    __slots__ = ("terminal", "moves", "idx", "P", "P_raw", "N", "W", "VL", "children",
                 "value", "n_sum", "w_sum", "vl_sum", "pending")

    def __init__(self, terminal: float | None = None):
        self.terminal = terminal  # valeur exacte si la position est terminale
        self.moves: list[chess.Move] | None = None  # None = pas encore développé
        self.idx = self.P = self.P_raw = self.N = self.W = self.VL = None
        self.children: list[Node | None] | None = None
        self.value = 0.0  # valeur du réseau pour ce nœud (camp au trait ici)
        self.n_sum = 0  # somme de N (visites des arêtes)
        self.w_sum = 0.0  # somme de W
        self.vl_sum = 0  # somme de VL
        self.pending = False  # en attente d'évaluation dans le lot en cours

    @property
    def expanded(self) -> bool:
        return self.moves is not None

    def expand(self, moves: list[chess.Move], idx: np.ndarray, priors: np.ndarray, value: float) -> None:
        n = len(moves)
        self.moves, self.idx = moves, idx
        self.P_raw = np.asarray(priors, dtype=np.float64)
        self.P = self.P_raw
        self.N = np.zeros(n, dtype=np.int64)
        self.W = np.zeros(n, dtype=np.float64)
        self.VL = np.zeros(n, dtype=np.int64)
        self.children = [None] * n
        self.value = float(value)

    def q(self) -> float:
        """Valeur moyenne du nœud pour le camp au trait dans ce nœud (sans perte virtuelle)."""
        if self.terminal is not None:
            return self.terminal
        return (self.value + self.w_sum) / (1 + self.n_sum)

    def fpu_value(self, fpu_reduction: float) -> float:
        """Q prêté aux coups jamais visités, du point de vue du camp qui choisit ici."""
        return self.q() - fpu_reduction

    def edge_q(self, virtual_loss: float, fpu_reduction: float) -> np.ndarray:
        """Q de chaque arête (camp qui choisit), perte virtuelle comprise ; FPU pour les arêtes non visitées."""
        n_eff = self.N + self.VL
        q = np.full(len(self.moves), self.fpu_value(fpu_reduction))
        seen = n_eff > 0
        q[seen] = (self.W[seen] - virtual_loss * self.VL[seen]) / n_eff[seen]
        return q

    def select(self, c_puct: float, virtual_loss: float, fpu_reduction: float) -> int:
        n_eff = self.N + self.VL
        u = c_puct * self.P * math.sqrt(1 + self.n_sum + self.vl_sum) / (1 + n_eff)
        return int(np.argmax(self.edge_q(virtual_loss, fpu_reduction) + u))


# --------------------------------------------------------------------------- recherche


class MCTS:
    """Arbre de recherche pour une partie. `advance(move)` après chaque coup joué (des deux camps)."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: MCTSConfig | None = None,
        board: chess.Board | None = None,
        rng: np.random.Generator | None = None,
        check_invariants: bool = False,
        tablebase=None,
    ):
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.board = board.copy() if board is not None else chess.Board()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.check = check_invariants  # vérifie l'arbre après chaque lot (tests)
        self.tablebase = tablebase  # chesszero.syzygy.Tablebase ou None
        self.root = Node(terminal_value(self.board))
        self.stats = {"batches": 0, "evaluated": 0, "terminal": 0, "collisions": 0, "seconds": 0.0}

    # ---------------------------------------------------------------- recherche

    def search(self, num_simulations: int | None = None) -> Node:
        """Ajoute exactement `num_simulations` simulations (config par défaut) à l'arbre."""
        if self.root.terminal is not None:
            raise ValueError(f"position terminale, rien à chercher : {self.board.fen()}")
        n = self.config.num_simulations if num_simulations is None else num_simulations
        start = time.perf_counter()
        if not self.root.expanded:
            self._evaluate_and_expand([([], self.root, self.board.copy(stack=False))])
        if self.config.dirichlet_epsilon > 0:
            self._add_root_noise()
        done = 0
        while done < n:
            done += self._run_batch(min(self.config.batch_size, n - done))
            if self.check:
                self.check_tree()
        self.stats["seconds"] += time.perf_counter() - start
        return self.root

    def _add_root_noise(self) -> None:
        """P = (1 - eps) * P_raw + eps * Dir(alpha), recalculé à partir de P_raw (pas de cumul)."""
        cfg, root = self.config, self.root
        noise = self.rng.dirichlet([cfg.dirichlet_alpha] * len(root.moves))
        root.P = (1 - cfg.dirichlet_epsilon) * root.P_raw + cfg.dirichlet_epsilon * noise

    def _run_batch(self, target: int) -> int:
        """Collecte jusqu'à `target` feuilles (perte virtuelle), les évalue en un lot, remonte.
        Renvoie le nombre de simulations faites (feuilles évaluées + feuilles terminales)."""
        cfg, board = self.config, self.board
        leaves: list[tuple[list, Node, chess.Board]] = []
        n_terminal = collisions = 0
        while len(leaves) + n_terminal < target and collisions < cfg.batch_size:
            node, path = self.root, []
            while node.expanded and node.terminal is None:
                a = node.select(cfg.c_puct, cfg.virtual_loss, cfg.fpu_reduction)
                node.VL[a] += 1
                node.vl_sum += 1
                path.append((node, a))
                board.push(node.moves[a])
                child = node.children[a]
                if child is None:
                    child = node.children[a] = Node(self._exact_value(board))
                node = child
            if node.terminal is not None:
                self._backup(path, node.terminal)
                n_terminal += 1
            elif node.pending:  # feuille déjà dans ce lot : on annule cette descente
                self._undo_virtual_loss(path)
                collisions += 1
            else:
                node.pending = True
                leaves.append((path, node, board.copy(stack=False)))
            for _ in path:
                board.pop()
        if leaves:
            self._evaluate_and_expand(leaves)
        self.stats["batches"] += 1
        self.stats["terminal"] += n_terminal
        self.stats["collisions"] += collisions
        return len(leaves) + n_terminal

    def _exact_value(self, board: chess.Board) -> float | None:
        """Valeur exacte d'une nouvelle feuille : fin de partie, sinon tables de finales (si présentes)."""
        value = terminal_value(board)
        if value is None and self.tablebase is not None:
            value = self.tablebase.value(board)
        return value

    def _evaluate_and_expand(self, leaves) -> None:
        legal = [legal_moves_and_indices(b) for _, _, b in leaves]
        priors, values = self.evaluator.evaluate([b for _, _, b in leaves], [idx for _, idx in legal])
        for (path, node, _), (moves, idx), p, v in zip(leaves, legal, priors, values):
            node.expand(moves, idx, p, float(v))
            node.pending = False
            self._backup(path, float(v))
        self.stats["evaluated"] += len(leaves)

    @staticmethod
    def _backup(path, value: float) -> None:
        """`value` : vue par le camp au trait dans la feuille. L'arête qui y mène appartient à
        l'adversaire : on change le signe AVANT de l'ajouter, puis à chaque niveau."""
        for node, a in reversed(path):
            value = -value
            node.N[a] += 1
            node.W[a] += value
            node.VL[a] -= 1
            node.n_sum += 1
            node.w_sum += value
            node.vl_sum -= 1

    @staticmethod
    def _undo_virtual_loss(path) -> None:
        for node, a in path:
            node.VL[a] -= 1
            node.vl_sum -= 1

    # ---------------------------------------------------------------- choix du coup

    def best_move(self, temperature: float | None = None) -> chess.Move:
        """T = 0 : coup le plus visité (égalité : plus grand prior) ; T > 0 : tirage selon N^(1/T).
        Avec des tables de finales et une racine couverte, le coup des tables est prioritaire."""
        root = self.root
        if not root.expanded or root.n_sum == 0:
            raise ValueError("aucune simulation : appeler search() d'abord")
        if self.tablebase is not None:  # racine couverte par les tables : coup exact (gain le plus direct)
            move = self.tablebase.choose(self.board, dict(zip(root.moves, root.N.tolist())))
            if move is not None:
                return move
        t = self.config.temperature if temperature is None else temperature
        if t == 0:
            a = int(np.lexsort((root.P_raw, root.N))[-1])
        else:
            weights = (root.N / root.N.max()) ** (1.0 / t)
            a = int(self.rng.choice(len(root.moves), p=weights / weights.sum()))
        return root.moves[a]

    def visit_distribution(self) -> np.ndarray:
        """Visites de la racine normalisées, vecteur (N_ACTIONS,) float32 : cible de politique du self-play."""
        out = np.zeros(N_ACTIONS, dtype=np.float32)
        out[self.root.idx] = self.root.N / self.root.n_sum
        return out

    def advance(self, move: chess.Move) -> None:
        """Joue `move` : son sous-arbre (s'il existe) devient la nouvelle racine, le reste est libéré."""
        root, child = self.root, None
        if root.expanded:
            try:
                a = root.moves.index(move)
            except ValueError:
                raise ValueError(f"coup illégal : {move} dans {self.board.fen()}") from None
            child = root.children[a]
        elif not self.board.is_legal(move):
            raise ValueError(f"coup illégal : {move} dans {self.board.fen()}")
        self.board.push(move)
        if child is not None and child.terminal is not None and terminal_value(self.board) is None:
            child = None  # feuille exacte des tables, pas une fin de partie : la racine doit être cherchée
        self.root = child if child is not None else Node(terminal_value(self.board))

    # ---------------------------------------------------------------- affichage et vérifications

    def root_stats(self) -> list[dict]:
        """Coups de la racine triés par visites : coup, SAN, prior (sans bruit), visites, Q (camp au trait)."""
        root = self.root
        out = []
        for a in np.argsort(-root.N, kind="stable"):
            n = int(root.N[a])
            out.append({"move": root.moves[a], "san": self.board.san(root.moves[a]), "prior": float(root.P_raw[a]),
                        "visits": n, "q": float(root.W[a] / n) if n else None})
        return out

    def principal_variation(self, max_len: int = 12) -> list[chess.Move]:
        """Suite des coups les plus visités depuis la racine."""
        pv, node = [], self.root
        while node is not None and node.expanded and node.n_sum > 0 and len(pv) < max_len:
            a = int(np.argmax(node.N))
            pv.append(node.moves[a])
            node = node.children[a]
        return pv

    def check_tree(self) -> int:
        """Vérifie l'arbre (perte virtuelle entièrement retirée, compteurs cohérents). Renvoie le nombre de nœuds."""
        count, stack = 0, [self.root]
        while stack:
            node = stack.pop()
            count += 1
            assert not node.pending, "nœud resté en attente"
            if not node.expanded:
                continue
            assert node.vl_sum == 0 and not node.VL.any(), "perte virtuelle non retirée"
            assert node.n_sum == int(node.N.sum()), "n_sum incohérent"
            assert math.isclose(node.w_sum, float(node.W.sum()), abs_tol=1e-6), "w_sum incohérent"
            for a, child in enumerate(node.children):
                assert (child is not None) == (node.N[a] > 0), "fils créé sans visite (ou l'inverse)"
                if child is None:
                    continue
                assert abs(node.W[a]) <= node.N[a] + 1e-9, "|Q| > 1"
                if child.terminal is None:
                    # chaque passage par l'arête = l'évaluation du fils + une visite d'un de ses coups
                    assert node.N[a] == 1 + child.n_sum, "N(arête) != 1 + N(fils)"
                    # Q(arête) = -q(fils)
                    assert math.isclose(node.W[a] / node.N[a], -child.q(), abs_tol=1e-9), "Q(arête) != -q(fils)"
                stack.append(child)
        return count
