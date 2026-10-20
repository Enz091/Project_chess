from pathlib import Path

import chess
import numpy as np
import pytest

from chesszero.encoding import N_ACTIONS, move_to_index
from chesszero.mcts import MCTS, MCTSConfig, NetEvaluator, policy_best_move, terminal_value

CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "kaggle_sup" / "best.pt"

BACK_RANK = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"  # Ra8# ; la politique seule joue Ra7
LEGAL_MATE = "rn1qkbnr/ppp2p1p/3p2p1/4N3/2B1P3/2N5/PPPP1PPP/R1BbK2R w KQkq - 0 6"  # 6.Bxf7+ Ke7 7.Nd5#
SMOTHERED = "1r5k/6pp/7N/3Q4/8/8/8/K7 w - - 0 1"  # 1.Qg8+ Rxg8 2.Nf7#
HANGING_QUEEN = "rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"  # Nxh4
QUEEN_UP = "4k3/pppp4/8/8/8/8/PPPP4/3QK3 w - - 0 1"  # Blancs au trait, une dame d'avance

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


# --------------------------------------------------------------------------- évaluateurs factices


class StubEvaluator:
    """Priors uniformes ; valeur (camp au trait) donnée par `value_fn(board)`."""

    def __init__(self, value_fn=lambda board: 0.0):
        self.value_fn = value_fn

    def evaluate(self, boards, legal_indices):
        priors = [np.full(len(idx), 1.0 / len(idx)) for idx in legal_indices]
        return priors, np.array([self.value_fn(b) for b in boards], dtype=np.float64)


def material(board: chess.Board) -> float:
    """Différence de matériel, vue par le camp au trait, ramenée dans [-1, 1]."""
    diff = sum((1 if p.color == board.turn else -1) * PIECE_VALUES[p.piece_type] for p in board.piece_map().values())
    return float(np.tanh(diff / 3))


def white_better(board: chess.Board) -> float:
    """+0,5 pour les Blancs quel que soit le camp au trait : +0,5 si Blancs au trait, -0,5 sinon."""
    return 0.5 if board.turn == chess.WHITE else -0.5


def make(fen=chess.STARTING_FEN, evaluator=None, check=True, **cfg):
    config = MCTSConfig(**cfg)
    return MCTS(evaluator or StubEvaluator(material), config, chess.Board(fen),
                rng=np.random.default_rng(0), check_invariants=check)


def mirrored(fen: str) -> str:
    """Même position, couleurs échangées et plateau retourné : c'est aux Noirs de jouer le même coup."""
    return chess.Board(fen).mirror().fen()


def iter_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        if node.expanded:
            stack.extend(c for c in node.children if c is not None)


# --------------------------------------------------------------------------- config


def test_config_loads_mcts_section_with_noise_and_temperature_off():
    for name in ("tiny", "kaggle"):
        cfg = MCTSConfig.named(name)
        assert cfg.num_simulations == 200 and cfg.batch_size == 8
        assert cfg.dirichlet_epsilon == 0.0 and cfg.temperature == 0.0


def test_config_rejects_unknown_key(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[mcts]\nnum_simulation = 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clés inconnues"):
        MCTSConfig.from_toml(path)


# --------------------------------------------------------------------------- positions terminales


@pytest.mark.parametrize("fen, expected", [
    ("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1", -1.0),  # mat du couloir : les Noirs sont matés
    ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0.0),  # pat
    ("8/8/4k3/8/8/3K4/8/8 w - - 0 1", 0.0),  # roi contre roi
    ("8/8/4k3/8/8/3KB3/8/8 w - - 0 1", 0.0),  # roi + fou contre roi
    ("8/8/4k3/8/8/3K4/3R4/8 w - - 100 80", 0.0),  # 50 coups
    ("8/8/4k3/8/8/3K4/3R4/8 w - - 99 80", None),  # pas encore
    (chess.STARTING_FEN, None),
])
def test_terminal_values(fen, expected):
    assert terminal_value(chess.Board(fen)) == expected


def test_threefold_repetition_is_a_draw():
    board = chess.Board()
    for i, uci in enumerate("g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8".split()):
        assert terminal_value(board) is None, f"nulle trop tôt au demi-coup {i}"
        board.push_uci(uci)
    assert terminal_value(board) == 0.0  # position initiale pour la 3e fois


def test_search_on_terminal_position_raises():
    with pytest.raises(ValueError, match="terminale"):
        make("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1").search(10)


def test_terminal_leaf_in_tree_uses_threefold_with_game_history():
    """Après Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1, la réponse Ng8 répète trois fois : le nœud vaut 0, pas l'avis du réseau."""
    board = chess.Board()
    for uci in "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1".split():
        board.push_uci(uci)
    mcts = MCTS(StubEvaluator(lambda b: 0.9), MCTSConfig(batch_size=4), board, check_invariants=True)
    mcts.search(300)
    a = mcts.root.moves.index(chess.Move.from_uci("f6g8"))
    child = mcts.root.children[a]
    assert child is not None and child.terminal == 0.0
    assert mcts.root.W[a] == 0.0  # que des nulles exactes sur cette arête


# --------------------------------------------------------------------------- comptage et perte virtuelle


@pytest.mark.parametrize("batch_size, sims", [(1, 60), (8, 100), (16, 200), (64, 50)])
def test_root_visits_equal_simulations(batch_size, sims):
    mcts = make(batch_size=batch_size)
    mcts.search(sims)
    assert mcts.root.n_sum == sims == int(mcts.root.N.sum())
    mcts.search(sims)  # une deuxième recherche ajoute exactement `sims`
    assert mcts.root.n_sum == 2 * sims


def test_visits_after_tree_reuse():
    mcts = make(batch_size=8)
    mcts.search(200)
    move = mcts.best_move()
    kept = int(mcts.root.N[mcts.root.moves.index(move)])
    mcts.advance(move)
    assert mcts.root.n_sum == kept - 1  # l'arête comptait aussi l'évaluation du nœud lui-même
    mcts.search(50)
    assert mcts.root.n_sum == kept - 1 + 50


def test_virtual_loss_fully_removed_after_each_batch(monkeypatch):
    mcts = make(batch_size=16, check=False)
    batches = []
    original = mcts._run_batch

    def checked(target):
        done = original(target)
        for node in iter_nodes(mcts.root):
            assert not node.pending
            if node.expanded:
                assert node.vl_sum == 0 and not node.VL.any()
        batches.append(done)
        return done

    monkeypatch.setattr(mcts, "_run_batch", checked)
    mcts.search(300)
    assert sum(batches) == 300 and len(batches) >= 300 // 16
    assert mcts.check_tree() > 100


def test_check_tree_detects_leftover_virtual_loss():
    mcts = make(batch_size=8)
    mcts.search(40)
    mcts.root.VL[0] += 1
    mcts.root.vl_sum += 1
    with pytest.raises(AssertionError, match="perte virtuelle"):
        mcts.check_tree()


def test_virtual_loss_spreads_a_batch_over_distinct_leaves():
    mcts = make(batch_size=16)
    mcts.search(1)  # développe la racine (1 simulation)
    mcts._run_batch(16)  # avec perte virtuelle, 16 feuilles différentes sont trouvées en un lot
    assert mcts.root.n_sum == 17 and mcts.stats["collisions"] == 0


# --------------------------------------------------------------------------- signes


def test_backup_sign_single_simulation():
    """La feuille vaut +0,5 pour son camp au trait : l'arête de la racine (adversaire) reçoit -0,5."""
    mcts = make(evaluator=StubEvaluator(lambda b: 0.5), batch_size=1)
    mcts.search(1)
    a = int(np.argmax(mcts.root.N))
    assert mcts.root.N[a] == 1 and mcts.root.W[a] == -0.5
    assert mcts.root.q() == pytest.approx(0.0)  # (0,5 du réseau à la racine - 0,5) / 2


@pytest.mark.parametrize("fen, root_q", [(chess.STARTING_FEN, 0.5), (mirrored(chess.STARTING_FEN), -0.5)])
def test_backup_sign_at_every_depth(fen, root_q):
    """Les Blancs valent +0,5 partout : toute arête vaut +0,5 si les Blancs la choisissent, -0,5 sinon."""
    mcts = make(fen, evaluator=StubEvaluator(white_better), batch_size=8)
    mcts.search(400)
    depth_seen = 0
    stack = [(mcts.root, chess.Board(fen).turn, 0)]
    while stack:
        node, turn, depth = stack.pop()
        if not node.expanded:
            continue
        depth_seen = max(depth_seen, depth)
        seen = node.N > 0
        expected = 0.5 if turn == chess.WHITE else -0.5
        np.testing.assert_allclose(node.W[seen] / node.N[seen], expected)
        assert node.q() == pytest.approx(expected)
        stack.extend((c, not turn, depth + 1) for c in node.children if c is not None)
    assert depth_seen >= 3
    assert mcts.root.q() == pytest.approx(root_q)


@pytest.mark.parametrize("fen", [QUEEN_UP, mirrored(QUEEN_UP)])
def test_fpu_has_the_sign_of_the_player_who_chooses(fen):
    """Position très gagnante pour le camp au trait : ses coups inexplorés reçoivent une valeur FPU POSITIVE
    (≈ +1 - 0,2). Au niveau suivant, c'est au perdant de choisir : sa FPU est négative. Un signe inversé
    (par ex. prendre le Q de l'arête entrante tel quel) donnerait l'inverse aux deux niveaux."""
    mcts = make(fen, batch_size=1)
    mcts.search(10)  # 18 coups à la racine : il en reste d'inexplorés
    root = mcts.root
    unvisited = root.N == 0
    assert unvisited.any()
    assert root.fpu_value(0.2) > 0.5
    np.testing.assert_allclose(root.edge_q(1.0, 0.2)[unvisited], root.fpu_value(0.2))
    assert root.fpu_value(0.2) == pytest.approx(root.q() - 0.2)

    child = root.children[int(np.argmax(root.N))]
    assert child.expanded and (child.N == 0).any()
    assert child.fpu_value(0.2) < -0.5
    a = int(np.argmax(root.N))
    assert child.q() == pytest.approx(-root.W[a] / root.N[a])  # base FPU = -Q de l'arête entrante


@pytest.mark.parametrize("fen, uci", [(BACK_RANK, "a1a8"), (mirrored(BACK_RANK), "a8a1")])
def test_finds_mate_in_one_with_uniform_priors(fen, uci):
    mcts = make(fen, evaluator=StubEvaluator(), batch_size=8)
    mcts.search(200)
    assert mcts.best_move() == chess.Move.from_uci(uci)
    assert mcts.root_stats()[0]["q"] == 1.0


@pytest.mark.parametrize("fen, uci", [(HANGING_QUEEN, "f3h4"), (mirrored(HANGING_QUEEN), "f6h5")])
def test_captures_hanging_queen_for_both_colors(fen, uci):
    mcts = make(fen, batch_size=8)
    mcts.search(300)
    assert mcts.best_move() == chess.Move.from_uci(uci)
    assert mcts.root_stats()[0]["q"] > 0.5


# --------------------------------------------------------------------------- bruit, température, réutilisation


def test_noise_is_off_by_default():
    mcts = make(batch_size=8)
    mcts.search(50)
    assert mcts.root.P is mcts.root.P_raw


def test_dirichlet_noise_changes_root_priors_reproducibly():
    def run():
        mcts = make(batch_size=8, dirichlet_epsilon=0.25)
        mcts.search(100)
        return mcts

    a, b = run(), run()
    assert not np.allclose(a.root.P, a.root.P_raw)
    assert a.root.P.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(a.root.P, b.root.P)
    np.testing.assert_array_equal(a.root.N, b.root.N)

    move = a.best_move()
    a.advance(move)
    a.search(50)  # le bruit de la nouvelle racine part de ses priors d'origine, sans cumul
    assert not np.allclose(a.root.P, a.root.P_raw)
    np.testing.assert_allclose(a.root.P_raw, 1.0 / len(a.root.moves))


def test_temperature_zero_picks_most_visited_and_one_samples_by_visits():
    mcts = make(batch_size=8)
    mcts.search(300)
    root = mcts.root
    assert root.N[root.moves.index(mcts.best_move())] == root.N.max()
    counts = np.zeros(len(root.moves))
    for _ in range(3000):
        counts[root.moves.index(mcts.best_move(temperature=1.0))] += 1
    np.testing.assert_allclose(counts / counts.sum(), root.N / root.n_sum, atol=0.03)


def test_visit_distribution():
    mcts = make(batch_size=8)
    mcts.search(100)
    dist = mcts.visit_distribution()
    assert dist.shape == (N_ACTIONS,) and dist.sum() == pytest.approx(1.0)
    for move, n in zip(mcts.root.moves, mcts.root.N):
        assert dist[move_to_index(move, mcts.board)] == pytest.approx(n / 100)


def test_tree_reuse_keeps_the_subtree():
    mcts = make(batch_size=8)
    mcts.search(200)
    move = mcts.best_move()
    child = mcts.root.children[mcts.root.moves.index(move)]
    n_before = child.n_sum
    mcts.advance(move)
    assert mcts.root is child and mcts.root.n_sum == n_before > 0
    assert mcts.board.move_stack[-1] == move

    # coup jamais exploré (ici à la racine noire) : nouvelle racine vide
    unexplored = next(m for m, c in zip(mcts.root.moves, mcts.root.children) if c is None)
    mcts.advance(unexplored)
    assert not mcts.root.expanded and mcts.root.n_sum == 0
    mcts.search(20)
    assert mcts.root.n_sum == 20

    with pytest.raises(ValueError, match="illégal"):
        mcts.advance(chess.Move.from_uci("a1a8"))


# --------------------------------------------------------------------------- avec le vrai réseau


needs_checkpoint = pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoints/kaggle_sup/best.pt absent (hors git)")


@pytest.fixture(scope="module")
def net():
    from chesszero.checkpoint import load_for_inference

    model, _ = load_for_inference(CHECKPOINT)
    return NetEvaluator(model)


@needs_checkpoint
@pytest.mark.parametrize("fen, mate, policy_choice", [
    (BACK_RANK, "a1a8", "a1a7"),  # mat du couloir : la politique préfère Ra7 (17,9 % contre 17,2 %)
    (mirrored(BACK_RANK), "a8a1", None),  # même chose pour les Noirs
    (LEGAL_MATE, "c4f7", "e5f7"),  # mat en deux : la politique préfère Nxf7 (59 %)
    (SMOTHERED, "d5g8", "h6f7"),  # mat en deux : la politique préfère Nf7+ (68 %)
])
@pytest.mark.parametrize("batch_size", [1, 8, 16])
def test_network_search_finds_mates(net, fen, mate, policy_choice, batch_size):
    board = chess.Board(fen)
    if policy_choice is not None:
        assert policy_best_move(net, board) == chess.Move.from_uci(policy_choice)
    mcts = MCTS(net, MCTSConfig(batch_size=batch_size), board, check_invariants=True)
    mcts.search(400)
    assert mcts.best_move() == chess.Move.from_uci(mate)
    assert mcts.root.n_sum == 400
    top = mcts.root_stats()
    assert top[0]["q"] > 0.85 and top[0]["visits"] > top[1]["visits"]
