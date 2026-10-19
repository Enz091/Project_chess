import random

import chess
import numpy as np
import pytest

from chesszero.encoding import (
    N_ACTIONS,
    N_PLANES,
    PACKED_SIZE,
    PLANE_CASTLE,
    PLANE_EP,
    PLANE_HALFMOVE,
    PLANE_OPP,
    PLANE_OWN,
    encode_board,
    index_to_move,
    legal_move_mask,
    move_to_index,
    pack_board,
    unpack_batch,
)


def random_positions(n_games: int, seed: int = 0, max_plies: int = 200):
    """Positions de parties jouées au hasard (toutes les positions de chaque partie)."""
    rng = random.Random(seed)
    for _ in range(n_games):
        board = chess.Board()
        for _ in range(max_plies):
            if board.is_game_over():
                break
            yield board.copy()
            board.push(rng.choice(list(board.legal_moves)))


# Positions ciblées : les cas spéciaux ne doivent pas dépendre du hasard.
SPECIAL_FENS = [
    # roques des deux camps, les deux côtés
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
    # prise en passant, Blancs (e5xd6) puis Noirs (d4xe3)
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
    # promotions (dame, tours, fous, cavaliers) avec et sans prise, Blancs puis Noirs
    "r1r4k/1P6/8/8/8/8/8/K7 w - - 0 1",
    "k7/8/8/8/8/8/1p6/R1R4K b - - 0 1",
    "1n2k3/P7/8/8/8/8/p7/1N2K3 w - - 0 1",
    "1n2k3/P7/8/8/8/8/p7/1N2K3 b - - 0 1",
]


def all_test_positions():
    for fen in SPECIAL_FENS:
        yield chess.Board(fen)
    yield from random_positions(60)


def test_special_fens_cover_special_moves():
    kinds = set()
    for fen in SPECIAL_FENS:
        b = chess.Board(fen)
        for m in b.legal_moves:
            if b.is_castling(m):
                kinds.add("castle")
            if b.is_en_passant(m):
                kinds.add("ep")
            if m.promotion == chess.QUEEN:
                kinds.add("promo_q")
            if m.promotion in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
                kinds.add("underpromo")
    assert kinds == {"castle", "ep", "promo_q", "underpromo"}


def test_move_roundtrip_all_legal_moves():
    n_pos = 0
    colors = set()
    for board in all_test_positions():
        n_pos += 1
        colors.add(board.turn)
        indices = set()
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            assert 0 <= idx < N_ACTIONS
            assert index_to_move(idx, board) == move, (board.fen(), move)
            indices.add(idx)
        assert len(indices) == board.legal_moves.count(), "deux coups légaux partagent un indice"
    assert n_pos >= 3000
    assert colors == {chess.WHITE, chess.BLACK}


def test_mask_matches_legal_moves():
    for board in all_test_positions():
        mask = legal_move_mask(board)
        assert mask.shape == (N_ACTIONS,) and mask.dtype == bool
        assert mask.sum() == board.legal_moves.count()
        for idx in np.nonzero(mask)[0]:
            assert index_to_move(int(idx), board) in board.legal_moves


def test_mirror_symmetry_tensor_and_moves():
    """Même position avec les couleurs inversées => même tenseur, mêmes indices de coups."""
    for board in all_test_positions():
        mirrored = board.mirror()
        np.testing.assert_array_equal(encode_board(board), encode_board(mirrored))
        np.testing.assert_array_equal(pack_board(board), pack_board(mirrored))
        np.testing.assert_array_equal(legal_move_mask(board), legal_move_mask(mirrored))
        for move in board.legal_moves:
            mirrored_move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                move.promotion,
            )
            assert move_to_index(move, board) == move_to_index(mirrored_move, mirrored)


def test_start_position_planes():
    t = encode_board(chess.Board())
    assert t.shape == (N_PLANES, 8, 8) and t.dtype == np.float32
    assert t[PLANE_OWN + 0].sum() == 8 and t[PLANE_OWN + 0][1].sum() == 8  # mes pions, rangée 2
    assert t[PLANE_OPP + 0][6].sum() == 8  # pions adverses, rangée 7
    assert t[PLANE_OWN + 5][0, 4] == 1  # mon roi en e1
    assert t[PLANE_CASTLE : PLANE_CASTLE + 4].min() == 1
    assert t[PLANE_EP].sum() == 0 and t[PLANE_HALFMOVE].sum() == 0
    # Noirs au trait après 1.e4 : « mon » roi est en e1 dans le repère retourné
    b = chess.Board()
    b.push_san("e4")
    t = encode_board(b)
    assert t[PLANE_OWN + 5][0, 4] == 1
    assert t[PLANE_OPP + 0][4, 4] == 1  # le pion e4 adverse est vu en e5 (rangée 5, indice 4)


def test_en_passant_plane_only_when_capture_is_legal():
    b = chess.Board()
    b.push_san("e4")  # python-chess pose ep_square=e3 mais aucune prise n'est possible
    assert encode_board(b)[PLANE_EP].sum() == 0
    b = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    ep = encode_board(b)[PLANE_EP]
    assert ep.sum() == 1 and ep[5, 3] == 1  # d6 = rangée 6 (indice 5), colonne d (3)
    b = chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
    ep = encode_board(b)[PLANE_EP]
    assert ep.sum() == 1 and ep[5, 4] == 1  # e3 retourné -> e6 : rangée indice 5


def test_halfmove_plane_is_clipped():
    b = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 40 1")
    assert encode_board(b)[PLANE_HALFMOVE].max() == pytest.approx(0.4)
    b = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 130 1")
    assert encode_board(b)[PLANE_HALFMOVE].max() == 1.0


def test_pack_unpack_batch_matches_encode():
    boards = list(random_positions(10, seed=1))
    packed = np.stack([pack_board(b) for b in boards])
    assert packed.dtype == np.uint8 and packed.shape == (len(boards), PACKED_SIZE)
    batch = unpack_batch(packed)
    for i, b in enumerate(boards):
        np.testing.assert_array_equal(batch[i], encode_board(b))
