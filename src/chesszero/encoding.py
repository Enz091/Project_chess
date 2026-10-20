"""Encodage des positions et des coups, façon AlphaZero.

Tout est exprimé du point de vue du camp au trait : quand c'est aux Noirs, on retourne
l'échiquier verticalement (rangée r -> 7-r) et on échange les couleurs. Le réseau voit donc
toujours « mes pièces » qui avancent vers le haut (rangées croissantes). Pas de miroir
gauche-droite : les roques ne sont pas symétriques (roi en e, côté dame != côté roi).

Numérotation des cases (comme python-chess) : sq = 8 * rangée + colonne, a1 = 0, h8 = 63.
Retourner verticalement une case = sq ^ 56 (chess.square_mirror).
"""

from __future__ import annotations

import chess
import numpy as np

# --------------------------------------------------------------------------- positions

# Plans du tenseur d'entrée (N_PLANES, 8, 8)
PLANE_OWN = 0  # 0..5  : mes P, N, B, R, Q, K
PLANE_OPP = 6  # 6..11 : P, N, B, R, Q, K adverses
PLANE_CASTLE = 12  # 12..15 : mon petit roque, mon grand roque, petit adverse, grand adverse
PLANE_EP = 16  # case de prise en passant (one-hot)
PLANE_HALFMOVE = 17  # compteur des 50 coups / 100, plafonné à 1 (plan constant)
N_PLANES = 18

# Format compact : 64 cases (0 vide, 1..6 mes pièces, 7..12 pièces adverses) + 3 octets.
PACKED_SIZE = 67
_IDX_CASTLE = 64  # masque de bits : 1 = mon petit, 2 = mon grand, 4 = petit adverse, 8 = grand adverse
_IDX_EP = 65  # case en passant dans le repère retourné, 64 = aucune
_IDX_HALFMOVE = 66  # compteur des 50 coups, plafonné à 255

_NO_EP = 64


def _oriented(board: chess.Board) -> chess.Board:
    """Position vue par le camp au trait (retournée + couleurs échangées si Noirs)."""
    return board if board.turn == chess.WHITE else board.mirror()


def pack_board(board: chess.Board) -> np.ndarray:
    """Position -> vecteur uint8 de PACKED_SIZE octets (format de stockage sur disque)."""
    me = board.turn
    flip = 56 if me == chess.BLACK else 0
    out = np.zeros(PACKED_SIZE, dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        mine = piece.color == me
        out[sq ^ flip] = piece.piece_type + (0 if mine else 6)

    castle = 0
    for bit, (color, kingside) in enumerate(
        [(me, True), (me, False), (not me, True), (not me, False)]
    ):
        has = board.has_kingside_castling_rights if kingside else board.has_queenside_castling_rights
        castle |= int(has(color)) << bit
    out[_IDX_CASTLE] = castle

    # On ne signale le en passant que s'il existe une prise légale : même position
    # « utile » = même tenseur, quelle que soit la façon dont on y est arrivé.
    ep = board.ep_square if board.has_legal_en_passant() else None
    out[_IDX_EP] = _NO_EP if ep is None else ep ^ flip
    out[_IDX_HALFMOVE] = min(board.halfmove_clock, 255)
    return out


def unpack_batch(packed: np.ndarray) -> np.ndarray:
    """(N, PACKED_SIZE) uint8 -> (N, N_PLANES, 8, 8) float32. Vectorisé, sans boucle Python.

    Accepte aussi un seul vecteur (PACKED_SIZE,) et renvoie alors (N_PLANES, 8, 8).
    """
    single = packed.ndim == 1
    packed = np.atleast_2d(packed)
    n = packed.shape[0]
    out = np.zeros((n, N_PLANES, 64), dtype=np.float32)

    squares = packed[:, :64]
    for code in range(1, 13):  # code 1..12 -> plan code-1 (mes pièces 0..5, adverses 6..11)
        out[:, code - 1, :] = squares == code

    castle = packed[:, _IDX_CASTLE]
    for bit in range(4):
        out[:, PLANE_CASTLE + bit, :] = ((castle >> bit) & 1)[:, None]

    ep = packed[:, _IDX_EP].astype(np.int64)
    rows = np.nonzero(ep != _NO_EP)[0]
    out[rows, PLANE_EP, ep[rows]] = 1.0

    out[:, PLANE_HALFMOVE, :] = (np.minimum(packed[:, _IDX_HALFMOVE], 100) / 100.0)[:, None]

    out = out.reshape(n, N_PLANES, 8, 8)
    return out[0] if single else out


def board_from_packed(vec: np.ndarray) -> chess.Board:
    """Inverse de `pack_board` : position VUE DU CAMP AU TRAIT (Blancs au trait, plateau retourné si
    c'était aux Noirs). Les coups légaux y portent les mêmes indices que dans la position d'origine."""
    board = chess.Board.empty()
    for sq in np.nonzero(vec[:64])[0]:
        code = int(vec[sq])
        board.set_piece_at(int(sq), chess.Piece((code - 1) % 6 + 1, chess.WHITE if code <= 6 else chess.BLACK))
    castle = int(vec[_IDX_CASTLE])
    rights = 0
    for bit, sq in enumerate([chess.H1, chess.A1, chess.H8, chess.A8]):
        if castle >> bit & 1:
            rights |= chess.BB_SQUARES[sq]
    board.castling_rights = rights
    if vec[_IDX_EP] != _NO_EP:
        board.ep_square = int(vec[_IDX_EP])
    board.halfmove_clock = int(vec[_IDX_HALFMOVE])
    return board


def legal_masks_from_packed(packed: np.ndarray) -> np.ndarray:
    """(N, PACKED_SIZE) uint8 -> (N, N_ACTIONS) bool : masque des coups légaux de chaque position."""
    return np.stack([legal_move_mask(board_from_packed(v)) for v in packed])


def encode_board(board: chess.Board) -> np.ndarray:
    """Position -> tenseur (N_PLANES, 8, 8) float32, du point de vue du camp au trait."""
    return unpack_batch(pack_board(board))


# --------------------------------------------------------------------------- coups

N_MOVE_PLANES = 73
N_ACTIONS = 64 * N_MOVE_PLANES  # 4672

# 8 directions (d_colonne, d_rangée) dans le repère du camp au trait : N, NE, E, SE, S, SO, O, NO
_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
# Plans 0..55 : direction * 7 + (distance - 1)
_KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
# Plans 56..63 : sauts de cavalier
_KNIGHT_BASE = 56
# Plans 64..72 : sous-promotions, (pièce, direction) avec direction = d_colonne in {-1, 0, +1}
_UNDERPROMO_BASE = 64
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

_QUEEN_PLANE = {}  # (dc, dr) -> plan
for _d, (_dc, _dr) in enumerate(_DIRECTIONS):
    for _dist in range(1, 8):
        _QUEEN_PLANE[(_dc * _dist, _dr * _dist)] = _d * 7 + (_dist - 1)
_KNIGHT_PLANE = {delta: _KNIGHT_BASE + i for i, delta in enumerate(_KNIGHT_DELTAS)}
_PLANE_DELTA = {plane: delta for delta, plane in _QUEEN_PLANE.items()}
_PLANE_DELTA.update({plane: delta for delta, plane in _KNIGHT_PLANE.items()})


def _flip_sq(sq: int, turn: chess.Color) -> int:
    return sq if turn == chess.WHITE else sq ^ 56


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Coup légal de `board` -> indice dans [0, 4672). Indice = case_départ * 73 + plan."""
    turn = board.turn
    frm = _flip_sq(move.from_square, turn)
    to = _flip_sq(move.to_square, turn)
    dc = chess.square_file(to) - chess.square_file(frm)
    dr = chess.square_rank(to) - chess.square_rank(frm)

    if move.promotion is not None and move.promotion != chess.QUEEN:
        plane = _UNDERPROMO_BASE + _UNDERPROMO_PIECES.index(move.promotion) * 3 + (dc + 1)
    elif (dc, dr) in _QUEEN_PLANE:
        plane = _QUEEN_PLANE[(dc, dr)]
    elif (dc, dr) in _KNIGHT_PLANE:
        plane = _KNIGHT_PLANE[(dc, dr)]
    else:
        raise ValueError(f"coup non encodable : {move} (dc={dc}, dr={dr})")
    return frm * N_MOVE_PLANES + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Indice -> coup (non vérifié légal : combiner avec legal_move_mask).

    Lève ValueError si l'indice sort de l'échiquier. Une poussée de pion qui arrive sur la
    dernière rangée avec un plan « dame » est décodée comme promotion en dame.
    """
    if not 0 <= index < N_ACTIONS:
        raise ValueError(f"indice hors limites : {index}")
    turn = board.turn
    frm, plane = divmod(index, N_MOVE_PLANES)
    promotion = None
    if plane >= _UNDERPROMO_BASE:
        piece, direction = divmod(plane - _UNDERPROMO_BASE, 3)
        promotion = _UNDERPROMO_PIECES[piece]
        dc, dr = direction - 1, 1
    else:
        dc, dr = _PLANE_DELTA[plane]

    f = chess.square_file(frm) + dc
    r = chess.square_rank(frm) + dr
    if not (0 <= f < 8 and 0 <= r < 8):
        raise ValueError(f"indice {index} : la case d'arrivée sort de l'échiquier")
    to = chess.square(f, r)

    real_from, real_to = _flip_sq(frm, turn), _flip_sq(to, turn)
    if (
        promotion is None
        and board.piece_type_at(real_from) == chess.PAWN
        and chess.square_rank(to) == 7
    ):
        promotion = chess.QUEEN
    return chess.Move(real_from, real_to, promotion)


def legal_move_indices(board: chess.Board) -> list[tuple[chess.Move, int]]:
    """Liste (coup légal, indice) pour la position."""
    return [(m, move_to_index(m, board)) for m in board.legal_moves]


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Masque booléen (N_ACTIONS,) : True pour les indices des coups légaux."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask
