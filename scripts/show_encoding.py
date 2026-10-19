"""Montre ce que « voit » le réseau pour une position donnée.

    python scripts/show_encoding.py                       # exemple : Noirs au trait
    python scripts/show_encoding.py --fen "<FEN>"
    python scripts/show_encoding.py --all-planes          # affiche les 18 plans

Les plans sont affichés rangée 8 en haut, comme un échiquier, mais dans le repère du camp
au trait : « mes » pièces sont toujours en bas et avancent vers le haut.
"""

from __future__ import annotations

import argparse

import chess

from chesszero.encoding import (
    N_MOVE_PLANES,
    N_PLANES,
    PLANE_CASTLE,
    PLANE_EP,
    PLANE_HALFMOVE,
    PLANE_OPP,
    PLANE_OWN,
    encode_board,
    legal_move_indices,
    pack_board,
)

# Exemple par défaut : partie italienne, trait aux Noirs (on voit bien le retournement).
DEFAULT_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"

PIECES = "PNBRQK"
PLANE_NAMES = (
    [f"mes {n}" for n in ("pions", "cavaliers", "fous", "tours", "dames", "roi")]
    + [f"adverses {n}" for n in ("pions", "cavaliers", "fous", "tours", "dames", "roi")]
    + ["mon petit roque", "mon grand roque", "petit roque adverse", "grand roque adverse"]
    + ["prise en passant", "compteur 50 coups / 100"]
)


def plane_text(plane) -> list[str]:
    rows = []
    for r in range(7, -1, -1):  # rangée du haut d'abord
        cells = ["." if v == 0 else ("1" if v == 1 else f"{v:.2f}"[1:]) for v in plane[r]]
        rows.append(f"{r + 1} " + " ".join(f"{c:>3}" for c in cells))
    rows.append("    " + "   ".join("abcdefgh"))
    return rows


def combined_board(t) -> list[str]:
    rows = []
    for r in range(7, -1, -1):
        line = []
        for f in range(8):
            ch = "."
            for i, p in enumerate(PIECES):
                if t[PLANE_OWN + i, r, f]:
                    ch = p
                if t[PLANE_OPP + i, r, f]:
                    ch = p.lower()
            line.append(ch)
        rows.append(f"{r + 1} " + " ".join(line))
    rows.append("  " + " ".join("abcdefgh"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", default=DEFAULT_FEN)
    parser.add_argument("--all-planes", action="store_true")
    args = parser.parse_args()

    board = chess.Board(args.fen)
    t = encode_board(board)
    side = "Blancs" if board.turn == chess.WHITE else "Noirs"

    print(f"FEN : {board.fen()}\nTrait aux {side}\n")
    print("Échiquier réel :")
    print(board)
    print("\nVue du réseau (MAJUSCULES = mes pièces, minuscules = adverses) :")
    print("\n".join(combined_board(t)))

    shown = range(N_PLANES) if args.all_planes else (PLANE_OWN, PLANE_OPP, PLANE_CASTLE, PLANE_EP, PLANE_HALFMOVE)
    for p in shown:
        print(f"\nPlan {p} : {PLANE_NAMES[p]}")
        print("\n".join(plane_text(t[p])))

    print(f"\nFormat compact : {len(pack_board(board))} octets (uint8) au lieu de {t.nbytes} en float32.")

    moves = sorted(legal_move_indices(board), key=lambda mi: mi[1])
    print(f"\n{len(moves)} coups légaux (indice = case_départ_retournée * {N_MOVE_PLANES} + plan) :")
    print(f"{'coup':8} {'SAN':8} {'indice':>6}  case(retournée)  plan")
    for move, idx in moves:
        frm, plane = divmod(idx, N_MOVE_PLANES)
        print(f"{move.uci():8} {board.san(move):8} {idx:6}  {chess.square_name(frm):>15}  {plane}")


if __name__ == "__main__":
    main()
