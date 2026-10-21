"""Livre d'ouvertures du self-play (étape 6b)."""

import chess
import numpy as np
import pytest

from chesszero.book import OpeningBook, book_entry, cut_ply, first_moves

# Format réel des bases Lichess : pendule et évaluation en commentaires, annotations, numéros « 1... ».
LICHESS_GAME = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abcdefgh"]
[White "a"]
[Black "b"]
[Result "1-0"]
[WhiteElo "2100"]
[BlackElo "2050"]
[TimeControl "180+2"]
[Termination "Normal"]

1. e4 { [%eval 0.2] [%clk 0:03:00] } 1... c5 { [%eval 0.3] [%clk 0:03:00] } 2. Nf3?! { [%clk 0:02:59] } 2... d6 \
3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6!? 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 $1 11. g4 b5 1-0
"""


def test_first_moves_reads_lichess_movetext():
    moves = first_moves(LICHESS_GAME, 40)
    board = chess.Board()
    sans = []
    for m in moves:
        sans.append(board.san(m))
        board.push(m)
    assert sans[:6] == ["e4", "c5", "Nf3", "d6", "d4", "cxd4"]
    assert sans[17] == "O-O" and sans[18] == "O-O-O" and len(moves) == 22
    assert len(first_moves(LICHESS_GAME, 5)) == 5


def test_first_moves_rejects_illegal_or_headerless():
    bad = LICHESS_GAME.replace("2... d6", "2... Qh4h4")
    assert first_moves(bad, 40) is None
    assert first_moves("1. e4 e5", 10) is None  # pas d'en-têtes : pas de partie


def test_cut_is_deterministic_and_in_range():
    plies = [cut_ply(f"https://lichess.org/{i}", 8, 20) for i in range(2000)]
    assert min(plies) == 8 and max(plies) == 20
    assert plies == [cut_ply(f"https://lichess.org/{i}", 8, 20) for i in range(2000)]
    counts = np.bincount(plies)[8:]
    assert counts.min() > 100  # à peu près uniforme (2000 / 13 ≈ 154)


def test_book_entry_keeps_move_sequence_and_needs_a_following_move():
    key = "https://lichess.org/abcdefgh"
    epd, moves = book_entry(LICHESS_GAME, key, 8, 20)
    board = chess.Board()
    for uci in moves.split():
        board.push_uci(uci)
    assert board.epd() == epd and len(moves.split()) == cut_ply(key, 8, 20)
    assert book_entry(LICHESS_GAME, key, 22, 22) is None  # 22 coups seulement : rien après la coupe


def test_opening_book_round_trip_and_uniform_sampling(tmp_path):
    lines = ["e2e4 e7e5 g1f3", "d2d4 d7d5", "c2c4"]
    book = OpeningBook(lines, np.array([10, 1, 1]), np.array([0.1, -0.2, 0.0], dtype=np.float32))
    book.save(tmp_path / "b.tsv.gz", {"essai": True})
    assert (tmp_path / "b.json").exists()
    loaded = OpeningBook.load(tmp_path / "b.tsv.gz")
    assert loaded.lines == lines and loaded.counts.tolist() == [10, 1, 1]
    np.testing.assert_allclose(loaded.values, [0.1, -0.2, 0.0])
    board = loaded.board(0)
    assert len(board.move_stack) == 3 and board.turn == chess.BLACK
    rng = np.random.default_rng(0)
    seen = np.bincount([len(loaded.sample(rng).move_stack) for _ in range(3000)], minlength=4)[1:]
    assert seen.min() > 900  # uniforme sur les positions, pas pondéré par le nombre de parties


def test_empty_book_is_refused():
    with pytest.raises(ValueError):
        OpeningBook([])


# --------------------------------------------------------------------------- exclusion des ouvertures de la mesure


def test_excluding_catches_prefixes_transpositions_and_partial_lines():
    openings = [("Quatre cavaliers (exemple)", "g1f3 g8f6 b1c3 b8c6")]
    lines = ["g1f3 g8f6 b1c3 b8c6 e2e4 e7e5",  # passe par la position (même ordre de coups)
             "b1c3 b8c6 g1f3 g8f6 d2d4",  # passe par la position par transposition
             "g1f3 g8f6 b1c3",  # s'arrête sur la ligne, avant sa fin
             "g1f3 g8f6 e2e4 e7e5",  # partage seulement le début : gardée
             "e2e4 e7e5 g1f3 b8c6"]  # sans rapport : gardée
    book, removed = OpeningBook(lines).excluding(openings)
    assert removed == {"through": 2, "on_path": 1}
    assert book.lines == lines[3:]
    assert book.counts.tolist() == [1, 1]


def test_committed_book_avoids_the_v2_measure_openings():
    """Aucune entrée du livre versionné ne passe par une des 40 ouvertures du protocole v2 (transpositions
    comprises) ni ne s'arrête sur une de leurs lignes : le self-play ne s'entraîne pas sur la mesure."""
    import sys
    from pathlib import Path

    from chesszero.book import opening_keys, touches_openings

    root = Path(__file__).resolve().parents[1]
    path = root / "books" / "lichess_2000.tsv.gz"
    if not path.exists():
        pytest.skip("livre absent")
    sys.path.insert(0, str(root / "scripts"))
    from ladder_openings import OPENINGS_V2

    final, on_path = opening_keys(OPENINGS_V2)
    assert len(final) == 40
    book = OpeningBook.load(path)
    assert len(book) > 200_000
    bad = [line for line in book.lines if touches_openings(line, final, on_path) is not None]
    assert bad == []
