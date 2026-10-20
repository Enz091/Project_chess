"""Protocoles de mesure contre Stockfish (scripts/match_stockfish.py) : figés et cohérents."""

import json
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import match_stockfish as ms  # noqa: E402


def test_v1_protocol_is_exactly_the_archived_one():
    archived = json.loads((ROOT / "docs" / "stockfish" / "v1" / "kaggle_sup.json").read_text(encoding="utf-8"))
    assert archived["protocol"] == ms.PROTOCOL_V1


def test_v2_openings_are_legal_and_distinct():
    openings = ms.PROTOCOL_V2["openings"]
    assert len(openings) == 40
    positions = set()
    for name, moves in openings:
        board = chess.Board()
        for uci in moves.split():
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves, (name, uci)
            board.push(move)
        assert 6 <= len(board.move_stack) <= 10
        positions.add(board.epd())
    assert len(positions) == 40 and len({n for n, _ in openings}) == 40


def test_v2_keeps_everything_else_from_v1():
    differing = {k for k in ms.PROTOCOL_V1 if ms.PROTOCOL_V1[k] != ms.PROTOCOL_V2[k]}
    assert differing == {"name", "openings", "refine_rounds"}
    assert ms.PROTOCOL_V2["simulations"] == 200 and ms.PROTOCOL_V2["stockfish_movetime_ms"] == 300


def test_round_sizes_and_tag_parsing():
    assert len(ms.round_tasks(ms.PROTOCOL_V1, 2000, 0, 1)) == 20
    assert len(ms.round_tasks(ms.PROTOCOL_V2, 2000, 0, 1)) == 80
    assert ms.parse_tag("v1/kaggle_sup", "v2") == ("v1", "kaggle_sup")
    assert ms.parse_tag("kaggle_sup", "v2") == ("v2", "kaggle_sup")
    r = ms.Results("v1/kaggle_sup")
    assert r.protocol is ms.PROTOCOL_V1 and len(r.games) == 220 and r.ai == "sans tables"
