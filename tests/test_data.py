"""Tests de la préparation des données (étape 3) sur un petit PGN synthétique au format Lichess."""

import random
from collections import Counter
from dataclasses import dataclass, replace

import chess
import chess.pgn
import numpy as np
import pytest
import zstandard

from chesszero.data import (
    SPLITS,
    FilterConfig,
    estimated_seconds,
    extract,
    load_split,
    parse_headers,
    reject_reason,
    shard_paths,
    split_of,
)
from chesszero.encoding import (
    N_ACTIONS,
    PLANE_OPP,
    PLANE_OWN,
    index_to_move,
    legal_move_mask,
    move_to_index,
    pack_board,
    unpack_batch,
)

CFG = FilterConfig(val_fraction=0.25)  # beaucoup de validation pour que les deux splits existent
KINDS = ["ok", "ok", "ok", "low_elo", "bullet", "abandoned", "short", "fen", "no_result", "bot"]
EXPECTED_REASON = {  # kind -> raison de rejet attendue (None = partie gardée)
    "ok": None, "low_elo": "elo", "bullet": "bullet", "abandoned": "termination",
    "short": "too_short", "fen": "variant", "no_result": "result", "bot": "bot",
}


@dataclass
class Game:
    kind: str
    site: str
    result: str
    moves: list
    text: str

    @property
    def reason(self):
        reason = EXPECTED_REASON[self.kind]
        if reason is None and len(self.moves) < CFG.min_plies:  # partie aléatoire finie trop tôt
            return "too_short"
        return reason

    @property
    def split(self):
        return split_of({"Site": self.site}, 0, CFG.val_fraction)


def random_moves(rng, n_plies):
    board, moves = chess.Board(), []
    while len(moves) < n_plies and not board.is_game_over():
        move = rng.choice(list(board.legal_moves))
        moves.append(move)
        board.push(move)
    return moves


def make_game(i, kind, rng):
    site = f"https://lichess.org/g{i:07d}"
    result = rng.choice(["1-0", "0-1", "1/2-1/2"])
    n_plies = 10 if kind == "short" else rng.randint(25, 140)
    moves = random_moves(rng, n_plies)
    game = chess.pgn.Game()
    game.headers.update({
        "Event": "Rated Blitz game", "Site": site, "Date": "2026.09.01", "Round": "-",
        "White": "a", "Black": "b", "Result": "*" if kind == "no_result" else result,
        "WhiteElo": "2300" if kind == "low_elo" else str(rng.randint(2000, 2700)),
        "BlackElo": "1900" if kind == "low_elo" else str(rng.randint(2000, 2700)),
        "TimeControl": "60+0" if kind == "bullet" else rng.choice(["600+0", "180+2", "300+3"]),
        "Termination": "Abandoned" if kind == "abandoned" else rng.choice(["Normal", "Time forfeit"]),
    })
    if kind == "bot":
        game.headers["BlackTitle"] = "BOT"
    if kind == "fen":
        game.headers["SetUp"], game.headers["FEN"] = "1", chess.STARTING_FEN
    node = game
    for move in moves:
        node = node.add_variation(move)
        node.comment = "[%eval 0.3] [%clk 0:05:00]"
    text = game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=True))
    return Game(kind, site, result, moves, text)


ILLEGAL_GAME = """[Event "Rated Blitz game"]
[Site "https://lichess.org/illegal0"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2200"]
[TimeControl "300+0"]
[Termination "Normal"]

1. e4 e5 2. Ke4 Nc6 1-0
"""


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    rng = random.Random(3)
    games = [make_game(i, KINDS[i % len(KINDS)], rng) for i in range(72)]
    path = tmp_path_factory.mktemp("pgn") / "sample.pgn"
    path.write_text("\n\n".join(g.text for g in games) + "\n\n" + ILLEGAL_GAME, encoding="utf-8")
    return path, games


@pytest.fixture(scope="module")
def extracted(corpus, tmp_path_factory):
    path, games = corpus
    out = tmp_path_factory.mktemp("shards")
    manifest = extract(str(path), out, CFG, shard_size=300, progress=False)
    return out, manifest


def board_from_packed(vec):
    """Reconstruit, à partir du seul format stocké, la position (Blancs au trait = vue du camp au trait)."""
    board = chess.Board.empty()
    for sq in range(64):
        code = int(vec[sq])
        if code:
            color = chess.WHITE if code <= 6 else chess.BLACK
            board.set_piece_at(sq, chess.Piece((code - 1) % 6 + 1, color))
    castle = int(vec[64])
    rights = 0
    for bit, sq in enumerate([chess.H1, chess.A1, chess.H8, chess.A8]):
        if castle >> bit & 1:
            rights |= chess.BB_SQUARES[sq]
    board.castling_rights = rights
    if vec[65] != 64:
        board.ep_square = int(vec[65])
    board.halfmove_clock = int(vec[66])
    return board


# --------------------------------------------------------------------------- filtrage


def test_estimated_seconds():
    assert estimated_seconds("180+0") == 180
    assert estimated_seconds("60+1") == 100
    assert estimated_seconds("-") is None


def headers(**kw):
    base = {"Result": "1-0", "WhiteElo": "2100", "BlackElo": "2050", "TimeControl": "300+0", "Termination": "Normal"}
    base.update(kw)
    return base


def test_reject_reason_cases():
    cfg = FilterConfig()
    assert reject_reason(headers(), cfg) is None
    assert reject_reason(headers(TimeControl="180+0"), cfg) is None  # 3+0 = blitz, pas bullet
    assert reject_reason(headers(TimeControl="120+1"), cfg) == "bullet"  # 160 s
    assert reject_reason(headers(TimeControl="-"), cfg) == "time_control"
    assert reject_reason(headers(BlackElo="1999"), cfg) == "elo"  # le plus faible des deux compte
    assert reject_reason(headers(WhiteElo="?"), cfg) == "elo"
    assert reject_reason(headers(Termination="Abandoned"), cfg) == "termination"
    assert reject_reason(headers(Termination="Time forfeit"), cfg) is None
    assert reject_reason(headers(Result="*"), cfg) == "result"
    assert reject_reason(headers(FEN="8/8/8/8/8/8/8/8 w - - 0 1"), cfg) == "variant"
    assert reject_reason(headers(BlackElo="1700"), replace(cfg, min_elo=1500)) is None


def test_bot_filter():
    cfg = FilterConfig()
    assert cfg.exclude_bots  # activé par défaut
    assert reject_reason(headers(WhiteTitle="BOT"), cfg) == "bot"
    assert reject_reason(headers(BlackTitle="BOT"), cfg) == "bot"
    assert reject_reason(headers(WhiteTitle="GM", BlackTitle="IM"), cfg) is None  # autres titres : gardés
    assert reject_reason(headers(WhiteTitle="BOT"), replace(cfg, exclude_bots=False)) is None
    assert reject_reason(headers(WhiteTitle="BOT", BlackElo="1500"), cfg) == "elo"  # le bot ne masque rien


def test_parse_headers_stops_at_moves():
    h = parse_headers('[Event "x"]\n[White "a [b]"]\n\n1. e4 { [%clk 0:01:00] } e5 *\n')
    assert h == {"Event": "x", "White": "a [b]"}


# --------------------------------------------------------------------------- extraction


def test_counts(corpus, extracted):
    _, games = corpus
    _, m = extracted
    assert m["games_read"] == len(games) + 1  # + la partie à coup illégal
    expected = Counter(g.reason for g in games)
    header_reasons = {k: v for k, v in expected.items() if k not in (None, "too_short")}
    assert m["games_rejected_by_header"] == header_reasons
    assert m["games_rejected_after_parse"]["too_short"] == expected["too_short"]
    assert m["games_rejected_after_parse"]["parse_error"] == 1
    assert m["games_kept"] == expected[None] > 10
    kept_by_term = m["kept_by_termination"]
    assert sum(kept_by_term["games"].values()) == m["games_kept"]
    assert sum(kept_by_term["positions"].values()) == sum(m["positions"].values())
    assert kept_by_term["games"] == dict(Counter(
        parse_headers(g.text)["Termination"] for g in games if g.reason is None))
    assert not m["stopped_early"]


def test_positions_valid_legal_and_signed(corpus, extracted):
    """Chaque position stockée est valide, son coup est légal, le résultat a le bon signe."""
    out, _ = extracted
    _, games = corpus
    kept = [g for g in games if g.reason is None]
    for split in SPLITS:
        data = load_split(out, split)
        boards, moves, results = data["boards"], data["moves"], data["results"]
        assert boards.dtype == np.uint8 and moves.dtype == np.int16 and results.dtype == np.int8
        n = len(boards)
        assert n == len(moves) == len(results) == len(data["games"]) > 0
        assert moves.min() >= 0 and moves.max() < N_ACTIONS

        # 1. Validité : vérifiée sur les seules données stockées.
        planes = unpack_batch(boards)
        assert (planes[:, PLANE_OWN + 5].sum(axis=(1, 2)) == 1).all()  # mon roi, une fois
        assert (planes[:, PLANE_OPP + 5].sum(axis=(1, 2)) == 1).all()  # roi adverse, une fois
        for plane in (PLANE_OWN, PLANE_OPP):  # pas de pion sur la 1re / 8e rangée
            assert planes[:, plane, 0].sum() == 0 and planes[:, plane, 7].sum() == 0
        for vec, idx in zip(boards, moves):
            board = board_from_packed(vec)
            assert board.is_valid(), board.status()
            assert board.turn == chess.WHITE
            # 2. Légalité du coup stocké dans la position décodée.
            assert legal_move_mask(board)[idx]
            assert index_to_move(int(idx), board) in board.legal_moves

        # 3. Relecture indépendante des parties : mêmes positions, mêmes coups, bon signe.
        i = 0
        assert data["value_valid"].dtype == np.uint8 and len(data["value_valid"]) == n
        for g in (g for g in kept if g.split == split):
            start = i
            board = chess.Board()
            score = {"1-0": (1, -1), "0-1": (-1, 1), "1/2-1/2": (0, 0)}[g.result]  # (Blancs, Noirs)
            seen = set()
            for move in g.moves:
                assert (boards[i] == pack_board(board)).all()
                assert moves[i] == move_to_index(move, board)
                assert results[i] == score[board.turn == chess.BLACK]
                seen.add(int(results[i]))
                board.push(move)
                i += 1
            assert seen == ({1, -1} if g.result != "1/2-1/2" else {0})  # signes opposés pour les 2 camps
            # value_valid : 1 pour toute la partie si elle s'est finie en « Normal », 0 si « Time forfeit »
            expected_valid = int(parse_headers(g.text)["Termination"] == "Normal")
            assert (data["value_valid"][start:i] == expected_valid).all()
        assert i == n  # aucune position en trop


def test_split_is_by_game(corpus, extracted):
    out, _ = extracted
    _, games = corpus
    ids = {s: set(load_split(out, s)["games"].tolist()) for s in SPLITS}
    assert ids["train"] and ids["val"]
    assert not ids["train"] & ids["val"]  # aucune partie des deux côtés
    kept = [g for g in games if g.reason is None]
    assert len(ids["val"]) == sum(g.split == "val" for g in kept)


def test_split_fraction_is_respected():
    cfg = FilterConfig()  # 2 %
    val = sum(split_of({"Site": f"https://lichess.org/{i}"}, i, cfg.val_fraction) == "val" for i in range(50_000))
    assert 0.015 < val / 50_000 < 0.025


def test_shards_have_fixed_size(extracted):
    out, m = extracted
    for split in SPLITS:
        sizes = [s["positions"] for s in m["shards"][split]]
        assert all(s == 300 for s in sizes[:-1]) and 0 < sizes[-1] <= 300
        assert len(shard_paths(out, split)) == len(sizes)
        assert sum(sizes) == m["positions"][split]
    assert m["bytes_on_disk"] > 0


def test_workers_give_identical_output(corpus, extracted, tmp_path):
    path, _ = corpus
    out1, _ = extracted
    extract(str(path), tmp_path, CFG, shard_size=300, workers=2, chunk_games=7, progress=False)
    for split in SPLITS:
        a, b = load_split(out1, split), load_split(tmp_path, split)
        for key in a:
            assert (a[key] == b[key]).all(), (split, key)


def test_zst_stream_and_target(corpus, tmp_path):
    path, _ = corpus
    zst = tmp_path / "sample.pgn.zst"
    zst.write_bytes(zstandard.ZstdCompressor().compress(path.read_bytes()))
    m = extract(str(zst), tmp_path / "out", CFG, shard_size=500, target_positions=1500, chunk_games=5, progress=False)
    total = sum(m["positions"].values())
    assert m["stopped_early"]
    assert 1500 <= total < 1500 + 5 * 140  # dépasse d'au plus un lot
    assert m["games_read"] < 73


def test_max_games_and_overwrite_guard(corpus, tmp_path):
    path, _ = corpus
    m = extract(str(path), tmp_path, CFG, max_games=10, progress=False)
    assert m["games_read"] == 10
    with pytest.raises(FileExistsError):
        extract(str(path), tmp_path, CFG, max_games=10, progress=False)
    m2 = extract(str(path), tmp_path, CFG, max_games=20, overwrite=True, progress=False)
    assert m2["games_read"] == 20
