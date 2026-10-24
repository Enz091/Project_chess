"""Self-play (étape 6c) : parties, enregistrement, abandon, arbitrage par les tables, tampon entre sessions."""

import json
import time
from dataclasses import replace

import chess
import numpy as np
import pytest

from chesszero.book import OpeningBook
from chesszero.mcts import MCTSConfig
from chesszero.selfplay import (T_MAX_PLIES, T_RESIGN, T_TABLES, SelfPlayConfig, ShardWriter, buffer_window,
                                end_of_game, list_shards, load_shard, run_selfplay, sync_buffer, verify_shard)
from chesszero.syzygy import DEFAULT_DIR, Tablebase

CURSED_WIN = "8/p7/3N4/8/8/N1K5/6k1/8 w - - 0 1"  # vraies tables : WDL +1, DTZ 102 (gain trop long : nulle)
BLESSED_LOSS = "8/k7/4p3/2N5/3K4/8/4N3/8 b - - 0 1"  # vraies tables : WDL -1 (perte trop longue : nulle)


class Stub:
    """Priors uniformes ; valeur donnée par `value_fn(board)` (camp au trait)."""

    def __init__(self, value_fn=lambda b: 0.0):
        self.value_fn = value_fn

    def evaluate(self, boards, legal_indices):
        return [np.full(len(i), 1.0 / len(i)) for i in legal_indices], np.array([self.value_fn(b) for b in boards])


class FakeProber:
    def __init__(self, wdl):
        self.wdl = wdl

    def probe_wdl(self, board):
        return self.wdl

    def probe_dtz(self, board):
        raise AssertionError("pas de DTZ en mode WDL seul")


def white_is_lost(board):
    return -0.99 if board.turn == chess.WHITE else 0.99


CFG = SelfPlayConfig(simulations=24, leaves_per_tree=4, games_per_worker=3, max_plies=60, resign_min_ply=20,
                     shard_games=100, temperature_plies=10)


def play(tmp_path, cfg=CFG, value_fn=lambda b: 0.0, book=None, tablebase=None, games=4, seed=0):
    writer = ShardWriter(tmp_path, "essai_w0", {"test": True}, cfg.shard_games, cfg.shard_minutes)
    mcts_cfg = cfg.mcts_config(MCTSConfig())
    stats = run_selfplay(Stub(value_fn), cfg, mcts_cfg, writer, np.random.default_rng(seed), "essai", book=book,
                         tablebase=tablebase, max_games=games)
    return stats, [load_shard(p) for p in writer.written]


# --------------------------------------------------------------------------- parties et fichiers


def test_selfplay_games_are_recorded_and_verified(tmp_path):
    stats, shards = play(tmp_path)
    assert stats["games"] == 4 and len(shards) == 1
    data = shards[0]
    assert verify_shard(data, CFG) == []
    assert len(data["game_uid"]) == 4 and stats["positions"] == len(data["played"]) > 0
    assert data["boards"].shape == (len(data["played"]), 67) and data["boards"].dtype == np.uint8
    assert data["policy_offsets"][-1] == len(data["policy_index"]) == len(data["policy_visits"])
    assert set(np.unique(data["result"])) <= {-1, 0, 1}
    assert (data["termination"] == T_MAX_PLIES).any() or len(set(data["termination"])) >= 1
    meta = json.loads((tmp_path / "essai_w0_00000.json").read_text(encoding="utf-8"))
    assert meta["positions"] == len(data["played"]) and meta["games"] == 4 and meta["test"] is True


def test_verify_detects_corruption(tmp_path):
    cfg = replace(CFG, no_resign_fraction=0.0, book_fraction=0.0)
    _, (data,) = play(tmp_path, cfg=cfg, value_fn=white_is_lost, games=2)  # parties décisives (abandon)
    assert verify_shard(data, CFG) == []
    bad = dict(data)
    bad["result"] = -data["result"]
    assert any("résultat" in e for e in verify_shard(bad, CFG))
    bad = dict(data)
    bad["played"] = data["played"].copy()
    bad["played"][0] = -1
    assert any("coup joué" in e for e in verify_shard(bad, CFG))
    bad = dict(data)
    bad["boards"] = data["boards"].copy()
    bad["boards"][0, :64] = np.roll(bad["boards"][0, :64], 1)
    assert any("position" in e for e in verify_shard(bad, CFG))


def test_games_start_from_the_book(tmp_path):
    book = OpeningBook(["e2e4 e7e5 g1f3 b8c6", "d2d4 d7d5 c2c4 e7e6", "c2c4 g8f6"])
    cfg = replace(CFG, book_fraction=1.0)
    _, (data,) = play(tmp_path, cfg=cfg, book=book, games=4)
    assert verify_shard(data, cfg) == []
    for g in range(4):
        i = int(data["start_index"][g])
        assert i >= 0 and str(data["game_moves"][g]).startswith(book.lines[i])
        assert int(data["start_plies"][g]) == len(book.lines[i].split())
    assert data["ply"].min() >= 2  # rien d'enregistré dans l'ouverture du livre


# --------------------------------------------------------------------------- abandon et parties de contrôle


def test_resignation(tmp_path):
    cfg = replace(CFG, no_resign_fraction=0.0, book_fraction=0.0, max_plies=80)
    _, (data,) = play(tmp_path, cfg=cfg, value_fn=white_is_lost, games=2)
    assert verify_shard(data, cfg) == []
    assert (data["termination"] == T_RESIGN).all()
    assert (data["result_white"] == -1).all()
    assert (data["plies"] == 20).all()  # les Blancs abandonnent dès que c'est permis (demi-coup 20)
    assert (data["would_resign"] == -1).all()


def test_control_games_never_resign_but_record_the_trigger(tmp_path):
    cfg = replace(CFG, no_resign_fraction=1.0, book_fraction=0.0, max_plies=40)
    _, (data,) = play(tmp_path, cfg=cfg, value_fn=white_is_lost, games=2)
    assert verify_shard(data, cfg) == []
    assert data["control"].all() and not (data["termination"] == T_RESIGN).any()
    assert (data["would_resign"][:, 0] == 20).all() and (data["would_resign"][:, 1] == -1).all()


# --------------------------------------------------------------------------- arbitrage par les tables


def test_tablebase_adjudication_wdl_only(tmp_path):
    """Faux sondeur qui couvre tout (32 pièces) et dit « le camp au trait perd » : la partie s'arrête à la première
    position sans droit de roque et de compteur nul (prise ou coup de pion), avec le résultat des tables. (Avec
    « le camp au trait gagne », le MCTS évite toute prise ou poussée de pion et la partie finit par répétition :
    preuve que les valeurs exactes des tables servent bien dans l'arbre.)"""
    book = OpeningBook(["e2e4 e7e5 e1e2 e8e7"])  # plus de droit de roque
    tb = Tablebase(prober=FakeProber(-2), max_pieces=32, wdl_only=True)
    cfg = replace(CFG, book_fraction=1.0, tablebase_pieces=32, max_plies=200)
    _, (data,) = play(tmp_path, cfg=cfg, book=book, tablebase=tb, games=3)
    assert (data["termination"] == T_TABLES).all()
    assert verify_shard(data, cfg, tb) == []
    for g in range(3):
        final = chess.Board()
        for u in str(data["game_moves"][g]).split():
            final.push_uci(u)
        assert final.halfmove_clock == 0
        assert int(data["result_white"][g]) == (-1 if final.turn == chess.WHITE else 1)  # le camp au trait perd


@pytest.mark.parametrize("wdl", [1, -1, 0])
def test_cursed_win_and_blessed_loss_are_draws_wdl_only(wdl):
    tb = Tablebase(prober=FakeProber(wdl), max_pieces=5, wdl_only=True)
    board = chess.Board("8/p7/3N4/8/8/N1K5/6k1/8 w - - 0 1")
    assert tb.value(board) == 0.0
    assert end_of_game(board, tb) == (0, T_TABLES)


def test_wdl_only_needs_a_zero_counter_and_never_chooses_moves():
    tb = Tablebase(prober=FakeProber(2), max_pieces=5, wdl_only=True)
    assert tb.value(chess.Board("8/p7/3N4/8/8/N1K5/6k1/8 w - - 0 1")) == 1.0
    assert tb.value(chess.Board("8/p7/3N4/8/8/N1K5/6k1/8 w - - 7 10")) is None  # compteur non nul : pas couvert
    assert tb.choose(chess.Board("8/p7/3N4/8/8/N1K5/6k1/8 w - - 0 1")) is None


real = pytest.mark.skipif(Tablebase.open(DEFAULT_DIR) is None, reason="tables Syzygy absentes de tools/syzygy")


@real
def test_real_tables_cursed_win_and_blessed_loss_wdl_only():
    tb = Tablebase.open(wdl_only=True)
    assert tb.wdl_only and tb.max_pieces == 5
    cursed, blessed = chess.Board(CURSED_WIN), chess.Board(BLESSED_LOSS)
    assert tb.prober.probe_wdl(cursed) == 1 and tb.prober.probe_wdl(blessed) == -1
    assert tb.value(cursed) == 0.0 and tb.value(blessed) == 0.0
    assert end_of_game(cursed, tb) == (0, T_TABLES) and end_of_game(blessed, tb) == (0, T_TABLES)
    won = chess.Board("8/8/8/4k3/8/8/2Q5/4K3 w - - 0 1")  # KQ contre K
    assert end_of_game(won, tb) == (1, T_TABLES)
    won.turn = chess.BLACK
    assert end_of_game(won, tb) == (1, T_TABLES)  # Noirs au trait, perdants : résultat toujours vu des Blancs


# --------------------------------------------------------------------------- tampon entre sessions


def fake_shard(directory, name, positions, created):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / f"{name}.npz", x=np.zeros(1))
    (directory / f"{name}.json").write_text(json.dumps({"created": created, "positions": positions}), encoding="utf-8")


def test_buffer_window_keeps_most_recent_up_to_the_limit(tmp_path):
    for i in range(6):
        fake_shard(tmp_path, f"s{i}", 100, created=1000 + i)
    names = [s["name"] for s in buffer_window(list_shards([tmp_path]), 250)]
    assert names == ["s5", "s4", "s3"]  # 300 >= 250 : le dernier shard peut dépasser


def test_sync_buffer_across_sessions(tmp_path):
    previous = tmp_path / "input" / "selfplay"  # sortie de la session précédente (lecture seule sur Kaggle)
    work = tmp_path / "working" / "selfplay"
    for i in range(5):
        fake_shard(previous, f"a{i}", 100, created=1000 + i)
    s = sync_buffer([previous], work, 300)  # début de session : on récupère la fenêtre
    assert s["copied"] == 3 and sorted(x["name"] for x in list_shards([work])) == ["a2", "a3", "a4"]
    for i in range(2):  # la session produit ses propres shards
        fake_shard(work, f"b{i}", 100, created=2000 + i)
    s = sync_buffer([], work, 300)  # fin de session : la sortie ne garde que la fenêtre
    assert s["removed"] == 2 and sorted(x["name"] for x in list_shards([work])) == ["a4", "b0", "b1"]
    assert sorted(p.name for p in previous.iterdir()) == sorted(f"a{i}.{e}" for i in range(5) for e in ("npz", "json"))


def test_incomplete_shard_is_ignored(tmp_path):
    fake_shard(tmp_path, "ok", 10, created=1)
    np.savez(tmp_path / "orphelin.npz", x=np.zeros(1))  # .json absent : écriture interrompue
    assert [s["name"] for s in list_shards([tmp_path])] == ["ok"]


def test_writer_flushes_every_n_games(tmp_path):
    cfg = replace(CFG, shard_games=2)
    stats, shards = play(tmp_path, cfg=cfg, games=5)
    assert stats["games"] == 5 and [len(s["game_uid"]) for s in shards] == [2, 2, 1]
    assert len(list_shards([tmp_path])) == 3
    assert all(verify_shard(s, cfg) == [] for s in shards)


def test_deadlines_stop_new_games_and_drop_unfinished(tmp_path):
    writer = ShardWriter(tmp_path, "t", {}, 100, 10)
    cfg = replace(CFG, games_per_worker=2)
    now = time.time()
    stats = run_selfplay(Stub(), cfg, cfg.mcts_config(MCTSConfig()), writer, np.random.default_rng(0), "t",
                         soft_deadline=now - 1, hard_deadline=now - 1)
    assert stats["games"] == 0 and writer.written == []


def test_positions_target_stops_new_games_and_lets_running_games_finish(tmp_path):
    """Tranche de la boucle (2026-10-24) : plus de nouvelles parties une fois l'objectif de positions couvert (parties
    finies + parties en cours estimées à la longueur moyenne), les parties en cours se terminent ; pas avant la durée
    minimale."""
    cfg = replace(CFG, games_per_worker=3)
    writer = ShardWriter(tmp_path / "a", "t", {}, 100, 10)
    stats = run_selfplay(Stub(), cfg, cfg.mcts_config(MCTSConfig()), writer, np.random.default_rng(0), "t",
                         max_games=50, target_positions=60)
    assert stats["unfinished_dropped"] == 0 and stats["target_reached_after"] is not None
    assert 60 <= stats["positions"] < 60 + 3 * cfg.max_plies  # au plus les parties en cours au moment de l'arrêt
    assert stats["games"] < 50
    # durée minimale pas encore écoulée : l'objectif ne coupe rien (ici : les 6 parties demandées sont jouées)
    writer = ShardWriter(tmp_path / "b", "t", {}, 100, 10)
    stats = run_selfplay(Stub(), cfg, cfg.mcts_config(MCTSConfig()), writer, np.random.default_rng(0), "t",
                         max_games=6, target_positions=1, min_seconds=3600)
    assert stats["games"] == 6 and stats["target_reached_after"] is None
    shards = [load_shard(p) for p in writer.written]
    assert all(verify_shard(s, cfg) == [] for s in shards)
    assert all("policy_prior" in s and len(s["policy_prior"]) == len(s["policy_index"]) for s in shards)


# --------------------------------------------------------------------------- indicateur des répétitions précoces


def test_repetition_stats_by_ply_and_material():
    from chesszero.selfplay import T_MATE, T_REPETITION, repetition_stats

    shuffle = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8"  # position initiale vue 3 fois : demi-coup 8, matériel 78
    data = {"game_uid": np.array(["a", "b", "c"]), "result_white": np.array([0, 0, 1], dtype=np.int8),
            "termination": np.array([T_REPETITION, T_MAX_PLIES, T_MATE], dtype=np.uint8),
            "game_moves": np.array([shuffle, "e2e4", "f2f3 e7e5 g2g4 d8h4"])}
    s = repetition_stats([data])
    assert s["games"] == 3 and s["draws"] == 2 and s["repetition_draws"] == 1
    assert s["share_of_games"] == pytest.approx(1 / 3) and s["share_of_draws"] == pytest.approx(0.5)
    assert s["by_ply"]["< 60"] == 1 and sum(s["by_ply"].values()) == 1
    assert s["by_material"][">= 60 (presque complet)"] == 1 and s["median_material"] == 78
