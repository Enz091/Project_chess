"""Entraînement sur le self-play (étape 6d) : tampon de parties + données Lichess, par tours, avec garde-fous.

Un TOUR = un réseau candidat :
  1. fenêtre du tampon (shards les plus récents, ~500 000 positions) ; 10 % des PARTIES sont mises de côté
     (choisies par leur identifiant : toujours les mêmes) pour mesurer le surapprentissage ;
  2. nombre de pas : chaque NOUVELLE position (pas encore utilisée par un tour accepté) doit être vue `reuse` (4)
     fois en moyenne : pas = reuse × nouvelles / (positions de self-play par lot) ; plafonné à `max_views_per_round`
     (1) passage moyen par position du tampon dans un même tour ; aucun tour tant que le tampon (hors parties mises
     de côté) a moins de `min_buffer_positions` (250 000) positions ;
  3. chaque lot : composition EXACTE, `round(B × part humaine)` positions Lichess (25 %, puis 10 % après
     `human_switch_generation` générations acceptées) et le reste tiré uniformément dans le tampon ; cible de
     politique = visites normalisées (self-play) ou coup joué (Lichess) ; cible de valeur = résultat de la partie
     (ou (résultat + Q racine) / 2 si `value_target = "blend"`) pour le self-play, résultat Lichess si
     `value_valid` ; perte moyenne par position (chaque part pèse selon sa place dans le lot) ;
  4. taux : AdamW neuf au départ, montée linéaire de `lr_start` (2e-5) à `lr` (2e-4) sur `warmup_steps` (300) pas,
     puis constant, multiplié par `lr_scale` (divisé par 2 par le garde-fou G1) ;
  5. garde-fous (`decide`) : passés -> le tour devient le CANDIDAT (`candidate.pt`), en attente de l'arène
     (étape 7) ; non passés -> retour aux poids et à l'optimiseur du dernier état qui les a passés (`line_ok.pt`),
     réaction automatique, et le tour sera refait sur les mêmes données ;
  6. verdict de l'arène (`commit_arena`) : accepté -> `generations/gen_XXXX.pt`, génération + 1 ; rejeté -> la
     ligne d'entraînement CONTINUE (comme AlphaGo Zero : rien n'est défait), seul le self-play garde l'ancien
     réseau. Les générations comptées (part humaine, garde-fous stricts) sont celles acceptées en arène.

Reprise : les lots d'un tour ne dépendent que de (graine, n° de tour, pas) et de la liste des shards fixée au début
du tour (enregistrée) ; `train_state.pt` est écrit tous les `save_every` pas : une reprise après coupure donne
exactement les mêmes poids qu'un tour d'une traite (testé).
"""

from __future__ import annotations

import json
import math
import os
import time
import tomllib
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chesszero.encoding import N_ACTIONS, legal_masks_from_packed, unpack_batch
from chesszero.mcts import CONFIG_DIR
from chesszero.model import ModelConfig, build_model, masked_log_softmax
from chesszero.selfplay import buffer_window, list_shards, load_shard


@dataclass(frozen=True)
class SelfPlayTrainConfig:
    """Section [selfplay_train] de configs/<nom>.toml."""

    batch_size: int = 1024
    lr: float = 2e-4
    lr_start: float = 2e-5
    warmup_steps: int = 300
    lr_min: float = 2.5e-5  # plancher du taux après des divisions par 2 (G1)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    buffer_positions: int = 500_000
    reuse: float = 4.0  # passages moyens par nouvelle position
    reduced_reuse: float = 2.0  # après un rejet pour surapprentissage (G3)
    max_views_per_round: float = 1.0  # plafond : passages moyens par position du tampon dans un même tour
    min_buffer_positions: int = 250_000  # pas de tour en dessous (hors parties mises de côté)
    human_fraction: float = 0.25
    human_fraction_late: float = 0.10
    human_switch_generation: int = 5  # 10 % à partir de la 6e génération acceptée
    value_target: str = "result"  # "result" ou "blend" = (résultat + Q racine) / 2
    heldout_fraction: float = 0.10  # parties de self-play mises de côté
    lichess_val_positions: int = 50_000
    early_generations: int = 5  # G1 / G2 stricts pendant ces générations
    g1_policy_loss: float = 0.05  # hausse de perte de politique Lichess (contre le réseau supervisé)
    g2_top1_points: float = 3.0  # baisse du top-1 Lichess, en points
    catastrophic_policy_loss: float = 0.15  # après les premières générations : rejet seulement au-delà
    catastrophic_top1_points: float = 10.0
    jump_policy_loss: float = 0.05  # alerte : chute brutale par rapport à la génération précédente
    jump_top1_points: float = 2.0
    g3_value_loss: float = 0.05  # alerte : hausse de la perte de valeur Lichess
    max_rejections: int = 3  # G4 : arrêt après autant de rejets d'affilée par les garde-fous
    max_arena_rejections: int = 5  # arrêt (stagnation) après autant de rejets en arène d'affilée
    anchor_stop_generations: int = 2  # arrêt si l'accepté est sous l'ancre sur autant de générations consécutives...
    anchor_stop_from_generation: int = 3  # ... en ne comptant que les générations >= celle-ci (1er arrêt possible : 4)
    save_every: int = 100
    log_every: int = 20
    amp: str = "auto"  # fp16 + GradScaler sur CUDA seulement
    num_workers: int = 3  # processus qui calculent les masques des positions Lichess
    seed: int = 0

    def __post_init__(self):
        if self.value_target not in ("result", "blend"):
            raise ValueError("value_target doit valoir 'result' ou 'blend'")

    @classmethod
    def from_toml(cls, path: str | Path) -> SelfPlayTrainConfig:
        with open(path, "rb") as f:
            section = tomllib.load(f).get("selfplay_train", {})
        unknown = set(section) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"clés inconnues dans [selfplay_train] : {sorted(unknown)}")
        return cls(**section)

    @classmethod
    def named(cls, name: str) -> SelfPlayTrainConfig:
        return cls.from_toml(CONFIG_DIR / f"{name}.toml")


# --------------------------------------------------------------------------- données


def is_heldout(game_uid: str, fraction: float) -> bool:
    """Partie mise de côté (jamais utilisée pour l'entraînement), choisie par son identifiant : stable."""
    return zlib.crc32(str(game_uid).encode()) % 1000 < fraction * 1000


class SelfPlayData:
    """Positions des shards d'une fenêtre du tampon, en mémoire."""

    def __init__(self, shards: list[dict], heldout_fraction: float):
        self.names = [s["name"] for s in shards]
        parts = [load_shard(s["npz"]) for s in shards]
        self.shard_of = np.concatenate([np.full(len(p["played"]), i, np.int32) for i, p in enumerate(parts)]) \
            if parts else np.zeros(0, np.int32)
        cat = (lambda k: np.concatenate([p[k] for p in parts])) if parts else (lambda k: np.zeros(0))
        self.boards = cat("boards") if parts else np.zeros((0, 67), np.uint8)
        self.result = cat("result").astype(np.float32) if parts else np.zeros(0, np.float32)
        self.root_q = cat("root_q").astype(np.float32) if parts else np.zeros(0, np.float32)
        # coups légaux et visites : décalages remis bout à bout
        offsets, index, visits, base = [np.zeros(1, np.int64)], [], [], 0
        uids = []
        for p in parts:
            offsets.append(p["policy_offsets"][1:].astype(np.int64) + base)
            base += len(p["policy_index"])
            index.append(p["policy_index"])
            visits.append(p["policy_visits"])
            uids.append(p["game_uid"][p["game"]])
        self.offsets = np.concatenate(offsets)
        self.index = np.concatenate(index).astype(np.int64) if index else np.zeros(0, np.int64)
        self.visits = np.concatenate(visits).astype(np.float32) if visits else np.zeros(0, np.float32)
        self.game_uid = np.concatenate(uids) if uids else np.zeros(0, dtype=str)
        held = {u: is_heldout(u, heldout_fraction) for u in set(self.game_uid.tolist())}
        self.heldout = np.array([held[u] for u in self.game_uid.tolist()], dtype=bool)
        self.train_rows = np.nonzero(~self.heldout)[0]
        self.heldout_rows = np.nonzero(self.heldout)[0]

    def __len__(self) -> int:
        return len(self.boards)

    def train_positions_in(self, names: set[str]) -> int:
        """Positions d'entraînement (hors parties mises de côté) des shards nommés."""
        wanted = np.array([n in names for n in self.names], dtype=bool)
        return int(wanted[self.shard_of[self.train_rows]].sum()) if len(self.train_rows) else 0

    def targets(self, rows: np.ndarray, value_target: str):
        """(masque des coups légaux, cible de politique dense, cible de valeur) pour ces positions."""
        n = len(rows)
        legal = np.zeros((n, N_ACTIONS), dtype=bool)
        policy = np.zeros((n, N_ACTIONS), dtype=np.float32)
        for i, r in enumerate(rows):
            a, b = self.offsets[r], self.offsets[r + 1]
            idx, v = self.index[a:b], self.visits[a:b]
            legal[i, idx] = True
            policy[i, idx] = v / v.sum()
        z = self.result[rows]
        value = (z + self.root_q[rows]) / 2 if value_target == "blend" else z
        return legal, policy, value.astype(np.float32)


def human_val_set(human_val: dict, size: int, seed: int, pool) -> dict:
    """Sous-ensemble FIXE de la validation Lichess, avec ses masques (calculés une fois)."""
    n = len(human_val["boards"])
    idx = np.arange(n) if size <= 0 or size >= n else np.sort(np.random.default_rng(seed).choice(n, size, replace=False))
    boards = human_val["boards"][idx]
    chunks = [boards[i:i + 2048] for i in range(0, len(boards), 2048)]
    masks = list(pool.map(legal_masks_from_packed, chunks)) if pool else [legal_masks_from_packed(c) for c in chunks]
    return {"boards": boards, "moves": human_val["moves"][idx].astype(np.int64),
            "results": human_val["results"][idx].astype(np.float32),
            "value_valid": human_val["value_valid"][idx].astype(np.float32),
            "legal": np.concatenate(masks) if masks else np.zeros((0, N_ACTIONS), bool)}


# --------------------------------------------------------------------------- lots


def batch_rows(cfg: SelfPlayTrainConfig, human_fraction: float, round_id: int, step: int,
               n_selfplay_train: int, n_human: int) -> tuple[np.ndarray, np.ndarray]:
    """Lignes (dans les positions d'entraînement du tampon, et dans les données Lichess) du lot n° `step` du tour
    `round_id`. Ne dépend que de (graine, tour, pas) : reprise exacte. Tirage uniforme avec remise."""
    n_h = int(round(cfg.batch_size * human_fraction))
    rng = np.random.default_rng([cfg.seed, round_id, step])
    sp = rng.integers(n_selfplay_train, size=cfg.batch_size - n_h)
    hu = rng.integers(n_human, size=n_h) if n_h else np.zeros(0, np.int64)
    return sp, hu


def make_batch(sp_data: SelfPlayData, human: dict, sp_pos: np.ndarray, hu_rows: np.ndarray, human_legal: np.ndarray,
               value_target: str) -> dict:
    sp_rows = sp_data.train_rows[sp_pos]
    legal_sp, pol_sp, val_sp = sp_data.targets(sp_rows, value_target)
    n_h = len(hu_rows)
    pol_h = np.zeros((n_h, N_ACTIONS), dtype=np.float32)
    pol_h[np.arange(n_h), human["moves"][hu_rows].astype(np.int64)] = 1.0
    boards = np.concatenate([sp_data.boards[sp_rows], human["boards"][hu_rows]])
    return {
        "x": unpack_batch(boards),
        "legal": np.concatenate([legal_sp, human_legal]),
        "policy": np.concatenate([pol_sp, pol_h]),
        "value": np.concatenate([val_sp, human["results"][hu_rows].astype(np.float32)]),
        "value_valid": np.concatenate([np.ones(len(sp_rows), np.float32),
                                       human["value_valid"][hu_rows].astype(np.float32)]),
        "is_human": np.concatenate([np.zeros(len(sp_rows), bool), np.ones(n_h, bool)]),
    }


def mixed_loss(logits, value, batch_t: dict):
    """Perte moyenne par position : entropie croisée (coups illégaux masqués) + erreur quadratique de valeur (positions
    à `value_valid` seulement). Renvoie (perte, détail par part : self-play / humain)."""
    log_p = masked_log_softmax(logits, batch_t["legal"])
    ce = -(batch_t["policy"] * log_p).sum(dim=-1)
    se = (value.float() - batch_t["value"]) ** 2
    valid = batch_t["value_valid"]
    loss = ce.mean() + (valid * se).sum() / valid.sum().clamp(min=1.0)
    parts = {}
    ce, se = ce.detach(), se.detach()  # le détail par part n'est qu'un suivi : pas de gradient
    for name, mask in (("sp", ~batch_t["is_human"]), ("human", batch_t["is_human"])):
        m = mask.float()
        parts[f"{name}_policy"] = float((ce * m).sum() / m.sum().clamp(min=1.0))
        vm = m * valid
        parts[f"{name}_value"] = float((se * vm).sum() / vm.sum().clamp(min=1.0))
    parts["human_share"] = float(batch_t["is_human"].float().mean())
    return loss, parts


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


# --------------------------------------------------------------------------- évaluation


@torch.no_grad()
def eval_human(model, val: dict, device, batch_size: int = 1024) -> dict:
    """Validation Lichess : perte de politique, top-1, perte de valeur (positions à value_valid)."""
    model.eval()
    n, pol, top1, val_se, val_n = len(val["boards"]), 0.0, 0, 0.0, 0.0
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        x = torch.from_numpy(unpack_batch(val["boards"][sl])).to(device)
        logits, value = model(x)
        legal = torch.from_numpy(val["legal"][sl]).to(device)
        moves = torch.from_numpy(val["moves"][sl]).to(device)
        log_p = masked_log_softmax(logits, legal)
        pol += float(F.nll_loss(log_p, moves, reduction="sum"))
        top1 += int((log_p.argmax(dim=1) == moves).sum())
        vv = torch.from_numpy(val["value_valid"][sl]).to(device)
        val_se += float((vv * (value.float() - torch.from_numpy(val["results"][sl]).to(device)) ** 2).sum())
        val_n += float(vv.sum())
    model.train()
    return {"policy_loss": pol / max(n, 1), "top1": top1 / max(n, 1), "value_loss": val_se / max(val_n, 1)}


@torch.no_grad()
def eval_selfplay(model, data: SelfPlayData, rows: np.ndarray, device, value_target: str,
                  batch_size: int = 1024, max_rows: int = 50_000) -> dict:
    """Self-play (parties mises de côté, ou échantillon d'entraînement) : perte de politique contre les visites,
    erreur de valeur contre le RÉSULTAT de la partie (quelle que soit la cible d'entraînement)."""
    if len(rows) == 0:
        return {"policy_loss": None, "value_mse": None, "positions": 0}
    if len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(0).choice(rows, max_rows, replace=False))
    model.eval()
    pol, se = 0.0, 0.0
    for i in range(0, len(rows), batch_size):
        r = rows[i:i + batch_size]
        legal, policy, _ = data.targets(r, value_target)
        x = torch.from_numpy(unpack_batch(data.boards[r])).to(device)
        logits, value = model(x)
        log_p = masked_log_softmax(logits, torch.from_numpy(legal).to(device))
        pol += float(-(torch.from_numpy(policy).to(device) * log_p).sum())
        se += float(((value.float() - torch.from_numpy(data.result[r]).to(device)) ** 2).sum())
    model.train()
    return {"policy_loss": pol / len(rows), "value_mse": se / len(rows), "positions": int(len(rows))}


# --------------------------------------------------------------------------- garde-fous


def decide(metrics: dict, baseline: dict, previous: dict | None, history: list[dict], generation: int,
           cfg: SelfPlayTrainConfig) -> dict:
    """Décision sur un candidat (fonction pure, testée).

    `metrics` / `baseline` (réseau supervisé) / `previous` (dernière génération acceptée) : dicts avec
    `lichess` = {policy_loss, top1, value_loss}, `heldout` = {value_mse}, `train` = {sp_value}. `history` : les
    métriques des candidats précédents (acceptés ou non), du plus ancien au plus récent. `generation` = nombre de
    générations déjà acceptées.

    Règles (validées par l'auteur) :
      - G1 / G2 stricts pendant les `early_generations` premières générations : perte de politique Lichess >
        réseau supervisé + 0,05 -> rejet, taux / 2 ; top-1 Lichess < supervisé − 3 points -> rejet, part humaine
        remise (et maintenue) à 25 % ;
      - ensuite : rejet seulement si chute catastrophique (top-1 −10 points ou perte de politique +0,15) ; les
        dépassements de G1 / G2 deviennent des alertes ;
      - chute brutale par rapport à la génération précédente (top-1 −2 points ou perte de politique +0,05) :
        alerte ;
      - G3 : perte de valeur Lichess > supervisé + 0,05 : alerte ; surapprentissage sur le self-play (erreur de
        valeur des parties mises de côté en hausse deux tours de suite pendant que celle d'entraînement baisse) :
        rejet, réutilisation ramenée à `reduced_reuse` ;
      - G4 (appliqué par l'appelant) : arrêt après `max_rejections` rejets d'affilée.
    """
    lic, base = metrics["lichess"], baseline["lichess"]
    d_policy = lic["policy_loss"] - base["policy_loss"]
    d_top1 = (base["top1"] - lic["top1"]) * 100  # baisse en points
    reasons, alerts, actions = [], [], []
    early = generation < cfg.early_generations
    if d_policy > cfg.g1_policy_loss:
        if early:
            reasons.append(f"G1 : perte de politique Lichess +{d_policy:.3f} (> {cfg.g1_policy_loss})")
            actions.append("halve_lr")
        else:
            alerts.append(f"G1 (alerte) : perte de politique Lichess +{d_policy:.3f} contre le réseau supervisé")
    if d_top1 > cfg.g2_top1_points:
        if early:
            reasons.append(f"G2 : top-1 Lichess −{d_top1:.1f} points (> {cfg.g2_top1_points})")
            actions.append("human_25")
        else:
            alerts.append(f"G2 (alerte) : top-1 Lichess −{d_top1:.1f} points contre le réseau supervisé")
    if not early and (d_top1 > cfg.catastrophic_top1_points or d_policy > cfg.catastrophic_policy_loss):
        reasons.append(f"chute catastrophique : top-1 −{d_top1:.1f} points, perte de politique +{d_policy:.3f}")
        actions.append("halve_lr")
    if previous is not None:
        prev = previous["lichess"]
        jump_top1 = (prev["top1"] - lic["top1"]) * 100
        jump_policy = lic["policy_loss"] - prev["policy_loss"]
        if jump_top1 > cfg.jump_top1_points or jump_policy > cfg.jump_policy_loss:
            alerts.append(f"chute brutale depuis la génération précédente : top-1 −{jump_top1:.1f} points, "
                          f"perte de politique +{jump_policy:.3f}")
    d_value = lic["value_loss"] - base["value_loss"]
    if d_value > cfg.g3_value_loss:
        alerts.append(f"G3 (alerte) : perte de valeur Lichess +{d_value:.3f} contre le réseau supervisé")
    seq = [h for h in history if h.get("heldout", {}).get("value_mse") is not None] + [metrics]
    if len(seq) >= 3 and metrics.get("heldout", {}).get("value_mse") is not None:
        h = [s["heldout"]["value_mse"] for s in seq[-3:]]
        t = [s["train"]["sp_value"] for s in seq[-3:]]
        if h[0] < h[1] < h[2] and t[0] > t[1] > t[2]:
            reasons.append(f"G3 : surapprentissage (erreur de valeur des parties mises de côté "
                           f"{h[0]:.3f} -> {h[1]:.3f} -> {h[2]:.3f} pendant que celle d'entraînement baisse)")
            actions.append("reduce_reuse")
    return {"accept": not reasons, "reasons": reasons, "alerts": alerts, "actions": sorted(set(actions))}


# --------------------------------------------------------------------------- tours d'entraînement


def lr_at(step: int, cfg: SelfPlayTrainConfig, lr_scale: float) -> float:
    if step < cfg.warmup_steps:
        base = cfg.lr_start + (cfg.lr - cfg.lr_start) * (step + 1) / cfg.warmup_steps
    else:
        base = cfg.lr
    return max(base * lr_scale, min(cfg.lr_min, base))


def plan_steps(new_positions: int, buffer_train_positions: int, cfg: SelfPlayTrainConfig, reuse: float,
               human_fraction: float) -> int:
    """Pas du tour : `reuse` passages par nouvelle position, plafonné à `max_views_per_round` passage par position
    du tampon ; 0 si le tampon est trop petit."""
    if buffer_train_positions < cfg.min_buffer_positions or new_positions <= 0:
        return 0
    n_sp = cfg.batch_size - int(round(cfg.batch_size * human_fraction))
    want = math.ceil(reuse * new_positions / n_sp)
    cap = max(1, math.floor(cfg.max_views_per_round * buffer_train_positions / n_sp))
    return min(want, cap)


def _atomic_torch_save(obj, path: Path) -> None:
    """Écrit dans `<path>.tmp`, force sur disque, puis renomme : `path` est soit l'ancien, soit le nouveau."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


STATE_FORMAT = 2  # 1 = étape 6d (acceptation par les garde-fous seuls) ; 2 = étape 7b (candidat puis arène)


def _new_state_fields(init_sha: str | None) -> dict:
    """Champs de l'état ajoutés en 7b (format 2)."""
    return {
        "format": STATE_FORMAT,
        "candidate": None,  # candidat en attente du verdict de l'arène (voir `_finish_round`)
        "arena_rejections": 0,  # rejets en arène d'affilée (les rejets par les garde-fous n'y comptent pas)
        "accepted": {"generation": 0, "file": None, "sha256": init_sha, "anchor": None, "anchor_source": "identité"},
        "anchor_history": [],  # mesure contre l'ancre de chaque génération acceptée
        "anchor_stop_ack": 0,  # générations <= cette valeur ignorées par l'arrêt sur l'ancre (après --clear-stop)
        "commits": [],  # verdicts d'arène déjà appliqués (un verdict rejoué ne fait rien)
        "line_ok_round": None,  # tour du point de retour `line_ok.pt` (None = réseau de départ)
        "stop_reason": None,
    }


def _sha256(path: Path) -> str:
    from chesszero.arena import sha256_file

    return sha256_file(path)


class SelfPlayTrainer:
    """État de l'entraînement sur le self-play dans `out_dir` (format 2, étape 7b) :
      train_state.pt  SOURCE DE VÉRITÉ : modèle et optimiseur de la ligne d'entraînement, pas, tour en cours,
                      candidat en attente, génération acceptée (fichier + SHA-256 + mesure contre l'ancre),
                      rejets en arène, historique contre l'ancre, arrêt (lu par `read_train_state`) ;
      line_ok.pt      modèle et optimiseur du dernier état qui a passé les garde-fous (point de retour d'un rejet
                      par les garde-fous ; régénéré depuis train_state.pt s'il ne correspond pas au tour attendu) ;
      candidate.pt    candidat en attente (format load_for_inference) ; ignoré s'il n'est pas référencé ;
      generations/gen_XXXX.pt  réseaux acceptés EN ARÈNE (format load_for_inference + résultats des arènes) ;
                      un fichier non référencé par train_state.pt (coupure) est ignoré puis écrasé ;
      rounds.jsonl    une ligne par tour et par verdict d'arène ; train_log.jsonl : pertes pendant les tours.
    Ordre d'écriture : fichiers de données (atomiques) puis train_state.pt (point de validation), puis line_ok.pt
    et les journaux (régénérés ou complétés sans doublon à la reprise).
    """

    def __init__(self, out_dir: str | Path, cfg: SelfPlayTrainConfig, device: torch.device | str,
                 init_checkpoint: str | Path | None = None, log=print):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cfg, self.device, self.log = cfg, torch.device(device), log
        self.use_amp = cfg.amp == "on" or (cfg.amp == "auto" and self.device.type == "cuda")
        state_path = self.out / "train_state.pt"
        fresh = not state_path.exists()
        if not fresh:
            self.state = torch.load(state_path, map_location="cpu", weights_only=False)
            self._migrate()
            model_cfg = ModelConfig(**self.state["model_config"])
        else:
            if init_checkpoint is None:
                raise ValueError("premier lancement : il faut --init (checkpoint de départ)")
            payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
            model_cfg = ModelConfig(**payload["config"]["model"])
            self.state = {
                "model_config": asdict(model_cfg), "model": payload["model"], "optimizer": None, "scaler": None,
                "global_step": 0, "generation": 0, "round_id": 0, "lr_scale": 1.0, "reuse": cfg.reuse,
                "human_sticky_25": False, "trained_shards": [], "consecutive_rejections": 0, "baseline": None,
                "previous": None, "history": [], "round": None, "status": "ok",
                "init_checkpoint": str(init_checkpoint), "init_step": payload.get("step"),
                **_new_state_fields(_sha256(Path(init_checkpoint))),
            }
        self.model = build_model(model_cfg).to(self.device)
        self.model.load_state_dict(self.state["model"])
        self.model.train()
        decay = [p for p in self.model.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.model.parameters() if p.ndim < 2]
        self.optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                                            {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr_start)
        if self.state["optimizer"] is not None:
            self.optimizer.load_state_dict(self.state["optimizer"])
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if self.state["scaler"] is not None and self.use_amp:
            self.scaler.load_state_dict(self.state["scaler"])
        if fresh:  # lisible tout de suite par read_train_state (réseau de départ et son SHA-256)
            self._save_state()
        self._check_line_ok()

    # ---------------------------------------------------------------- état

    def _migrate(self) -> None:
        """État de l'étape 6d (format 1) -> format 2. Refusé s'il contient des générations : elles ont été
        acceptées par les garde-fous seuls, sans arène."""
        st = self.state
        if st.get("format", 1) >= STATE_FORMAT:
            return
        if st["generation"] > 0:
            raise RuntimeError(f"{self.out} : état de l'étape 6d avec {st['generation']} génération(s) acceptées sans "
                               "arène ; repartir d'un dossier neuf (le tampon, lui, reste utilisable)")
        init = Path(st.get("init_checkpoint") or "")
        st.update(_new_state_fields(_sha256(init) if init.is_file() else None))
        old = self.out / "accepted.pt"
        if old.exists():
            if not (self.out / "line_ok.pt").exists():
                payload = torch.load(old, map_location="cpu", weights_only=False)
                _atomic_torch_save({**payload, "round": None}, self.out / "line_ok.pt")
            old.replace(old.with_name("accepted.pt.migre"))  # gardé (renommé), jamais supprimé
        _atomic_torch_save(st, self.out / "train_state.pt")
        self.log(f"État de l'étape 6d migré au format {STATE_FORMAT} (accepted.pt -> line_ok.pt ; ancien fichier "
                 "gardé sous accepted.pt.migre)")

    def _check_line_ok(self) -> None:
        """`line_ok.pt` doit être le point du tour `line_ok_round`. Après une coupure entre la validation d'un
        candidat (train_state.pt) et l'écriture de line_ok.pt, il est régénéré depuis train_state.pt, qui contient
        à ce moment-là exactement l'état du candidat (aucun tour n'a pu commencer depuis)."""
        path = self.out / "line_ok.pt"
        expected = self.state["line_ok_round"]
        if path.exists():
            current = torch.load(path, map_location="cpu", weights_only=False).get("round", "absent")
            if current == expected:
                return
            if self.state["round"] is not None:
                raise RuntimeError(f"{path} (tour {current}) ne correspond pas au tour {expected} et un tour est en "
                                   "cours : état incohérent")
        elif self.state["round"] is not None and expected is not None:
            raise RuntimeError(f"{path} manquant pendant un tour : état incohérent")
        self._save_line_ok()

    def human_fraction(self) -> float:
        if self.state["human_sticky_25"] or self.state["generation"] < self.cfg.human_switch_generation:
            return self.cfg.human_fraction
        return self.cfg.human_fraction_late

    def _save_state(self) -> None:
        self.state["model"] = self.model.state_dict()
        self.state["optimizer"] = self.optimizer.state_dict()
        self.state["scaler"] = self.scaler.state_dict() if self.use_amp else None
        _atomic_torch_save(self.state, self.out / "train_state.pt")

    def _save_line_ok(self) -> None:
        _atomic_torch_save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                            "scaler": self.scaler.state_dict() if self.use_amp else None,
                            "global_step": self.state["global_step"], "round": self.state["line_ok_round"]},
                           self.out / "line_ok.pt")

    def _restore_line_ok(self) -> None:
        acc = torch.load(self.out / "line_ok.pt", map_location="cpu", weights_only=False)
        if acc.get("round") != self.state["line_ok_round"]:
            raise RuntimeError("line_ok.pt ne correspond pas au dernier état qui a passé les garde-fous")
        self.model.load_state_dict(acc["model"])
        self.optimizer.load_state_dict(acc["optimizer"])
        if self.use_amp and acc.get("scaler"):
            self.scaler.load_state_dict(acc["scaler"])

    def _append(self, name: str, row: dict) -> None:
        with open(self.out / name, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _append_once(self, row: dict, key: tuple) -> None:
        """Ajoute `row` à rounds.jsonl sauf si une ligne de même clé (type, n°) y est déjà (reprise)."""
        from chesszero.arena import read_jsonl

        rows = read_jsonl(self.out / "rounds.jsonl", repair=True)
        if not any((r.get("type", "round"), r.get("round", r.get("candidate"))) == key for r in rows):
            self._append("rounds.jsonl", row)

    def _model_payload(self, metrics: dict, **extra) -> dict:
        """Checkpoint au format load_for_inference (modèle courant)."""
        return {"step": self.state["global_step"], "epoch": 0, "model": self.model.state_dict(),
                "config": {"model": self.state["model_config"], "selfplay_train": asdict(self.cfg)},
                "metrics": metrics, **extra}

    # ---------------------------------------------------------------- un tour

    def run_round(self, buffer_dirs, human_train: dict, human_val: dict, pool: ProcessPoolExecutor | None = None,
                  steps_override: int | None = None, stop_after: int | None = None, force: bool = False) -> dict:
        """Un tour complet (ou sa reprise). `stop_after` : s'arrête après ce nombre de pas de CE lancement (coupure
        simulée). `steps_override` / `force` : pour les essais (nombre de pas imposé, tampon minimal ignoré)."""
        cfg, st = self.cfg, self.state
        if st["status"] == "stop":
            raise RuntimeError(f"arrêt demandé ({st.get('stop_reason') or 'G4'}) : décision humaine nécessaire "
                               "(train_selfplay.py --clear-stop)")
        if st["candidate"] is not None:
            raise RuntimeError(f"candidat {st['candidate']['id']} en attente du verdict de l'arène : commit_arena "
                               "d'abord")
        if st["round"] is None:  # nouveau tour : on fige la fenêtre du tampon
            shards = buffer_window(list_shards(buffer_dirs), cfg.buffer_positions)
            names = sorted(s["name"] for s in shards)
            data = SelfPlayData([s for s in shards if s["name"] in set(names)], cfg.heldout_fraction)
            new = data.train_positions_in(set(names) - set(st["trained_shards"]))
            hf = self.human_fraction()
            if steps_override is not None:
                steps = steps_override
            else:
                steps = plan_steps(new, len(data.train_rows), replace(cfg, min_buffer_positions=0) if force else cfg,
                                   st["reuse"], hf)
            if steps == 0:
                return {"status": "waiting", "buffer_train_positions": int(len(data.train_rows)),
                        "new_positions": new, "min_buffer_positions": cfg.min_buffer_positions}
            st["round"] = {"id": st["round_id"], "shards": names, "steps": steps, "done": 0, "human_fraction": hf,
                           "new_positions": new, "value_target": cfg.value_target}
            if st["baseline"] is None:
                st["baseline"] = {"lichess": eval_human(self.model, human_val, self.device)}
                self.log(f"Réseau de départ sur la validation Lichess : {st['baseline']['lichess']}")
            self._save_state()
        rnd = st["round"]
        by_name = {s["name"]: s for s in list_shards(buffer_dirs)}
        missing = [n for n in rnd["shards"] if n not in by_name]
        if missing:
            raise FileNotFoundError(f"shards du tour en cours introuvables : {missing[:3]}...")
        data = SelfPlayData([by_name[n] for n in rnd["shards"]], cfg.heldout_fraction)
        self.log(f"Tour {rnd['id']} : {rnd['steps']} pas ({rnd['done']} déjà faits), tampon {len(data)} positions "
                 f"({len(data.train_rows)} d'entraînement, {len(data.heldout_rows)} mises de côté), "
                 f"{rnd['new_positions']} nouvelles, part humaine {rnd['human_fraction']:.0%}, cible de valeur "
                 f"{rnd['value_target']}")

        n_human = len(human_train["boards"])
        rows = [batch_rows(cfg, rnd["human_fraction"], rnd["id"], s, len(data.train_rows), n_human)
                for s in range(rnd["done"], rnd["steps"])]
        jobs: dict[int, object] = {}  # masques Lichess calculés quelques lots à l'avance par le pool

        def submit(k: int) -> None:
            if pool is not None and k < len(rows) and k not in jobs:
                jobs[k] = pool.submit(legal_masks_from_packed, human_train["boards"][rows[k][1]])

        for k in range(6):
            submit(k)
        # fenêtre de moyenne du journal : reprise avec le tour pour que le journal soit lui aussi identique
        done_now, window, t0 = 0, list(rnd.get("log_window", [])), time.perf_counter()
        for k, (sp_pos, hu) in enumerate(rows):
            submit(k + 6)
            job = jobs.pop(k, None)
            human_legal = job.result() if job is not None else legal_masks_from_packed(human_train["boards"][hu])
            batch = make_batch(data, human_train, sp_pos, hu, human_legal, rnd["value_target"])
            assert int(batch["is_human"].sum()) == len(hu) == int(round(cfg.batch_size * rnd["human_fraction"]))
            bt = to_device(batch, self.device)
            for g in self.optimizer.param_groups:
                g["lr"] = lr_at(st["global_step"], cfg, st["lr_scale"])
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits, value = self.model(bt["x"])
            loss, parts = mixed_loss(logits, value, bt)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            rnd["done"] += 1
            st["global_step"] += 1
            done_now += 1
            window.append({"loss": float(loss.detach()), **parts})
            if rnd["done"] % cfg.log_every == 0 or rnd["done"] == rnd["steps"]:
                row = {"round": rnd["id"], "step": rnd["done"], "global_step": st["global_step"],
                       "lr": self.optimizer.param_groups[0]["lr"],
                       **{k2: float(np.mean([w[k2] for w in window])) for k2 in window[0]},
                       "positions_per_s": len(window) * cfg.batch_size / max(time.perf_counter() - t0, 1e-9)}
                self._append("train_log.jsonl", row)
                self.log(f"  pas {rnd['done']}/{rnd['steps']} : perte {row['loss']:.3f} (self-play politique "
                         f"{row['sp_policy']:.3f} valeur {row['sp_value']:.3f} ; humain politique "
                         f"{row['human_policy']:.3f}) ; part humaine {row['human_share']:.3f} ; lr {row['lr']:.2e}")
                window, t0 = [], time.perf_counter()
            if rnd["done"] % cfg.save_every == 0 and rnd["done"] < rnd["steps"]:
                rnd["log_window"] = window
                self._save_state()
            if stop_after is not None and done_now >= stop_after and rnd["done"] < rnd["steps"]:
                for job in jobs.values():
                    job.cancel()
                rnd["log_window"] = window
                self._save_state()
                return {"status": "interrupted", "round": rnd["id"], "done": rnd["done"], "steps": rnd["steps"]}
        return self._finish_round(data, human_val)

    def _finish_round(self, data: SelfPlayData, human_val: dict) -> dict:
        """Garde-fous. Passés : le tour devient le CANDIDAT en attente de l'arène (candidate.pt), ses shards sont
        marqués comme utilisés et il devient le point de retour (line_ok.pt) : la ligne d'entraînement continue
        quel que soit le verdict de l'arène. Non passés : retour à line_ok.pt et réactions (comme en 6d)."""
        cfg, st, rnd = self.cfg, self.state, self.state["round"]
        lines = [json.loads(l) for l in (self.out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
        last = [l for l in lines if l["round"] == rnd["id"]][-1:]
        metrics = {
            "lichess": eval_human(self.model, human_val, self.device),
            "heldout": eval_selfplay(self.model, data, data.heldout_rows, self.device, rnd["value_target"]),
            "train": {"sp_value": last[0]["sp_value"] if last else None,
                      "sp_policy": last[0]["sp_policy"] if last else None},
        }
        # `previous` = métriques du dernier état de la ligne qui a passé les garde-fous (chute brutale)
        decision = decide(metrics, st["baseline"], st["previous"], st["history"], st["generation"], cfg)
        row = {"type": "round", "round": rnd["id"], "generation_before": st["generation"], "steps": rnd["steps"],
               "new_positions": rnd["new_positions"], "human_fraction": rnd["human_fraction"],
               "value_target": rnd["value_target"], "lr_scale": st["lr_scale"], "reuse": st["reuse"],
               "metrics": metrics, "baseline": st["baseline"], **decision, "time": time.time()}
        st["history"].append(metrics)
        passed = decision["accept"]
        if passed:
            cid = rnd["id"]  # numéro du candidat = numéro du tour (graine des ouvertures de l'arène)
            path = self.out / "candidate.pt"
            _atomic_torch_save(self._model_payload(metrics, candidate=cid, round=cid), path)
            st["candidate"] = {"id": cid, "round": cid, "file": "candidate.pt", "sha256": _sha256(path),
                               "metrics": metrics, "global_step": st["global_step"],
                               "opponent": dict(st["accepted"])}
            st["trained_shards"] = sorted(set(st["trained_shards"]) | set(rnd["shards"]))
            st["previous"] = metrics
            st["consecutive_rejections"] = 0
            st["line_ok_round"] = cid
            row["candidate"] = cid
        else:
            self._restore_line_ok()
            if "halve_lr" in decision["actions"]:
                st["lr_scale"] /= 2
            if "human_25" in decision["actions"]:
                st["human_sticky_25"] = True
            if "reduce_reuse" in decision["actions"]:
                st["reuse"] = min(st["reuse"], cfg.reduced_reuse)
            st["consecutive_rejections"] += 1
            if st["consecutive_rejections"] >= cfg.max_rejections:
                st["status"], st["stop_reason"] = "stop", "G4"
                decision["alerts"].append(f"G4 : {st['consecutive_rejections']} rejets d'affilée par les garde-fous : "
                                          "arrêt, décision humaine nécessaire")
                row["alerts"] = decision["alerts"]
        row["generation_after"] = st["generation"]
        row["loop_status"] = st["status"]
        st["round"] = None
        st["round_id"] += 1
        self._save_state()  # point de validation
        if passed:
            self._save_line_ok()  # régénéré à la reprise si une coupure tombe avant
        self._append_once(row, ("round", rnd["id"]))
        return {"status": "candidate" if passed else "rejected", **row}

    # ---------------------------------------------------------------- verdict de l'arène

    def commit_arena(self, candidate_id: int, gate: dict, anchor: dict | None, arena_cfg) -> dict:
        """Applique le verdict de l'arène au candidat `candidate_id` (rejouable sans effet).

        `gate` : résumé (`arena.summarize`) du candidat contre le réseau accepté ; `anchor` : contre l'ancre
        `kaggle_sup`. Tant que le réseau accepté est le réseau de départ (génération 0, qui EST l'ancre), l'arène
        contre l'accepté est aussi la mesure contre l'ancre (`anchor` = None) : cette mesure est gonflée par la
        sélection (elle a servi à accepter ; ~10 à 20 Elo), à garder en tête si la génération 2 subit un veto.
        Ensuite `anchor` est obligatoire quand le candidat gagne l'arène (veto de l'ancre, `arena.gate_decision`).
        Accepté : generations/gen_XXXX.pt, génération + 1, rejets en arène à 0 ; rejeté : la ligne d'entraînement
        continue (rien n'est défait), rejets en arène + 1. Arrêts : `max_arena_rejections` rejets en arène
        d'affilée ; réseau accepté sous l'ancre (borne haute < 0) sur `anchor_stop_generations` générations
        acceptées consécutives, en ne comptant que les générations >= `anchor_stop_from_generation`."""
        from chesszero.arena import gate_decision

        cfg, st = self.cfg, self.state
        done = [c for c in st["commits"] if c["candidate"] == candidate_id]
        if done:
            self._append_once({"type": "arena", **done[0]}, ("arena", candidate_id))
            return done[0]
        cand = st["candidate"]
        if cand is None or cand["id"] != candidate_id:
            raise ValueError(f"candidat {candidate_id} pas en attente (en attente : {cand and cand['id']})")
        path = self.out / cand["file"]
        if not path.exists() or _sha256(path) != cand["sha256"]:
            raise RuntimeError(f"{path} ne correspond pas au candidat {candidate_id} enregistré (SHA-256)")
        first = st["accepted"]["generation"] == 0  # l'accepté est le réseau de départ = l'ancre
        if first:
            decision = gate_decision(gate, arena_cfg)
            cand_anchor, source = gate, "arène contre l'accepté (= l'ancre), biaisée par la sélection (~10 à 20 Elo)"
        else:
            decision = gate_decision(gate, arena_cfg)
            if decision["accept"]:
                if anchor is None:
                    raise ValueError("arène ancre manquante : obligatoire quand le candidat gagne l'arène")
                decision = gate_decision(gate, arena_cfg, anchor, st["accepted"]["anchor"])
            cand_anchor, source = anchor, "arène ancre (ouvertures fixes)"
        commit = {"candidate": candidate_id, "accepted": decision["accept"], "reasons": decision["reasons"],
                  "anchor_diff": decision["anchor_diff"], "anchor_diff_se": decision["anchor_diff_se"],
                  "gate": {k: gate[k] for k in ("games", "score", "points", "elo", "elo_low", "elo_high", "los")},
                  "anchor": ({k: cand_anchor[k] for k in ("games", "score", "elo", "elo_se", "elo_low", "elo_high")}
                             if cand_anchor is not None else None),
                  "opponent": cand["opponent"], "alerts": [], "time": time.time()}
        if decision["accept"]:
            g = st["generation"] + 1
            gen_dir = self.out / "generations"
            gen_dir.mkdir(exist_ok=True)
            gen_path = gen_dir / f"gen_{g:04d}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload.update(generation=g, arena={"gate": gate, "anchor": cand_anchor, "anchor_source": source,
                                                "decision": decision})
            _atomic_torch_save(payload, gen_path)  # un fichier orphelin (coupure avant la validation) est écrasé
            st["generation"] = g
            st["accepted"] = {"generation": g, "file": str(gen_path.relative_to(self.out)).replace("\\", "/"),
                              "sha256": _sha256(gen_path), "anchor": cand_anchor, "anchor_source": source,
                              "candidate": candidate_id}
            st["anchor_history"].append({"generation": g, "candidate": candidate_id, "elo": cand_anchor["elo"],
                                         "elo_se": cand_anchor["elo_se"], "elo_low": cand_anchor["elo_low"],
                                         "elo_high": cand_anchor["elo_high"], "source": source})
            st["arena_rejections"] = 0
            below = [h for h in st["anchor_history"] if h["generation"] >= cfg.anchor_stop_from_generation
                     and h["generation"] > st["anchor_stop_ack"]][-cfg.anchor_stop_generations:]
            if (len(below) == cfg.anchor_stop_generations
                    and [h["generation"] for h in below] == list(range(g - cfg.anchor_stop_generations + 1, g + 1))
                    and all(h["elo_high"] is not None and h["elo_high"] < 0 for h in below)):
                st["status"], st["stop_reason"] = "stop", "ancre"
                commit["alerts"].append(f"arrêt : réseau accepté sous l'ancre kaggle_sup (borne haute < 0) sur "
                                        f"{cfg.anchor_stop_generations} générations consécutives ({below[0]['generation']} "
                                        f"à {g}) : décision humaine nécessaire")
        else:
            st["arena_rejections"] += 1
            if st["arena_rejections"] >= cfg.max_arena_rejections:
                st["status"], st["stop_reason"] = "stop", "stagnation"
                commit["alerts"].append(f"arrêt : {st['arena_rejections']} rejets en arène d'affilée (stagnation) : "
                                        "décision humaine nécessaire")
        commit.update(generation_after=st["generation"], arena_rejections=st["arena_rejections"],
                      loop_status=st["status"])
        st["candidate"] = None
        st["commits"].append(commit)
        self._save_state()  # point de validation
        self._append_once({"type": "arena", **commit}, ("arena", candidate_id))
        return commit

    def clear_stop(self, note: str = "") -> dict:
        """Reprise après un arrêt, sur décision humaine : remet à zéro le compteur qui a déclenché l'arrêt."""
        st = self.state
        if st["status"] != "stop":
            raise ValueError("aucun arrêt en cours")
        reason = st.get("stop_reason") or "G4"
        if reason == "G4":
            st["consecutive_rejections"] = 0
        elif reason == "stagnation":
            st["arena_rejections"] = 0
        elif reason == "ancre":
            st["anchor_stop_ack"] = st["generation"]  # il faudra de nouvelles générations sous l'ancre
        st["status"], st["stop_reason"] = "ok", None
        self._save_state()
        row = {"type": "clear_stop", "round": f"{reason}@{st['round_id']}", "reason": reason, "note": note,
               "time": time.time()}
        self._append_once(row, ("clear_stop", row["round"]))
        return row


def read_train_state(out_dir: str | Path) -> dict:
    """Lecture de l'état d'entraînement (SOURCE DE VÉRITÉ pour l'orchestrateur, étape 7d : ne pas recopier ces
    informations dans loop_state.json) : réseau accepté (chemin absolu, None = réseau de départ), candidat en
    attente, compteurs, historique contre l'ancre, arrêt. Ne charge aucun modèle sur un device."""
    out = Path(out_dir)
    path = out / "train_state.pt"
    if not path.exists():
        return {"exists": False}
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st.get("format", 1) < STATE_FORMAT:
        return {"exists": True, "format": st.get("format", 1), "needs_migration": True,
                "generation": st["generation"]}
    acc, cand, rnd = st["accepted"], st["candidate"], st["round"]
    return {
        "exists": True, "format": st["format"], "status": st["status"], "stop_reason": st["stop_reason"],
        "generation": st["generation"],
        "accepted": {**{k: v for k, v in acc.items() if k != "file"},
                     "path": str(out / acc["file"]) if acc["file"] else None,
                     "is_init": acc["file"] is None, "init_checkpoint": st["init_checkpoint"]},
        "candidate": ({"id": cand["id"], "path": str(out / cand["file"]), "sha256": cand["sha256"],
                       "metrics": cand["metrics"], "opponent_sha256": cand["opponent"]["sha256"]} if cand else None),
        "round_in_progress": ({"id": rnd["id"], "done": rnd["done"], "steps": rnd["steps"]} if rnd else None),
        "arena_rejections": st["arena_rejections"], "guard_rejections": st["consecutive_rejections"],
        "anchor_history": st["anchor_history"], "commits": len(st["commits"]),
        "round_id": st["round_id"], "global_step": st["global_step"], "lr_scale": st["lr_scale"],
        "reuse": st["reuse"], "human_sticky_25": st["human_sticky_25"],
    }
