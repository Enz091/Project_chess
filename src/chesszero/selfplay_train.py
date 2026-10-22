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
  5. évaluation et décision (`decide`), puis : accepté -> `generations/gen_XXXX.pt` (+ état « accepté ») ;
     rejeté -> retour aux poids et à l'optimiseur du dernier réseau accepté, réaction automatique, et le tour
     sera refait sur les mêmes données.

Reprise : les lots d'un tour ne dépendent que de (graine, n° de tour, pas) et de la liste des shards fixée au début
du tour (enregistrée) ; `train_state.pt` est écrit tous les `save_every` pas : une reprise après coupure donne
exactement les mêmes poids qu'un tour d'une traite (testé).
"""

from __future__ import annotations

import json
import math
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
    max_rejections: int = 3  # G4 : arrêt après autant de rejets d'affilée
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
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


class SelfPlayTrainer:
    """État de l'entraînement sur le self-play dans `out_dir` :
      train_state.pt  modèle et optimiseur courants, pas, tour en cours (shards, pas faits), compteurs ;
      accepted.pt     modèle et optimiseur du dernier réseau accepté (point de retour d'un rejet) ;
      generations/gen_XXXX.pt  réseaux acceptés (format de checkpoint habituel : load_for_inference) ;
      rounds.jsonl    une ligne par tour (métriques, décision) ; train_log.jsonl : pertes pendant les tours.
    """

    def __init__(self, out_dir: str | Path, cfg: SelfPlayTrainConfig, device: torch.device | str,
                 init_checkpoint: str | Path | None = None, log=print):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cfg, self.device, self.log = cfg, torch.device(device), log
        self.use_amp = cfg.amp == "on" or (cfg.amp == "auto" and self.device.type == "cuda")
        state_path = self.out / "train_state.pt"
        if state_path.exists():
            self.state = torch.load(state_path, map_location="cpu", weights_only=False)
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
        if not (self.out / "accepted.pt").exists():
            self._save_accepted()

    # ---------------------------------------------------------------- état

    def human_fraction(self) -> float:
        if self.state["human_sticky_25"] or self.state["generation"] < self.cfg.human_switch_generation:
            return self.cfg.human_fraction
        return self.cfg.human_fraction_late

    def _save_state(self) -> None:
        self.state["model"] = self.model.state_dict()
        self.state["optimizer"] = self.optimizer.state_dict()
        self.state["scaler"] = self.scaler.state_dict() if self.use_amp else None
        _atomic_torch_save(self.state, self.out / "train_state.pt")

    def _save_accepted(self) -> None:
        _atomic_torch_save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                            "scaler": self.scaler.state_dict() if self.use_amp else None,
                            "global_step": self.state["global_step"]}, self.out / "accepted.pt")

    def _restore_accepted(self) -> None:
        acc = torch.load(self.out / "accepted.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(acc["model"])
        self.optimizer.load_state_dict(acc["optimizer"])
        if self.use_amp and acc.get("scaler"):
            self.scaler.load_state_dict(acc["scaler"])

    def _append(self, name: str, row: dict) -> None:
        with open(self.out / name, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- un tour

    def run_round(self, buffer_dirs, human_train: dict, human_val: dict, pool: ProcessPoolExecutor | None = None,
                  steps_override: int | None = None, stop_after: int | None = None, force: bool = False) -> dict:
        """Un tour complet (ou sa reprise). `stop_after` : s'arrête après ce nombre de pas de CE lancement (coupure
        simulée). `steps_override` / `force` : pour les essais (nombre de pas imposé, tampon minimal ignoré)."""
        cfg, st = self.cfg, self.state
        if st["status"] == "stop":
            raise RuntimeError("arrêt demandé par le garde-fou G4 : décision humaine nécessaire (train_state.pt)")
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
        cfg, st, rnd = self.cfg, self.state, self.state["round"]
        lines = [json.loads(l) for l in (self.out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
        last = [l for l in lines if l["round"] == rnd["id"]][-1:]
        metrics = {
            "lichess": eval_human(self.model, human_val, self.device),
            "heldout": eval_selfplay(self.model, data, data.heldout_rows, self.device, rnd["value_target"]),
            "train": {"sp_value": last[0]["sp_value"] if last else None,
                      "sp_policy": last[0]["sp_policy"] if last else None},
        }
        decision = decide(metrics, st["baseline"], st["previous"], st["history"], st["generation"], cfg)
        row = {"round": rnd["id"], "generation_before": st["generation"], "steps": rnd["steps"],
               "new_positions": rnd["new_positions"], "human_fraction": rnd["human_fraction"],
               "value_target": rnd["value_target"], "lr_scale": st["lr_scale"], "reuse": st["reuse"],
               "metrics": metrics, "baseline": st["baseline"], **decision, "time": time.time()}
        st["history"].append(metrics)
        if decision["accept"]:
            st["generation"] += 1
            st["trained_shards"] = sorted(set(st["trained_shards"]) | set(rnd["shards"]))
            st["previous"] = metrics
            st["consecutive_rejections"] = 0
            gen_dir = self.out / "generations"
            gen_dir.mkdir(exist_ok=True)
            path = gen_dir / f"gen_{st['generation']:04d}.pt"
            _atomic_torch_save({"step": st["global_step"], "epoch": 0, "model": self.model.state_dict(),
                                "config": {"model": st["model_config"], "selfplay_train": asdict(cfg)},
                                "metrics": metrics, "generation": st["generation"]}, path)
            self._save_accepted()
            row["checkpoint"] = str(path.relative_to(self.out))
        else:
            self._restore_accepted()
            if "halve_lr" in decision["actions"]:
                st["lr_scale"] /= 2
            if "human_25" in decision["actions"]:
                st["human_sticky_25"] = True
            if "reduce_reuse" in decision["actions"]:
                st["reuse"] = min(st["reuse"], cfg.reduced_reuse)
            st["consecutive_rejections"] += 1
            if st["consecutive_rejections"] >= cfg.max_rejections:
                st["status"] = "stop"
                decision["alerts"].append(f"G4 : {st['consecutive_rejections']} rejets d'affilée : arrêt, "
                                          "décision humaine nécessaire")
                row["alerts"] = decision["alerts"]
        row["generation_after"] = st["generation"]
        row["loop_status"] = st["status"]
        st["round"] = None
        st["round_id"] += 1
        self._append("rounds.jsonl", row)
        self._save_state()
        return {"status": "accepted" if decision["accept"] else "rejected", **row}
