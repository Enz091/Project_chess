"""Essai B2 : valeur entraînée sur des évaluations Stockfish (docs/options_apres_session3.md, journal du 2026-11-07).

Trois briques, utilisées par scripts/label_stockfish.py (étiquetage) et scripts/finetune_sf_value.py (affinage) :

1. Positions : tirées de la partie ENTRAÎNEMENT des données Lichess (jamais de la validation, sauf pour le jeu de
   contrôle étiqueté à part), en retirant toute position (pièces, trait, roques, en passant ; le compteur des
   50 coups est ignoré) qui apparaît le long d'une ligne d'ouverture du protocole v2 ou de l'arène ancre, position
   initiale comprise (`opening_keys`).
2. Valeur : v = tanh(cp / s), cp = évaluation de Stockfish (camp au trait) à nombre de NŒUDS fixe ; mat pour le camp
   au trait = +1, mat contre lui = −1. L'échelle s est ajustée sur nos données (`fit_scale` : moindres carrés de
   tanh(cp / s) contre le résultat des parties Lichess terminées normalement), pour que la valeur garde son sens
   actuel (score attendu d'une partie entre joueurs 2000+, celui avec lequel le MCTS, l'abandon et l'arène ont été
   réglés) : Stockfish ne sert qu'à débruiter cette cible, position par position.
3. Affinage (`finetune`) : depuis `kaggle_sup`, perte de politique sur le coup humain (comme au pré-entraînement) +
   erreur quadratique de valeur contre λ · v_Stockfish + (1 − λ) · résultat (λ = 1 par défaut : Stockfish seul ;
   avec λ < 1, les positions « Time forfeit » gardent la cible Stockfish seule). AdamW neuf, montée linéaire puis
   cosinus ; lots déterminés par (graine, pas) : relancer = reprendre au dernier point de sauvegarde.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np

from chesszero.encoding import PACKED_SIZE, pack_board

ROOT = Path(__file__).resolve().parents[2]
KEY_BYTES = PACKED_SIZE - 1  # sans le compteur des 50 coups (dernier octet)


# --------------------------------------------------------------------------- positions exclues


def position_keys(boards: np.ndarray) -> list[bytes]:
    """Clé de chaque position compacte (N, 67) : ses 66 premiers octets (le compteur des 50 coups est ignoré)."""
    b = np.ascontiguousarray(boards[:, :KEY_BYTES])
    return b.view(np.dtype((np.void, KEY_BYTES))).ravel().tolist()


def line_keys(lines: list[str]) -> set:
    """Clés de toutes les positions le long des lignes (suites de coups UCI depuis la position initiale), position
    initiale et position finale comprises."""
    keys = set()
    for line in lines:
        board = chess.Board()
        packed = [pack_board(board)]
        for uci in line.split():
            board.push_uci(uci)
            packed.append(pack_board(board))
        keys.update(position_keys(np.stack(packed)))
    return keys


def protected_lines(anchor_pairs: int = 100) -> dict[str, list[str]]:
    """Lignes des ouvertures à protéger : les 40 du protocole v2 et les `anchor_pairs` de l'arène ancre."""
    import sys

    from chesszero.arena import anchor_openings
    from chesszero.book import OpeningBook, arena_reserve_path

    sys.path.insert(0, str(ROOT / "scripts"))
    from ladder_openings import OPENINGS_V2  # noqa: E402

    reserve = OpeningBook.load(arena_reserve_path(ROOT / "books" / "lichess_2000.tsv.gz"))
    anchor = [reserve.lines[i] for i in anchor_openings(len(reserve), anchor_pairs)]
    return {"v2": [moves for _, moves in OPENINGS_V2], "anchor": anchor}


def opening_keys(anchor_pairs: int = 100) -> set:
    lines = protected_lines(anchor_pairs)
    return line_keys(lines["v2"] + lines["anchor"])


def sample_rows(boards: np.ndarray, n: int, seed: int, excluded: set) -> tuple[np.ndarray, int]:
    """`n` lignes tirées sans remise (graine fixe), triées, hors positions exclues. Renvoie (lignes, nombre de
    tirages écartés parce que la position était exclue)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(boards))
    keep, dropped, start = [], 0, 0
    while len(keep) < n and start < len(order):
        part = order[start:start + 2 * (n - len(keep)) + 1000]
        start += len(part)
        ok = np.array([k not in excluded for k in position_keys(boards[part])], bool)
        dropped += int((~ok).sum())
        keep.extend(part[ok][: n - len(keep)].tolist())
    if len(keep) < n:
        raise ValueError(f"{len(keep)} positions disponibles, {n} demandées")
    return np.sort(np.array(keep, dtype=np.int64)), dropped


# --------------------------------------------------------------------------- centipions -> valeur


def cp_to_value(cp: np.ndarray, mate: np.ndarray, scale: float) -> np.ndarray:
    """Valeur dans [−1, 1] du camp au trait : tanh(cp / scale) ; mat = ±1 (signe de `mate`, 0 = pas de mat)."""
    v = np.tanh(np.asarray(cp, np.float64) / scale)
    mate = np.asarray(mate)
    return np.where(mate != 0, np.sign(mate), v).astype(np.float32)


def fit_scale(cp: np.ndarray, mate: np.ndarray, results: np.ndarray, grid=None) -> dict:
    """Échelle s qui minimise l'erreur quadratique de tanh(cp / s) contre le résultat (positions sans mat) ;
    recherche sur une grille puis affinage local. Renvoie s, l'erreur, et celle de la formule de Lichess
    (s ≈ 543 : 2 / (1 + exp(−0,00368208 cp)) − 1 = tanh(0,00184104 cp)) pour comparaison."""
    m = np.asarray(mate) == 0
    x, y = np.asarray(cp, np.float64)[m], np.asarray(results, np.float64)[m]

    def mse(s):
        return float(((np.tanh(x / s) - y) ** 2).mean())

    grid = np.arange(100, 2001, 25) if grid is None else np.asarray(grid)
    best = float(grid[np.argmin([mse(s) for s in grid])])
    fine = np.arange(max(10.0, best - 25), best + 25.01, 1.0)
    best = float(fine[np.argmin([mse(s) for s in fine])])
    return {"scale": best, "mse": mse(best), "positions": int(m.sum()), "lichess_scale": 1 / 0.00184104,
            "mse_lichess_formula": mse(1 / 0.00184104), "mse_zero": float((y ** 2).mean())}


def value_target(labels: dict, scale: float, sf_weight: float) -> np.ndarray:
    """Cible de valeur : λ · v_Stockfish + (1 − λ) · résultat, le résultat seulement s'il est valide (Normal)."""
    v = cp_to_value(labels["cp"], labels["mate"], scale)
    if sf_weight >= 1.0:
        return v
    res = labels["results"].astype(np.float32)
    mixed = sf_weight * v + (1.0 - sf_weight) * res
    return np.where(labels["value_valid"] == 1, mixed, v).astype(np.float32)


# --------------------------------------------------------------------------- étiquettes sur disque


def load_labels(folder: str | Path) -> dict:
    """Concatène les morceaux `chunk_XXXX.npz` d'un dossier d'étiquettes (dans l'ordre)."""
    files = sorted(Path(folder).glob("chunk_*.npz"))
    if not files:
        raise FileNotFoundError(f"aucune étiquette dans {folder}")
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# --------------------------------------------------------------------------- affinage


@dataclass
class FinetuneConfig:
    steps: int = 1000
    batch_size: int = 1024
    lr: float = 1e-4  # pic (après la montée)
    lr_start: float = 1e-5
    lr_end: float = 1e-5  # fin du cosinus
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    sf_weight: float = 1.0  # λ : part de Stockfish dans la cible de valeur
    seed: int = 0
    eval_every: int = 250
    save_every: int = 50
    lichess_val: int = 50_000  # même sous-ensemble fixe que les tours de self-play (graine 0)


def lr_at(step: int, cfg: FinetuneConfig) -> float:
    """Taux du pas `step` (0 = premier pas) : montée linéaire lr_start -> lr, puis cosinus lr -> lr_end."""
    if step < cfg.warmup_steps:
        return cfg.lr_start + (cfg.lr - cfg.lr_start) * step / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr_end + 0.5 * (cfg.lr - cfg.lr_end) * (1 + math.cos(math.pi * min(t, 1.0)))


def _atomic_torch_save(obj, path: Path) -> None:
    import torch

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _optimizer(model, cfg: FinetuneConfig):
    import torch

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr_start)


def eval_value_sf(model, boards: np.ndarray, target: np.ndarray, results: np.ndarray, valid: np.ndarray,
                  device, batch_size: int = 1024) -> dict:
    """Erreur quadratique de la valeur contre la cible Stockfish et contre le résultat (positions valides)."""
    import torch

    from chesszero.encoding import unpack_batch

    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(boards), batch_size):
            out.append(model(torch.from_numpy(unpack_batch(boards[i:i + batch_size])).to(device))[1].float().cpu().numpy())
    model.train()
    v = np.concatenate(out).astype(np.float64)
    vv = valid == 1
    return {"mse_vs_stockfish": float(((v - target) ** 2).mean()),
            "mse_vs_result": float(((v[vv] - results[vv]) ** 2).mean()),
            "mean_abs_value": float(np.abs(v).mean()), "mean_abs_target": float(np.abs(target).mean()),
            "positions": int(len(v))}


def finetune(init: str | Path, labels: dict, val_labels: dict, human_val: dict, out: str | Path,
             cfg: FinetuneConfig, scale: float, device="cpu", workers: int = 4, log=print,
             stop_after: int | None = None) -> dict:
    """Affine `init` ; reprend `out/state.pt` s'il existe. Écrit `metrics.jsonl` (une ligne par évaluation) et, à la
    fin, `final.pt` (format load_for_inference). Renvoie le dernier bloc de métriques."""
    from concurrent.futures import ProcessPoolExecutor

    import torch
    from torch import nn

    from chesszero.model import ModelConfig, build_model
    from chesszero.selfplay_train import eval_human
    from chesszero.train import forward_loss, stream_batches, train_indices

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    payload = torch.load(init, map_location="cpu", weights_only=False)
    model_cfg = payload["config"]["model"]
    model = build_model(ModelConfig(**model_cfg)).to(device)
    model.load_state_dict(payload["model"])
    optimizer = _optimizer(model, cfg)
    step = 0
    state_path = out / "state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        if st["config"] != asdict(cfg) or st["scale"] != scale:
            raise ValueError(f"{state_path} a été créé avec d'autres réglages : {st['config']} (échelle {st['scale']})")
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        step = st["step"]
        log(f"reprise au pas {step}/{cfg.steps}")
    model.train()
    data = {"boards": labels["boards"], "moves": labels["moves"],
            "results": value_target(labels, scale, cfg.sf_weight),
            "value_valid": np.ones(len(labels["boards"]), np.uint8)}
    val_target = cp_to_value(val_labels["cp"], val_labels["mate"], scale).astype(np.float64)
    val_results = val_labels["results"].astype(np.float64)
    metrics_path = out / "metrics.jsonl"

    def evaluate() -> dict:
        t = time.perf_counter()
        m = {"step": step, "lr": lr_at(max(step - 1, 0), cfg), "lichess": eval_human(model, human_val, device),
             "value_sf_val": eval_value_sf(model, val_labels["boards"], val_target, val_results,
                                           val_labels["value_valid"], device)}
        model.train()
        m["eval_seconds"] = round(time.perf_counter() - t, 1)
        done = {json.loads(l)["step"] for l in metrics_path.read_text(encoding="utf-8").splitlines()} \
            if metrics_path.exists() else set()
        if step not in done:
            with metrics_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(m) + "\n")
        lic, vs = m["lichess"], m["value_sf_val"]
        log(f"  éval. pas {step}: Lichess politique {lic['policy_loss']:.4f} top-1 {lic['top1']:.2%} valeur "
            f"{lic['value_loss']:.4f} | contrôle Stockfish : MSE {vs['mse_vs_stockfish']:.4f} (résultat "
            f"{vs['mse_vs_result']:.4f}), |v| {vs['mean_abs_value']:.3f} ({m['eval_seconds']:.0f} s)")
        return m

    def save_state() -> None:
        _atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                            "config": asdict(cfg), "scale": scale}, state_path)

    last = evaluate() if step == 0 else None
    pool = ProcessPoolExecutor(workers) if workers > 0 else None
    t0, n0 = time.perf_counter(), step
    try:
        batches = stream_batches(data, train_indices(len(data["boards"]), cfg.batch_size, cfg.seed, step, cfg.steps), pool)
        window = []
        for batch in batches:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, cfg)
            loss, _, parts, _, _ = forward_loss(model, batch, device, False)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            step += 1
            window.append((parts["policy"], parts["value"]))
            if step % 25 == 0:
                rate = (step - n0) * cfg.batch_size / (time.perf_counter() - t0)
                p, v = np.mean(window, axis=0)
                log(f"pas {step}/{cfg.steps}  politique {p:.4f}  valeur {v:.4f}  lr {lr_at(step - 1, cfg):.2e}  "
                    f"{rate:,.0f} positions/s  (reste ~{(cfg.steps - step) * cfg.batch_size / rate / 60:.0f} min)")
                window = []
            if step % cfg.save_every == 0 or step == cfg.steps:
                save_state()
            if step % cfg.eval_every == 0 or step == cfg.steps:
                last = evaluate()
            if stop_after is not None and step >= stop_after:
                save_state()
                return {"interrupted": True, "step": step}
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    if last is None or last["step"] != step:
        last = evaluate()
    _atomic_torch_save({"step": step, "epoch": 0, "model": model.state_dict(),
                        "config": {"model": model_cfg, "finetune_sf_value": asdict(cfg), "scale": scale,
                                   "init": str(init)},
                        "metrics": {k: last[k] for k in ("lichess", "value_sf_val")}}, out / "final.pt")
    return last
