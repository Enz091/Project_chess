"""Pré-entraînement supervisé : le réseau apprend à imiter les coups et les résultats de parties Lichess.

Ce que fait `train()` :
  - charge les shards (`data.load_split`), tout en RAM ;
  - à chaque époque, mélange les positions (permutation fixée par `(seed, époque)`), donc le lot
    de l'étape `s` se déduit de `s` seul : la reprise n'a pas d'« état de lecture » à restaurer ;
  - perte = entropie croisée sur le coup joué (coups illégaux masqués, sauf `mask_train_loss = false`)
    + MSE sur le résultat ;
  - AdamW, montée linéaire du taux d'apprentissage puis décroissance en cosinus ;
  - précision mixte fp16 sur CUDA seulement ;
  - écrit `metrics.csv`, sauvegarde `last.pt` / `best.pt` et reprend tout seul après une coupure.

Les masques de coups légaux (python-chess, ~7 000 positions/s) sont calculés à l'avance par un petit
pool de processus ; les lots produits sont identiques avec ou sans pool.

Diagnostic de débit (colonnes `gpu_util`, `data_wait_pct`, `mask_wait_pct` de metrics.csv) : le temps
d'un pas est coupé en « attente des données » (obtenir le lot suivant, dont l'attente des masques) et
« calcul » (passe avant/arrière + métriques, synchrone). Si l'attente est grande, le GPU est affamé.
"""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
import threading
import time
import tomllib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple

import numpy as np
import torch
from torch import nn

from chesszero.checkpoint import find_resume_checkpoint, load_checkpoint, save_checkpoint
from chesszero.data import load_split
from chesszero.device import describe_device
from chesszero.encoding import legal_masks_from_packed, unpack_batch
from chesszero.model import _MASK_VALUE, ChessNet, ModelConfig, build_model, policy_value_loss

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
CSV_COLUMNS = ["step", "epoch", "split", "loss", "policy_loss", "value_loss", "top1", "top3", "lr", "pos_per_s", "elapsed_s",
               "gpu_util", "data_wait_pct", "mask_wait_pct"]


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class TrainConfig:
    """Section [train] de configs/<nom>.toml."""

    batch_size: int = 256
    lr: float = 1e-3  # taux d'apprentissage de pointe (après la montée)
    weight_decay: float = 0.01  # AdamW, appliquée aux poids des convolutions / couches linéaires seulement
    warmup_steps: int = 300  # montée linéaire de ~0 à `lr`
    min_lr_ratio: float = 0.01  # le cosinus descend jusqu'à lr * min_lr_ratio
    total_steps: int = 10_000  # nombre total de pas d'optimisation
    grad_clip: float = 1.0  # norme maximale du gradient (0 = pas de coupure)
    log_every: int = 50  # une ligne « train » (moyenne sur la fenêtre) tous les N pas
    eval_every: int = 500  # validation + sauvegarde (best.pt) tous les N pas
    save_every: int = 250  # sauvegarde de reprise (last.pt) tous les N pas
    keep_last: int = 3  # nombre de ckpt_XXXXXXXX.pt gardés
    val_positions: int = 20_000  # taille du sous-ensemble fixe de validation (0 = tout)
    amp: str = "auto"  # "auto" = fp16 sur CUDA seulement, "off" = jamais
    num_workers: int = 2  # processus qui préparent les masques de coups légaux (0 = dans le processus)
    mask_train_loss: bool = True  # false : perte politique sans masque (plus de masques à calculer à l'entraînement)
    seed: int = 0
    # Garde-fou de débit (B1, quota Kaggle) : au pas `throughput_check_step`, si le débit mesuré depuis le pas
    # `throughput_skip_steps` (attente des données + calcul ; validations et sauvegardes exclues) est sous
    # `min_pos_per_s`, sauvegarde de last.pt et arrêt (`stopped_low_throughput`). 0 = pas de garde-fou.
    min_pos_per_s: float = 0.0
    throughput_check_step: int = 2000
    throughput_skip_steps: int = 200

    def __post_init__(self):
        if self.amp not in ("auto", "off"):
            raise ValueError(f"amp doit valoir « auto » ou « off », pas {self.amp!r}")
        for name in ("save_every", "eval_every", "throughput_check_step"):
            if getattr(self, name) % self.log_every:
                # sinon une reprise recommencerait une fenêtre de moyenne en cours de route
                raise ValueError(f"{name} doit être un multiple de log_every ({self.log_every})")

    @classmethod
    def from_toml(cls, path: str | Path) -> TrainConfig:
        with open(path, "rb") as f:
            section = tomllib.load(f).get("train", {})
        unknown = set(section) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"clés inconnues dans [train] : {sorted(unknown)}")
        return cls(**section)

    @classmethod
    def named(cls, name: str) -> TrainConfig:
        return cls.from_toml(CONFIG_DIR / f"{name}.toml")


def lr_factor(step: int, cfg: TrainConfig) -> float:
    """Facteur multiplicatif du taux d'apprentissage au pas `step` (0 = 1er pas)."""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(progress, 1.0)
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    """AdamW avec décroissance de poids sur les tenseurs ≥ 2D seulement (pas BatchNorm ni biais)."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, cfg))
    return optimizer, scheduler


# --------------------------------------------------------------------------- lots de données


class Batch(NamedTuple):
    x: np.ndarray  # (B, 18, 8, 8) float32
    legal: np.ndarray | None  # (B, 4672) bool ; None si l'entraînement se fait sans masque
    moves: np.ndarray  # (B,) int64 : indice du coup joué
    results: np.ndarray  # (B,) float32 : résultat du point de vue du camp au trait
    value_valid: np.ndarray  # (B,) float32 : 1 si le résultat sert de cible de valeur (Normal), 0 sinon (Time forfeit)


def train_indices(n: int, batch_size: int, seed: int, start_step: int, end_step: int) -> Iterator[np.ndarray]:
    """Indices des lots des pas [start_step, end_step) : une permutation par époque, dernier lot incomplet écarté."""
    per_epoch = n // batch_size
    if per_epoch == 0:
        raise ValueError(f"{n} positions < taille de lot {batch_size}")
    order, order_epoch = None, -1
    for step in range(start_step, end_step):
        epoch, i = divmod(step, per_epoch)
        if epoch != order_epoch:
            order, order_epoch = np.random.default_rng([seed, epoch]).permutation(n), epoch
        yield order[i * batch_size : (i + 1) * batch_size]


class PhaseTimer:
    """Temps cumulés sur une fenêtre : attente du lot suivant (dont les masques) et calcul."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.data_s = self.mask_s = self.compute_s = 0.0

    @property
    def busy_s(self) -> float:
        return self.data_s + self.compute_s

    def pct(self, seconds: float) -> float:
        return 100 * seconds / self.busy_s if self.busy_s else float("nan")


def stream_batches(
    data: dict,
    index_batches: Iterable[np.ndarray],
    pool: ProcessPoolExecutor | None,
    depth: int = 6,
    *,
    with_masks: bool = True,
    timer: PhaseTimer | None = None,
) -> Iterator[Batch]:
    """Construit les lots dans l'ordre ; avec un pool, les masques sont calculés `depth` lots à l'avance.

    `with_masks=False` : aucun masque (`Batch.legal` vaut None). `timer.mask_s` cumule le temps passé
    à attendre (ou calculer, sans pool) les masques.
    """
    source = iter(index_batches)
    pending: deque = deque()
    use_pool = pool is not None and with_masks

    def submit() -> None:
        idx = next(source, None)
        if idx is not None:
            boards = data["boards"][idx]
            pending.append((idx, boards, pool.submit(legal_masks_from_packed, boards) if use_pool else None))

    for _ in range(depth if use_pool else 1):
        submit()
    while pending:
        idx, boards, future = pending.popleft()
        legal = None
        if with_masks:
            t = time.perf_counter()
            legal = future.result() if future else legal_masks_from_packed(boards)
            if timer is not None:
                timer.mask_s += time.perf_counter() - t
        submit()
        yield Batch(unpack_batch(boards), legal, data["moves"][idx].astype(np.int64),
                    data["results"][idx].astype(np.float32), data["value_valid"][idx].astype(np.float32))


def timed_batches(batches: Iterable[Batch], timer: PhaseTimer) -> Iterator[Batch]:
    """Cumule dans `timer.data_s` le temps passé à obtenir chaque lot."""
    it = iter(batches)
    while True:
        t = time.perf_counter()
        batch = next(it, None)
        timer.data_s += time.perf_counter() - t
        if batch is None:
            return
        yield batch


class GpuSampler:
    """Mesure le taux d'utilisation du GPU (%) dans un fil d'exécution, par pynvml ou `nvidia-smi`.

    `mean()` = moyenne des échantillons depuis le dernier `reset()` (nan s'il n'y en a pas : CPU, ou ni
    pynvml ni nvidia-smi). `read` (fonction sans argument qui renvoie un pourcentage) est injectable pour
    les tests. Le taux NVML est la part du temps où au moins un noyau CUDA tournait : 100 % ne prouve
    pas que le GPU est saturé, mais un taux bas prouve qu'il attend.
    """

    def __init__(self, device: torch.device, read=None, interval: float = 0.5):
        self._read = read if read is not None else self._default_reader(device)
        self._interval = interval
        self._values: list[float] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if self._read is not None:
            threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _default_reader(device: torch.device):
        if device.type != "cuda":
            return None
        index = device.index or 0
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            pynvml.nvmlDeviceGetUtilizationRates(handle)  # échoue ici plutôt qu'en plein entraînement
            return lambda: float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:  # noqa: BLE001 - pynvml absent ou NVML indisponible : repli sur nvidia-smi
            pass
        if shutil.which("nvidia-smi"):
            cmd = ["nvidia-smi", f"--id={index}", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
            return lambda: float(subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True).stdout.split()[0])
        return None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                value = self._read()
            except Exception:  # noqa: BLE001 - un échantillon raté n'arrête pas l'entraînement
                continue
            with self._lock:
                self._values.append(value)

    def reset(self) -> None:
        with self._lock:
            self._values = []

    def mean(self) -> float:
        with self._lock:
            return sum(self._values) / len(self._values) if self._values else float("nan")

    def close(self) -> None:
        self._stop.set()


class ValSet:
    """Positions de validation, avec leurs masques calculés une fois pour toutes."""

    def __init__(self, data: dict, size: int, pool: ProcessPoolExecutor | None, seed: int = 0):
        n = len(data["boards"])
        idx = np.arange(n) if size <= 0 or size >= n else np.sort(np.random.default_rng(seed).choice(n, size, replace=False))
        self.boards, self.moves, self.results = data["boards"][idx], data["moves"][idx], data["results"][idx]
        self.value_valid = data["value_valid"][idx]
        chunks = [idx[i : i + 512] for i in range(0, len(idx), 512)]
        self.legal = np.concatenate([b.legal for b in stream_batches(data, chunks, pool)])

    def __len__(self) -> int:
        return len(self.boards)

    def batches(self, batch_size: int) -> Iterator[Batch]:
        for i in range(0, len(self), batch_size):
            sl = slice(i, i + batch_size)
            yield Batch(unpack_batch(self.boards[sl]), self.legal[sl], self.moves[sl].astype(np.int64),
                        self.results[sl].astype(np.float32), self.value_valid[sl].astype(np.float32))


# --------------------------------------------------------------------------- évaluation


def topk_correct(logits: torch.Tensor, legal: torch.Tensor | None, moves: torch.Tensor) -> tuple[int, int]:
    """Nombre de positions où le coup joué est le 1er choix / dans les 3 premiers (parmi les coups légaux ;
    `legal=None` : parmi toutes les sorties, sans masque)."""
    logits = logits.float()
    top = (logits if legal is None else logits.masked_fill(~legal, _MASK_VALUE)).topk(3, dim=1).indices
    hit = top == moves[:, None]
    return int(hit[:, 0].sum()), int(hit.any(dim=1).sum())


class Meter:
    """Moyennes pondérées. La perte de valeur est moyennée sur les positions à value_valid = 1 seulement,
    la perte totale = politique (toutes positions) + valeur (positions valides)."""

    def __init__(self):
        self.n = 0  # toutes les positions
        self.n_value = 0  # positions à value_valid = 1
        self.sums = dict.fromkeys(("policy_loss", "value_loss", "top1", "top3"), 0.0)

    def add(self, n: int, n_value: int, policy: float, value: float, top1: int, top3: int) -> None:
        self.n += n
        self.n_value += n_value
        self.sums["policy_loss"] += policy * n
        self.sums["value_loss"] += value * n_value  # `value` est déjà une moyenne sur les n_value positions
        self.sums["top1"] += top1
        self.sums["top3"] += top3

    def means(self) -> dict[str, float]:
        out = {
            "policy_loss": self.sums["policy_loss"] / max(self.n, 1),
            "value_loss": self.sums["value_loss"] / max(self.n_value, 1),
            "top1": self.sums["top1"] / max(self.n, 1),
            "top3": self.sums["top3"] / max(self.n, 1),
        }
        return {"loss": out["policy_loss"] + out["value_loss"], **out}


def forward_loss(model: ChessNet, batch: Batch, device: torch.device, use_amp: bool):
    """Passe avant + perte. Retourne (perte tensor, logits, parties de la perte, masque ou None, coups)."""
    x = torch.from_numpy(batch.x).to(device)
    legal = None if batch.legal is None else torch.from_numpy(batch.legal).to(device)
    moves = torch.from_numpy(batch.moves).to(device)
    results = torch.from_numpy(batch.results).to(device)
    value_valid = torch.from_numpy(batch.value_valid).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        logits, value = model(x)
    loss, parts = policy_value_loss(logits, value, moves, results, legal, value_valid)  # perte en float32
    return loss, logits, parts, legal, moves


@torch.no_grad()
def evaluate(model: ChessNet, batches: Iterable[Batch], device: torch.device, use_amp: bool = False) -> dict[str, float]:
    was_training = model.training
    model.eval()
    meter = Meter()
    for batch in batches:
        loss, logits, parts, legal, moves = forward_loss(model, batch, device, use_amp)
        top1, top3 = topk_correct(logits, legal, moves)
        meter.add(len(batch.moves), int(batch.value_valid.sum()), parts["policy"], parts["value"], top1, top3)
    model.train(was_training)
    return meter.means()


# --------------------------------------------------------------------------- plusieurs GPU, garde-fou de débit


def wrap_data_parallel(model: nn.Module, device: torch.device, gpus: int, force: bool = False, log=print) -> nn.Module:
    """`nn.DataParallel` sur les GPU 0..gpus-1 si gpus > 1 (ou si `force`, même sans GPU : sur CPU l'enveloppe
    appelle simplement le réseau). Refus si moins de `gpus` GPU sont visibles."""
    if gpus > 1:
        visible = torch.cuda.device_count() if device.type == "cuda" else 0
        if visible < gpus:
            raise ValueError(f"{gpus} GPU demandés, {visible} visibles")
        log(f"DataParallel sur {gpus} GPU : chaque lot est coupé en {gpus}")
        return nn.DataParallel(model, device_ids=list(range(gpus)))
    if force:
        return nn.DataParallel(model)
    return model


class ThroughputGuard:
    """Débit (positions/s, attente des données + calcul) des pas `throughput_skip_steps`+1 .. `throughput_check_step`,
    jugé au pas de contrôle ; écrit `throughput_check.json`. Inactif si `min_pos_per_s` = 0, si l'entraînement a
    repris au-delà du pas de contrôle, ou s'il a repris si près du contrôle que moins de 100 pas seraient mesurés."""

    def __init__(self, cfg: TrainConfig, start_step: int, out_dir: Path, log=print):
        self.cfg, self.out_dir, self.log = cfg, Path(out_dir), log
        self.first = max(start_step, cfg.throughput_skip_steps)
        measurable = cfg.throughput_check_step - self.first
        self.active = cfg.min_pos_per_s > 0 and measurable > 0 and (start_step <= cfg.throughput_skip_steps
                                                                    or measurable >= 100)
        if cfg.min_pos_per_s > 0 and not self.active and start_step < cfg.throughput_check_step:
            log(f"Garde-fou de débit inactif : reprise au pas {start_step}, trop près du contrôle")
        self.n, self.seconds, self.rate = 0, 0.0, None

    def add(self, step: int, positions: int, seconds: float) -> None:
        if self.active and self.first < step <= self.cfg.throughput_check_step:
            self.n += positions
            self.seconds += seconds

    def failed(self, step: int) -> bool:
        if not self.active or step != self.cfg.throughput_check_step:
            return False
        import json

        self.rate = self.n / max(self.seconds, 1e-9)
        ok = self.rate >= self.cfg.min_pos_per_s
        (self.out_dir / "throughput_check.json").write_text(json.dumps(
            {"step": step, "measured_from_step": self.first, "positions": self.n, "seconds": round(self.seconds, 2),
             "pos_per_s": round(self.rate, 1), "min_pos_per_s": self.cfg.min_pos_per_s, "ok": ok}, indent=1) + "\n",
            encoding="utf-8")
        self.log(f"CONTRÔLE DU DÉBIT au pas {step} : {self.rate:,.0f} positions/s (seuil {self.cfg.min_pos_per_s:,.0f}) "
                 f"-> {'on continue' if ok else 'ARRÊT'}")
        return not ok


def bench_throughput(model_cfg: ModelConfig, cfg: TrainConfig, data_dir: str | Path, device: torch.device,
                     gpus_list=(1, 2), steps: int = 300, skip: int = 50, shards: int = 4, log=print) -> dict:
    """Essai de débit d'entraînement (réseau neuf, rien sauvegardé) sur les `shards` premiers shards : pour chaque
    nombre de GPU, `steps` pas complets (passe avant/arrière, optimiseur, masques par le pool), débit mesuré après
    `skip` pas. Avant de mesurer DataParallel, contrôle de cohérence : sorties du réseau seul et enveloppé sur le
    même lot, en mode évaluation (écart attendu ~1e-3 en fp16, seuil 0,05). Renvoie {gpus: résultat}."""
    from chesszero.data import shard_paths

    skip = min(skip, steps // 5)  # au moins 4/5 des pas mesurés
    files = shard_paths(data_dir, "train")[:shards]
    parts = [dict(np.load(f)) for f in files]
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("boards", "moves", "results", "value_valid")}
    use_amp = cfg.amp == "auto" and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    pool = ProcessPoolExecutor(cfg.num_workers) if cfg.num_workers > 0 else None
    out = {}
    try:
        for gpus in gpus_list:
            visible = torch.cuda.device_count() if device.type == "cuda" else 0
            if gpus > 1 and visible < gpus:
                out[gpus] = {"skipped": f"{visible} GPU visible(s)"}
                log(f"{gpus} GPU : non mesuré ({visible} GPU visible(s))")
                continue
            torch.manual_seed(cfg.seed)
            base = build_model(model_cfg).to(device)
            model = wrap_data_parallel(base, device, gpus, log=log)
            optimizer, _ = build_optimizer(base, cfg)
            scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
            res = {}
            if gpus > 1:  # cohérence : même lot, réseau seul contre réseau enveloppé (mode évaluation)
                x = torch.from_numpy(unpack_batch(data["boards"][:512])).to(device)
                base.eval()
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    l1, v1 = base(x)
                    l2, v2 = model(x)
                res["max_diff_logits"] = float((l1.float() - l2.float()).abs().max())
                res["max_diff_value"] = float((v1.float() - v2.float()).abs().max())
                res["consistent"] = res["max_diff_logits"] < 0.05 and res["max_diff_value"] < 0.05
            base.train()
            batches = stream_batches(data, train_indices(len(data["boards"]), cfg.batch_size, cfg.seed, 0, steps), pool)
            t0, n, losses = time.perf_counter(), 0, []
            for i, batch in enumerate(batches):
                if i == skip:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0, n = time.perf_counter(), 0
                loss, _, _, _, _ = forward_loss(model, batch, device, use_amp)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(base.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
                n += len(batch.moves)
            if device.type == "cuda":
                torch.cuda.synchronize()
            res.update({"pos_per_s": n / (time.perf_counter() - t0), "steps": steps, "skip": skip,
                        "loss_first": losses[0], "loss_last": float(np.mean(losses[-20:])),
                        "finite": bool(np.isfinite(losses).all())})
            out[gpus] = res
            log(f"{gpus} GPU : {res['pos_per_s']:,.0f} positions/s ; perte {res['loss_first']:.3f} -> {res['loss_last']:.3f}"
                + (f" ; cohérence DataParallel : écarts {res['max_diff_logits']:.1e} / {res['max_diff_value']:.1e} "
                   f"({'OK' if res['consistent'] else 'ÉCHEC'})" if gpus > 1 else ""))
            del model, base, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    return out


def choose_gpus(bench: dict, min_gain: float = 0.10) -> tuple[int, str]:
    """Nombre de GPU retenu d'après `bench_throughput` : 2 seulement si l'essai à 2 GPU est cohérent, à perte finie
    et au moins `min_gain` plus rapide qu'à 1 GPU ; sinon 1."""
    one, two = bench.get(1, {}), bench.get(2, {})
    if "pos_per_s" not in one:
        raise ValueError("pas de mesure à 1 GPU")
    if "pos_per_s" not in two:
        return 1, f"2 GPU non mesurés ({two.get('skipped', 'absent')}) : 1 GPU"
    if not (two.get("consistent") and two.get("finite")):
        return 1, "DataParallel incohérent ou perte non finie : 1 GPU"
    gain = two["pos_per_s"] / one["pos_per_s"] - 1
    if gain < min_gain:
        return 1, f"gain de 2 GPU {gain:+.0%} < {min_gain:.0%} : 1 GPU"
    return 2, f"gain de 2 GPU {gain:+.0%} : 2 GPU"


# --------------------------------------------------------------------------- journal CSV


class MetricsLog:
    """metrics.csv : une ligne par mesure. À la reprise, on retire les lignes postérieures au checkpoint."""

    def __init__(self, path: Path, resume_step: int | None):
        self.path = path
        rows = []
        if resume_step is not None and path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                # « val_full » est recalculée à chaque fin de run : on ne la garde pas
                rows = [r for r in csv.DictReader(f) if int(r["step"]) <= resume_step and r["split"] != "val_full"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def write(self, **row) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, CSV_COLUMNS).writerow(row)


def read_metrics(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: (v if k == "split" else float(v or "nan")) for k, v in r.items()} for r in csv.DictReader(f)]


# --------------------------------------------------------------------------- boucle d'entraînement


def train(
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    data_dir: str | Path,
    out_dir: str | Path,
    device: torch.device,
    *,
    stop_after: int | None = None,
    max_minutes: float | None = None,
    gpus: int = 1,
    force_data_parallel: bool = False,
    log=lambda *a: print(*a, flush=True),
) -> dict:
    """Entraîne (ou reprend) jusqu'à `cfg.total_steps`. Retourne un résumé.

    `stop_after` simule une coupure : on s'arrête à ce pas SANS sauvegarde ni évaluation finales.
    `max_minutes` : budget de temps d'entraînement (hors chargement des données). Une fois dépassé, on
    sauvegarde `last.pt` proprement et on s'arrête (`timed_out` vaut True) ; relancer reprend. Sert aux
    essais de débit et à finir une session Kaggle (12 h) avant la coupure.
    `gpus` > 1 : `nn.DataParallel` sur les GPU 0..gpus-1 (chaque lot est coupé entre eux) ; les checkpoints restent
    au format d'un seul réseau (`checkpoint._unwrap`), donc une reprise peut changer de nombre de GPU.
    `force_data_parallel` : enveloppe DataParallel même sans GPU (tests sur CPU, où elle appelle simplement le réseau).
    Garde-fou de débit : `TrainConfig.min_pos_per_s` (résultat `stopped_low_throughput`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = cfg.amp == "auto" and device.type == "cuda"
    log(f"Device : {describe_device(device)} | précision mixte : {'fp16' if use_amp else 'non'} | "
        f"perte politique {'masquée' if cfg.mask_train_loss else 'SANS masque'}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # formes fixes : cuDNN choisit les convolutions les plus rapides

    torch.manual_seed(cfg.seed)
    model = build_model(model_cfg).to(device)
    optimizer, scheduler = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    config = {"model": model_cfg.to_dict(), "train": asdict(cfg), "data": str(data_dir)}

    step, best_val, elapsed_before = 0, math.inf, 0.0
    resume = find_resume_checkpoint(out_dir)
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=False)["config"]  # avant de charger les poids
        if saved["model"] != model_cfg.to_dict():
            raise ValueError(f"{resume} a été créé avec un autre réseau : {saved['model']}")
        if saved["train"] != asdict(cfg):
            log("Attention : la config [train] a changé depuis la sauvegarde.")
        meta = load_checkpoint(resume, model, optimizer, scheduler, scaler=scaler)
        step = meta["step"]
        best_val = meta["metrics"].get("best_val_loss", math.inf)
        elapsed_before = meta["metrics"].get("elapsed_s", 0.0)
        log(f"Reprise depuis {resume} (pas {step}/{cfg.total_steps})")
    model = wrap_data_parallel(model, device, gpus, force_data_parallel, log)
    start_step = step
    metrics_log = MetricsLog(out_dir / "metrics.csv", resume_step=step if resume else None)

    t0 = time.time()
    train_data = load_split(data_dir, "train")
    val_data = load_split(data_dir, "val")
    if "value_valid" not in train_data:
        raise ValueError(f"{data_dir} n'a pas de champ value_valid : relancer scripts/prepare_data.py")
    n_train = len(train_data["boards"])
    steps_per_epoch = n_train // cfg.batch_size
    log(f"{n_train:,} positions d'entraînement ({steps_per_epoch:,} pas/époque), {len(val_data['boards']):,} de validation | "
        f"{cfg.total_steps:,} pas = {cfg.total_steps / steps_per_epoch:.2f} époques")

    pool = ProcessPoolExecutor(cfg.num_workers) if cfg.num_workers > 0 else None
    sampler = GpuSampler(device)
    try:
        val_set = ValSet(val_data, cfg.val_positions, pool)  # la validation est toujours masquée
        log(f"Validation sur {len(val_set):,} positions (sous-ensemble fixe)")

        window, timer = Meter(), PhaseTimer()
        elapsed = lambda: elapsed_before + time.time() - t0  # noqa: E731
        nan_to_blank = lambda v, nd=1: "" if v != v else round(v, nd)  # noqa: E731

        def flush_window() -> None:
            if not window.n:
                return
            pos_per_s = window.n / timer.busy_s  # hors validation et sauvegardes
            gpu, data_wait, mask_wait = sampler.mean(), timer.pct(timer.data_s), timer.pct(timer.mask_s)
            metrics_log.write(step=step, epoch=round(step / steps_per_epoch, 4), split="train", **window.means(),
                              lr=scheduler.get_last_lr()[0], pos_per_s=round(pos_per_s, 1), elapsed_s=round(elapsed(), 1),
                              gpu_util=nan_to_blank(gpu), data_wait_pct=nan_to_blank(data_wait), mask_wait_pct=nan_to_blank(mask_wait))
            m = window.means()
            gpu_txt = "GPU n/d" if gpu != gpu else f"GPU {gpu:.0f}%"
            log(f"pas {step:6d}/{cfg.total_steps}  perte {m['loss']:.3f} (pol {m['policy_loss']:.3f} val {m['value_loss']:.3f})  "
                f"top1 {m['top1']:.1%} top3 {m['top3']:.1%}  lr {scheduler.get_last_lr()[0]:.2e}  "
                f"{pos_per_s:,.0f} pos/s | {gpu_txt} | attente données {data_wait:.0f}% (dont masques {mask_wait:.0f}%)")

        def save(is_best: bool, val: dict | None) -> None:
            save_checkpoint(out_dir, model, optimizer, scheduler, scaler=scaler, step=step,
                            epoch=step // steps_per_epoch, config=config, is_best=is_best, keep_last=cfg.keep_last,
                            metrics={"best_val_loss": best_val, "elapsed_s": elapsed(), **({"val": val} if val else {})})

        index_batches = train_indices(n_train, cfg.batch_size, cfg.seed, step, cfg.total_steps)
        batches = timed_batches(
            stream_batches(train_data, index_batches, pool, with_masks=cfg.mask_train_loss, timer=timer), timer)
        model.train()
        is_best = False
        deadline = None if max_minutes is None else time.time() + 60 * max_minutes
        guard = ThroughputGuard(cfg, start_step, out_dir, log)
        t_mark = time.perf_counter()
        for batch in batches:
            t_compute = time.perf_counter()
            loss, logits, parts, legal, moves = forward_loss(model, batch, device, use_amp)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            top1, top3 = topk_correct(logits, legal, moves)
            window.add(len(batch.moves), int(batch.value_valid.sum()), parts["policy"], parts["value"], top1, top3)
            timer.compute_s += time.perf_counter() - t_compute  # synchrone : les .item() attendent le GPU
            guard.add(step, len(batch.moves), time.perf_counter() - t_mark)
            if stop_after is not None and step >= stop_after:
                log(f"Arrêt simulé au pas {step}")
                return {"step": step, "interrupted": True}

            last = step == cfg.total_steps
            flushed = step % cfg.log_every == 0 or last
            if flushed:
                flush_window()
            do_eval = step % cfg.eval_every == 0 or last
            is_best = False
            if do_eval:
                val = evaluate(model, val_set.batches(1024), device, use_amp)
                metrics_log.write(step=step, epoch=round(step / steps_per_epoch, 4), split="val", **val,
                                  lr=scheduler.get_last_lr()[0], pos_per_s="", elapsed_s=round(elapsed(), 1))
                is_best = val["loss"] < best_val
                best_val = min(best_val, val["loss"])
                log(f"  VALIDATION pas {step}: perte {val['loss']:.3f} (pol {val['policy_loss']:.3f} val {val['value_loss']:.3f})  "
                    f"top1 {val['top1']:.1%} top3 {val['top3']:.1%}{'  <- meilleure' if is_best else ''}")
            saved_now = do_eval or step % cfg.save_every == 0 or last
            if saved_now:
                save(is_best, val if do_eval else None)
            if flushed:  # nouvelle fenêtre de mesure, validation et sauvegardes exclues
                window = Meter()
                timer.reset()
                sampler.reset()
            if guard.failed(step):
                if not saved_now:
                    flush_window()
                    save(False, None)
                log(f"DÉBIT INSUFFISANT : {guard.rate:,.0f} positions/s < {cfg.min_pos_per_s:,.0f} au pas {step} : "
                    f"arrêt (last.pt sauvegardé ; relancer reprendrait sans nouveau contrôle)")
                return {"step": step, "interrupted": True, "stopped_low_throughput": True, "pos_per_s": guard.rate,
                        "elapsed_s": elapsed()}
            t_mark = time.perf_counter()
            if deadline is not None and time.time() >= deadline and not last:
                if not saved_now:
                    flush_window()
                    save(False, None)
                log(f"Budget de {max_minutes:g} min atteint : arrêt propre au pas {step} (last.pt sauvegardé)")
                return {"step": step, "interrupted": True, "timed_out": True, "elapsed_s": elapsed()}

        # Évaluation finale sur TOUTE la validation (le sous-ensemble ne sert qu'au suivi).
        full = val_set if len(val_set) == len(val_data["boards"]) else ValSet(val_data, 0, pool)
        final = evaluate(model, full.batches(1024), device, use_amp)
        metrics_log.write(step=step, epoch=round(step / steps_per_epoch, 4), split="val_full", **final,
                          lr=scheduler.get_last_lr()[0], pos_per_s="", elapsed_s=round(elapsed(), 1))
        log(f"FINAL (toute la validation, {len(full):,} positions) : perte {final['loss']:.3f} "
            f"(pol {final['policy_loss']:.3f} val {final['value_loss']:.3f})  top1 {final['top1']:.1%} top3 {final['top3']:.1%}")
        return {"step": step, "interrupted": False, "best_val_loss": best_val, "final_val": final, "elapsed_s": elapsed()}
    finally:
        sampler.close()
        if pool:
            pool.shutdown(cancel_futures=True)
