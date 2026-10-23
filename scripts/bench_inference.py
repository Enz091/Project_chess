r"""Vitesse d'inférence du réseau (positions évaluées par seconde), une par une et par lots.

C'est ce chiffre qui dicte le budget du MCTS : nombre de simulations par seconde ~ positions/s.

    .venv\Scripts\python scripts\bench_inference.py                 # tiny et kaggle
    .venv\Scripts\python scripts\bench_inference.py --configs tiny --threads 1
"""

from __future__ import annotations

import argparse
import time

import torch

from chesszero.device import describe_device, get_device
from chesszero.encoding import N_PLANES
from chesszero.model import build_model, count_parameters


def positions_per_second(model, x: torch.Tensor, min_time: float = 1.5) -> float:
    with torch.inference_mode():
        for _ in range(3):  # échauffement
            model(x)
        n_calls, start = 0, time.perf_counter()
        while time.perf_counter() - start < min_time:
            model(x)
            n_calls += 1
        elapsed = time.perf_counter() - start
    return n_calls * x.shape[0] / elapsed


def main() -> None:
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--configs", nargs="+", default=["tiny", "kaggle"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 64, 128, 256])
    parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = get_device(args.device)
    print(f"device : {describe_device(device)}")

    for name in args.configs:
        model = build_model(name).to(device).eval()
        print(f"\n{name} : {model.config}, {count_parameters(model):,} paramètres")
        print(f"{'lot':>6} {'positions/s':>13} {'ms/appel':>10}")
        for b in args.batches:
            x = torch.randn(b, N_PLANES, 8, 8, device=device)
            pps = positions_per_second(model, x)
            print(f"{b:>6} {pps:>13,.0f} {1000 * b / pps:>10.2f}")


if __name__ == "__main__":
    main()
