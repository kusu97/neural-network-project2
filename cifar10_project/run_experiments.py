"""Convenience launcher for the required CIFAR-10 ablation experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    "baseline_relu.json",
    "gelu_activation.json",
    "small_filters.json",
    "adamw_regularized.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run several CIFAR-10 experiments sequentially.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    for config_name in EXPERIMENTS:
        command = [
            sys.executable,
            "-m",
            "cifar10_project.train_cifar10",
            "--config",
            str(base / "configs" / config_name),
            "--data-dir",
            args.data_dir,
            "--num-workers",
            str(args.num_workers),
        ]
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

