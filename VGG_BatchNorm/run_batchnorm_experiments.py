"""Run VGG-A Batch Normalization experiments on CIFAR-10.

This script implements the requirements in Project 2 section 2:

1. Compare VGG-A with and without Batch Normalization.
2. Train both variants with several learning rates.
3. Save per-step losses and plot the loss-landscape bands.
4. Probe local linearization error and gradient variation around checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .loaders import CifarLoaderConfig, create_loaders
from .models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters


@dataclass
class ExperimentConfig:
    data_dir: str = "./data"
    output_dir: str = "./outputs/batchnorm_vgg"
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 20
    n_items: int = 0
    seed: int = 2020
    learning_rates: tuple[float, ...] = (1e-3, 2e-3, 1e-4, 5e-4)
    optimizer: str = "adam"
    weight_decay: float = 0.0
    amp: bool = True
    analysis_every: int = 5
    download: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_model(variant: str) -> nn.Module:
    if variant == "vgg_a":
        return VGG_A()
    if variant == "vgg_a_bn":
        return VGG_A_BatchNorm()
    raise ValueError(f"Unknown model variant: {variant}")


def build_optimizer(model: nn.Module, config: ExperimentConfig, lr: float) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size
    return total_loss / total, total_correct / total


def compute_grad_vector(model: nn.Module) -> torch.Tensor:
    grads = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            grads.append(parameter.grad.detach().flatten())
    if not grads:
        return torch.empty(0)
    return torch.cat(grads)


@contextmanager
def perturb_parameters(model: nn.Module, direction: list[torch.Tensor], alpha: float):
    with torch.no_grad():
        for parameter, delta in zip(model.parameters(), direction):
            parameter.add_(delta, alpha=alpha)
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, delta in zip(model.parameters(), direction):
                parameter.add_(delta, alpha=-alpha)


def local_smoothness_probe(
    model: nn.Module,
    criterion: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    alphas: Iterable[float] = (1e-3, 2e-3, 5e-3, 1e-2),
) -> list[dict[str, float]]:
    """Measure loss variation, linearization error, and gradient change."""

    was_training = model.training
    model.eval()
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    model.zero_grad(set_to_none=True)
    base_loss = criterion(model(inputs), targets)
    base_loss.backward()
    base_grad = compute_grad_vector(model)
    grad_norm = base_grad.norm().item()
    if grad_norm == 0.0:
        if was_training:
            model.train()
        return []

    direction = []
    for parameter in model.parameters():
        if parameter.grad is None:
            direction.append(torch.zeros_like(parameter))
        else:
            direction.append(parameter.grad.detach() / grad_norm)

    rows = []
    for alpha in alphas:
        with perturb_parameters(model, direction, alpha):
            model.zero_grad(set_to_none=True)
            shifted_loss = criterion(model(inputs), targets)
            shifted_loss.backward()
            shifted_grad = compute_grad_vector(model)

        actual_change = shifted_loss.item() - base_loss.item()
        predicted_change = alpha * grad_norm
        grad_diff = (shifted_grad - base_grad).norm().item()
        rows.append(
            {
                "alpha": float(alpha),
                "base_loss": float(base_loss.item()),
                "shifted_loss": float(shifted_loss.item()),
                "actual_change": float(actual_change),
                "predicted_change": float(predicted_change),
                "linearization_error": float(abs(actual_change - predicted_change)),
                "grad_diff": float(grad_diff),
                "grad_diff_per_alpha": float(grad_diff / alpha),
            }
        )

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return rows


def train_one_run(
    variant: str,
    lr: float,
    train_loader,
    test_loader,
    analysis_batch: tuple[torch.Tensor, torch.Tensor],
    config: ExperimentConfig,
    device: torch.device,
    run_dir: Path,
) -> dict:
    set_seed(config.seed)
    model = build_model(variant).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config, lr)
    scaler = GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    metrics_path = run_dir / f"metrics_{variant}_lr{lr:g}.csv"
    step_path = run_dir / f"step_losses_{variant}_lr{lr:g}.csv"
    smoothness_path = run_dir / f"smoothness_{variant}_lr{lr:g}.csv"

    best_acc = 0.0
    best_epoch = 0
    global_step = 0
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        pbar = tqdm(train_loader, desc=f"{variant} lr={lr:g} epoch={epoch}", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9).item()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_seen += batch_size
            append_csv(
                step_path,
                {
                    "global_step": global_step,
                    "epoch": epoch,
                    "loss": f"{loss.item():.8f}",
                    "grad_norm": f"{grad_norm:.8f}",
                },
            )
            global_step += 1
            pbar.set_postfix(loss=f"{total_loss / total_seen:.4f}", acc=f"{total_correct / total_seen:.4f}")

        train_loss = total_loss / total_seen
        train_acc = total_correct / total_seen
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        append_csv(
            metrics_path,
            {
                "epoch": epoch,
                "lr": f"{lr:.8f}",
                "train_loss": f"{train_loss:.8f}",
                "train_acc": f"{train_acc:.8f}",
                "test_loss": f"{test_loss:.8f}",
                "test_acc": f"{test_acc:.8f}",
                "elapsed_min": f"{(time.time() - start) / 60:.2f}",
            },
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            module = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "variant": variant,
                    "lr": lr,
                    "epoch": epoch,
                    "model_state": module.state_dict(),
                    "test_acc": test_acc,
                    "parameters": get_number_of_parameters(module),
                },
                run_dir / f"best_{variant}_lr{lr:g}.pth",
            )

        if config.analysis_every > 0 and (epoch % config.analysis_every == 0 or epoch == config.epochs):
            probe_rows = local_smoothness_probe(model, criterion, analysis_batch, device)
            for row in probe_rows:
                append_csv(
                    smoothness_path,
                    {
                        "epoch": epoch,
                        **{k: f"{v:.10f}" for k, v in row.items()},
                    },
                )

        print(
            f"{variant} lr={lr:g} epoch={epoch:03d}/{config.epochs} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )

    module = model.module if isinstance(model, nn.DataParallel) else model
    return {
        "variant": variant,
        "lr": lr,
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "parameters": get_number_of_parameters(module),
        "elapsed_minutes": (time.time() - start) / 60,
    }


def read_step_losses(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return np.array([float(row["loss"]) for row in reader], dtype=np.float64)


def moving_average(values: np.ndarray, window: int = 25) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def build_loss_band(run_dir: Path, variant: str, learning_rates: Iterable[float]) -> dict[str, np.ndarray]:
    curves = []
    for lr in learning_rates:
        path = run_dir / f"step_losses_{variant}_lr{lr:g}.csv"
        if path.exists():
            curves.append(read_step_losses(path))
    min_len = min(len(curve) for curve in curves)
    stacked = np.stack([curve[:min_len] for curve in curves])
    return {
        "min": stacked.min(axis=0),
        "max": stacked.max(axis=0),
        "mean": stacked.mean(axis=0),
    }


def plot_loss_landscape(run_dir: Path, learning_rates: Iterable[float]) -> None:
    plt.figure(figsize=(9, 5))
    labels = {"vgg_a": "VGG-A without BN", "vgg_a_bn": "VGG-A with BN"}
    colors = {"vgg_a": "#d95f02", "vgg_a_bn": "#1b9e77"}
    for variant in ("vgg_a", "vgg_a_bn"):
        band = build_loss_band(run_dir, variant, learning_rates)
        lo = moving_average(band["min"])
        hi = moving_average(band["max"])
        mean = moving_average(band["mean"])
        steps = np.arange(len(mean))
        plt.plot(steps, mean, color=colors[variant], label=labels[variant])
        plt.fill_between(steps, lo, hi, color=colors[variant], alpha=0.18)
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Loss landscape band across learning rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_landscape_band.png", dpi=180)
    plt.close()


def plot_best_training_curves(run_dir: Path, summaries: list[dict]) -> None:
    best_by_variant = {}
    for row in summaries:
        variant = row["variant"]
        if variant not in best_by_variant or row["best_test_acc"] > best_by_variant[variant]["best_test_acc"]:
            best_by_variant[variant] = row

    plt.figure(figsize=(9, 4))
    for variant, row in best_by_variant.items():
        metrics_path = run_dir / f"metrics_{variant}_lr{row['lr']:g}.csv"
        with metrics_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        epochs = [int(item["epoch"]) for item in rows]
        test_acc = [float(item["test_acc"]) for item in rows]
        plt.plot(epochs, test_acc, marker="o", label=f"{variant} lr={row['lr']:g}")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("VGG-A with vs. without BatchNorm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy_comparison.png", dpi=180)
    plt.close()


def plot_smoothness(run_dir: Path, summaries: list[dict]) -> None:
    rows = []
    for path in run_dir.glob("smoothness_*.csv"):
        parts = path.stem.replace("smoothness_", "").split("_lr")
        variant = parts[0]
        lr = float(parts[1])
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "variant": variant,
                        "lr": lr,
                        "epoch": int(row["epoch"]),
                        "alpha": float(row["alpha"]),
                        "linearization_error": float(row["linearization_error"]),
                        "grad_diff_per_alpha": float(row["grad_diff_per_alpha"]),
                    }
                )
    if not rows:
        return

    best_lrs = {}
    for summary in summaries:
        variant = summary["variant"]
        if variant not in best_lrs or summary["best_test_acc"] > best_lrs[variant]["best_test_acc"]:
            best_lrs[variant] = summary

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for variant, summary in best_lrs.items():
        selected = [row for row in rows if row["variant"] == variant and abs(row["lr"] - summary["lr"]) < 1e-12]
        if not selected:
            continue
        epochs = sorted(set(row["epoch"] for row in selected))
        lin = [np.mean([row["linearization_error"] for row in selected if row["epoch"] == epoch]) for epoch in epochs]
        grad = [np.mean([row["grad_diff_per_alpha"] for row in selected if row["epoch"] == epoch]) for epoch in epochs]
        axes[0].plot(epochs, lin, marker="o", label=variant)
        axes[1].plot(epochs, grad, marker="o", label=variant)
    axes[0].set_title("Gradient predictiveness")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean linearization error")
    axes[1].set_title("Gradient change")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean ||grad shift|| / distance")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "smoothness_probe.png", dpi=180)
    plt.close()


def parse_lrs(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VGG-A BatchNorm experiments.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./outputs/batchnorm_vgg")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-items", type=int, default=0, help="Use a subset for quick debugging; 0 means full dataset.")
    parser.add_argument("--learning-rates", default="0.001,0.002,0.0001,0.0005")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--analysis-every", type=int, default=5)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.set_defaults(amp=True, download=True)
    args = parser.parse_args()

    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        n_items=args.n_items,
        seed=args.seed,
        learning_rates=parse_lrs(args.learning_rates),
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        amp=args.amp,
        analysis_every=args.analysis_every,
        download=args.download,
    )

    set_seed(config.seed)
    run_dir = ensure_dir(config.output_dir)
    save_json(asdict(config), run_dir / "experiment_config.json")
    loader_cfg = CifarLoaderConfig(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        n_items=config.n_items,
        seed=config.seed,
        download=config.download,
    )
    train_loader, test_loader = create_loaders(loader_cfg)
    analysis_batch = next(iter(test_loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, gpu_count={torch.cuda.device_count()}")

    summaries = []
    for variant in ("vgg_a", "vgg_a_bn"):
        for lr in config.learning_rates:
            summary = train_one_run(variant, lr, train_loader, test_loader, analysis_batch, config, device, run_dir)
            summaries.append(summary)
            save_json({"runs": summaries}, run_dir / "summary.json")

    plot_loss_landscape(run_dir, config.learning_rates)
    plot_best_training_curves(run_dir, summaries)
    plot_smoothness(run_dir, summaries)
    save_json({"runs": summaries}, run_dir / "summary.json")

    best = sorted(summaries, key=lambda item: item["best_test_acc"], reverse=True)[0]
    print(f"Best run: {best}")
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
