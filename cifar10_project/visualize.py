"""Visualization utilities for reports."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .data import CLASS_NAMES, CIFAR10_MEAN, CIFAR10_STD
from .utils import ensure_dir


def plot_curves(metrics_csv: str | Path, out_dir: str | Path) -> None:
    metrics_csv = Path(metrics_csv)
    out_dir = ensure_dir(out_dir)
    rows = []
    with metrics_csv.open("r", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_acc = [float(row["train_acc"]) for row in rows]
    val_acc = [float(row["val_acc"]) for row in rows]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, val_acc, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=180)
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray, out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(ticks, CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close()


def visualize_first_layer_filters(model: nn.Module, out_dir: str | Path, max_filters: int = 32) -> None:
    out_dir = ensure_dir(out_dir)
    first_conv = next((m for m in model.modules() if isinstance(m, nn.Conv2d)), None)
    if first_conv is None:
        return

    weights = first_conv.weight.detach().cpu()
    count = min(max_filters, weights.shape[0])
    filters = weights[:count]
    filters = (filters - filters.amin(dim=(1, 2, 3), keepdim=True)) / (filters.amax(dim=(1, 2, 3), keepdim=True) - filters.amin(dim=(1, 2, 3), keepdim=True) + 1e-8)

    cols = 8
    rows = int(np.ceil(count / cols))
    plt.figure(figsize=(cols, rows))
    for i in range(count):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(filters[i].permute(1, 2, 0).numpy())
        plt.axis("off")
    plt.tight_layout(pad=0.2)
    plt.savefig(out_dir / "first_layer_filters.png", dpi=180)
    plt.close()


def unnormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)

