"""Train and evaluate CIFAR-10 classifiers."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

try:
    from .data import CLASS_NAMES, LoaderConfig, create_loaders
    from .models import ModelConfig, build_model, count_parameters
    from .utils import AverageMeter, accuracy, append_metrics, ensure_dir, load_json, save_json, set_seed
    from .visualize import plot_confusion_matrix, plot_curves, visualize_first_layer_filters
except ImportError:
    from data import CLASS_NAMES, LoaderConfig, create_loaders
    from models import ModelConfig, build_model, count_parameters
    from utils import AverageMeter, accuracy, append_metrics, ensure_dir, load_json, save_json, set_seed
    from visualize import plot_confusion_matrix, plot_curves, visualize_first_layer_filters


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 0.1
    min_lr: float = 1e-5
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    clip_grad_norm: float = 0.0
    amp: bool = True
    seed: int = 42
    output_dir: str = "./outputs/cifar10_resnet"


def mixup(inputs: torch.Tensor, targets: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[indices]
    return mixed_inputs, targets, targets[indices], float(lam)


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay, nesterov=True)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer, scheduler, scaler: GradScaler, device: torch.device, config: TrainConfig) -> tuple[float, float]:
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    pbar = tqdm(loader, desc="train", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mixed_inputs, target_a, target_b, lam = mixup(inputs, targets, config.mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
            logits = model(mixed_inputs)
            loss = lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)

        scaler.scale(loss).backward()
        if config.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = inputs.size(0)
        losses.update(loss.item(), batch_size)
        accs.update(accuracy(logits.detach(), targets), batch_size)
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{accs.avg:.4f}")
    return losses.avg, accs.avg


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> tuple[float, float, np.ndarray]:
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for inputs, targets in tqdm(loader, desc="eval", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        losses.update(loss.item(), inputs.size(0))
        accs.update(accuracy(logits, targets), inputs.size(0))
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = targets.cpu().numpy()
        for true_label, pred_label in zip(labels, preds):
            matrix[true_label, pred_label] += 1
    return losses.avg, accs.avg, matrix


def load_config(path: str | None) -> dict:
    return {} if path is None else load_json(path)


def merge_config(args: argparse.Namespace) -> tuple[LoaderConfig, ModelConfig, TrainConfig]:
    raw = load_config(args.config)
    loader_cfg = LoaderConfig(**{**raw.get("data", {}), **{k: v for k, v in vars(args).items() if k in LoaderConfig.__annotations__ and v is not None}})
    model_cfg = ModelConfig(**{**raw.get("model", {}), **{k: v for k, v in vars(args).items() if k in ModelConfig.__annotations__ and v is not None}})
    train_cfg = TrainConfig(**{**raw.get("train", {}), **{k: v for k, v in vars(args).items() if k in TrainConfig.__annotations__ and v is not None}})
    return loader_cfg, model_cfg, train_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 classifier.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON experiment config.")
    parser.add_argument("--data-dir", type=str, dest="data_dir")
    parser.add_argument("--output-dir", type=str, dest="output_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--num-workers", type=int, dest="num_workers")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"])
    parser.add_argument("--weight-decay", type=float, dest="weight_decay")
    parser.add_argument("--activation", choices=["relu", "gelu", "silu", "leaky_relu"])
    parser.add_argument("--base-channels", type=int, dest="base_channels")
    parser.add_argument("--blocks-per-stage", type=int, dest="blocks_per_stage")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--mixup-alpha", type=float, dest="mixup_alpha")
    parser.add_argument("--label-smoothing", type=float, dest="label_smoothing")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.set_defaults(amp=None)
    args = parser.parse_args()

    loader_cfg, model_cfg, train_cfg = merge_config(args)
    set_seed(train_cfg.seed)
    output_dir = ensure_dir(train_cfg.output_dir)
    figure_dir = ensure_dir(output_dir / "figures")

    train_loader, val_loader, test_loader = create_loaders(loader_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    parameter_count = count_parameters(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    optimizer = build_optimizer(model, train_cfg)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs * steps_per_epoch, eta_min=train_cfg.min_lr)
    scaler = GradScaler("cuda", enabled=train_cfg.amp and device.type == "cuda")

    save_json({"data": asdict(loader_cfg), "model": asdict(model_cfg), "train": asdict(train_cfg), "parameters": parameter_count}, output_dir / "config.json")
    metrics_path = output_dir / "metrics.csv"
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, train_cfg)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": f"{current_lr:.8f}",
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "elapsed_min": f"{(time.time() - start_time) / 60:.2f}",
        }
        append_metrics(metrics_path, row)
        print(f"epoch {epoch:03d}/{train_cfg.epochs} lr={current_lr:.5f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            module = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": module.state_dict(),
                    "model_config": asdict(model_cfg),
                    "train_config": asdict(train_cfg),
                    "val_acc": val_acc,
                    "parameters": parameter_count,
                },
                output_dir / "best_model.pth",
            )

    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, matrix = evaluate(model, test_loader, criterion, device)
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    plot_curves(metrics_path, figure_dir)
    plot_confusion_matrix(matrix, figure_dir)
    visualize_first_layer_filters(module, figure_dir)

    summary = {
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_error": 1.0 - test_acc,
        "parameters": parameter_count,
        "best_epoch": int(checkpoint["epoch"]),
        "elapsed_minutes": (time.time() - start_time) / 60,
    }
    save_json(summary, output_dir / "summary.json")
    print(f"Best epoch: {summary['best_epoch']}, test_acc={test_acc:.4f}, test_error={1.0 - test_acc:.4f}, params={parameter_count:,}")


if __name__ == "__main__":
    main()
