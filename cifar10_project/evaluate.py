"""Evaluate a saved CIFAR-10 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .data import LoaderConfig, create_loaders
    from .models import ModelConfig, build_model, count_parameters
    from .train_cifar10 import evaluate
    from .utils import ensure_dir, save_json
    from .visualize import plot_confusion_matrix, visualize_first_layer_filters
except ImportError:
    from data import LoaderConfig, create_loaders
    from models import ModelConfig, build_model, count_parameters
    from train_cifar10 import evaluate
    from utils import ensure_dir, save_json
    from visualize import plot_confusion_matrix, visualize_first_layer_filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="./outputs/eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss()
    _, _, test_loader = create_loaders(LoaderConfig(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, augment="none", cutout=False))
    test_loss, test_acc, matrix = evaluate(model, test_loader, criterion, device)

    output_dir = ensure_dir(args.output_dir)
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    plot_confusion_matrix(matrix, output_dir / "figures")
    visualize_first_layer_filters(model, output_dir / "figures")
    save_json({"test_loss": test_loss, "test_acc": test_acc, "test_error": 1.0 - test_acc, "parameters": count_parameters(model)}, output_dir / "summary.json")
    print(f"test_acc={test_acc:.4f}, test_error={1.0 - test_acc:.4f}")


if __name__ == "__main__":
    main()

