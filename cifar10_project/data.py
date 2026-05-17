"""Data loading and augmentation for CIFAR-10."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class Cutout:
    """Randomly masks one square region in an image tensor."""

    def __init__(self, size: int = 16, p: float = 0.5) -> None:
        self.size = size
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return image
        _, height, width = image.shape
        y = torch.randint(0, height, (1,)).item()
        x = torch.randint(0, width, (1,)).item()
        y1 = max(0, y - self.size // 2)
        y2 = min(height, y + self.size // 2)
        x1 = max(0, x - self.size // 2)
        x2 = min(width, x + self.size // 2)
        image = image.clone()
        image[:, y1:y2, x1:x2] = 0.0
        return image


@dataclass(frozen=True)
class LoaderConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    val_size: int = 5000
    seed: int = 42
    download: bool = True
    augment: str = "strong"
    cutout: bool = True


def build_transforms(augment: str = "strong", cutout: bool = True) -> tuple[transforms.Compose, transforms.Compose]:
    train_ops: list[object] = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if augment == "strong":
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    elif augment not in {"basic", "none"}:
        raise ValueError(f"Unknown augmentation policy: {augment}")

    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    if cutout and augment != "none":
        train_ops.append(Cutout(size=16, p=0.5))

    test_ops = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def create_loaders(config: LoaderConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = Path(config.data_dir)
    train_transform, test_transform = build_transforms(config.augment, config.cutout)

    train_full = datasets.CIFAR10(root=data_dir, train=True, download=config.download, transform=train_transform)
    val_full = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=test_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=config.download, transform=test_transform)

    train_size = len(train_full) - config.val_size
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(train_full), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_subset = Subset(train_full, train_indices)
    val_subset = Subset(val_full, val_indices)

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(train_subset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
