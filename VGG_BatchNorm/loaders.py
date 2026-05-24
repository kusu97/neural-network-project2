"""CIFAR-10 loaders used by the VGG BatchNorm experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class CifarLoaderConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    n_items: int = 0
    seed: int = 2020
    download: bool = True


def _make_transform(train: bool) -> transforms.Compose:
    ops: list[object] = []
    if train:
        ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return transforms.Compose(ops)


def _maybe_subset(dataset: datasets.CIFAR10, n_items: int, seed: int) -> datasets.CIFAR10 | Subset:
    if n_items <= 0 or n_items >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n_items].tolist()
    return Subset(dataset, indices)


def get_cifar_loader(
    train: bool = True,
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    n_items: int = 0,
    seed: int = 2020,
    download: bool = True,
) -> DataLoader:
    dataset = datasets.CIFAR10(
        root=Path(data_dir),
        train=train,
        download=download,
        transform=_make_transform(train),
    )
    dataset = _maybe_subset(dataset, n_items=n_items, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def create_loaders(config: CifarLoaderConfig) -> tuple[DataLoader, DataLoader]:
    train_loader = get_cifar_loader(
        train=True,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        n_items=config.n_items,
        seed=config.seed,
        download=config.download,
    )
    test_items = max(1, config.n_items // 5) if config.n_items > 0 else 0
    test_loader = get_cifar_loader(
        train=False,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        n_items=test_items,
        seed=config.seed,
        download=config.download,
    )
    return train_loader, test_loader

