"""Model definitions for CIFAR-10 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = make_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = make_activation(activation)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.act2(out + residual)
        return out


class CifarResidualNet(nn.Module):
    """Compact ResNet-style CNN customized for 32x32 CIFAR-10 images."""

    def __init__(
        self,
        num_classes: int = 10,
        base_channels: int = 64,
        blocks_per_stage: int = 3,
        activation: str = "relu",
        dropout: float = 0.1,
        fc_dim: int = 256,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            make_activation(activation),
        )
        channels = [base_channels, base_channels * 2, base_channels * 4]
        self.stage1 = self._make_stage(base_channels, channels[0], blocks_per_stage, stride=1, activation=activation, dropout=dropout)
        self.stage2 = self._make_stage(channels[0], channels[1], blocks_per_stage, stride=2, activation=activation, dropout=dropout)
        self.stage3 = self._make_stage(channels[1], channels[2], blocks_per_stage, stride=2, activation=activation, dropout=dropout)
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=2), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[2], fc_dim),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )
        self.apply(self._init_weights)

    def _make_stage(self, in_channels: int, out_channels: int, blocks: int, stride: int, activation: str, dropout: float) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride, activation, dropout)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, activation, dropout))
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.classifier(x)


class SimpleConvNet(nn.Module):
    """A smaller CNN baseline for ablation on filter counts and activations."""

    def __init__(self, num_classes: int = 10, width: int = 48, activation: str = "relu", dropout: float = 0.2, fc_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, width, activation, dropout=0.0),
            ConvBlock(width, width, activation, dropout=dropout / 2),
            nn.MaxPool2d(2),
            ConvBlock(width, width * 2, activation, dropout=dropout / 2),
            ConvBlock(width * 2, width * 2, activation, dropout=dropout),
            nn.MaxPool2d(2),
            ConvBlock(width * 2, width * 4, activation, dropout=dropout),
            ConvBlock(width * 4, width * 4, activation, dropout=dropout),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4, fc_dim),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )
        self.apply(CifarResidualNet._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass(frozen=True)
class ModelConfig:
    name: str = "resnet"
    base_channels: int = 64
    blocks_per_stage: int = 3
    activation: str = "relu"
    dropout: float = 0.1
    fc_dim: int = 256


def build_model(config: ModelConfig) -> nn.Module:
    if config.name == "resnet":
        return CifarResidualNet(
            base_channels=config.base_channels,
            blocks_per_stage=config.blocks_per_stage,
            activation=config.activation,
            dropout=config.dropout,
            fc_dim=config.fc_dim,
        )
    if config.name == "simple":
        return SimpleConvNet(width=config.base_channels, activation=config.activation, dropout=config.dropout, fc_dim=config.fc_dim)
    raise ValueError(f"Unknown model: {config.name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

