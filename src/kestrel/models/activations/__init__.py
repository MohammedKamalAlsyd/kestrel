"""Activation function registry (`ACT2FN`) and custom activations for Kestrel."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):
    """Squared ReLU activation: `relu(x) ** 2` (high natural sparsity for wide FFNs)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class NewGELU(nn.Module):
    """Tanh-approximated GELU used in GPT-2 / classic Transformer FFNs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


ACT2FN: dict[str, Callable[[], nn.Module]] = {
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
    "gelu_new": NewGELU,
    "gelu_tanh": NewGELU,
    "relu": nn.ReLU,
    "relu2": SquaredReLU,
    "squared_relu": SquaredReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


def get_activation(name: str) -> nn.Module:
    """Instantiate an activation module by name from `ACT2FN`."""
    key = name.strip().lower()
    if key not in ACT2FN:
        raise KeyError(f"Unknown activation '{name}'. Available: {sorted(ACT2FN)}")
    return ACT2FN[key]()


__all__ = [
    "ACT2FN",
    "NewGELU",
    "SquaredReLU",
    "get_activation",
]
