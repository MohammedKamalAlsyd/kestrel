"""Dense feed-forward network blocks (`SwiGLUFFN`, `ReLU2FFN`) for Kestrel."""

from __future__ import annotations

import torch
import torch.nn as nn

from kestrel.models.activations import get_activation


class SwiGLUFFN(nn.Module):
    """Gated SwiGLU / GLU-style FFN: `down_proj(act(gate_proj(x)) * up_proj(x))`."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        bias: bool = False,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class ReLU2FFN(nn.Module):
    """Two-layer sparse FFN using Squared ReLU (or any `ACT2FN` activation)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        bias: bool = False,
        activation: str = "relu2",
    ) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))
