"""Value-residual gating (SATFormer / ResFormer) for cross-layer value mixing."""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueResidualGate(nn.Module):
    """Mix current-layer values `v_l` with layer-0 values `v_0`: `(1 - g) * v_l + g * v_0`."""

    def __init__(
        self,
        num_kv_heads: int,
        hidden_size: int | None = None,
        mode: str = "satformer",
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.num_kv_heads = num_kv_heads
        if self.mode == "satformer":
            # Named `gate_scalar` so `optim.py` routes it to AdamW without weight decay
            self.gate_scalar = nn.Parameter(torch.zeros(1, num_kv_heads, 1, 1))
            self.proj = None
        elif self.mode == "resformer":
            if hidden_size is None:
                raise ValueError("hidden_size is required for 'resformer' value residual gating.")
            self.gate_scalar = None
            self.proj = nn.Linear(hidden_size, num_kv_heads, bias=True)
        else:
            raise ValueError(f"Unknown value_residual_gating '{mode}'; expected 'satformer' or 'resformer'.")

    def forward(
        self,
        v_current: torch.Tensor,
        v_first: torch.Tensor | None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mix `v_current` `(B, H_kv, T, D)` with `v_first` if provided."""
        if v_first is None:
            return v_current
        if self.mode == "satformer":
            assert self.gate_scalar is not None
            g = torch.sigmoid(self.gate_scalar).to(v_current.dtype)
        else:
            assert self.proj is not None and hidden_states is not None
            # (B, T, H_kv) -> (B, H_kv, T, 1)
            g = torch.sigmoid(self.proj(hidden_states)).transpose(1, 2).unsqueeze(-1).to(v_current.dtype)
        return (1.0 - g) * v_current + g * v_first.to(v_current.dtype)
