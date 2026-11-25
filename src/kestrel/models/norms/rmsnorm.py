"""Normalization layers (fused RMSNorm) for Kestrel."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_FUSED_RMS_NORM = hasattr(F, "rms_norm")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization using native fused `F.rms_norm` when available."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = (hidden_size,)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_FUSED_RMS_NORM:
            return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (normed * self.weight.float()).to(input_dtype)
