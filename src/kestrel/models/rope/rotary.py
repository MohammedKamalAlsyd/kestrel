"""Rotary Position Embeddings (RoPE) with FP32 inverse frequencies and cos/sin caching."""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first half (negated)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply precomputed `(cos, sin)` rotary embeddings to `(q, k)` of shape `(B, H, T, D)`."""
    cos = cos.unsqueeze(1)  # (B or 1, 1, T, D)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    """Cached RoPE module keeping `inv_freq` in FP32 and lazily caching `(cos, sin)` tables."""

    def __init__(self, dim: int, theta: float = 10_000.0, max_seq_len: int = 4096) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {dim}")
        self.dim = dim
        self.theta = float(theta)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max_seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._build_cache(max_seq_len, device=inv_freq.device)

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()  # (L, D) in fp32
        self._sin_cached = emb.sin()  # (L, D) in fp32
        self._max_seq_len_cached = seq_len

    @torch.no_grad()
    def forward(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(cos, sin)` tensors of shape `(B_or_1, seq_len, dim)` in `dtype`."""
        needed = int(position_ids.max().item()) + 1 if position_ids is not None else seq_len
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or needed > self._max_seq_len_cached
            or self._cos_cached.device != device
        ):
            new_len = max(needed, self._max_seq_len_cached * 2 if self._max_seq_len_cached else 2048)
            self._build_cache(new_len, device=device)

        assert self._cos_cached is not None and self._sin_cached is not None
        if position_ids is None:
            cos = self._cos_cached[:seq_len].unsqueeze(0)
            sin = self._sin_cached[:seq_len].unsqueeze(0)
        else:
            pos = position_ids.to(device=device, dtype=torch.long)
            cos = self._cos_cached[pos]
            sin = self._sin_cached[pos]
        return cos.to(dtype), sin.to(dtype)
