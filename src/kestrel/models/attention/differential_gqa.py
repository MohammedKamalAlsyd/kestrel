"""Differential Grouped-Query Attention (ICLR 2025) using two SDPA passes."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kestrel.models.cache import KVCache
from kestrel.models.masking import create_causal_mask
from kestrel.models.norms import RMSNorm
from kestrel.models.rope import apply_rotary_pos_emb


_UNSET_MASK = object()


class DifferentialGQA(nn.Module):
    """Differential GQA: splits Q/K into two halves `(q1, q2)` and computes `attn1 - lambda * attn2`."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        *,
        bias: bool = False,
        qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        diff_attn_lambda_init: float = 0.8,
        sliding_window: int | None = None,
        use_rope: bool = True,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if head_dim % 4 != 0:
            raise ValueError(f"DifferentialGQA head_dim must be divisible by 4 (for half-dim RoPE), got {head_dim}")

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        self.sliding_window = sliding_window
        self.use_rope = use_rope
        self.layer_idx = layer_idx
        self.lambda_init = diff_attn_lambda_init

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=bias)

        self.q_norm = RMSNorm(self.half_dim, eps=rms_norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(self.half_dim, eps=rms_norm_eps) if qk_norm else None
        self.subln = RMSNorm(head_dim, eps=rms_norm_eps)

        # Named with `lambda_` so `optim.py` assigns them to AdamW without weight decay
        self.lambda_q1 = nn.Parameter(torch.zeros(self.half_dim).normal_(std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.half_dim).normal_(std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.half_dim).normal_(std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.half_dim).normal_(std=0.1))

    def _compute_lambda(self, dtype: torch.dtype) -> torch.Tensor:
        l1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dtype=torch.float32))
        l2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dtype=torch.float32))
        return (l1 - l2 + self.lambda_init).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        precomputed_mask: torch.Tensor | None | object = _UNSET_MASK,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, 2, self.half_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, 2, self.half_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q1, q2 = q[:, :, :, 0].transpose(1, 2), q[:, :, :, 1].transpose(1, 2)
        k1, k2 = k[:, :, :, 0].transpose(1, 2), k[:, :, :, 1].transpose(1, 2)

        if self.q_norm is not None and self.k_norm is not None:
            q1, q2 = self.q_norm(q1), self.q_norm(q2)
            k1, k2 = self.k_norm(k1), self.k_norm(k2)

        if self.use_rope and cos is not None and sin is not None:
            q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
            q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)

        past_len = kv_cache.get_cached_length(self.layer_idx) if kv_cache is not None else 0
        if kv_cache is not None:
            k_cat = torch.cat((k1, k2), dim=-1)
            k_cat, v = kv_cache.update(self.layer_idx, k_cat, v, sliding_window=self.sliding_window)
            k1, k2 = k_cat.split(self.half_dim, dim=-1)
            past_len = k1.shape[-2] - seq_len

        kv_len = k1.shape[-2]
        if precomputed_mask is _UNSET_MASK or kv_cache is not None:
            mask = create_causal_mask(
                bsz,
                seq_len,
                kv_len,
                past_len=past_len,
                sliding_window=self.sliding_window,
                document_ids=document_ids,
                attention_mask=attention_mask,
                device=x.device,
                dtype=q1.dtype,
            )
        else:
            mask = precomputed_mask  # type: ignore[assignment]

        scale = 1.0 / math.sqrt(self.half_dim)
        gqa = self.num_kv_heads != self.num_heads
        attn1 = F.scaled_dot_product_attention(q1, k1, v, attn_mask=mask, is_causal=(mask is None), scale=scale, enable_gqa=gqa)
        attn2 = F.scaled_dot_product_attention(q2, k2, v, attn_mask=mask, is_causal=(mask is None), scale=scale, enable_gqa=gqa)

        diff = attn1 - self._compute_lambda(attn1.dtype) * attn2
        diff = self.subln(diff) * (1.0 - self.lambda_init)
        out = diff.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)
