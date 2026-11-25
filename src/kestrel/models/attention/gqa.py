"""Grouped-Query Attention (GQA) with optional QK-Norm, RoPE, SWA, and Value Residuals."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kestrel.models.cache import KVCache
from kestrel.models.masking import create_causal_mask
from kestrel.models.norms import RMSNorm
from kestrel.models.residuals import ValueResidualGate
from kestrel.models.rope import apply_rotary_pos_emb


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention supporting QK-Norm, RNoPE-SWA, Value Residuals, and KV caching."""

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
        sliding_window: int | None = None,
        use_rope: bool = True,
        value_residual_gating: str | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.use_rope = use_rope
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=bias)

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps) if qk_norm else None
        self.value_gate = (
            ValueResidualGate(num_key_value_heads, hidden_size=hidden_size, mode=value_residual_gating)
            if value_residual_gating is not None and layer_idx > 0
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        v_first: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(attn_output, v_raw)` where `v_raw` has shape `(B, H_kv, T, D)`."""
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_raw = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope and cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        v = self.value_gate(v_raw, v_first, x) if self.value_gate is not None else v_raw

        past_len = kv_cache.get_cached_length(self.layer_idx) if kv_cache is not None else 0
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v, sliding_window=self.sliding_window)
            past_len = k.shape[-2] - seq_len

        kv_len = k.shape[-2]
        mask = create_causal_mask(
            bsz,
            seq_len,
            kv_len,
            past_len=past_len,
            sliding_window=self.sliding_window,
            document_ids=document_ids,
            attention_mask=attention_mask,
            device=x.device,
            dtype=q.dtype,
        )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=(mask is None),
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out), v_raw
