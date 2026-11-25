"""Gated DeltaNet linear-time sequence mixer for Track B hybrid architectures."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kestrel.models.cache import KVCache
from kestrel.models.norms import RMSNorm


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet recurrence layer: fast-weight delta-rule associative memory."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 8,
        head_dim: int = 128,
        conv_kernel: int = 4,
        use_gate: bool = True,
        use_short_conv: bool = True,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.conv_kernel = conv_kernel
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.alpha_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)

        if use_short_conv:
            # Depthwise causal conv over concatenated (q, k, v)
            self.conv1d = nn.Conv1d(
                3 * self.inner_dim,
                3 * self.inner_dim,
                kernel_size=conv_kernel,
                groups=3 * self.inner_dim,
                padding=0,
                bias=True,
            )
        else:
            self.conv1d = None

        self.g_proj = nn.Linear(hidden_size, self.inner_dim, bias=False) if use_gate else None
        self.o_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.o_proj = nn.Linear(self.inner_dim, hidden_size, bias=False)

    def _apply_short_conv(
        self,
        qkv: torch.Tensor,
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        """Apply causal depthwise Conv1d + SiLU on `qkv` of shape `(B, T, 3*D)`."""
        assert self.conv1d is not None
        x = qkv.transpose(1, 2)  # (B, C, T)
        k_size = self.conv_kernel
        if kv_cache is not None:
            rstate = kv_cache.get_recurrent_state(self.layer_idx)
            if rstate.conv_state is None:
                padded = F.pad(x, (k_size - 1, 0))
            else:
                padded = torch.cat((rstate.conv_state, x), dim=-1)
            rstate.conv_state = padded[:, :, -(k_size - 1) :].detach() if k_size > 1 else None
        else:
            padded = F.pad(x, (k_size - 1, 0))
        return F.silu(self.conv1d(padded)).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        document_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Run gated delta-rule recurrence over `x` of shape `(B, T, D)`."""
        bsz, seq_len, _ = x.shape

        qkv = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)
        if self.conv1d is not None:
            qkv = self._apply_short_conv(qkv, kv_cache)

        q, k, v = qkv.split(self.inner_dim, dim=-1)
        q = F.normalize(q.view(bsz, seq_len, self.num_heads, self.head_dim).float(), dim=-1)
        k = F.normalize(k.view(bsz, seq_len, self.num_heads, self.head_dim).float(), dim=-1)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).float()

        alpha = torch.sigmoid(self.alpha_proj(x).float())  # (B, T, H)
        beta = torch.sigmoid(self.beta_proj(x).float())    # (B, T, H)

        if kv_cache is not None:
            rstate = kv_cache.get_recurrent_state(self.layer_idx)
            state = (
                rstate.ssm_state.float()
                if rstate.ssm_state is not None
                else x.new_zeros(bsz, self.num_heads, self.head_dim, self.head_dim, dtype=torch.float32)
            )
        else:
            rstate = None
            state = x.new_zeros(bsz, self.num_heads, self.head_dim, self.head_dim, dtype=torch.float32)

        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            # Reset recurrent state when crossing packed document boundaries
            if document_ids is not None and t > 0:
                same_doc = (document_ids[:, t] == document_ids[:, t - 1]).float().view(bsz, 1, 1, 1)
                state = state * same_doc

            k_t = k[:, t]  # (B, H, D)
            v_t = v[:, t]  # (B, H, D)
            q_t = q[:, t]  # (B, H, D)
            a_t = alpha[:, t].view(bsz, self.num_heads, 1, 1)
            b_t = beta[:, t].view(bsz, self.num_heads, 1)

            v_pred = torch.einsum("bhij,bhj->bhi", state, k_t)
            err = b_t * (v_t - v_pred)
            state = a_t * state + torch.einsum("bhi,bhj->bhij", err, k_t)
            outputs.append(torch.einsum("bhij,bhj->bhi", state, q_t))

        if rstate is not None:
            rstate.ssm_state = state.detach()

        out = torch.stack(outputs, dim=1).to(x.dtype)  # (B, T, H, D)
        out = self.o_norm(out).reshape(bsz, seq_len, self.inner_dim)
        if self.g_proj is not None:
            out = out * F.silu(self.g_proj(x))
        return self.o_proj(out)
