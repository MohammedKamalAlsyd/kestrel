"""Gated DeltaNet linear-time sequence mixer with WY chunkwise parallelization and doc-boundary isolation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kestrel.models.cache import KVCache
from kestrel.models.norms import RMSNorm

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gdn  # type: ignore[import-not-found]
except ImportError:
    _fla_chunk_gdn = None


def _delta_rule_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    document_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact token-by-token Gated DeltaNet recurrence: `S_t = a_t S_{t-1}(I - b_t k_t k_t^T) + b_t v_t k_t^T`."""
    bsz, seq_len, num_heads, _ = q.shape
    outputs: list[torch.Tensor] = []

    for t in range(seq_len):
        if document_ids is not None and t > 0:
            same_doc = (document_ids[:, t] == document_ids[:, t - 1]).float().view(bsz, 1, 1, 1)
            state = state * same_doc

        k_t = k[:, t]
        v_t = v[:, t]
        q_t = q[:, t]
        a_t = alpha[:, t].view(bsz, num_heads, 1, 1)
        b_t = beta[:, t].view(bsz, num_heads, 1)

        decayed_state = a_t * state
        v_pred = torch.einsum("bhij,bhj->bhi", decayed_state, k_t)
        u_t = b_t * (v_t - v_pred)
        state = decayed_state + torch.einsum("bhi,bhj->bhij", u_t, k_t)
        outputs.append(torch.einsum("bhij,bhj->bhi", state, q_t))

    return torch.stack(outputs, dim=1), state


def _delta_rule_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    document_ids: torch.Tensor | None = None,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact WY-representation chunkwise parallel Gated DeltaNet in pure PyTorch."""
    bsz, seq_len, num_heads, head_dim = q.shape
    chunks_out: list[torch.Tensor] = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        c_len = end - start

        q_c = q[:, start:end]
        k_c = k[:, start:end]
        v_c = v[:, start:end]
        a_c = alpha[:, start:end]
        b_c = beta[:, start:end]

        # Check if a packed document boundary crosses inside this chunk
        doc_c = document_ids[:, start:end] if document_ids is not None else None
        has_boundary = False
        if doc_c is not None:
            if start > 0:
                same_prev = (document_ids[:, start] == document_ids[:, start - 1]).float().view(bsz, 1, 1, 1)
                state = state * same_prev
            if c_len > 1 and bool((doc_c[:, 1:] != doc_c[:, :-1]).any().item()):
                has_boundary = True

        if has_boundary or c_len == 1:
            out_c, state = _delta_rule_recurrent(q_c, k_c, v_c, a_c, b_c, state, document_ids=doc_c)
            chunks_out.append(out_c)
            continue

        # Parallel WY formulation inside same-document chunk: (B, H, C, D)
        qh = q_c.transpose(1, 2)
        kh = k_c.transpose(1, 2)
        vh = v_c.transpose(1, 2)
        log_a = torch.log(a_c.transpose(1, 2).clamp(min=1e-6))  # (B, H, C)
        gamma = torch.cumsum(log_a, dim=-1)  # (B, H, C)
        bh = b_c.transpose(1, 2)  # (B, H, C)

        # Pairwise decay ratio exp(gamma_i - gamma_j) for j <= i (clamped strictly lower/causal)
        diff = gamma.unsqueeze(-1) - gamma.unsqueeze(-2)  # (B, H, C, C)
        causal_mask = torch.tril(torch.ones(c_len, c_len, device=q.device, dtype=torch.bool))
        strict_lower = torch.tril(torch.ones(c_len, c_len, device=q.device, dtype=torch.bool), diagonal=-1)
        decay_tril = torch.exp(diff.masked_fill(~causal_mask, -1e9)) * causal_mask.float()
        decay_strict = torch.exp(diff.masked_fill(~strict_lower, -1e9)) * strict_lower.float()

        # Solve unit lower-triangular system (I + M) [U, W] = [b*V, b*exp(gamma)*K]
        kk = torch.matmul(kh, kh.transpose(-1, -2))  # (B, H, C, C)
        eye = torch.eye(c_len, device=q.device, dtype=q.dtype)
        lhs = eye + (bh.unsqueeze(-1) * decay_strict * kk)

        exp_gamma = torch.exp(gamma).unsqueeze(-1)  # (B, H, C, 1)
        rhs_v = bh.unsqueeze(-1) * vh
        rhs_k = (bh.unsqueeze(-1) * exp_gamma) * kh
        rhs = torch.cat((rhs_v, rhs_k), dim=-1)  # (B, H, C, 2D)

        solved = torch.linalg.solve_triangular(lhs, rhs, upper=False, unitriangular=True)
        u_mat, w_mat = solved.split(head_dim, dim=-1)

        # Effective chunk updates U_eff = U - W @ S_0^T
        u_eff = u_mat - torch.matmul(w_mat, state.transpose(-1, -2))

        # Parallel chunk outputs: (exp(gamma) * Q) @ S_0^T + (tril(Q K^T) * decay) @ U_eff
        qk = torch.matmul(qh, kh.transpose(-1, -2)) * decay_tril
        out_h = torch.matmul(exp_gamma * qh, state.transpose(-1, -2)) + torch.matmul(qk, u_eff)
        chunks_out.append(out_h.transpose(1, 2))

        # Exact end-of-chunk state update
        gamma_end = gamma[:, :, -1:].unsqueeze(-1)  # (B, H, 1, 1)
        k_scaled = torch.exp(gamma_end.squeeze(-1) - gamma).unsqueeze(-1) * kh  # (B, H, C, D)
        state = torch.exp(gamma_end) * state + torch.matmul(u_eff.transpose(-1, -2), k_scaled)

    return torch.cat(chunks_out, dim=1), state


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet recurrence layer with chunkwise WY fast-path and optional `fla` Triton kernel."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 8,
        head_dim: int = 128,
        conv_kernel: int = 4,
        use_gate: bool = True,
        use_short_conv: bool = True,
        chunk_size: int = 32,
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
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.alpha_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)

        if use_short_conv:
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
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply causal depthwise Conv1d + SiLU on `qkv` `(B, T, 3*D)` with zero cross-doc leakage."""
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
            return F.silu(self.conv1d(padded)).transpose(1, 2)

        padded = F.pad(x, (k_size - 1, 0))
        if document_ids is not None and k_size > 1:
            # Zero out conv taps that reach across packed document boundaries
            padded_docs = F.pad(document_ids, (k_size - 1, 0), value=-1)
            doc_win = padded_docs.unfold(dimension=1, size=k_size, step=1)  # (B, T, K)
            same_doc = (doc_win == document_ids.unsqueeze(-1)).to(x.dtype)  # (B, T, K)
            x_win = padded.unfold(dimension=2, size=k_size, step=1)  # (B, C, T, K)
            w = self.conv1d.weight.squeeze(1)  # (C, K)
            conv_out = torch.einsum("bctk,btk,ck->bct", x_win, same_doc, w)
            if self.conv1d.bias is not None:
                conv_out = conv_out + self.conv1d.bias.view(1, -1, 1)
            return F.silu(conv_out).transpose(1, 2)

        return F.silu(self.conv1d(padded)).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        document_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Run gated delta-rule mixer over `x` of shape `(B, T, D)`."""
        bsz, seq_len, _ = x.shape

        qkv = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)
        if self.conv1d is not None:
            qkv = self._apply_short_conv(qkv, kv_cache, document_ids=document_ids)

        q, k, v = qkv.split(self.inner_dim, dim=-1)
        q = F.normalize(q.view(bsz, seq_len, self.num_heads, self.head_dim).float(), dim=-1)
        k = F.normalize(k.view(bsz, seq_len, self.num_heads, self.head_dim).float(), dim=-1)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).float()

        alpha = torch.sigmoid(self.alpha_proj(x).float())  # (B, T, H)
        beta = torch.sigmoid(self.beta_proj(x).float())    # (B, T, H)

        rstate = kv_cache.get_recurrent_state(self.layer_idx) if kv_cache is not None else None
        state = (
            rstate.ssm_state.float()
            if (rstate is not None and rstate.ssm_state is not None)
            else x.new_zeros(bsz, self.num_heads, self.head_dim, self.head_dim, dtype=torch.float32)
        )

        # Optional `flash-linear-attention` Triton fast path (training on CUDA without doc_ids/cache)
        if _fla_chunk_gdn is not None and x.is_cuda and kv_cache is None and document_ids is None and seq_len >= 32:
            g_log = torch.log(alpha.clamp(min=1e-6))
            out_fp32, _ = _fla_chunk_gdn(q, k, v, g=g_log, beta=beta)
        elif seq_len >= 8:
            out_fp32, state = _delta_rule_chunkwise(
                q, k, v, alpha, beta, state, document_ids=document_ids, chunk_size=self.chunk_size
            )
        else:
            out_fp32, state = _delta_rule_recurrent(q, k, v, alpha, beta, state, document_ids=document_ids)

        if rstate is not None:
            rstate.ssm_state = state.detach()

        out = out_fp32.to(x.dtype)
        out = self.o_norm(out).reshape(bsz, seq_len, self.inner_dim)
        if self.g_proj is not None:
            out = out * F.silu(self.g_proj(x))
        return self.o_proj(out)
