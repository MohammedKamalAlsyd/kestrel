"""Multi-Token Prediction (MTP) auxiliary training head (discarded at inference)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kestrel.models.ffn import ReLU2FFN
from kestrel.models.norms import RMSNorm


class _MTPDepthModule(nn.Module):
    """Single MTP depth module: fuses previous hidden state with shifted token embedding."""

    def __init__(self, hidden_size: int, intermediate_size: int, rms_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.h_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.e_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.ffn_pre_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ffn = ReLU2FFN(hidden_size, intermediate_size, bias=False)
        self.ffn_post_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.out_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, h: torch.Tensor, tok_emb: torch.Tensor) -> torch.Tensor:
        x = self.proj(torch.cat((self.h_norm(h), self.e_norm(tok_emb)), dim=-1))
        x = x + self.ffn_post_norm(self.ffn(self.ffn_pre_norm(x)))
        return self.out_norm(x)


class MultiTokenPredictionHead(nn.Module):
    """Predicts future tokens `t+2 .. t+K` during training using shared embeddings & LM head."""

    def __init__(
        self,
        hidden_size: int,
        num_predict_tokens: int = 2,
        *,
        intermediate_size: int | None = None,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_extra_tokens = max(0, int(num_predict_tokens) - 1)
        ffn_dim = intermediate_size or (2 * hidden_size)
        self.modules_by_depth = nn.ModuleList(
            [_MTPDepthModule(hidden_size, ffn_dim, rms_norm_eps=rms_norm_eps) for _ in range(self.num_extra_tokens)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        *,
        embed_tokens: nn.Embedding,
        lm_head: nn.Linear,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Compute mean MTP cross-entropy across extra depths `k = 1 .. K-1`.

        `next_token_ids` has shape `(B, T)` where `next_token_ids[:, t]` is token `x_{t+1}`.
        For depth `k`, position `t` conditions on `x_{t+k}` and predicts `x_{t+k+1}`.
        """
        if self.num_extra_tokens == 0 or hidden_states.shape[1] < 2:
            return None

        h = hidden_states
        losses: list[torch.Tensor] = []
        seq_len = hidden_states.shape[1]

        for depth, mod in enumerate(self.modules_by_depth, start=1):
            valid_len = seq_len - depth
            if valid_len <= 0:
                break

            cond_ids = next_token_ids[:, depth - 1 : depth - 1 + valid_len]
            targets = next_token_ids[:, depth : depth + valid_len]

            # Guard against ignore_index (-100) in conditioning ids before embedding lookup
            safe_cond = cond_ids.clamp(min=0)
            tok_emb = embed_tokens(safe_cond)

            h_slice = h[:, :valid_len, :]
            h_next = mod(h_slice, tok_emb)
            logits = lm_head(h_next)

            masked_targets = targets.clone()
            masked_targets = masked_targets.masked_fill(cond_ids < 0, -100)
            if document_ids is not None:
                same_doc = document_ids[:, :valid_len] == document_ids[:, depth : depth + valid_len]
                masked_targets = masked_targets.masked_fill(~same_doc, -100)

            if (masked_targets != -100).any():
                loss_k = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    masked_targets.reshape(-1),
                    ignore_index=-100,
                )
                losses.append(loss_k)
            h = h_next

        if not losses:
            return None
        return sum(losses) / len(losses)
