"""Mixture-of-Experts SwiGLU FFN with shared expert and Auxiliary-Loss-Free Load Balancing."""

from __future__ import annotations

import torch
import torch.nn as nn

from kestrel.models.ffn.dense import SwiGLUFFN


class MoESwiGLUFFN(nn.Module):
    """Fine-grained MoE with always-on shared SwiGLU expert and ALF-LB routing."""

    def __init__(
        self,
        hidden_size: int,
        expert_intermediate_size: int,
        *,
        num_routed_experts: int = 16,
        num_experts_per_tok: int = 2,
        num_shared_experts: int = 1,
        shared_expert_intermediate_size: int = 1024,
        router_scoring: str = "sigmoid",
        bias: bool = False,
        alf_lb_bias_update_rate: float = 1e-3,
        seq_aux_loss_alpha: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_scoring = router_scoring.lower()
        self.alf_lb_bias_update_rate = alf_lb_bias_update_rate
        self.seq_aux_loss_alpha = seq_aux_loss_alpha

        self.router = nn.Linear(hidden_size, num_routed_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUFFN(hidden_size, expert_intermediate_size, bias=bias) for _ in range(num_routed_experts)]
        )
        self.shared_expert = (
            SwiGLUFFN(hidden_size, shared_expert_intermediate_size * num_shared_experts, bias=bias)
            if num_shared_experts > 0 and shared_expert_intermediate_size > 0
            else None
        )

        self.register_buffer("expert_bias", torch.zeros(num_routed_experts, dtype=torch.float32))
        self.register_buffer("_load_accum", torch.zeros(num_routed_experts, dtype=torch.float32), persistent=False)
        self.register_buffer("_load_steps", torch.zeros((), dtype=torch.long), persistent=False)
        self._last_aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(output, seq_aux_loss)` for input `x` of shape `(B, T, D)`."""
        bsz, seq_len, dim = x.shape
        flat_x = x.reshape(-1, dim)  # (N, D)

        router_logits = self.router(flat_x).float()  # (N, E)
        scores = torch.softmax(router_logits, dim=-1) if self.router_scoring == "softmax" else torch.sigmoid(router_logits)
        biased_scores = scores + self.expert_bias.unsqueeze(0)
        _, topk_idx = torch.topk(biased_scores, self.num_experts_per_tok, dim=-1)

        topk_weights = torch.gather(scores, dim=-1, index=topk_idx)
        topk_weights = (topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)).to(x.dtype)

        # Accumulate expert load counts for post-step ALF-LB bias updates
        counts = torch.bincount(topk_idx.reshape(-1), minlength=self.num_routed_experts).float()
        if self.training:
            with torch.no_grad():
                self._load_accum.add_(counts)
                self._load_steps.add_(1)

        # Tiny sequence-level balance loss (complementary to ALF-LB)
        f_i = counts / max(1, flat_x.shape[0] * self.num_experts_per_tok)
        p_i = (scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)).mean(dim=0)
        aux_loss = self.seq_aux_loss_alpha * self.num_routed_experts * (f_i * p_i).sum()
        self._last_aux_loss = aux_loss.detach()

        # Sort routed assignments once by expert id (eliminates 16 per-expert CUDA syncs)
        flat_expert_ids = topk_idx.reshape(-1)  # (N * K,)
        flat_weights = topk_weights.reshape(-1, 1)  # (N * K, 1)
        token_ids = (
            torch.arange(flat_x.shape[0], device=x.device)
            .unsqueeze(1)
            .expand(-1, self.num_experts_per_tok)
            .reshape(-1)
        )
        order = torch.argsort(flat_expert_ids, stable=False)
        sorted_tokens = token_ids[order]
        sorted_weights = flat_weights[order]
        sorted_x = flat_x[sorted_tokens]

        routed_out = torch.zeros_like(flat_x)
        counts_list = counts.long().tolist()
        offset = 0
        for expert_id, count in enumerate(counts_list):
            if count == 0:
                continue
            next_offset = offset + count
            expert_out = self.experts[expert_id](sorted_x[offset:next_offset])
            routed_out.index_add_(0, sorted_tokens[offset:next_offset], expert_out * sorted_weights[offset:next_offset])
            offset = next_offset

        out = routed_out.reshape(bsz, seq_len, dim)
        if self.shared_expert is not None:
            out = out + self.shared_expert(x)
        return out, aux_loss

    @torch.no_grad()
    def update_moe_biases(self) -> None:
        """Adjust `expert_bias` via ALF-LB based on accumulated routing counts."""
        if int(self._load_steps.item()) == 0:
            return
        avg_load = self._load_accum.mean()
        delta = torch.sign(avg_load - self._load_accum) * self.alf_lb_bias_update_rate
        self.expert_bias.add_(delta)
        self._load_accum.zero_()
        self._load_steps.zero_()

    def get_aux_metrics(self) -> dict[str, float]:
        """Return latest router load-balance statistics for logging."""
        metrics = {"bias_std": float(self.expert_bias.std().item())}
        if self._last_aux_loss is not None:
            metrics["seq_aux_loss"] = float(self._last_aux_loss.item())
        return metrics
