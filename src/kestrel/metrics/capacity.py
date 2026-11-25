"""Representation capacity and stratified validation loss diagnostics for 350M-700M models."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kestrel.models.ffn import MoESwiGLUFFN, ReLU2FFN
from kestrel.utils.train.checkpoint import unwrap_model


@torch.no_grad()
def effective_rank(hidden_states: torch.Tensor, max_samples: int = 1024) -> tuple[float, float]:
    """Compute Roy-Vetterli spectral entropy effective rank `(eRank, eRank / d)` of `(..., D)` states."""
    flat = hidden_states.detach().float().reshape(-1, hidden_states.shape[-1])
    if flat.shape[0] > max_samples:
        idx = torch.linspace(0, flat.shape[0] - 1, max_samples, device=flat.device).long()
        flat = flat[idx]
    # Center before SVD so mean shift does not dominate singular values
    flat = flat - flat.mean(dim=0, keepdim=True)
    svals = torch.linalg.svdvals(flat)
    total = svals.sum()
    if float(total.item()) <= 1e-12:
        return 0.0, 0.0
    probs = (svals / total).clamp(min=1e-12)
    entropy = -(probs * probs.log()).sum()
    erank = float(torch.exp(entropy).item())
    dim = int(hidden_states.shape[-1])
    return erank, erank / max(dim, 1)


@torch.no_grad()
def stratified_token_losses(
    token_losses: torch.Tensor,
    labels: torch.Tensor,
    *,
    common_token_threshold: int = 1000,
    early_range: tuple[int, int] = (64, 256),
    late_range: tuple[int, int] = (1024, 2048),
) -> dict[str, float]:
    """Break down per-token `(B, T)` CE loss into common, rare/tail, and long-context buckets."""
    valid = labels >= 0
    if not valid.any():
        return {}

    metrics: dict[str, float] = {}
    common_mask = valid & (labels < common_token_threshold)
    tail_mask = valid & (labels >= common_token_threshold)
    if common_mask.any():
        metrics["val/loss_common"] = float(token_losses[common_mask].mean().item())
    if tail_mask.any():
        metrics["val/loss_tail"] = float(token_losses[tail_mask].mean().item())

    seq_len = token_losses.shape[1]
    e0, e1 = early_range
    l0, l1 = late_range
    if seq_len >= l0 + 16:
        early_slice = token_losses[:, e0 : min(e1, seq_len)]
        early_valid = valid[:, e0 : min(e1, seq_len)]
        late_slice = token_losses[:, l0 : min(l1, seq_len)]
        late_valid = valid[:, l0 : min(l1, seq_len)]
        if early_valid.any() and late_valid.any():
            early_loss = float(early_slice[early_valid].mean().item())
            late_loss = float(late_slice[late_valid].mean().item())
            metrics["val/loss_early"] = early_loss
            metrics["val/loss_late"] = late_loss
            # Positive gain means the model uses longer context to predict better
            metrics["val/long_context_gain"] = early_loss - late_loss
    return metrics


@torch.no_grad()
def measure_model_capacity(
    model: nn.Module,
    input_ids: torch.Tensor,
    document_ids: torch.Tensor | None,
    autocast_ctx: Any,
) -> dict[str, float]:
    """Measure hidden effective rank, ReLU^2 dead-neuron fraction, and MoE routing entropy on 1 batch."""
    raw = unwrap_model(model)
    was_training = raw.training
    raw.eval()

    relu2_active_counts: list[torch.Tensor] = []
    moe_entropies: list[float] = []
    handles: list[Any] = []

    def _relu2_hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        # Track fraction of hidden units that fired at least once across the batch
        fired = (out.detach() > 0).reshape(-1, out.shape[-1]).any(dim=0)
        relu2_active_counts.append(fired.float())

    def _moe_hook(mod: MoESwiGLUFFN, inp: tuple[torch.Tensor, ...], _out: Any) -> None:
        x = inp[0].detach().reshape(-1, inp[0].shape[-1])
        logits = mod.router(x).float()
        scores = torch.softmax(logits, dim=-1) if mod.router_scoring == "softmax" else torch.sigmoid(logits)
        _, topk_idx = torch.topk(scores + mod.expert_bias.unsqueeze(0), mod.num_experts_per_tok, dim=-1)
        counts = torch.bincount(topk_idx.reshape(-1), minlength=mod.num_routed_experts).float()
        probs = (counts / counts.sum().clamp(min=1.0)).clamp(min=1e-12)
        entropy = float(-(probs * probs.log()).sum().item())
        max_ent = math.log(max(mod.num_routed_experts, 2))
        moe_entropies.append(entropy / max_ent)

    for mod in raw.modules():
        if isinstance(mod, ReLU2FFN):
            handles.append(mod.act.register_forward_hook(_relu2_hook))
        elif isinstance(mod, MoESwiGLUFFN):
            handles.append(mod.register_forward_hook(_moe_hook))

    metrics: dict[str, float] = {}
    try:
        with autocast_ctx():
            try:
                out = raw(input_ids, document_ids=document_ids, use_cache=False, compute_logits=False)
            except TypeError:
                out = raw(input_ids, document_ids=document_ids)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is not None:
            erank, erank_ratio = effective_rank(hidden)
            metrics["capacity/erank"] = erank
            metrics["capacity/erank_ratio"] = erank_ratio
        if relu2_active_counts:
            dead_frac = 1.0 - float(torch.cat(relu2_active_counts).mean().item())
            metrics["capacity/dead_relu2_frac"] = dead_frac
        if moe_entropies:
            metrics["capacity/moe_routing_entropy_norm"] = sum(moe_entropies) / len(moe_entropies)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            raw.train()

    return metrics
