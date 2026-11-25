"""Optimizer construction (Muon + AdamW) and LR schedules (WSD / Cosine) for Kestrel.

- **Muon** (native `torch.optim.Muon`, requires PyTorch >= 2.9) optimizes 2D hidden-layer weight matrices.
- **AdamW** optimizes input embeddings, the LM head (tied or untied), 1D norms, biases, and scalars.
- **Warmup-Stable-Decay (WSD)** and **Cosine** schedules are applied to both optimizers in sync.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
import torch.nn as nn

from kestrel.utils.checkpoint import unwrap_model


def _embedding_and_head_param_ids(model: nn.Module) -> set[int]:
    """Return `id(p)` for input embeddings and output LM heads so Muon never touches them."""
    excluded: set[int] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding) or name.endswith(("lm_head", "output_head", "embed_out")):
            excluded.update(id(p) for p in mod.parameters(recurse=False))
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        if hasattr(model, getter):
            emb = getattr(model, getter)()
            if isinstance(emb, nn.Module):
                excluded.update(id(p) for p in emb.parameters(recurse=False))
    return excluded


def build_optimizers(model: nn.Module, cfg: Any) -> dict[str, torch.optim.Optimizer]:
    """Partition parameters into Muon (2D hidden weights) and AdamW (embeddings, heads, 1D params)."""
    raw = unwrap_model(model)
    emb_head_ids = _embedding_and_head_param_ids(raw)

    muon_params: list[nn.Parameter] = []
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []
    seen: set[int] = set()

    for name, p in raw.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))

        is_emb_or_head = id(p) in emb_head_ids
        is_1d_or_norm = (
            p.ndim < 2
            or name.endswith(".bias")
            or any(tag in name.lower() for tag in ("norm", "lambda", "gate_scalar"))
        )

        if cfg.optimizer_type == "muon_adamw" and p.ndim == 2 and not is_emb_or_head and not is_1d_or_norm:
            muon_params.append(p)
        elif is_1d_or_norm or is_emb_or_head:
            adamw_no_decay.append(p)
        else:
            adamw_decay.append(p)

    optimizers: dict[str, torch.optim.Optimizer] = {}

    if cfg.optimizer_type == "muon_adamw" and muon_params:
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                f"torch.optim.Muon requires PyTorch >= 2.9 (found {torch.__version__}). "
                "Upgrade with `pip install -U torch` or set optimizer_type='adamw'."
            )
        opt_muon = torch.optim.Muon(
            muon_params,
            lr=cfg.muon_lr,
            weight_decay=cfg.muon_weight_decay,
            momentum=cfg.muon_momentum,
            nesterov=cfg.muon_nesterov,
            ns_steps=cfg.muon_ns_steps,
            adjust_lr_fn=cfg.muon_adjust_lr_fn,  # None -> original Muon scaling (pair with lr ~0.02)
        )
        for group in opt_muon.param_groups:
            group["base_lr"] = cfg.muon_lr
        optimizers["muon"] = opt_muon

    adamw_groups: list[dict[str, Any]] = []
    if adamw_decay:
        adamw_groups.append({"params": adamw_decay, "weight_decay": cfg.adamw_weight_decay, "base_lr": cfg.adamw_lr})
    if adamw_no_decay:
        adamw_groups.append({"params": adamw_no_decay, "weight_decay": 0.0, "base_lr": cfg.adamw_lr})

    if adamw_groups:
        use_fused = torch.cuda.is_available() and "fused" in inspect.signature(torch.optim.AdamW).parameters
        optimizers["adamw"] = torch.optim.AdamW(
            adamw_groups,
            lr=cfg.adamw_lr,
            betas=tuple(cfg.adamw_betas),
            eps=cfg.adamw_eps,
            fused=use_fused,
        )

    if not optimizers:
        raise ValueError("Model has no trainable parameters.")
    return optimizers


def compute_lr_multiplier(step: int, cfg: Any) -> float:
    """Return LR scale factor in `[min_lr_ratio, 1.0]` for Warmup-Stable-Decay (WSD) or Cosine."""
    step = max(0, step)
    max_steps = max(1, cfg.max_steps)
    warmup = min(max(0, cfg.warmup_steps), max_steps)

    # Phase 1: linear warmup
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)

    if cfg.lr_schedule == "cosine":
        progress = min(1.0, max(0.0, (step - warmup) / max(1, max_steps - warmup)))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Phase 2 & 3 (WSD): constant plateau followed by cooldown decay
    decay = min(max(0, cfg.decay_steps), max(0, max_steps - warmup))
    stable_end = max_steps - decay
    if step < stable_end or decay == 0:
        return 1.0

    progress = min(1.0, max(0.0, (step - stable_end) / max(1, decay)))
    if cfg.wsd_decay_type == "linear":
        factor = 1.0 - progress
    elif cfg.wsd_decay_type == "sqrt":
        factor = 1.0 - math.sqrt(progress)
    else:
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * factor


def apply_lr_schedule(optimizers: dict[str, torch.optim.Optimizer], mult: float) -> dict[str, float]:
    """Apply multiplier `mult` to every optimizer's `base_lr` and return current LR metrics."""
    lrs: dict[str, float] = {}
    for name, opt in optimizers.items():
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * mult
        lrs[f"lr/{name}"] = float(opt.param_groups[0]["lr"])
    return lrs
