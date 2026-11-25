"""Gradient flow auditing, update ratios (`||dW||/||W||`), and trajectory straightness."""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn

from kestrel.utils.train.checkpoint import unwrap_model


@torch.no_grad()
def audit_layer_gradients(model: nn.Module) -> dict[str, object]:
    """Inspect gradient norms by layer & component and detect dead or NaN/Inf gradients."""
    raw = unwrap_model(model)
    per_layer: dict[int, float] = defaultdict(float)
    per_group: dict[str, float] = defaultdict(float)
    dead_params: list[str] = []
    nonfinite_params: list[str] = []
    total_sq = 0.0

    for name, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            dead_params.append(name)
            continue
        g = p.grad.detach().float()
        if not torch.isfinite(g).all():
            nonfinite_params.append(name)
            continue
        gnorm_sq = float(g.pow(2).sum().item())
        if gnorm_sq == 0.0:
            dead_params.append(name)
        total_sq += gnorm_sq

        # Group by component tag
        low = name.lower()
        if "embed" in low or "lm_head" in low:
            group = "embed_head"
        elif "mtp" in low:
            group = "mtp"
        elif "mhc" in low:
            group = "mhc"
        elif "router" in low or "experts" in low:
            group = "moe"
        elif "norm" in low:
            group = "norms"
        elif "mixer" in low or "attn" in low:
            group = "mixer"
        elif "ffn" in low:
            group = "ffn"
        else:
            group = "other"
        per_group[group] += gnorm_sq

        # Extract layer index if name follows `model.layers.<idx>.*`
        parts = name.split(".")
        for i, part in enumerate(parts[:-1]):
            if part == "layers" and parts[i + 1].isdigit():
                per_layer[int(parts[i + 1])] += gnorm_sq
                break

    layer_norms = {k: math.sqrt(v) for k, v in sorted(per_layer.items())}
    group_norms = {k: math.sqrt(v) for k, v in sorted(per_group.items())}
    depth_ratio = None
    if len(layer_norms) >= 2:
        first_idx, last_idx = min(layer_norms), max(layer_norms)
        depth_ratio = layer_norms[first_idx] / max(layer_norms[last_idx], 1e-12)

    return {
        "total_grad_norm": math.sqrt(total_sq),
        "depth_grad_ratio": depth_ratio,
        "layer_norms": layer_norms,
        "group_norms": group_norms,
        "dead_param_count": len(dead_params),
        "dead_params_sample": dead_params[:10],
        "nonfinite_param_count": len(nonfinite_params),
        "nonfinite_params": nonfinite_params[:10],
    }


class GradientHealthMonitor:
    """Lightweight tracker for per-optimizer update ratios, gradient cosine, and trajectory straightness."""

    def __init__(
        self,
        model: nn.Module,
        optimizers: dict[str, torch.optim.Optimizer],
        *,
        straightness_window: int = 50,
    ) -> None:
        self.model = unwrap_model(model)
        self.optimizers = optimizers
        self.straightness_window = max(2, straightness_window)

        self._prev_weights: dict[str, list[torch.Tensor]] = {}
        self._window_anchor: list[torch.Tensor] | None = None
        self._window_path_len: float = 0.0
        self._window_steps: int = 0
        self._prev_grad_vec: torch.Tensor | None = None

    @staticmethod
    def _sample_params(params: list[nn.Parameter], max_tensors: int = 16) -> list[nn.Parameter]:
        """Select an evenly spaced subset of parameters so monitoring overhead stays <0.5%."""
        if len(params) <= max_tensors:
            return params
        step = max(1, len(params) // max_tensors)
        return params[::step][:max_tensors]

    @torch.no_grad()
    def before_step(self) -> dict[str, float]:
        """Snapshot sampled weights and compute macro-step gradient cosine similarity."""
        metrics: dict[str, float] = {}

        # Snapshot weights per optimizer for `||dW|| / ||W||` update-ratio tracking
        self._prev_weights.clear()
        for name, opt in self.optimizers.items():
            all_p = [p for g in opt.param_groups for p in g["params"] if p.requires_grad]
            self._prev_weights[name] = [p.detach().float().clone() for p in self._sample_params(all_p)]

        # Macro-step gradient direction cosine similarity (sampled params for minimal memory)
        sampled = self._sample_params([p for p in self.model.parameters() if p.grad is not None])
        if sampled:
            gvec = torch.cat([p.grad.detach().float().reshape(-1) for p in sampled])
            if self._prev_grad_vec is not None and self._prev_grad_vec.shape == gvec.shape:
                denom = (gvec.norm() * self._prev_grad_vec.norm()).clamp(min=1e-12)
                metrics["grad/step_cosine"] = float(torch.dot(gvec, self._prev_grad_vec).item() / denom.item())
            self._prev_grad_vec = gvec

        if self._window_anchor is None:
            anchor_p = self._sample_params([p for p in self.model.parameters() if p.requires_grad])
            self._window_anchor = [p.detach().float().clone() for p in anchor_p]
            self._window_path_len = 0.0
            self._window_steps = 0

        return metrics

    @torch.no_grad()
    def after_step(self) -> dict[str, float]:
        """Compute per-optimizer update ratios `||dW||/||W||` and trajectory straightness."""
        metrics: dict[str, float] = {}
        for name, opt in self.optimizers.items():
            prev = self._prev_weights.get(name)
            if not prev:
                continue
            all_p = [p for g in opt.param_groups for p in g["params"] if p.requires_grad]
            cur = self._sample_params(all_p)
            delta_sq, weight_sq = 0.0, 0.0
            for p_old, p_new in zip(prev, cur):
                diff = p_new.detach().float() - p_old
                delta_sq += float(diff.pow(2).sum().item())
                weight_sq += float(p_old.pow(2).sum().item())
            metrics[f"grad/update_ratio_{name}"] = math.sqrt(delta_sq) / max(math.sqrt(weight_sq), 1e-12)

        # Trajectory straightness over `straightness_window` steps
        anchor_p = self._sample_params([p for p in self.model.parameters() if p.requires_grad])
        if self._window_anchor is not None and len(anchor_p) == len(self._window_anchor):
            # Step displacement across all optimizers combined
            step_disp_sq = sum(
                float((p_new.detach().float() - p_old).pow(2).sum().item())
                for prev_list, (_, opt) in zip(self._prev_weights.values(), self.optimizers.items())
                for p_old, p_new in zip(
                    prev_list,
                    self._sample_params([p for g in opt.param_groups for p in g["params"] if p.requires_grad]),
                )
            )
            self._window_path_len += math.sqrt(step_disp_sq)
            self._window_steps += 1
            if self._window_steps >= self.straightness_window:
                net_disp_sq = sum(
                    float((p.detach().float() - a).pow(2).sum().item())
                    for p, a in zip(anchor_p, self._window_anchor)
                )
                metrics["plateau/trajectory_straightness"] = math.sqrt(net_disp_sq) / max(self._window_path_len, 1e-12)
                self._window_anchor = [p.detach().float().clone() for p in anchor_p]
                self._window_path_len = 0.0
                self._window_steps = 0

        return metrics
