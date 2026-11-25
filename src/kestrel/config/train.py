"""Training configuration schema and CLI `key=value` override utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kestrel.config.tracking import TrackingConfig


@dataclass
class TrainConfig:
    """Training recipe shared by every Kestrel candidate. All fields can be set from JSON."""

    model_config_path: str | None = None  # Path to a model config JSON (HF AutoConfig format)

    # Streaming dataset & sequence packing
    dataset_repo: str = "karpathy/climbmix-400b-shuffle"
    dataset_config: str | None = None
    dataset_split: str = "train"
    text_field: str = "text"
    shuffle_buffer_size: int = 2_048
    seq_len: int = 2048
    val_eval_batches: int = 16

    # Batching & Muon (2D hidden weights) + AdamW (embeddings / heads / norms)
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 1_000
    optimizer_type: str = "muon_adamw"  # "muon_adamw" | "adamw"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_weight_decay: float = 0.01
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str | None = None  # None = original Muon scaling (pair with lr ~0.02)
    adamw_lr: float = 3e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # LR schedule (WSD or Cosine)
    lr_schedule: str = "wsd"
    warmup_steps: int = 100
    decay_steps: int = 100
    wsd_decay_type: str = "cosine"  # "cosine" | "linear" | "sqrt"
    min_lr_ratio: float = 0.1

    # Auxiliary losses, precision & speed ("auto" picks bf16 on Ampere+, fp16 on T4, fp32 on CPU)
    z_loss_weight: float = 1e-4
    aux_loss_weight: float = 1.0  # MoE balance term returned by the model (if any)
    mtp_loss_weight: float = 0.3  # Multi-token-prediction term returned by the model (if any)
    precision: str = "auto"  # "auto" | "bf16" | "fp16" | "fp32"
    gradient_checkpointing: bool = False
    compile_model: bool = False  # Enable torch.compile() on Linux/Kaggle for kernel fusion
    loss_chunk_size: int = 512  # Chunked lm_head + CE + z-loss (0 disables chunking)
    prefetch_batches: int = 4  # Background thread prefetch depth (0 disables async prefetch)

    # Logging, checkpointing, capacity/plateau metrics & tracking
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    track_capacity_metrics: bool = True
    plateau_delta_per_doubling: float = 0.02
    output_dir: str = "artifacts/checkpoints"
    resume_from: str | None = None
    seed: int = 42
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        tracking = TrackingConfig(**(raw.pop("tracking", None) or {}))
        if isinstance(raw.get("adamw_betas"), list):
            raw["adamw_betas"] = tuple(raw["adamw_betas"])
        return cls(**raw, tracking=tracking)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_overrides(cfg: TrainConfig, overrides: list[str]) -> TrainConfig:
    """Apply `key=value` overrides; use dots for nested fields (e.g. `tracking.hf_repo_id=me/model`)."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        *parents, leaf = key.strip().split(".")
        target: Any = cfg
        for name in parents:
            target = getattr(target, name)
        if not hasattr(target, leaf):
            raise KeyError(f"Unknown config field '{key}'")
        try:
            value = json.loads(raw)  # numbers, true/false, null, lists
        except json.JSONDecodeError:
            value = raw  # plain strings such as "owner/repo"
        setattr(target, leaf, value)
    return cfg
