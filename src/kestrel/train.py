"""Unified Kestrel pretraining entrypoint (Tracks A Dense, B Hybrid, C MoE).

Designed to be short, readable, and Kaggle-ready (`python -m kestrel.train ...`):
- Optimizer/LR logic lives in `kestrel.utils.optim` (Muon + AdamW, WSD / Cosine).
- Checkpoint save/load & Hugging Face Hub publishing live in `kestrel.utils.checkpoint`.
- DagsHub/MLflow logging & GitHub snapshots live in `kestrel.tracking`.
- Training data is streamed directly from Hugging Face (`karpathy/climbmix-400b-shuffle`)
  with on-the-fly GPT-2 tokenization and packed `document_ids` (zero disk downloads).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from kestrel.tracking.mlflow_tracker import KestrelMLflowTracker, TrackingConfig
from kestrel.utils.checkpoint import (
    load_checkpoint,
    resolve_hf_repo_id,
    save_checkpoint,
    unwrap_model,
)
from kestrel.utils.optim import apply_lr_schedule, build_optimizers, compute_lr_multiplier
from kestrel.utils.text import get_gpt2_tokenizer

GPT2_EOT_TOKEN_ID = 50_256
GPT2_PADDED_VOCAB_SIZE = 50_304


# ---------------------------------------------------------------------------
# 1. Training Configuration & Streaming Data
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """JSON-serializable training recipe shared across all Kestrel candidate tracks."""

    # Model loading (HF AutoModelForCausalLM or explicit "module:Class")
    model_config_path: str | None = None
    model_class_path: str | None = None
    vocab_size: int = GPT2_PADDED_VOCAB_SIZE

    # Streaming dataset & sequence packing (zero multi-TB disk downloads)
    dataset_repo: str = "karpathy/climbmix-400b-shuffle"
    dataset_config: str | None = None
    dataset_split: str = "train"
    text_field: str = "text"
    shuffle_buffer_size: int = 2_048
    seq_len: int = 2048
    val_eval_batches: int = 16
    emit_document_ids: bool = True

    # Batching & optimization (Muon on 2D hidden weights + AdamW on embeddings/heads/norms)
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 1_000
    optimizer_type: str = "muon_adamw"  # "muon_adamw" | "adamw"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_weight_decay: float = 0.01
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str | None = "match_rms_adamw"
    adamw_lr: float = 3e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Schedule (WSD or Cosine)
    lr_schedule: str = "wsd"  # "wsd" | "cosine"
    warmup_steps: int = 100
    decay_steps: int = 100
    wsd_decay_type: str = "cosine"  # "cosine" | "linear" | "sqrt"
    min_lr_ratio: float = 0.1

    # Stability & multi-track auxiliary losses
    z_loss_weight: float = 1e-4  # Output logit stability (Peri-RMSNorm + QK-Norm + z-loss)
    aux_loss_weight: float = 1.0  # Optional MoE balance term returned by model
    mtp_loss_weight: float = 0.3  # Optional Multi-Token Prediction (t+2) head loss
    precision: str = "bf16"  # "bf16" | "fp16" | "fp32"
    compile_model: bool = False
    gradient_checkpointing: bool = False

    # Logging, checkpointing & tracking
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "artifacts/checkpoints"
    resume_from: str | None = None
    seed: int = 42
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        tracking_raw = raw.pop("tracking", {}) or {}
        if isinstance(raw.get("adamw_betas"), list):
            raw["adamw_betas"] = tuple(raw["adamw_betas"])
        return cls(**raw, tracking=TrackingConfig(**tracking_raw))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_gpt2_encoder() -> tuple[Any, int]:
    """Return `(encode_fn, eot_id)` using `tiktoken` (fast) or HF `gpt2` tokenizer."""
    try:
        enc = get_gpt2_tokenizer()
        return (lambda txt: enc.encode(txt, disallowed_special=())), int(enc.eot_token)
    except ImportError:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        eot_id = int(tok.eos_token_id if tok.eos_token_id is not None else GPT2_EOT_TOKEN_ID)
        return (lambda txt: tok.encode(txt, add_special_tokens=False)), eot_id


def stream_packed_batches(
    cfg: TrainConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    split_seed_offset: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Stream Hugging Face text, tokenize with GPT-2 BPE, and pack into `(x, y, doc_ids)` batches."""
    from datasets import load_dataset

    encode_fn, eot_id = _get_gpt2_encoder()

    def _packed_windows() -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        tok_buf: list[int] = []
        doc_buf: list[int] = []
        doc_id = 0
        epoch = 0
        chunk_len = cfg.seq_len + 1
        while True:
            ds = (
                load_dataset(cfg.dataset_repo, cfg.dataset_config, split=cfg.dataset_split, streaming=True)
                if cfg.dataset_config is not None
                else load_dataset(cfg.dataset_repo, split=cfg.dataset_split, streaming=True)
            )
            if world_size > 1 and hasattr(ds, "shard"):
                ds = ds.shard(num_shards=world_size, index=rank)
            if cfg.shuffle_buffer_size > 1 and hasattr(ds, "shuffle"):
                ds = ds.shuffle(seed=cfg.seed + epoch * 97 + split_seed_offset, buffer_size=cfg.shuffle_buffer_size)

            for row in ds:
                txt = row.get(cfg.text_field)
                if not isinstance(txt, str) or not txt.strip():
                    continue
                doc_tokens = [*encode_fn(txt), eot_id]
                tok_buf.extend(doc_tokens)
                doc_buf.extend([doc_id] * len(doc_tokens))
                doc_id += 1

                while len(tok_buf) >= chunk_len:
                    tok_chunk, doc_chunk = tok_buf[:chunk_len], doc_buf[: cfg.seq_len]
                    tok_buf, doc_buf = tok_buf[cfg.seq_len :], doc_buf[cfg.seq_len :]
                    base_doc = doc_chunk[0]
                    yield (
                        torch.tensor(tok_chunk[:-1], dtype=torch.long),
                        torch.tensor(tok_chunk[1:], dtype=torch.long),
                        torch.tensor([d - base_doc for d in doc_chunk], dtype=torch.long),
                    )
            epoch += 1

    window_iter = _packed_windows()
    while True:
        inps, lbls, docs = zip(*(next(window_iter) for _ in range(cfg.micro_batch_size)))
        yield torch.stack(inps), torch.stack(lbls), torch.stack(docs)


# ---------------------------------------------------------------------------
# 2. Architecture-Agnostic Model Loading & Unified Loss
# ---------------------------------------------------------------------------


def build_model_from_config(cfg: TrainConfig) -> tuple[nn.Module, dict[str, Any]]:
    """Load any HF-style or custom causal LM from `cfg.model_config_path`."""
    if not cfg.model_config_path:
        raise ValueError("Provide a pre-built `model` or set `TrainConfig.model_config_path`.")

    config_path = Path(cfg.model_config_path)
    raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))

    if cfg.model_class_path:
        mod_name, sym_name = cfg.model_class_path.split(":", 1)
        cls_or_fn = getattr(importlib.import_module(mod_name), sym_name)
        if hasattr(cls_or_fn, "from_config_dict"):
            model = cls_or_fn.from_config_dict(raw_cfg)
        elif hasattr(cls_or_fn, "config_class"):
            model = cls_or_fn(cls_or_fn.config_class(**raw_cfg))
        else:
            model = cls_or_fn(**raw_cfg)
        return model, raw_cfg

    from transformers import AutoConfig, AutoModelForCausalLM

    hf_cfg = AutoConfig.from_pretrained(str(config_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(hf_cfg, trust_remote_code=True)
    merged = hf_cfg.to_dict()
    for key in ("hf_repo_id", "hub_model_id", "blueprint", "track"):
        if key in raw_cfg and key not in merged:
            merged[key] = raw_cfg[key]
    return model, merged


def _extract_model_config(model: nn.Module, fallback_path: str | None = None) -> dict[str, Any]:
    raw = unwrap_model(model)
    if hasattr(raw, "config"):
        if hasattr(raw.config, "to_dict"):
            return dict(raw.config.to_dict())
        if isinstance(raw.config, dict):
            return dict(raw.config)
    if fallback_path and Path(fallback_path).is_file():
        return json.loads(Path(fallback_path).read_text(encoding="utf-8"))
    return {"model_class": raw.__class__.__name__}


def _accepts_document_ids(model: nn.Module) -> bool:
    params = inspect.signature(unwrap_model(model).forward).parameters
    return "document_ids" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def compute_unified_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    document_ids: torch.Tensor,
    cfg: TrainConfig,
    *,
    pass_doc_ids: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one forward pass and combine LM cross-entropy + z-loss + optional aux/MTP losses."""
    out = model(input_ids, document_ids=document_ids) if (cfg.emit_document_ids and pass_doc_ids) else model(input_ids)

    if isinstance(out, torch.Tensor):
        logits, lm_loss, aux_loss, mtp_loss = out, None, None, None
    elif isinstance(out, dict):
        logits, lm_loss, aux_loss, mtp_loss = out.get("logits"), out.get("loss"), out.get("aux_loss"), out.get("mtp_loss")
    else:
        logits = getattr(out, "logits", None)
        lm_loss = getattr(out, "loss", None)
        aux_loss = getattr(out, "aux_loss", None)
        mtp_loss = getattr(out, "mtp_loss", None)

    if lm_loss is None:
        if logits is None:
            raise ValueError("Model forward must return `logits` or `loss`.")
        lm_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

    total = lm_loss
    stats = {"lm": float(lm_loss.detach().item()), "z_loss": 0.0, "aux": 0.0, "mtp": 0.0}

    if cfg.z_loss_weight > 0.0 and logits is not None:
        z_term = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        total = total + cfg.z_loss_weight * z_term
        stats["z_loss"] = float(z_term.detach().item())
    if aux_loss is not None and cfg.aux_loss_weight > 0.0:
        total = total + cfg.aux_loss_weight * aux_loss
        stats["aux"] = float(aux_loss.detach().item())
    if mtp_loss is not None and cfg.mtp_loss_weight > 0.0:
        total = total + cfg.mtp_loss_weight * mtp_loss
        stats["mtp"] = float(mtp_loss.detach().item())

    return total, stats


@torch.no_grad()
def _post_step_hooks(model: nn.Module) -> dict[str, float]:
    """Run optional Track C ALF-LB bias updates and collect custom model telemetry."""
    raw = unwrap_model(model)
    for mod in raw.modules():
        if hasattr(mod, "update_moe_biases") and callable(mod.update_moe_biases):
            mod.update_moe_biases()
    if hasattr(raw, "get_aux_metrics") and callable(raw.get_aux_metrics):
        return {str(k): float(v) for k, v in (raw.get_aux_metrics() or {}).items() if isinstance(v, (int, float))}
    return {}


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    val_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: TrainConfig,
    device: torch.device,
    autocast_ctx: Any,
    *,
    pass_doc_ids: bool,
    world_size: int = 1,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_lm = 0.0
    for x, y, docs in val_batches:
        with autocast_ctx():
            _, stats = compute_unified_loss(
                model, x.to(device, non_blocking=True), y.to(device, non_blocking=True),
                docs.to(device, non_blocking=True), cfg, pass_doc_ids=pass_doc_ids,
            )
        total_lm += stats["lm"]

    val_loss = total_lm / max(len(val_batches), 1)
    if world_size > 1 and dist.is_initialized():
        t = torch.tensor([val_loss], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        val_loss = float(t.item())
    if was_training:
        model.train()
    return val_loss, math.exp(min(val_loss, 20.0))


# ---------------------------------------------------------------------------
# 3. Runtime Helpers & Unified Training Loop
# ---------------------------------------------------------------------------


def _setup_runtime(seed: int, precision: str) -> tuple[int, int, int, torch.device, Any, torch.amp.GradScaler | None]:
    """Initialize DDP (if launched via `torchrun`), RNG seeds, and AMP autocast."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank, world_size, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if device.type != "cuda" or precision == "fp32":
        return rank, world_size, local_rank, device, contextlib.nullcontext, None
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return rank, world_size, local_rank, device, (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)), None
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    return rank, world_size, local_rank, device, (lambda: torch.amp.autocast("cuda", dtype=torch.float16)), scaler


def train(cfg: TrainConfig, model: nn.Module | None = None) -> dict[str, Any]:
    """Run Kestrel's unified pretraining loop with MLflow/DagsHub + HF Hub publishing."""
    rank, world_size, local_rank, device, autocast_ctx, scaler = _setup_runtime(cfg.seed, cfg.precision)
    is_main = rank == 0

    # 1. Build model & auto-fill blueprint/track/hf_repo_id from model JSON if unset
    model, model_cfg = (build_model_from_config(cfg) if model is None else (model, _extract_model_config(model, cfg.model_config_path)))
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model = model.to(device)
    pass_doc_ids = _accepts_document_ids(model)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # 2. Optimizers & optional mid-run checkpoint resume
    optimizers = build_optimizers(model, cfg)
    start_step, tokens_seen, best_val_loss = (
        load_checkpoint(cfg.resume_from, model, optimizers, scaler, device)
        if cfg.resume_from
        else (0, 0, float("inf"))
    )

    # 3. Streamed packed train stream & fixed deterministic validation batches
    train_iter = stream_packed_batches(cfg, rank=rank, world_size=world_size, split_seed_offset=0)
    val_iter = stream_packed_batches(cfg, rank=rank, world_size=world_size, split_seed_offset=10_000)
    val_batches = [next(val_iter) for _ in range(max(0, cfg.val_eval_batches))]

    # 4. Prepare MLflow/DagsHub tracking config (inheriting per-model hf_repo_id/blueprint/track)
    tcfg = TrackingConfig(**asdict(cfg.tracking))
    tcfg.enabled = cfg.tracking.enabled and is_main
    tcfg.hf_repo_id = resolve_hf_repo_id(tcfg.hf_repo_id, model_config=model_cfg)
    if tcfg.blueprint == "custom" and isinstance(model_cfg.get("blueprint"), str):
        tcfg.blueprint = model_cfg["blueprint"]
    if tcfg.track == "track_a_dense" and isinstance(model_cfg.get("track"), str):
        tcfg.track = model_cfg["track"]

    tokens_per_step = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accum_steps * world_size
    t_run_start = time.perf_counter()
    final_train_loss, best_val_ppl = float("nan"), float("nan")
    throughputs: list[float] = []
    last_hf_dir: Path | None = None

    with KestrelMLflowTracker(tcfg) as tracker:
        train_cfg_dict = cfg.to_dict()
        param_stats = tracker.log_model_summary(model)
        tracker.log_configs(model_config=model_cfg, train_config=train_cfg_dict)
        run_ckpt_root = Path(cfg.output_dir) / tracker.run_name

        if is_main:
            print(
                f"[kestrel.train] Run={tracker.run_name} | Device={device} | "
                f"Params={param_stats['total']/1e6:.2f}M (active={param_stats['active']/1e6:.2f}M) | "
                f"Opt={list(optimizers)} | Tok/step={tokens_per_step:,}"
            )

        model.train()
        for step in range(start_step + 1, cfg.max_steps + 1):
            t_step = time.perf_counter()
            current_lrs = apply_lr_schedule(optimizers, compute_lr_multiplier(step - 1, cfg))
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)

            acc = {"total": 0.0, "lm": 0.0, "z_loss": 0.0, "aux": 0.0, "mtp": 0.0}
            for micro_idx in range(cfg.grad_accum_steps):
                x, y, docs = next(train_iter)
                sync_ctx = (
                    model.no_sync()
                    if (world_size > 1 and isinstance(model, DDP) and micro_idx < cfg.grad_accum_steps - 1)
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    with autocast_ctx():
                        total_loss, stats = compute_unified_loss(
                            model, x.to(device, non_blocking=True), y.to(device, non_blocking=True),
                            docs.to(device, non_blocking=True), cfg, pass_doc_ids=pass_doc_ids,
                        )
                        scaled = total_loss / cfg.grad_accum_steps
                    (scaler.scale(scaled) if scaler is not None else scaled).backward()

                acc["total"] += float(total_loss.detach().item()) / cfg.grad_accum_steps
                for k in ("lm", "z_loss", "aux", "mtp"):
                    acc[k] += stats[k] / cfg.grad_accum_steps

            if scaler is not None:
                for opt in optimizers.values():
                    scaler.unscale_(opt)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.grad is not None],
                    cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf"),
                ).item()
            )
            for opt in optimizers.values():
                scaler.step(opt) if scaler is not None else opt.step()
            if scaler is not None:
                scaler.update()

            hook_metrics = _post_step_hooks(model)
            tokens_seen += tokens_per_step
            final_train_loss = acc["lm"]
            step_sec = max(1e-6, time.perf_counter() - t_step)
            tok_per_sec = tokens_per_step / step_sec
            throughputs.append(tok_per_sec)
            wall_sec = time.perf_counter() - t_run_start
            gpu_hours = (wall_sec * world_size) / 3600.0

            if is_main and (step == 1 or step % cfg.log_interval == 0 or step == cfg.max_steps):
                metrics: dict[str, float] = {
                    "train/loss": acc["total"],
                    "train/lm_loss": acc["lm"],
                    "train/ppl": math.exp(min(acc["lm"], 20.0)),
                    "train/grad_norm": grad_norm,
                    "perf/tokens_per_sec": tok_per_sec,
                    "perf/step_ms": step_sec * 1000.0,
                    "progress/tokens_seen": float(tokens_seen),
                    "progress/wall_clock_sec": wall_sec,
                    "progress/gpu_hours": gpu_hours,
                    **current_lrs,
                    **hook_metrics,
                }
                if cfg.z_loss_weight > 0.0:
                    metrics["train/z_loss"] = acc["z_loss"]
                if acc["aux"] > 0.0:
                    metrics["train/aux_loss"] = acc["aux"]
                if acc["mtp"] > 0.0:
                    metrics["train/mtp_loss"] = acc["mtp"]
                if device.type == "cuda":
                    metrics["perf/peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
                tracker.log_metrics(metrics, step=step)
                print(
                    f"step {step:>5d}/{cfg.max_steps} | lm={acc['lm']:.4f} | ppl={metrics['train/ppl']:.2f} "
                    f"| gnorm={grad_norm:.2f} | tok/s={tok_per_sec:,.0f} | seen={tokens_seen:,}"
                )

            if cfg.eval_interval > 0 and val_batches and (step % cfg.eval_interval == 0 or step == cfg.max_steps):
                val_loss, val_ppl = _evaluate(model, val_batches, cfg, device, autocast_ctx, pass_doc_ids=pass_doc_ids, world_size=world_size)
                if is_main:
                    tracker.log_metrics({"val/loss": val_loss, "val/ppl": val_ppl, "progress/tokens_seen": float(tokens_seen), "progress/gpu_hours": gpu_hours}, step=step)
                    print(f"  [eval] step {step:>5d} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")
                    if val_loss < best_val_loss:
                        best_val_loss, best_val_ppl = val_loss, val_ppl
                        last_hf_dir = save_checkpoint(run_ckpt_root / "best", model, optimizers, scaler, step=step, tokens_seen=tokens_seen, best_val_loss=best_val_loss, model_config=model_cfg, train_config=train_cfg_dict)

            if is_main and cfg.save_interval > 0 and (step % cfg.save_interval == 0 or step == cfg.max_steps):
                last_hf_dir = save_checkpoint(run_ckpt_root / f"step_{step:07d}", model, optimizers, scaler, step=step, tokens_seen=tokens_seen, best_val_loss=best_val_loss, model_config=model_cfg, train_config=train_cfg_dict)

        # 5. Final MLflow model logging, HF Hub push & GitHub snapshot
        wall_sec = time.perf_counter() - t_run_start
        final_metrics = {
            "steps_completed": cfg.max_steps,
            "tokens_seen": tokens_seen,
            "final_train_loss": round(final_train_loss, 6) if math.isfinite(final_train_loss) else None,
            "best_val_loss": round(best_val_loss, 6) if math.isfinite(best_val_loss) else None,
            "best_val_ppl": round(best_val_ppl, 4) if math.isfinite(best_val_ppl) else None,
            "mean_tokens_per_sec": round(float(np.mean(throughputs)), 2) if throughputs else 0.0,
            "wall_clock_sec": round(wall_sec, 2),
            "gpu_hours": round((wall_sec * world_size) / 3600.0, 6),
        }

        model_info, hf_hub_info, snapshot_dir = None, None, None
        if is_main:
            hf_arg = last_hf_dir if (last_hf_dir and last_hf_dir.name == "hf_checkpoint") else None
            model_info = tracker.log_model(
                model,
                model_name=f"{tcfg.blueprint}_{tcfg.track}",
                step=cfg.max_steps,
                extra_metadata={"stage": tcfg.stage, "track": tcfg.track, "blueprint": tcfg.blueprint, "tokens_seen": tokens_seen},
                hf_checkpoint_dir=hf_arg,
            )
            hf_hub_info = tracker.publish_to_hub(
                model,
                model_config=model_cfg,
                train_config=train_cfg_dict,
                final_metrics=final_metrics,
                hf_checkpoint_dir=hf_arg,
            )
            snapshot_dir = tracker.export_github_snapshot(model_cfg, train_cfg_dict, final_metrics)
            print(f"[kestrel.train] Finished run {tracker.run_name} (MLflow run_id={tracker.run_id})")
            if hf_hub_info:
                print(f"[kestrel.train] Pushed HF Hub checkpoint  : {hf_hub_info['repo_url']} ({hf_hub_info['commit_sha'][:8]})")
            if snapshot_dir:
                print(f"[kestrel.train] GitHub snapshot exported to: {snapshot_dir}")

        return {
            "run_name": tracker.run_name,
            "mlflow_run_id": tracker.run_id,
            "tracking_uri": tracker.tracking_uri,
            "model_uri": getattr(model_info, "model_uri", None),
            "hf_hub": hf_hub_info,
            "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
            "parameters": param_stats,
            "metrics": final_metrics,
        }


# ---------------------------------------------------------------------------
# 4. Concise CLI Entrypoint (`python -m kestrel.train ...`)
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m kestrel.train", description="Unified Kestrel pretraining CLI.")
    p.add_argument("--train-config", type=str, default=None, help="Training config JSON (e.g. configs/train/climbmix_pretrain.json)")
    p.add_argument("--model-config", type=str, default=None, help="Model config JSON (e.g. configs/model/kestrel_350m_dense.json)")
    p.add_argument("--model-class", type=str, default=None, help="Optional 'package.module:ClassName'")
    p.add_argument("--dataset-repo", type=str, default=None, help="Hugging Face streaming dataset repo")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--optimizer", choices=["muon_adamw", "adamw"], default=None)
    p.add_argument("--lr-schedule", choices=["wsd", "cosine"], default=None)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    # Tracking & Hub flags
    p.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI (HTTP, DagsHub, or sqlite:///)")
    p.add_argument("--dagshub-repo", type=str, default=None, help="DagsHub shorthand 'owner/repo' (reads DAGSHUB_TOKEN)")
    p.add_argument("--experiment", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--stage", type=str, default=None)
    p.add_argument("--track", type=str, default=None)
    p.add_argument("--blueprint", type=str, default=None)
    p.add_argument("--register-model", type=str, default=None)
    p.add_argument("--hf-repo-id", type=str, default=None, help="HF Hub model repo ('owner/model'); overrides model JSON & enables push")
    p.add_argument("--push-to-hub", action="store_true", help="Push final checkpoint to HF Hub")
    p.add_argument("--no-push-to-hub", action="store_true", help="Disable HF Hub push")
    p.add_argument("--hf-revision", type=str, default=None, help="Target HF Hub branch")
    p.add_argument("--hf-path-in-repo", type=str, default=None, help="Optional subfolder inside HF Hub repo")
    p.add_argument("--hf-private", action="store_true", help="Create HF Hub repo as private")
    p.add_argument("--no-tracking", action="store_true")
    p.add_argument("--no-log-model", action="store_true", help="Skip uploading weights to MLflow (recommended with --push-to-hub)")
    p.add_argument("--no-github-snapshot", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = TrainConfig.from_json(args.train_config) if args.train_config else TrainConfig()

    for attr, arg_val in (
        ("model_config_path", args.model_config),
        ("model_class_path", args.model_class),
        ("dataset_repo", args.dataset_repo),
        ("max_steps", args.max_steps),
        ("seq_len", args.seq_len),
        ("micro_batch_size", args.micro_batch_size),
        ("grad_accum_steps", args.grad_accum_steps),
        ("optimizer_type", args.optimizer),
        ("lr_schedule", args.lr_schedule),
        ("precision", args.precision),
        ("output_dir", args.output_dir),
        ("resume_from", args.resume_from),
    ):
        if arg_val is not None:
            setattr(cfg, attr, arg_val)

    for attr, arg_val in (
        ("tracking_uri", args.tracking_uri),
        ("dagshub_repo", args.dagshub_repo),
        ("experiment_name", args.experiment),
        ("run_name", args.run_name),
        ("stage", args.stage),
        ("track", args.track),
        ("blueprint", args.blueprint),
        ("registered_model_name", args.register_model),
        ("hf_revision", args.hf_revision),
        ("hf_path_in_repo", args.hf_path_in_repo),
    ):
        if arg_val is not None:
            setattr(cfg.tracking, attr, arg_val)

    if args.hf_repo_id is not None:
        cfg.tracking.hf_repo_id = args.hf_repo_id
        if not args.no_push_to_hub:
            cfg.tracking.push_to_hub = True
    if args.push_to_hub:
        cfg.tracking.push_to_hub = True
    if args.no_push_to_hub:
        cfg.tracking.push_to_hub = False
    if args.hf_private:
        cfg.tracking.hf_private = True
    if args.no_tracking:
        cfg.tracking.enabled = False
    if args.no_log_model:
        cfg.tracking.log_model_to_mlflow = False
    if args.no_github_snapshot:
        cfg.tracking.export_github_snapshot = False

    train(cfg)


if __name__ == "__main__":
    main()
