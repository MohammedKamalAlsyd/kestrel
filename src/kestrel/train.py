"""Unified Kestrel pretraining entrypoint (Tracks A Dense, B Hybrid, C MoE).

Run on Kaggle (1 or 2 GPUs) with everything driven by JSON configs plus optional overrides:

    python -m kestrel.train --train-config configs/train/climbmix_pretrain.json \
        --model-config configs/model/kestrel_350m_dense.json --set max_steps=500

    torchrun --nproc_per_node=2 -m kestrel.train --train-config ... --model-config ...

Related modules:
- `kestrel.config`            TrainConfig, TrackingConfig, model JSON loading, CLI overrides
- `kestrel.models`            KestrelConfig, KestrelForCausalLM, KestrelModelBuilder
- `kestrel.metrics`           Plateau estimation, capacity eRank, stratified val loss, grad health
- `kestrel.utils.train`       device, dtype (bf16/fp16/fp32), DDP, seeds, Muon/AdamW, checkpoints
- `kestrel.utils.data`        GPT-2 tokenization and text helpers
- `kestrel.tracking`          DagsHub MLflow logging and GitHub run snapshots
Training text is streamed from Hugging Face and tokenized on the fly (no disk downloads).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import queue
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from kestrel.config import (
    TrackingConfig,
    TrainConfig,
    apply_overrides,
    extract_model_metadata,
    load_model_config,
)
from kestrel.metrics import (
    GradientHealthMonitor,
    PlateauTracker,
    measure_model_capacity,
    stratified_token_losses,
)
from kestrel.models import KestrelForCausalLM, KestrelModelBuilder, register_kestrel_models
from kestrel.tracking.mlflow_tracker import KestrelMLflowTracker
from kestrel.utils.data import get_gpt2_tokenizer
from kestrel.utils.train import (
    apply_lr_schedule,
    build_optimizers,
    cleanup_distributed,
    compute_lr_multiplier,
    load_checkpoint,
    save_checkpoint,
    setup_runtime,
    unwrap_model,
)

GPT2_EOT_TOKEN_ID = 50_256


# ---------------------------------------------------------------------------
# 1. Streaming data: HF text -> GPT-2 tokens -> packed (input, label, document_id) windows
# ---------------------------------------------------------------------------


def _get_gpt2_encoder() -> tuple[Any, int]:
    """Return `(encode_fn, eot_id)` using `tiktoken` if installed, else the HF `gpt2` tokenizer."""
    try:
        enc = get_gpt2_tokenizer()
        return (lambda txt: enc.encode(txt, disallowed_special=())), int(enc.eot_token)
    except ImportError:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        eot = tok.eos_token_id if tok.eos_token_id is not None else GPT2_EOT_TOKEN_ID
        return (lambda txt: tok.encode(txt, add_special_tokens=False)), int(eot)


def stream_packed_batches(
    cfg: TrainConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    seed_offset: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield infinite `(input_ids, labels, document_ids)` batches streamed straight from Hugging Face."""
    from datasets import load_dataset

    encode_fn, eot_id = _get_gpt2_encoder()
    chunk_len = cfg.seq_len + 1

    def windows() -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        tok_buf: list[int] = []
        doc_buf: list[int] = []
        doc_id, epoch = 0, 0
        while True:
            args = (cfg.dataset_repo, cfg.dataset_config) if cfg.dataset_config else (cfg.dataset_repo,)
            ds = load_dataset(*args, split=cfg.dataset_split, streaming=True)
            if world_size > 1 and hasattr(ds, "shard"):
                ds = ds.shard(num_shards=world_size, index=rank)  # each GPU reads different shards
            if cfg.shuffle_buffer_size > 1 and hasattr(ds, "shuffle"):
                ds = ds.shuffle(seed=cfg.seed + epoch * 97 + seed_offset, buffer_size=cfg.shuffle_buffer_size)

            for row in ds:
                text = row.get(cfg.text_field)
                if not isinstance(text, str) or not text.strip():
                    continue
                doc = [*encode_fn(text), eot_id]
                tok_buf.extend(doc)
                doc_buf.extend([doc_id] * len(doc))
                doc_id += 1

                while len(tok_buf) >= chunk_len:  # emit full windows, keep the remainder
                    toks, docs = tok_buf[:chunk_len], doc_buf[: cfg.seq_len]
                    tok_buf, doc_buf = tok_buf[cfg.seq_len :], doc_buf[cfg.seq_len :]
                    yield (
                        torch.tensor(toks[:-1]),
                        torch.tensor(toks[1:]),
                        torch.tensor([d - docs[0] for d in docs]),  # 0-based ids per window
                    )
            epoch += 1

    stream = windows()
    while True:
        xs, ys, ds_ = zip(*(next(stream) for _ in range(cfg.micro_batch_size)))
        yield torch.stack(xs), torch.stack(ys), torch.stack(ds_)


def prefetch_batches(
    batch_iter: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    prefetch_size: int = 4,
    pin_memory: bool = False,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Overlap remote HF streaming & tokenization with GPU compute via a background queue."""
    if prefetch_size <= 0:
        yield from batch_iter
        return

    buf: queue.Queue[Any] = queue.Queue(maxsize=prefetch_size)
    stop_event = threading.Event()
    sentinel = object()

    def _worker() -> None:
        try:
            for x, y, docs in batch_iter:
                if stop_event.is_set():
                    break
                if pin_memory:
                    x, y, docs = x.pin_memory(), y.pin_memory(), docs.pin_memory()
                buf.put((x, y, docs))
        except BaseException as exc:
            buf.put(exc)
        finally:
            buf.put(sentinel)

    thread = threading.Thread(target=_worker, name="kestrel-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = buf.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop_event.set()


# ---------------------------------------------------------------------------
# 2. Model loading, unified loss, eval, and post-step hooks
# ---------------------------------------------------------------------------


def build_model_from_config(cfg: TrainConfig) -> tuple[nn.Module, dict[str, Any]]:
    """Build a causal LM from a model config JSON via `KestrelModelBuilder` or HF `AutoModelForCausalLM`."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if not cfg.model_config_path:
        raise ValueError("Set `model_config_path` (or `--model-config`), or pass a pre-built model to train().")

    register_kestrel_models()
    raw = load_model_config(cfg.model_config_path)
    if raw.get("model_type") == "kestrel":
        return KestrelModelBuilder.build(cfg.model_config_path)

    hf_cfg = AutoConfig.from_pretrained(cfg.model_config_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(hf_cfg, trust_remote_code=True)

    model_cfg = hf_cfg.to_dict()
    for key, val in extract_model_metadata(raw).items():
        model_cfg.setdefault(key, val)
    return model, model_cfg


def _model_config_dict(model: nn.Module) -> dict[str, Any]:
    conf = getattr(unwrap_model(model), "config", {})
    return conf.to_dict() if hasattr(conf, "to_dict") else dict(conf)


def _field(out: Any, name: str) -> Any:
    return out.get(name) if isinstance(out, dict) else getattr(out, name, None)


def _chunked_lm_and_z_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Linear,
    *,
    chunk_size: int,
    z_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute exact CE + z-loss in sequence chunks without materializing full `(B, T, V)` FP32 logits."""
    seq_len = hidden_states.shape[1]
    total_valid = (labels != -100).sum().clamp(min=1)
    total_tokens = max(1, labels.numel())

    def _chunk_fn(h_chunk: torch.Tensor, y_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = lm_head(h_chunk).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_chunk.reshape(-1), ignore_index=-100, reduction="sum")
        z = torch.logsumexp(logits, dim=-1).pow(2).sum() if z_loss_weight > 0 else ce.new_zeros(())
        return ce, z

    ce_total = hidden_states.new_zeros((), dtype=torch.float32)
    z_total = hidden_states.new_zeros((), dtype=torch.float32)
    use_ckpt = hidden_states.requires_grad and torch.is_grad_enabled()

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        h_c = hidden_states[:, start:end, :]
        y_c = labels[:, start:end]
        if use_ckpt:
            ce_c, z_c = torch_checkpoint(_chunk_fn, h_c, y_c, use_reentrant=False)
        else:
            ce_c, z_c = _chunk_fn(h_c, y_c)
        ce_total = ce_total + ce_c
        if z_loss_weight > 0:
            z_total = z_total + z_c

    lm_loss = ce_total / total_valid
    z_loss = (z_total / total_tokens) if z_loss_weight > 0 else None
    return lm_loss, z_loss


def compute_unified_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    document_ids: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One forward pass -> LM loss + z-loss + optional aux (MoE) and MTP losses.

    Uses chunked `lm_head` + CE + z-loss on `KestrelForCausalLM` when `cfg.loss_chunk_size > 0`
    and keeps stats as detached GPU tensors so zero GPU->CPU syncs happen per microbatch.
    """
    raw = unwrap_model(model)
    is_kestrel = isinstance(raw, KestrelForCausalLM)
    use_chunked = is_kestrel and cfg.loss_chunk_size > 0 and cfg.loss_chunk_size < input_ids.shape[1]

    if is_kestrel:
        out = model(
            input_ids,
            document_ids=document_ids,
            mtp_labels=labels,
            use_cache=False,
            compute_logits=(not use_chunked),
        )
    else:
        out = model(input_ids, document_ids=document_ids)

    logits = out if isinstance(out, torch.Tensor) else _field(out, "logits")
    lm_loss = None if isinstance(out, torch.Tensor) else _field(out, "loss")
    z_loss: torch.Tensor | None = None

    if use_chunked:
        hidden = _field(out, "last_hidden_state")
        lm_loss, z_loss = _chunked_lm_and_z_loss(
            hidden,
            labels,
            raw.lm_head,
            chunk_size=cfg.loss_chunk_size,
            z_loss_weight=cfg.z_loss_weight,
        )
    else:
        if lm_loss is None:
            lm_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        if cfg.z_loss_weight > 0 and logits is not None:
            z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()

    total = lm_loss
    stats: dict[str, torch.Tensor] = {"lm": lm_loss.detach()}

    if z_loss is not None and cfg.z_loss_weight > 0:
        total = total + cfg.z_loss_weight * z_loss
        stats["z_loss"] = z_loss.detach()
    for name, weight in (("aux_loss", cfg.aux_loss_weight), ("mtp_loss", cfg.mtp_loss_weight)):
        extra = None if isinstance(out, torch.Tensor) else _field(out, name)
        if extra is not None and weight > 0:
            total = total + weight * extra
            stats[name] = extra.detach()
    return total, stats


@torch.no_grad()
def _post_step_hooks(model: nn.Module) -> dict[str, float]:
    """Run Track C's ALF-LB bias update (if present) and collect any model-reported metrics."""
    raw = unwrap_model(model)
    for mod in raw.modules():
        if callable(getattr(mod, "update_moe_biases", None)):
            mod.update_moe_biases()
    report = raw.get_aux_metrics() if callable(getattr(raw, "get_aux_metrics", None)) else {}
    return {str(k): float(v) for k, v in (report or {}).items() if isinstance(v, (int, float))}


@torch.no_grad()
def _evaluate(model, val_batches, cfg, device, autocast_ctx, world_size) -> tuple[float, float, dict[str, float]]:
    """Compute validation LM loss, perplexity, stratified token losses, and capacity metrics."""
    was_training = model.training
    model.eval()
    raw = unwrap_model(model)
    is_kestrel = isinstance(raw, KestrelForCausalLM)

    losses: list[float] = []
    strat_accum: dict[str, list[float]] = defaultdict(list)

    for x, y, docs in val_batches:
        xd, yd, dd = x.to(device, non_blocking=True), y.to(device, non_blocking=True), docs.to(device, non_blocking=True)
        with autocast_ctx():
            if is_kestrel and cfg.track_capacity_metrics:
                out = model(xd, document_ids=dd, use_cache=False, compute_logits=False)
                hidden = out.last_hidden_state
                chunk = cfg.loss_chunk_size if cfg.loss_chunk_size > 0 else yd.shape[1]
                tok_losses = []
                for s in range(0, yd.shape[1], chunk):
                    e = min(s + chunk, yd.shape[1])
                    lg = raw.lm_head(hidden[:, s:e, :]).float()
                    tok_losses.append(
                        F.cross_entropy(lg.reshape(-1, lg.size(-1)), yd[:, s:e].reshape(-1), ignore_index=-100, reduction="none")
                        .view(yd.shape[0], e - s)
                    )
                per_tok = torch.cat(tok_losses, dim=1)
                valid = yd != -100
                losses.append(float(per_tok[valid].mean().item()))
                for k, v in stratified_token_losses(per_tok, yd).items():
                    strat_accum[k].append(v)
            else:
                _, stats = compute_unified_loss(model, xd, yd, dd, cfg)
                losses.append(float(stats["lm"].item()))

    val_loss = sum(losses) / max(len(losses), 1)
    if world_size > 1 and dist.is_initialized():
        t = torch.tensor([val_loss], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        val_loss = float(t.item())

    extra_metrics = {k: sum(vs) / len(vs) for k, vs in strat_accum.items() if vs}
    if cfg.track_capacity_metrics and val_batches:
        x0, _, d0 = val_batches[0]
        extra_metrics.update(measure_model_capacity(model, x0.to(device), d0.to(device), autocast_ctx))

    if was_training:
        model.train()
    return val_loss, math.exp(min(val_loss, 20.0)), extra_metrics


# ---------------------------------------------------------------------------
# 3. Training loop (runtime initialized via `kestrel.utils.train`)
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig, model: nn.Module | None = None) -> dict[str, Any]:
    """Run pretraining; log to DagsHub MLflow, save checkpoints, and optionally push to HF Hub."""
    runtime = setup_runtime(seed=cfg.seed, precision=cfg.precision)
    try:
        return _train_inner(cfg, runtime, model=model)
    finally:
        if runtime.is_distributed:
            cleanup_distributed()


def _train_inner(cfg: TrainConfig, runtime: Any, model: nn.Module | None = None) -> dict[str, Any]:
    rank, world_size, local_rank = runtime.rank, runtime.world_size, runtime.local_rank
    device, autocast_ctx, scaler = runtime.device, runtime.autocast, runtime.scaler
    is_main = runtime.is_main

    # Model (+ metadata such as hf_repo_id / blueprint / track read from the model JSON)
    model, model_cfg = build_model_from_config(cfg) if model is None else (model, None)
    model_cfg = model_cfg or _model_config_dict(model)
    if cfg.gradient_checkpointing and hasattr(unwrap_model(model), "gradient_checkpointing_enable"):
        unwrap_model(model).gradient_checkpointing_enable()
    model = model.to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizers = build_optimizers(model, cfg)
    start_step, tokens_seen, best_val_loss = (
        load_checkpoint(cfg.resume_from, model, optimizers, scaler, device) if cfg.resume_from else (0, 0, float("inf"))
    )

    # Data: async prefetched training stream + fixed held-out validation batches
    raw_train_iter = stream_packed_batches(cfg, rank=rank, world_size=world_size)
    train_iter = prefetch_batches(
        raw_train_iter,
        prefetch_size=cfg.prefetch_batches,
        pin_memory=(device.type == "cuda"),
    )
    val_batches = list(islice(stream_packed_batches(cfg, rank=rank, world_size=world_size, seed_offset=10_000), cfg.val_eval_batches))

    # Capacity, gradient health & scaling/plateau monitors
    plateau_tracker = PlateauTracker(plateau_delta_per_doubling=cfg.plateau_delta_per_doubling)
    grad_monitor = GradientHealthMonitor(model, optimizers) if (is_main and cfg.track_capacity_metrics) else None

    # Tracking config: rank 0 only, inheriting hf_repo_id / blueprint / track from the model JSON
    tcfg = TrackingConfig(**asdict(cfg.tracking))
    tcfg.enabled = cfg.tracking.enabled and is_main
    tcfg.hf_repo_id = tcfg.hf_repo_id or model_cfg.get("hf_repo_id")
    if tcfg.blueprint == "custom" and isinstance(model_cfg.get("blueprint"), str):
        tcfg.blueprint = model_cfg["blueprint"]
    if isinstance(model_cfg.get("track"), str):
        tcfg.track = model_cfg["track"]

    tokens_per_step = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accum_steps * world_size
    t_start = time.perf_counter()
    final_train_loss, best_val_ppl = float("nan"), float("nan")
    throughputs: list[float] = []
    last_hf_dir: Path | None = None

    with KestrelMLflowTracker(tcfg) as tracker:
        train_cfg = cfg.to_dict()
        params = tracker.log_model_summary(model)
        tracker.log_configs(model_config=model_cfg, train_config=train_cfg)
        ckpt_root = Path(cfg.output_dir) / tracker.run_name

        def save(tag: str) -> Path:
            return save_checkpoint(
                ckpt_root / tag, model, optimizers, scaler, step=step, tokens_seen=tokens_seen,
                best_val_loss=best_val_loss, model_config=model_cfg, train_config=train_cfg,
            )

        if is_main:
            print(
                f"[kestrel.train] Run={tracker.run_name} | {runtime.summary()} | "
                f"Params={params['total']/1e6:.2f}M (active={params['active']/1e6:.2f}M) | "
                f"Opt={list(optimizers)} | Tok/step={tokens_per_step:,}"
            )

        model.train()
        for step in range(start_step + 1, cfg.max_steps + 1):
            t_step = time.perf_counter()
            lrs = apply_lr_schedule(optimizers, compute_lr_multiplier(step - 1, cfg))
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)

            # Gradient accumulation (keep loss stats on GPU to avoid per-microbatch sync stalls)
            acc_tensors: dict[str, torch.Tensor] = {}
            for micro in range(cfg.grad_accum_steps):
                x, y, docs = (t.to(device, non_blocking=True) for t in next(train_iter))
                skip_sync = isinstance(model, DDP) and micro < cfg.grad_accum_steps - 1
                with model.no_sync() if skip_sync else contextlib.nullcontext():
                    with autocast_ctx():
                        loss, stats = compute_unified_loss(model, x, y, docs, cfg)
                    scaled = loss / cfg.grad_accum_steps
                    (scaler.scale(scaled) if scaler else scaled).backward()
                inv_accum = 1.0 / cfg.grad_accum_steps
                acc_tensors["loss"] = acc_tensors.get("loss", 0) + loss.detach() * inv_accum
                for k, v in stats.items():
                    acc_tensors[k] = acc_tensors.get(k, 0) + v * inv_accum

            if scaler:
                for opt in optimizers.values():
                    scaler.unscale_(opt)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.grad is not None],
                    cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf"),
                )
            )
            should_log = is_main and (step == 1 or step % cfg.log_interval == 0 or step == cfg.max_steps)
            grad_health: dict[str, float] = {}
            if grad_monitor is not None and should_log:
                grad_health.update(grad_monitor.before_step())
            for opt in optimizers.values():
                scaler.step(opt) if scaler else opt.step()
            if scaler:
                scaler.update()
            if grad_monitor is not None and should_log:
                grad_health.update(grad_monitor.after_step())

            hook_metrics = _post_step_hooks(model)  # Track C ALF-LB updates
            tokens_seen += tokens_per_step
            step_sec = max(1e-6, time.perf_counter() - t_step)
            throughputs.append(tokens_per_step / step_sec)
            wall = time.perf_counter() - t_start

            if should_log:
                acc = {k: float(v.item()) for k, v in acc_tensors.items()}
                final_train_loss = acc["lm"]
                metrics = {
                    **{f"train/{k}": v for k, v in acc.items() if k != "loss" or len(acc) > 2},
                    "train/ppl": math.exp(min(acc["lm"], 20.0)),
                    "train/grad_norm": grad_norm,
                    "perf/tokens_per_sec": throughputs[-1],
                    "progress/tokens_seen": float(tokens_seen),
                    "progress/gpu_hours": wall * world_size / 3600,
                    **lrs,
                    **hook_metrics,
                    **grad_health,
                }
                tracker.log_metrics(metrics, step=step)
                print_every = cfg.print_interval if cfg.print_interval is not None else cfg.log_interval
                if step == 1 or step % max(1, print_every) == 0 or step == cfg.max_steps:
                    print(
                        f"step {step:>5d}/{cfg.max_steps} | lm={acc['lm']:.3f} | ppl={metrics['train/ppl']:.1f} "
                        f"| gnorm={grad_norm:.2f} | tok/s={throughputs[-1]:,.0f} | tok={tokens_seen/1e6:.2f}M"
                    )

            if cfg.eval_interval > 0 and val_batches and (step % cfg.eval_interval == 0 or step == cfg.max_steps):
                val_loss, val_ppl, extra_eval = _evaluate(model, val_batches, cfg, device, autocast_ctx, world_size)
                if is_main:
                    in_cooldown = cfg.lr_schedule == "wsd" and step > (cfg.max_steps - cfg.decay_steps)
                    plateau_metrics = plateau_tracker.update(step, float(tokens_seen), val_loss, is_cooldown=in_cooldown)
                    tracker.log_metrics(
                        {
                            "val/loss": val_loss,
                            "val/ppl": val_ppl,
                            "progress/tokens_seen": float(tokens_seen),
                            "progress/gpu_hours": wall * world_size / 3600,
                            **extra_eval,
                            **plateau_metrics,
                        },
                        step=step,
                    )
                    alpha_str = f" | alpha={plateau_metrics['plateau/alpha']:.3f}" if "plateau/alpha" in plateau_metrics else ""
                    erank_str = f" | eRank={extra_eval['capacity/erank']:.1f}" if "capacity/erank" in extra_eval else ""
                    print(f"  [eval] step {step:>5d} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}{erank_str}{alpha_str}")
                    if val_loss < best_val_loss:
                        best_val_loss, best_val_ppl = val_loss, val_ppl
                        last_hf_dir = save("best")

            if is_main and cfg.save_interval > 0 and (step % cfg.save_interval == 0 or step == cfg.max_steps):
                last_hf_dir = save(f"step_{step:07d}")

        # Finish: optional MLflow weights, HF Hub push, and GitHub run snapshot
        wall = time.perf_counter() - t_start
        finite = lambda v, nd: round(v, nd) if math.isfinite(v) else None
        final_metrics = {
            "steps_completed": cfg.max_steps,
            "tokens_seen": tokens_seen,
            "final_train_loss": finite(final_train_loss, 6),
            "best_val_loss": finite(best_val_loss, 6),
            "best_val_ppl": finite(best_val_ppl, 4),
            "mean_tokens_per_sec": round(float(np.mean(throughputs)), 2) if throughputs else 0.0,
            "wall_clock_sec": round(wall, 2),
            "gpu_hours": round(wall * world_size / 3600, 6),
            **({"peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)} if device.type == "cuda" else {}),
        }

        model_info = hub_info = snapshot_dir = None
        if is_main:
            hf_dir = last_hf_dir if (last_hf_dir and last_hf_dir.name == "hf_checkpoint") else None
            model_info = tracker.log_model(
                model, model_name=f"{tcfg.blueprint}_{tcfg.track}", step=cfg.max_steps,
                extra_metadata={"stage": tcfg.stage, "track": tcfg.track, "tokens_seen": tokens_seen},
                hf_checkpoint_dir=hf_dir,
            )
            hub_info = tracker.publish_to_hub(
                model, model_config=model_cfg, train_config=train_cfg, final_metrics=final_metrics, hf_checkpoint_dir=hf_dir
            )
            snapshot_dir = tracker.export_github_snapshot(model_cfg, train_cfg, final_metrics)
            print(f"[kestrel.train] Finished run {tracker.run_name} (MLflow run_id={tracker.run_id})")
            if hub_info:
                print(f"[kestrel.train] Pushed HF Hub checkpoint  : {hub_info['repo_url']} ({hub_info['commit_sha'][:8]})")
            if snapshot_dir:
                print(f"[kestrel.train] GitHub snapshot exported to: {snapshot_dir}")

        return {
            "run_name": tracker.run_name,
            "mlflow_run_id": tracker.run_id,
            "tracking_uri": tracker.tracking_uri,
            "model_uri": getattr(model_info, "model_uri", None),
            "hf_hub": hub_info,
            "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
            "parameters": params,
            "metrics": final_metrics,
        }


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m kestrel.train", description="Kestrel pretraining.")
    parser.add_argument("--train-config", help="Training JSON, e.g. configs/train/climbmix_pretrain.json")
    parser.add_argument("--model-config", help="Model JSON, e.g. configs/model/kestrel_350m_dense.json")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override any config field (repeatable), e.g. --set max_steps=500 --set tracking.push_to_hub=true",
    )
    args = parser.parse_args(argv)

    cfg = TrainConfig.from_json(args.train_config) if args.train_config else TrainConfig()
    if args.model_config:
        cfg.model_config_path = args.model_config
    train(apply_overrides(cfg, args.overrides))


if __name__ == "__main__":
    main()
