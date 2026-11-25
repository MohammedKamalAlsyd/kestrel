"""Unified Kestrel pretraining entrypoint (Tracks A Dense, B Hybrid, C MoE).

Run on Kaggle (1 or 2 GPUs) with everything driven by JSON configs plus optional overrides:

    python -m kestrel.train --train-config configs/train/climbmix_pretrain.json \
        --model-config configs/model/kestrel_350m_dense.json --set max_steps=500

    torchrun --nproc_per_node=2 -m kestrel.train --train-config ... --model-config ...

Related modules:
- `kestrel.config`            TrainConfig, TrackingConfig, model JSON loading, CLI overrides
- `kestrel.utils.train`       device, dtype (bf16/fp16/fp32), DDP, seeds, Muon/AdamW, checkpoints
- `kestrel.utils.data`        GPT-2 tokenization and text helpers
- `kestrel.tracking`          DagsHub MLflow logging and GitHub run snapshots
Training text is streamed from Hugging Face and tokenized on the fly (no disk downloads).
"""

from __future__ import annotations

import argparse
import contextlib
import math
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

from kestrel.config import (
    TrackingConfig,
    TrainConfig,
    apply_overrides,
    extract_model_metadata,
    load_model_config,
)
from kestrel.tracking.mlflow_tracker import KestrelMLflowTracker
from kestrel.utils.data import get_gpt2_tokenizer
from kestrel.utils.train import (
    apply_lr_schedule,
    build_optimizers,
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


# ---------------------------------------------------------------------------
# 2. Model loading, unified loss, eval, and post-step hooks
# ---------------------------------------------------------------------------


def build_model_from_config(cfg: TrainConfig) -> tuple[nn.Module, dict[str, Any]]:
    """Build a causal LM from a model config JSON via HF `AutoModelForCausalLM` (supports custom code)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if not cfg.model_config_path:
        raise ValueError("Set `model_config_path` (or `--model-config`), or pass a pre-built model to train().")

    raw = load_model_config(cfg.model_config_path)
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


def compute_unified_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    document_ids: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One forward pass -> LM loss + z-loss + optional aux (MoE) and MTP losses.

    Models receive `document_ids` so they can block attention across packed documents.
    """
    out = model(input_ids, document_ids=document_ids)
    logits = out if isinstance(out, torch.Tensor) else _field(out, "logits")
    lm_loss = None if isinstance(out, torch.Tensor) else _field(out, "loss")
    if lm_loss is None:
        lm_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

    total = lm_loss
    stats = {"lm": float(lm_loss.detach())}

    if cfg.z_loss_weight > 0 and logits is not None:  # keeps final logits from drifting
        z = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        total = total + cfg.z_loss_weight * z
        stats["z_loss"] = float(z.detach())
    for name, weight in (("aux_loss", cfg.aux_loss_weight), ("mtp_loss", cfg.mtp_loss_weight)):
        extra = None if isinstance(out, torch.Tensor) else _field(out, name)
        if extra is not None and weight > 0:
            total = total + weight * extra
            stats[name] = float(extra.detach())
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
def _evaluate(model, val_batches, cfg, device, autocast_ctx, world_size) -> tuple[float, float]:
    """Mean validation LM loss and perplexity over a fixed set of held-out batches."""
    was_training = model.training
    model.eval()
    losses = []
    for x, y, docs in val_batches:
        with autocast_ctx():
            _, stats = compute_unified_loss(model, x.to(device), y.to(device), docs.to(device), cfg)
        losses.append(stats["lm"])
    val_loss = sum(losses) / max(len(losses), 1)

    if world_size > 1 and dist.is_initialized():
        t = torch.tensor([val_loss], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        val_loss = float(t.item())
    if was_training:
        model.train()
    return val_loss, math.exp(min(val_loss, 20.0))


# ---------------------------------------------------------------------------
# 3. Training loop (runtime initialized via `kestrel.utils.train`)
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig, model: nn.Module | None = None) -> dict[str, Any]:
    """Run pretraining; log to DagsHub MLflow, save checkpoints, and optionally push to HF Hub."""
    runtime = setup_runtime(seed=cfg.seed, precision=cfg.precision)
    rank, world_size, local_rank = runtime.rank, runtime.world_size, runtime.local_rank
    device, autocast_ctx, scaler = runtime.device, runtime.autocast, runtime.scaler
    is_main = runtime.is_main

    # Model (+ metadata such as hf_repo_id / blueprint / track read from the model JSON)
    model, model_cfg = build_model_from_config(cfg) if model is None else (model, None)
    model_cfg = model_cfg or _model_config_dict(model)
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizers = build_optimizers(model, cfg)
    start_step, tokens_seen, best_val_loss = (
        load_checkpoint(cfg.resume_from, model, optimizers, scaler, device) if cfg.resume_from else (0, 0, float("inf"))
    )

    # Data: training stream + fixed held-out validation batches (different shuffle seed)
    train_iter = stream_packed_batches(cfg, rank=rank, world_size=world_size)
    val_batches = list(islice(stream_packed_batches(cfg, rank=rank, world_size=world_size, seed_offset=10_000), cfg.val_eval_batches))

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

            # Gradient accumulation (average every logged loss term over micro-batches)
            acc: dict[str, float] = defaultdict(float)
            for micro in range(cfg.grad_accum_steps):
                x, y, docs = (t.to(device, non_blocking=True) for t in next(train_iter))
                skip_sync = isinstance(model, DDP) and micro < cfg.grad_accum_steps - 1
                with model.no_sync() if skip_sync else contextlib.nullcontext():
                    with autocast_ctx():
                        loss, stats = compute_unified_loss(model, x, y, docs, cfg)
                    scaled = loss / cfg.grad_accum_steps
                    (scaler.scale(scaled) if scaler else scaled).backward()
                acc["loss"] += float(loss.detach()) / cfg.grad_accum_steps
                for k, v in stats.items():
                    acc[k] += v / cfg.grad_accum_steps

            if scaler:
                for opt in optimizers.values():
                    scaler.unscale_(opt)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.grad is not None],
                    cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf"),
                )
            )
            for opt in optimizers.values():
                scaler.step(opt) if scaler else opt.step()
            if scaler:
                scaler.update()

            hook_metrics = _post_step_hooks(model)  # Track C ALF-LB updates
            tokens_seen += tokens_per_step
            final_train_loss = acc["lm"]
            step_sec = max(1e-6, time.perf_counter() - t_step)
            throughputs.append(tokens_per_step / step_sec)
            wall = time.perf_counter() - t_start

            if is_main and (step == 1 or step % cfg.log_interval == 0 or step == cfg.max_steps):
                metrics = {
                    **{f"train/{k}": v for k, v in acc.items() if k != "loss" or len(acc) > 2},
                    "train/ppl": math.exp(min(acc["lm"], 20.0)),
                    "train/grad_norm": grad_norm,
                    "perf/tokens_per_sec": throughputs[-1],
                    "progress/tokens_seen": float(tokens_seen),
                    "progress/gpu_hours": wall * world_size / 3600,
                    **lrs,
                    **hook_metrics,
                }
                tracker.log_metrics(metrics, step=step)
                print(
                    f"step {step:>5d}/{cfg.max_steps} | lm={acc['lm']:.4f} | ppl={metrics['train/ppl']:.2f} "
                    f"| gnorm={grad_norm:.2f} | tok/s={throughputs[-1]:,.0f} | seen={tokens_seen:,}"
                )

            if cfg.eval_interval > 0 and val_batches and (step % cfg.eval_interval == 0 or step == cfg.max_steps):
                val_loss, val_ppl = _evaluate(model, val_batches, cfg, device, autocast_ctx, world_size)
                if is_main:
                    tracker.log_metrics(
                        {"val/loss": val_loss, "val/ppl": val_ppl, "progress/tokens_seen": float(tokens_seen),
                         "progress/gpu_hours": wall * world_size / 3600},
                        step=step,
                    )
                    print(f"  [eval] step {step:>5d} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")
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
