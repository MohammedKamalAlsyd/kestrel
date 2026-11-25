"""Local pre-flight gradient, leakage, AMP parity, and VRAM diagnostics (no DagsHub needed).

Run locally before launching on Kaggle:
    python -m kestrel.metrics.preflight --model-config configs/model/kestrel_350m_dense.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from kestrel.config import TrainConfig, load_model_config
from kestrel.metrics.gradients import GradientHealthMonitor, audit_layer_gradients
from kestrel.models import KestrelForCausalLM, KestrelModelBuilder
from kestrel.utils.train import build_optimizers, resolve_device, seed_everything, setup_runtime


def _shrink_config_for_local(raw: dict[str, Any], *, full_size: bool = False) -> dict[str, Any]:
    cfg = dict(raw)
    if full_size:
        return cfg
    cfg.update(
        {
            "vocab_size": 256,
            "max_position_embeddings": 128,
            "num_hidden_layers": min(4, int(cfg.get("num_hidden_layers", 4))),
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "intermediate_size": 256,
            "swa_window_size": 16,
            "global_attn_every_n_layers": 4,
            "hybrid_attn_every_n_layers": 4,
            "deltanet_num_heads": 4,
            "deltanet_head_dim": 32,
            "num_routed_experts": 4,
            "num_experts_per_tok": 2,
            "expert_intermediate_size": 64,
            "shared_expert_intermediate_size": 128,
        }
    )
    return cfg


def check_causal_and_doc_leakage(model: KestrelForCausalLM, device: torch.device) -> dict[str, Any]:
    """Mathematically verify zero future-token and zero cross-document gradient leakage."""
    model.eval()
    model.zero_grad(set_to_none=True)
    vocab = model.config.vocab_size
    x = torch.randint(0, vocab, (1, 12), device=device)
    docs = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], device=device)

    emb_out: list[torch.Tensor] = []

    def _capture(_mod: Any, _inp: Any, out: torch.Tensor) -> None:
        out.retain_grad()
        emb_out.append(out)

    handle = model.get_input_embeddings().register_forward_hook(_capture)
    try:
        out = model(x, document_ids=docs, use_cache=False)
        # Backprop only from token 8 (inside document 1)
        out.logits[0, 8].sum().backward()
        grad = emb_out[0].grad
        assert grad is not None
        cross_doc_max = float(grad[0, :6].abs().max().item())
        future_max = float(grad[0, 9:].abs().max().item())
        self_and_past_max = float(grad[0, 6:9].abs().max().item())
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    passed = cross_doc_max == 0.0 and future_max == 0.0 and self_and_past_max > 0.0
    return {
        "passed": passed,
        "cross_doc_grad_max": cross_doc_max,
        "future_grad_max": future_max,
        "valid_grad_max": self_and_past_max,
    }


def check_single_batch_overfit(
    model: KestrelForCausalLM,
    device: torch.device,
    *,
    steps: int = 40,
    optimizer_type: str = "muon_adamw",
) -> dict[str, Any]:
    """Train on 1 fixed batch to confirm smooth loss descent and healthy Muon/AdamW update ratios."""
    model.train()
    cfg = TrainConfig(optimizer_type=optimizer_type, max_steps=steps)
    optimizers = build_optimizers(model, cfg)
    monitor = GradientHealthMonitor(model, optimizers, straightness_window=min(20, steps))

    vocab = model.config.vocab_size
    x = torch.randint(0, vocab, (2, 32), device=device)
    y = torch.randint(0, vocab, (2, 32), device=device)
    docs = torch.zeros_like(x)

    losses: list[float] = []
    last_ratios: dict[str, float] = {}
    last_audit: dict[str, Any] = {}

    for step in range(1, steps + 1):
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        out = model(x, document_ids=docs, mtp_labels=y, use_cache=False)
        loss = F.cross_entropy(out.logits.float().reshape(-1, vocab), y.reshape(-1))
        if out.aux_loss is not None:
            loss = loss + out.aux_loss
        if out.mtp_loss is not None:
            loss = loss + 0.3 * out.mtp_loss
        loss.backward()

        if step == steps:
            last_audit = audit_layer_gradients(model)

        monitor.before_step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers.values():
            opt.step()
        last_ratios = monitor.after_step()
        losses.append(float(loss.detach().item()))

    return {
        "passed": losses[-1] < losses[0] * 0.5 and last_audit.get("nonfinite_param_count", 0) == 0,
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
        "update_ratios": {k: round(v, 6) for k, v in last_ratios.items()},
        "grad_audit": last_audit,
    }


def check_amp_fp16_parity(raw_cfg: dict[str, Any], device: torch.device, steps: int = 10) -> dict[str, Any]:
    """Verify CUDA FP16 + GradScaler stability (simulates Kaggle T4) vs BF16/FP32."""
    if device.type != "cuda":
        return {"skipped": True, "reason": "CUDA unavailable"}

    results: dict[str, Any] = {}
    for prec in ("fp16", "bf16"):
        rt = setup_runtime(seed=42, precision=prec, device=device, distributed=False)
        model, _ = KestrelModelBuilder.build(raw_cfg)
        model = model.to(rt.device).train()
        cfg = TrainConfig(optimizer_type="muon_adamw", precision=prec)
        optimizers = build_optimizers(model, cfg)

        x = torch.randint(0, model.config.vocab_size, (2, 32), device=rt.device)
        y = torch.randint(0, model.config.vocab_size, (2, 32), device=rt.device)
        skipped, max_logit, final_loss = 0, 0.0, float("nan")

        for _ in range(steps):
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            with rt.autocast():
                out = model(x, mtp_labels=y, use_cache=False)
                loss = F.cross_entropy(out.logits.float().reshape(-1, out.logits.size(-1)), y.reshape(-1))
            max_logit = max(max_logit, float(out.logits.detach().abs().max().item()))
            (rt.scaler.scale(loss) if rt.scaler else loss).backward()
            if rt.scaler:
                prev_scale = rt.scaler.get_scale()
                for opt in optimizers.values():
                    rt.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for opt in optimizers.values():
                    rt.scaler.step(opt)
                rt.scaler.update()
                if rt.scaler.get_scale() < prev_scale:
                    skipped += 1
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for opt in optimizers.values():
                    opt.step()
            final_loss = float(loss.detach().item())

        results[prec] = {
            "resolved_precision": rt.precision,
            "final_loss": round(final_loss, 4),
            "max_abs_logit": round(max_logit, 3),
            "scaler_skipped_steps": skipped,
        }
    return {"passed": all(v["final_loss"] == v["final_loss"] for v in results.values()), "modes": results}


def run_preflight_check(
    model_config_path: str | Path,
    *,
    device: str = "auto",
    full_size: bool = False,
    overfit_steps: int = 40,
    output_dir: str | Path = "experiments/preflight",
) -> dict[str, Any]:
    """Run all local gradient & numerical pre-flight diagnostics and write a JSON receipt."""
    seed_everything(42)
    dev = resolve_device(device)
    raw = load_model_config(model_config_path)
    test_cfg = _shrink_config_for_local(raw, full_size=full_size)

    t0 = time.perf_counter()
    model_leak, _ = KestrelModelBuilder.build(test_cfg)
    leak_res = check_causal_and_doc_leakage(model_leak.to(dev), dev)

    seed_everything(42)
    model_fit, _ = KestrelModelBuilder.build(test_cfg)
    overfit_res = check_single_batch_overfit(model_fit.to(dev), dev, steps=overfit_steps)

    amp_res = check_amp_fp16_parity(test_cfg, dev)

    report = {
        "model_config": str(model_config_path),
        "blueprint": raw.get("blueprint", Path(model_config_path).stem),
        "device": str(dev),
        "full_size": full_size,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "all_passed": bool(leak_res["passed"] and overfit_res["passed"] and amp_res.get("passed", True)),
        "causal_doc_leakage": leak_res,
        "single_batch_overfit": overfit_res,
        "amp_parity": amp_res,
    }

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_file = out_root / f"{Path(model_config_path).stem}_preflight.json"
    out_file.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    report["report_path"] = str(out_file)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m kestrel.metrics.preflight", description="Local pre-flight gradient checks.")
    parser.add_argument("--model-config", required=True, help="Path to model JSON in configs/model/")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--full-size", action="store_true", help="Run on full production dimensions (needs GPU VRAM)")
    parser.add_argument("--steps", type=int, default=40, help="Single-batch overfit steps")
    parser.add_argument("--output-dir", default="experiments/preflight", help="Where to write local JSON report")
    args = parser.parse_args(argv)

    rep = run_preflight_check(
        args.model_config,
        device=args.device,
        full_size=args.full_size,
        overfit_steps=args.steps,
        output_dir=args.output_dir,
    )
    print(json.dumps({k: v for k, v in rep.items() if k != "single_batch_overfit"}, indent=2))
    print("Overfit:", rep["single_batch_overfit"]["initial_loss"], "->", rep["single_batch_overfit"]["final_loss"])
    print("Update ratios:", rep["single_batch_overfit"]["update_ratios"])
    print("Saved report to:", rep["report_path"])


if __name__ == "__main__":
    main()
