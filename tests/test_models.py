"""End-to-end tests for all 4 Kestrel architectures, builder, HF Auto classes, KV cache, and train.py integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from kestrel.config import TrainConfig, load_model_config
from kestrel.models import (
    BLUEPRINT_CONFIGS,
    KestrelConfig,
    KestrelForCausalLM,
    KestrelModelBuilder,
)
from kestrel.train import _post_step_hooks, build_model_from_config, compute_unified_loss
from kestrel.utils.train import build_optimizers, load_checkpoint, save_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_tiny_from_blueprint(blueprint: str) -> KestrelForCausalLM:
    """Clone a real blueprint JSON and shrink dimensions for fast CPU/GPU unit testing."""
    raw = load_model_config(REPO_ROOT / BLUEPRINT_CONFIGS[blueprint])
    raw.update(
        {
            "vocab_size": 128,
            "max_position_embeddings": 64,
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "intermediate_size": 128,
            "swa_window_size": 8,
            "global_attn_every_n_layers": 4,
            "hybrid_attn_every_n_layers": 4,
            "deltanet_num_heads": 4,
            "deltanet_head_dim": 16,
            "num_routed_experts": 4,
            "num_experts_per_tok": 2,
            "expert_intermediate_size": 32,
            "shared_expert_intermediate_size": 64,
        }
    )
    model, _ = KestrelModelBuilder.build(raw)
    return model


@pytest.mark.parametrize("blueprint", ["dense", "hybrid_gdn", "moe16", "diff_mhc_relu2"])
def test_all_four_architectures_train_step_and_kv_cache(blueprint: str) -> None:
    model = _make_tiny_from_blueprint(blueprint)
    model.train()
    cfg = TrainConfig(optimizer_type="muon_adamw", z_loss_weight=1e-4, aux_loss_weight=1.0, mtp_loss_weight=0.3)

    x = torch.randint(0, 128, (2, 10))
    y = torch.randint(0, 128, (2, 10))
    docs = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0] * 10])

    # Forward + unified loss + backward + Muon/AdamW step + post-step hooks
    optimizers = build_optimizers(model, cfg)
    loss, stats = compute_unified_loss(model, x, y, docs, cfg)
    assert torch.isfinite(loss)
    assert "lm" in stats and "z_loss" in stats
    if blueprint == "moe16":
        assert "aux_loss" in stats
    if blueprint == "diff_mhc_relu2":
        assert "mtp_loss" in stats

    loss.backward()
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad(set_to_none=True)

    hooks = _post_step_hooks(model)
    if blueprint == "moe16":
        assert "moe/bias_std_mean" in hooks

    # KV-cached inference matches uncached last-token logits closely (atol for fp32)
    model.eval()
    prompt = x[:1, :6]
    with torch.no_grad():
        full_out = model(prompt, use_cache=False).logits
        prefill = model(prompt[:, :4], use_cache=True)
        step1 = model(prompt[:, 4:5], past_key_values=prefill.past_key_values, use_cache=True)
        step2 = model(prompt[:, 5:6], past_key_values=step1.past_key_values, use_cache=True)
    assert torch.allclose(full_out[:, -1, :], step2.logits[:, -1, :], atol=1e-4, rtol=1e-3)

    # Fast KV-cached autoregressive generation
    gen = model.generate_fast(prompt[:, :4], max_new_tokens=3, temperature=0.0, eos_token_id=-1)
    assert gen.shape == (1, 7)


def test_hf_auto_registration_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = _make_tiny_from_blueprint("dense")
    cfg = TrainConfig(optimizer_type="adamw")
    optimizers = build_optimizers(model, cfg)

    ckpt_dir = tmp_path / "ckpt"
    hf_dir = save_checkpoint(
        ckpt_dir,
        model,
        optimizers,
        scaler=None,
        step=12,
        tokens_seen=1024,
        best_val_loss=3.21,
        model_config=model.config.to_dict(),
        train_config=cfg.to_dict(),
    )
    assert hf_dir.is_dir()

    # Load back via HF AutoConfig + AutoModelForCausalLM
    reloaded_cfg = AutoConfig.from_pretrained(str(hf_dir))
    assert isinstance(reloaded_cfg, KestrelConfig)
    reloaded_model = AutoModelForCausalLM.from_pretrained(str(hf_dir))
    assert isinstance(reloaded_model, KestrelForCausalLM)

    # Load back via kestrel checkpoint utility
    fresh = _make_tiny_from_blueprint("dense")
    fresh_opts = build_optimizers(fresh, cfg)
    step, tokens_seen, best_val = load_checkpoint(ckpt_dir, fresh, fresh_opts, None, torch.device("cpu"))
    assert (step, tokens_seen) == (12, 1024)
    assert best_val == pytest.approx(3.21)


def test_full_configs_build_and_parameter_counts() -> None:
    """Instantiate all 4 production JSON configs on meta device to check exact param counts fast."""
    expected_ranges = {
        "dense": (340e6, 355e6, 340e6, 355e6),
        "hybrid_gdn": (400e6, 415e6, 400e6, 415e6),
        "moe16": (675e6, 695e6, 215e6, 230e6),
        "diff_mhc_relu2": (340e6, 365e6, 340e6, 360e6),
    }
    for name, (min_tot, max_tot, min_act, max_act) in expected_ranges.items():
        with torch.device("meta"):
            cfg = TrainConfig(model_config_path=str(REPO_ROOT / BLUEPRINT_CONFIGS[name]))
            model, model_cfg = build_model_from_config(cfg)
            summary = KestrelModelBuilder.summarize(model)
        assert model_cfg["blueprint"] == summary["blueprint"]
        assert min_tot <= summary["total_params"] <= max_tot, f"{name} total={summary['total_params']:,}"
        assert min_act <= summary["active_params"] <= max_act, f"{name} active={summary['active_params']:,}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP/GradScaler verification")
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_cuda_amp_muon_gradscaler_step(precision: str) -> None:
    """Verify real CUDA mixed-precision (including Kaggle T4 FP16+GradScaler) with Muon + AdamW."""
    from kestrel.utils.train import setup_runtime

    rt = setup_runtime(seed=42, precision=precision, device="cuda:0", distributed=False)
    model = _make_tiny_from_blueprint("dense").to(rt.device)
    model.train()
    cfg = TrainConfig(optimizer_type="muon_adamw", precision=precision)
    optimizers = build_optimizers(model, cfg)

    x = torch.randint(0, 128, (2, 16), device=rt.device)
    y = torch.randint(0, 128, (2, 16), device=rt.device)
    docs = torch.zeros_like(x)

    with rt.autocast():
        loss, _ = compute_unified_loss(model, x, y, docs, cfg)
    assert torch.isfinite(loss)
    (rt.scaler.scale(loss) if rt.scaler else loss).backward()
    if rt.scaler:
        for opt in optimizers.values():
            rt.scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for opt in optimizers.values():
        rt.scaler.step(opt) if rt.scaler else opt.step()
    if rt.scaler:
        rt.scaler.update()

