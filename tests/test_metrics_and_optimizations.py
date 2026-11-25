"""Tests for `kestrel.metrics` (plateau, capacity, gradients, preflight) and training optimizations."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
import torch

from kestrel.config import TrainConfig
from kestrel.metrics import (
    GradientHealthMonitor,
    PlateauTracker,
    audit_layer_gradients,
    check_causal_and_doc_leakage,
    effective_rank,
    measure_model_capacity,
    run_preflight_check,
    stratified_token_losses,
)
from kestrel.models import BLUEPRINT_CONFIGS, KestrelModelBuilder
from kestrel.models.attention.gated_deltanet import _delta_rule_chunkwise, _delta_rule_recurrent
from kestrel.train import compute_unified_loss, prefetch_batches
from kestrel.utils.train import build_optimizers

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_chunkwise_deltanet_matches_exact_recurrence() -> None:
    """Verify WY-representation chunkwise parallel GatedDeltaNet matches token-by-token recurrence."""
    torch.manual_seed(0)
    b, t, h, d = 2, 48, 4, 16
    q = torch.nn.functional.normalize(torch.randn(b, t, h, d), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, t, h, d), dim=-1)
    v = torch.randn(b, t, h, d)
    alpha = torch.sigmoid(torch.randn(b, t, h))
    beta = torch.sigmoid(torch.randn(b, t, h))
    docs = torch.cat([torch.zeros(b, 20, dtype=torch.long), torch.ones(b, 28, dtype=torch.long)], dim=1)

    s0 = torch.zeros(b, h, d, d)
    out_rec, st_rec = _delta_rule_recurrent(q, k, v, alpha, beta, s0.clone(), document_ids=docs)
    out_chk, st_chk = _delta_rule_chunkwise(q, k, v, alpha, beta, s0.clone(), document_ids=docs, chunk_size=16)

    assert torch.allclose(out_rec, out_chk, atol=1e-4, rtol=1e-4)
    assert torch.allclose(st_rec, st_chk, atol=1e-4, rtol=1e-4)


def test_chunked_loss_matches_full_logits_loss_and_grads() -> None:
    """Verify chunked lm_head + CE + z-loss produces identical loss and gradients to full logits."""
    torch.manual_seed(1)
    raw_cfg = {
        "vocab_size": 128,
        "max_position_embeddings": 64,
        "num_hidden_layers": 2,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 64,
    }
    m_full, _ = KestrelModelBuilder.build(raw_cfg)
    m_chunk, _ = KestrelModelBuilder.build(raw_cfg)
    m_chunk.load_state_dict(m_full.state_dict())

    x = torch.randint(0, 128, (2, 24))
    y = torch.randint(0, 128, (2, 24))
    docs = torch.zeros_like(x)

    loss_full, stats_full = compute_unified_loss(m_full, x, y, docs, TrainConfig(loss_chunk_size=0, z_loss_weight=1e-4))
    loss_chunk, stats_chunk = compute_unified_loss(m_chunk, x, y, docs, TrainConfig(loss_chunk_size=8, z_loss_weight=1e-4))

    assert torch.allclose(loss_full, loss_chunk, atol=1e-5)
    assert torch.allclose(stats_full["lm"], stats_chunk["lm"], atol=1e-5)
    assert torch.allclose(stats_full["z_loss"], stats_chunk["z_loss"], atol=1e-5)

    loss_full.backward()
    loss_chunk.backward()
    for (n1, p1), (_, p2) in zip(m_full.named_parameters(), m_chunk.named_parameters()):
        assert p1.grad is not None and p2.grad is not None, n1
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4, rtol=1e-3), f"grad mismatch in {n1}"


def test_plateau_tracker_power_law_and_capacity_metrics() -> None:
    tracker = PlateauTracker(plateau_delta_per_doubling=0.02)
    # Synthetic scaling curve L(D) = 2.0 + 50 * D^(-0.35)
    for step, tokens in enumerate([1e7, 2e7, 4e7, 8e7, 1.6e8, 3.2e8], start=1):
        val_loss = 2.0 + 50.0 * (tokens ** -0.35)
        res = tracker.update(step * 100, tokens, val_loss)
    assert "plateau/alpha" in res and "plateau/l_inf" in res
    assert 0.2 < res["plateau/alpha"] < 0.55
    assert 1.5 < res["plateau/l_inf"] < 2.3
    assert res["plateau/delta_per_2x_tokens"] > 0

    # Effective rank: isotropic random matrix has much higher eRank than rank-1 collapsed matrix
    full_rank = torch.randn(2, 64, 32)
    collapsed = torch.randn(2, 64, 1).expand(-1, -1, 32)
    er_full, ratio_full = effective_rank(full_rank)
    er_col, _ = effective_rank(collapsed)
    assert er_full > er_col * 5.0 and 0.0 < ratio_full <= 1.0

    # Stratified losses
    tok_losses = torch.ones(2, 128) * 3.0
    tok_losses[:, 100:] = 2.0
    labels = torch.cat([torch.full((2, 64), 50), torch.full((2, 64), 5000)], dim=1)
    strat = stratified_token_losses(tok_losses, labels, early_range=(16, 48), late_range=(96, 128))
    assert strat["val/loss_common"] > strat["val/loss_tail"]
    assert strat["val/long_context_gain"] > 0


@pytest.mark.parametrize("blueprint", ["dense", "hybrid_gdn", "moe16", "diff_mhc_relu2"])
def test_preflight_causal_doc_leakage_and_report(blueprint: str, tmp_path: Path) -> None:
    cfg_path = REPO_ROOT / BLUEPRINT_CONFIGS[blueprint]
    report = run_preflight_check(
        cfg_path,
        device="cpu",
        full_size=False,
        overfit_steps=12,
        output_dir=tmp_path,
    )
    assert report["causal_doc_leakage"]["passed"] is True, report["causal_doc_leakage"]
    assert report["causal_doc_leakage"]["cross_doc_grad_max"] == 0.0
    assert report["causal_doc_leakage"]["future_grad_max"] == 0.0
    assert Path(report["report_path"]).is_file()


def test_prefetch_batches_and_capacity_hooks() -> None:
    def _gen():
        for i in range(3):
            t = torch.full((2, 8), i)
            yield t, t, torch.zeros_like(t)

    items = list(prefetch_batches(_gen(), prefetch_size=2, pin_memory=False))
    assert len(items) == 3 and int(items[2][0][0, 0].item()) == 2

    model, _ = KestrelModelBuilder.build(
        {
            "vocab_size": 128,
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "ffn_type": "relu2",
            "intermediate_size": 64,
        }
    )
    x = torch.randint(0, 128, (2, 16))
    cap = measure_model_capacity(model, x, torch.zeros_like(x), contextlib.nullcontext)
    assert "capacity/erank" in cap and "capacity/dead_relu2_frac" in cap

    opts = build_optimizers(model, TrainConfig(optimizer_type="adamw"))
    mon = GradientHealthMonitor(model, opts, straightness_window=2)
    for _ in range(2):
        loss, _ = compute_unified_loss(model, x, x, torch.zeros_like(x), TrainConfig(loss_chunk_size=8))
        loss.backward()
        audit = audit_layer_gradients(model)
        assert audit["nonfinite_param_count"] == 0
        mon.before_step()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        after = mon.after_step()
    assert "plateau/trajectory_straightness" in after
