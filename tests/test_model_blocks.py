"""Unit tests for Kestrel modular model building blocks."""

from __future__ import annotations

import torch

from kestrel.models.activations import ACT2FN, get_activation
from kestrel.models.attention import (
    ATTENTION_REGISTRY,
    DifferentialGQA,
    GatedDeltaNet,
    GroupedQueryAttention,
)
from kestrel.models.cache import KVCache
from kestrel.models.ffn import FFN_REGISTRY, MoESwiGLUFFN, ReLU2FFN, SwiGLUFFN
from kestrel.models.masking import create_causal_mask
from kestrel.models.norms import RMSNorm
from kestrel.models.residuals import ManifoldHyperConnection, ValueResidualGate, sinkhorn_knopp
from kestrel.models.rope import RotaryEmbedding


def test_activations_and_norms() -> None:
    x = torch.tensor([-2.0, 0.0, 3.0])
    relu2 = get_activation("relu2")
    assert torch.allclose(relu2(x), torch.tensor([0.0, 0.0, 9.0]))
    assert "silu" in ACT2FN and "relu2" in ACT2FN

    norm = RMSNorm(16)
    h = torch.randn(2, 5, 16)
    out = norm(h)
    assert out.shape == h.shape


def test_causal_mask_document_boundaries_and_swa() -> None:
    # Fast path returns None when no custom masking is needed
    assert create_causal_mask(2, 8, 8) is None

    doc_ids = torch.tensor([[0, 0, 1, 1]])
    mask = create_causal_mask(1, 4, 4, document_ids=doc_ids, sliding_window=2)
    assert mask is not None and mask.shape == (1, 1, 4, 4)
    # Token 2 (doc 1) cannot attend to token 1 (doc 0)
    assert mask[0, 0, 2, 1] < -1e9
    # Token 3 (doc 1) can attend to token 2 (doc 1), within window=2
    assert mask[0, 0, 3, 2] == 0.0


def test_gqa_rope_value_residual_and_kv_cache() -> None:
    rope = RotaryEmbedding(dim=16)
    attn0 = GroupedQueryAttention(64, 4, 2, 16, value_residual_gating="satformer", layer_idx=0)
    attn1 = GroupedQueryAttention(64, 4, 2, 16, value_residual_gating="satformer", layer_idx=1)

    x = torch.randn(2, 6, 64)
    cos, sin = rope(6, device=x.device, dtype=x.dtype)
    out0, v0 = attn0(x, cos=cos, sin=sin)
    out1, _ = attn1(out0, cos=cos, sin=sin, v_first=v0)
    assert out0.shape == x.shape and out1.shape == x.shape

    # Incremental KV-cache decoding
    cache = KVCache()
    _, v_prefill = attn0(x[:, :4], cos=cos[:, :4], sin=sin[:, :4], kv_cache=cache)
    assert cache.get_seq_length(0) == 4
    pos = torch.tensor([[4]])
    cos_step, sin_step = rope(1, device=x.device, dtype=x.dtype, position_ids=pos)
    step_out, _ = attn0(x[:, 4:5], cos=cos_step, sin=sin_step, kv_cache=cache)
    assert step_out.shape == (2, 1, 64)
    assert cache.get_seq_length(0) == 5
    assert v_prefill.shape == (2, 2, 4, 16)


def test_differential_gqa_and_gated_deltanet() -> None:
    rope_half = RotaryEmbedding(dim=8)
    diff = DifferentialGQA(64, 4, 2, 16, sliding_window=4, layer_idx=0)
    x = torch.randn(2, 6, 64)
    cos, sin = rope_half(6, device=x.device, dtype=x.dtype)
    out_diff = diff(x, cos=cos, sin=sin)
    assert out_diff.shape == x.shape

    gdn = GatedDeltaNet(64, num_heads=4, head_dim=16, conv_kernel=4, layer_idx=1)
    cache = KVCache()
    out_gdn = gdn(x, document_ids=torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]]), kv_cache=cache)
    assert out_gdn.shape == x.shape
    assert cache.get_recurrent_state(1).ssm_state is not None
    assert "gated_deltanet" in ATTENTION_REGISTRY


def test_ffn_and_moe_alf_lb() -> None:
    x = torch.randn(2, 5, 32)
    assert SwiGLUFFN(32, 64)(x).shape == x.shape
    assert ReLU2FFN(32, 128)(x).shape == x.shape

    moe = MoESwiGLUFFN(32, 16, num_routed_experts=4, num_experts_per_tok=2, shared_expert_intermediate_size=32)
    moe.train()
    out, aux_loss = moe(x)
    assert out.shape == x.shape
    assert aux_loss.ndim == 0
    moe.update_moe_biases()
    assert "bias_std" in moe.get_aux_metrics()
    assert "moe_swiglu" in FFN_REGISTRY


def test_residuals_mhc_and_value_gate() -> None:
    mhc = ManifoldHyperConnection(num_streams=4, sinkhorn_iters=10)
    mat = mhc.mixing_matrix()
    assert torch.allclose(mat.sum(dim=-1), torch.ones(4), atol=1e-4)
    assert torch.allclose(mat.sum(dim=-2), torch.ones(4), atol=1e-4)

    x = torch.randn(2, 5, 16)
    h = ManifoldHyperConnection.expand(x, 4)
    h2 = mhc(h, lambda t: t * 0.5)
    assert h2.shape == (2, 5, 4, 16)
    assert ManifoldHyperConnection.collapse(h2).shape == x.shape

    gate = ValueResidualGate(num_kv_heads=2, hidden_size=16, mode="resformer")
    v = torch.randn(2, 2, 5, 8)
    assert gate(v, v * 0.5, x).shape == v.shape
