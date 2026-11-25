"""Hugging Face `PretrainedConfig` for all four Kestrel architectures."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class KestrelConfig(PretrainedConfig):
    """Unified configuration covering Dense, Hybrid GDN, MoE-16, and Diff-mHC-ReLU2 Kestrel models."""

    model_type = "kestrel"

    def __init__(
        self,
        # Core vocabulary & dimensions
        vocab_size: int = 50_304,
        hidden_size: int = 960,
        num_hidden_layers: int = 30,
        max_position_embeddings: int = 2048,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 50_256,
        eos_token_id: int = 50_256,
        pad_token_id: int = 50_256,
        # Attention & sequence mixer
        mixer_type: str = "attention",  # "attention" | "gated_deltanet_hybrid"
        attention_type: str = "gqa",  # "gqa" | "differential_gqa"
        num_attention_heads: int = 15,
        num_key_value_heads: int = 3,
        head_dim: int = 64,
        attention_bias: bool = False,
        diff_attn_lambda_init: float = 0.8,
        # Track B: Gated DeltaNet hybrid parameters
        hybrid_attn_every_n_layers: int = 4,
        deltanet_num_heads: int = 8,
        deltanet_head_dim: int = 128,
        deltanet_conv_kernel: int = 4,
        deltanet_use_gate: bool = True,
        deltanet_use_short_conv: bool = True,
        # Positional encoding & sliding window
        positional_encoding: str = "rnope_swa",  # "rnope_swa" | "rope_swa" | "rope"
        rope_theta: float = 10_000.0,
        rope_fp32_inv_freq: bool = True,
        swa_window_size: int | None = 512,
        global_attn_every_n_layers: int = 4,
        global_layers_use_nope: bool = True,
        # Normalization & residuals
        norm_type: str = "peri_rmsnorm",  # "peri_rmsnorm" | "rmsnorm"
        qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        residual_type: str = "standard",  # "standard" | "value_residual" | "mhc"
        value_residual_gating: str = "satformer",  # "satformer" | "resformer"
        mhc_num_streams: int = 4,
        mhc_sinkhorn_iters: int = 10,
        # Feed-forward & Track C MoE
        ffn_type: str = "swiglu",  # "swiglu" | "relu2" | "moe_swiglu"
        intermediate_size: int = 2688,
        ffn_bias: bool = False,
        num_routed_experts: int = 16,
        num_experts_per_tok: int = 2,
        expert_intermediate_size: int = 512,
        num_shared_experts: int = 1,
        shared_expert_intermediate_size: int = 1024,
        router_scoring: str = "sigmoid",  # "sigmoid" | "softmax"
        load_balancing: str = "aux_loss_free",
        alf_lb_bias_update_rate: float = 1e-3,
        seq_aux_loss_alpha: float = 1e-4,
        # Multi-Token Prediction & initialization
        mtp_num_predict_tokens: int = 1,
        initializer_range: float = 0.02,
        residual_init_scale: bool = True,
        use_cache: bool = True,
        # Kestrel metadata fields (preserved in config.json)
        blueprint: str | None = None,
        track: str | None = None,
        hf_repo_id: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.mixer_type = mixer_type
        self.attention_type = attention_type
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.diff_attn_lambda_init = diff_attn_lambda_init

        self.hybrid_attn_every_n_layers = hybrid_attn_every_n_layers
        self.deltanet_num_heads = deltanet_num_heads
        self.deltanet_head_dim = deltanet_head_dim
        self.deltanet_conv_kernel = deltanet_conv_kernel
        self.deltanet_use_gate = deltanet_use_gate
        self.deltanet_use_short_conv = deltanet_use_short_conv

        self.positional_encoding = positional_encoding
        self.rope_theta = rope_theta
        self.rope_fp32_inv_freq = rope_fp32_inv_freq
        self.swa_window_size = swa_window_size
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.global_layers_use_nope = global_layers_use_nope

        self.norm_type = norm_type
        self.qk_norm = qk_norm
        self.rms_norm_eps = rms_norm_eps
        self.residual_type = residual_type
        self.value_residual_gating = value_residual_gating
        self.mhc_num_streams = mhc_num_streams
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters

        self.ffn_type = ffn_type
        self.intermediate_size = intermediate_size
        self.ffn_bias = ffn_bias
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_intermediate_size = expert_intermediate_size
        self.num_shared_experts = num_shared_experts
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.router_scoring = router_scoring
        self.load_balancing = load_balancing
        self.alf_lb_bias_update_rate = alf_lb_bias_update_rate
        self.seq_aux_loss_alpha = seq_aux_loss_alpha

        self.mtp_num_predict_tokens = mtp_num_predict_tokens
        self.initializer_range = initializer_range
        self.residual_init_scale = residual_init_scale
        self.use_cache = use_cache

        self.blueprint = blueprint
        self.track = track
        self.hf_repo_id = hf_repo_id
        self.description = description

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
