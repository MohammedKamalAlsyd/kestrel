"""Unified Kestrel decoder layer composing mixer/attention, FFN/MoE, Peri-RMSNorm, and residuals."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kestrel.models.attention import DifferentialGQA, GatedDeltaNet, GroupedQueryAttention
from kestrel.models.cache import KVCache
from kestrel.models.ffn import MoESwiGLUFFN, ReLU2FFN, SwiGLUFFN
from kestrel.models.norms import RMSNorm
from kestrel.models.residuals import ManifoldHyperConnection


class KestrelDecoderLayer(nn.Module):
    """Single Kestrel layer supporting Tracks A (Dense), B (Hybrid GDN), C (MoE), and Frontier (Diff+mHC)."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.use_peri_norm = getattr(config, "norm_type", "peri_rmsnorm") == "peri_rmsnorm"
        self.use_mhc = getattr(config, "residual_type", "standard") == "mhc"

        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=eps) if self.use_peri_norm else nn.Identity()
        self.pre_ffn_norm = RMSNorm(config.hidden_size, eps=eps)
        self.post_ffn_norm = RMSNorm(config.hidden_size, eps=eps) if self.use_peri_norm else nn.Identity()

        # Sequence mixer / attention selection
        self.mixer_kind, self.mixer = self._build_mixer(config, layer_idx)
        self.sliding_window: int | None = getattr(self.mixer, "sliding_window", None)

        # Feed-forward selection
        self.is_moe = getattr(config, "ffn_type", "swiglu") == "moe_swiglu"
        self.ffn = self._build_ffn(config)

        # Residual connection wrapper (mHC 4-stream or standard residual)
        if self.use_mhc:
            streams = int(getattr(config, "mhc_num_streams", 4))
            iters = int(getattr(config, "mhc_sinkhorn_iters", 10))
            self.mhc_attn = ManifoldHyperConnection(num_streams=streams, sinkhorn_iters=iters)
            self.mhc_ffn = ManifoldHyperConnection(num_streams=streams, sinkhorn_iters=iters)
        else:
            self.mhc_attn = None
            self.mhc_ffn = None

    @staticmethod
    def _build_mixer(config: Any, layer_idx: int) -> tuple[str, nn.Module]:
        mixer_type = getattr(config, "mixer_type", "attention")
        eps = float(getattr(config, "rms_norm_eps", 1e-6))

        # Track B: 3:1 Gated DeltaNet + local SWA-GQA hybrid
        if mixer_type == "gated_deltanet_hybrid":
            every_n = int(getattr(config, "hybrid_attn_every_n_layers", 4))
            is_attn_layer = (layer_idx + 1) % every_n == 0
            if not is_attn_layer:
                return (
                    "gated_deltanet",
                    GatedDeltaNet(
                        config.hidden_size,
                        num_heads=int(getattr(config, "deltanet_num_heads", 8)),
                        head_dim=int(getattr(config, "deltanet_head_dim", 128)),
                        conv_kernel=int(getattr(config, "deltanet_conv_kernel", 4)),
                        use_gate=bool(getattr(config, "deltanet_use_gate", True)),
                        use_short_conv=bool(getattr(config, "deltanet_use_short_conv", True)),
                        rms_norm_eps=eps,
                        layer_idx=layer_idx,
                    ),
                )
            return (
                "gqa",
                GroupedQueryAttention(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    bias=bool(getattr(config, "attention_bias", False)),
                    qk_norm=bool(getattr(config, "qk_norm", True)),
                    rms_norm_eps=eps,
                    sliding_window=getattr(config, "swa_window_size", 512),
                    use_rope=True,
                    value_residual_gating=None,
                    layer_idx=layer_idx,
                ),
            )

        # RNoPE-SWA vs RoPE-SWA sliding window & RoPE scheduling
        pos_enc = getattr(config, "positional_encoding", "rnope_swa")
        swa = getattr(config, "swa_window_size", None)
        every_n = int(getattr(config, "global_attn_every_n_layers", 4))
        global_nope = bool(getattr(config, "global_layers_use_nope", True))

        if pos_enc == "rnope_swa":
            is_global = (layer_idx + 1) % every_n == 0
            layer_swa = None if is_global else swa
            layer_rope = (not global_nope) if is_global else True
        elif pos_enc == "rope_swa":
            layer_swa = swa
            layer_rope = True
        else:
            layer_swa = None
            layer_rope = True

        attn_type = getattr(config, "attention_type", "gqa")
        if attn_type == "differential_gqa":
            return (
                "differential_gqa",
                DifferentialGQA(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    bias=bool(getattr(config, "attention_bias", False)),
                    qk_norm=bool(getattr(config, "qk_norm", True)),
                    rms_norm_eps=eps,
                    diff_attn_lambda_init=float(getattr(config, "diff_attn_lambda_init", 0.8)),
                    sliding_window=layer_swa,
                    use_rope=layer_rope,
                    layer_idx=layer_idx,
                ),
            )

        val_gating = (
            getattr(config, "value_residual_gating", None)
            if getattr(config, "residual_type", "standard") == "value_residual"
            else None
        )
        return (
            "gqa",
            GroupedQueryAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                bias=bool(getattr(config, "attention_bias", False)),
                qk_norm=bool(getattr(config, "qk_norm", True)),
                rms_norm_eps=eps,
                sliding_window=layer_swa,
                use_rope=layer_rope,
                value_residual_gating=val_gating,
                layer_idx=layer_idx,
            ),
        )

    @staticmethod
    def _build_ffn(config: Any) -> nn.Module:
        ffn_type = getattr(config, "ffn_type", "swiglu")
        bias = bool(getattr(config, "ffn_bias", False))
        if ffn_type == "moe_swiglu":
            return MoESwiGLUFFN(
                config.hidden_size,
                int(getattr(config, "expert_intermediate_size", 512)),
                num_routed_experts=int(getattr(config, "num_routed_experts", 16)),
                num_experts_per_tok=int(getattr(config, "num_experts_per_tok", 2)),
                num_shared_experts=int(getattr(config, "num_shared_experts", 1)),
                shared_expert_intermediate_size=int(getattr(config, "shared_expert_intermediate_size", 1024)),
                router_scoring=str(getattr(config, "router_scoring", "sigmoid")),
                bias=bias,
                alf_lb_bias_update_rate=float(getattr(config, "alf_lb_bias_update_rate", 1e-3)),
                seq_aux_loss_alpha=float(getattr(config, "seq_aux_loss_alpha", 1e-4)),
            )
        if ffn_type == "relu2":
            return ReLU2FFN(config.hidden_size, int(config.intermediate_size), bias=bias)
        return SwiGLUFFN(config.hidden_size, int(config.intermediate_size), bias=bias)

    def _run_mixer(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        v_first: torch.Tensor | None,
        document_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        precomputed_mask: Any = None,
        has_precomputed_mask: bool = False,
        kv_cache: KVCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed = self.pre_attn_norm(x)
        mask_kwargs = {"precomputed_mask": precomputed_mask} if has_precomputed_mask else {}
        if self.mixer_kind == "gqa":
            out, v_raw = self.mixer(
                normed,
                cos=cos,
                sin=sin,
                v_first=v_first,
                document_ids=document_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                **mask_kwargs,
            )
            return self.post_attn_norm(out), v_raw
        if self.mixer_kind == "differential_gqa":
            out = self.mixer(
                normed,
                cos=cos,
                sin=sin,
                document_ids=document_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                **mask_kwargs,
            )
            return self.post_attn_norm(out), None
        out = self.mixer(normed, document_ids=document_ids, kv_cache=kv_cache)
        return self.post_attn_norm(out), None

    def _run_ffn(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed = self.pre_ffn_norm(x)
        if self.is_moe:
            out, aux_loss = self.ffn(normed)
            return self.post_ffn_norm(out), aux_loss
        return self.post_ffn_norm(self.ffn(normed)), None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        v_first: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        precomputed_mask: Any = None,
        has_precomputed_mask: bool = False,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return `(hidden_states, v_raw_or_None, aux_loss_or_None)`."""
        v_raw: torch.Tensor | None = None
        aux_loss: torch.Tensor | None = None

        if self.use_mhc:
            assert self.mhc_attn is not None and self.mhc_ffn is not None
            x_in = self.mhc_attn.pre_aggregate(hidden_states)
            attn_out, v_raw = self._run_mixer(
                x_in,
                cos=cos,
                sin=sin,
                v_first=v_first,
                document_ids=document_ids,
                attention_mask=attention_mask,
                precomputed_mask=precomputed_mask,
                has_precomputed_mask=has_precomputed_mask,
                kv_cache=kv_cache,
            )
            hidden_states = self.mhc_attn.post_update(hidden_states, attn_out)

            f_in = self.mhc_ffn.pre_aggregate(hidden_states)
            ffn_out, aux_loss = self._run_ffn(f_in)
            hidden_states = self.mhc_ffn.post_update(hidden_states, ffn_out)
        else:
            attn_out, v_raw = self._run_mixer(
                hidden_states,
                cos=cos,
                sin=sin,
                v_first=v_first,
                document_ids=document_ids,
                attention_mask=attention_mask,
                precomputed_mask=precomputed_mask,
                has_precomputed_mask=has_precomputed_mask,
                kv_cache=kv_cache,
            )
            hidden_states = hidden_states + attn_out

            ffn_out, aux_loss = self._run_ffn(hidden_states)
            hidden_states = hidden_states + ffn_out

        return hidden_states, v_raw, aux_loss
