"""Unified `KestrelModel` and `KestrelForCausalLM` built from modular Kestrel blocks."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import GenerationMixin, PreTrainedModel

from kestrel.models.blocks import KestrelDecoderLayer
from kestrel.models.cache import KVCache
from kestrel.models.ffn import MoESwiGLUFFN
from kestrel.models.heads import MultiTokenPredictionHead
from kestrel.models.kestrel.configuration_kestrel import KestrelConfig
from kestrel.models.masking import create_causal_mask
from kestrel.models.norms import RMSNorm
from kestrel.models.outputs import KestrelCausalLMOutput
from kestrel.models.residuals import ManifoldHyperConnection
from kestrel.models.rope import RotaryEmbedding


class KestrelPreTrainedModel(PreTrainedModel):
    """Base HF `PreTrainedModel` with Kestrel weight initialization and checkpointing hooks."""

    config_class = KestrelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KestrelDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        std = float(self.config.initializer_range)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _apply_residual_init_scale(self) -> None:
        """Apply GPT-2/modern depth scaling (`std / sqrt(2L)`) to residual output projections."""
        if not getattr(self.config, "residual_init_scale", True):
            return
        scale_std = float(self.config.initializer_range) / math.sqrt(2.0 * max(1, self.config.num_hidden_layers))
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear) and name.endswith(("o_proj", "down_proj")):
                nn.init.normal_(mod.weight, mean=0.0, std=scale_std)


class KestrelModel(KestrelPreTrainedModel):
    """Backbone transformer / hybrid / MoE decoder returning hidden states, cache, and aux loss."""

    def __init__(self, config: KestrelConfig) -> None:
        super().__init__(config)
        self.config = config
        self.use_mhc = config.residual_type == "mhc"
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        rope_dim = config.head_dim // 2 if config.attention_type == "differential_gqa" else config.head_dim
        self.rotary_emb = RotaryEmbedding(
            dim=rope_dim,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
        )
        self.layers = nn.ModuleList(
            [KestrelDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()
        self._apply_residual_init_scale()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        document_ids: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None, torch.Tensor | None]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if use_cache and past_key_values is None:
            past_key_values = KVCache()

        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None and past_len > 0:
            position_ids = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(0).expand(bsz, -1)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(seq_len, device=device, dtype=hidden_states.dtype, position_ids=position_ids)

        if self.use_mhc:
            hidden_states = ManifoldHyperConnection.expand(hidden_states, self.config.mhc_num_streams)

        # Precompute shared attention masks ONCE per microbatch (instead of 30x per layer)
        shared_masks: dict[int | None, torch.Tensor | None] = {}
        use_shared_masks = past_key_values is None
        if use_shared_masks:
            unique_windows = {
                layer.sliding_window
                for layer in self.layers
                if layer.mixer_kind in ("gqa", "differential_gqa")
            }
            for win in unique_windows:
                shared_masks[win] = create_causal_mask(
                    bsz,
                    seq_len,
                    seq_len,
                    past_len=0,
                    sliding_window=win,
                    document_ids=document_ids,
                    attention_mask=attention_mask,
                    device=device,
                    dtype=hidden_states.dtype,
                )

        v_first: torch.Tensor | None = None
        aux_losses: list[torch.Tensor] = []

        for idx, layer in enumerate(self.layers):
            layer_mask = shared_masks.get(layer.sliding_window) if use_shared_masks else None
            if self.gradient_checkpointing and self.training and past_key_values is None:
                def _layer_fn(
                    h: torch.Tensor,
                    c: torch.Tensor,
                    s: torch.Tensor,
                    vf: torch.Tensor | None,
                    layer_mod: KestrelDecoderLayer = layer,
                    lmask: torch.Tensor | None = layer_mask,
                ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
                    return layer_mod(
                        h,
                        cos=c,
                        sin=s,
                        v_first=vf,
                        document_ids=document_ids,
                        attention_mask=attention_mask,
                        precomputed_mask=lmask,
                        has_precomputed_mask=True,
                        kv_cache=None,
                    )

                hidden_states, v_raw, layer_aux = torch_checkpoint(
                    _layer_fn, hidden_states, cos, sin, v_first, use_reentrant=False
                )
            else:
                hidden_states, v_raw, layer_aux = layer(
                    hidden_states,
                    cos=cos,
                    sin=sin,
                    v_first=v_first,
                    document_ids=document_ids,
                    attention_mask=attention_mask,
                    precomputed_mask=layer_mask,
                    has_precomputed_mask=use_shared_masks,
                    kv_cache=past_key_values,
                )

            if idx == 0 and v_raw is not None:
                v_first = v_raw
            if layer_aux is not None:
                aux_losses.append(layer_aux)

        if self.use_mhc:
            hidden_states = ManifoldHyperConnection.collapse(hidden_states)

        hidden_states = self.norm(hidden_states)
        total_aux = sum(aux_losses) if aux_losses else None
        return hidden_states, (past_key_values if use_cache else None), total_aux


class KestrelForCausalLM(KestrelPreTrainedModel, GenerationMixin):
    """Causal LM wrapper with tied LM head, optional MTP training head, and active-param stats."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: KestrelConfig) -> None:
        super().__init__(config)
        self.model = KestrelModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_head = (
            MultiTokenPredictionHead(
                config.hidden_size,
                num_predict_tokens=config.mtp_num_predict_tokens,
                intermediate_size=config.intermediate_size,
                rms_norm_eps=config.rms_norm_eps,
            )
            if config.mtp_num_predict_tokens > 1
            else None
        )

        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def count_active_parameters(self, *, include_mtp: bool = False) -> int:
        """Count unique active parameters per forward token (accounting for MoE top-k routing)."""
        routed_ids: set[int] = set()
        mtp_ids: set[int] = set()
        active_routed = 0

        if self.mtp_head is not None and not include_mtp:
            mtp_ids.update(id(p) for p in self.mtp_head.parameters())

        for mod in self.modules():
            if isinstance(mod, MoESwiGLUFFN):
                for exp in mod.experts:
                    routed_ids.update(id(p) for p in exp.parameters())
                if len(mod.experts) > 0:
                    one_expert_params = sum(p.numel() for p in mod.experts[0].parameters())
                    active_routed += one_expert_params * mod.num_experts_per_tok

        seen: set[int] = set()
        dense_active = 0
        for p in self.parameters():
            pid = id(p)
            if pid in seen or pid in routed_ids or pid in mtp_ids:
                continue
            seen.add(pid)
            dense_active += p.numel()
        return dense_active + active_routed

    def get_aux_metrics(self) -> dict[str, float]:
        """Aggregate MoE router load-balancing metrics across layers for MLflow logging."""
        bias_stds: list[float] = []
        aux_vals: list[float] = []
        for mod in self.modules():
            if isinstance(mod, MoESwiGLUFFN):
                m = mod.get_aux_metrics()
                if "bias_std" in m:
                    bias_stds.append(m["bias_std"])
                if "seq_aux_loss" in m:
                    aux_vals.append(m["seq_aux_loss"])
        report: dict[str, float] = {}
        if bias_stds:
            report["moe/bias_std_mean"] = sum(bias_stds) / len(bias_stds)
        if aux_vals:
            report["moe/seq_aux_loss_mean"] = sum(aux_vals) / len(aux_vals)
        return report

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        document_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        mtp_labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        compute_logits: bool = True,
        **kwargs: Any,
    ) -> KestrelCausalLMOutput:
        use_cache_flag = self.config.use_cache if use_cache is None else bool(use_cache)
        if self.training:
            use_cache_flag = False

        hidden_states, cache_out, aux_loss = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            document_ids=document_ids,
            use_cache=use_cache_flag,
        )
        logits = self.lm_head(hidden_states) if (compute_logits or labels is not None) else None

        loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].float().contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        # Multi-Token Prediction auxiliary loss (training only; zero inference cost)
        mtp_loss: torch.Tensor | None = None
        if self.training and self.mtp_head is not None:
            if mtp_labels is not None:
                mtp_loss = self.mtp_head(
                    hidden_states,
                    mtp_labels,
                    embed_tokens=self.model.embed_tokens,
                    lm_head=self.lm_head,
                    document_ids=document_ids,
                )
            elif labels is not None and labels.shape[1] > 1:
                mtp_loss = self.mtp_head(
                    hidden_states[:, :-1, :],
                    labels[:, 1:],
                    embed_tokens=self.model.embed_tokens,
                    lm_head=self.lm_head,
                    document_ids=document_ids[:, :-1] if document_ids is not None else None,
                )

        return KestrelCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=cache_out,
            aux_loss=aux_loss,
            mtp_loss=mtp_loss,
            last_hidden_state=hidden_states,
        )

    @torch.no_grad()
    def generate_fast(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Fast KV-cached autoregressive generation loop supporting all 4 Kestrel tracks."""
        was_training = self.training
        self.eval()
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id

        cache = KVCache()
        generated = input_ids
        out = self(input_ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values

        for _ in range(max_new_tokens):
            next_logits = out.logits[:, -1, :]
            if temperature > 0.0:
                probs = torch.softmax(next_logits.float() / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_tok), dim=1)
            if eos is not None and (next_tok == eos).all():
                break
            out = self(next_tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values

        if was_training:
            self.train()
        return generated
