"""Kestrel model building blocks, unified architectures, and model builder."""

from kestrel.models import (
    activations,
    attention,
    blocks,
    cache,
    ffn,
    heads,
    kestrel,
    masking,
    norms,
    outputs,
    residuals,
    rope,
)
from kestrel.models.builder import (
    BLUEPRINT_CONFIGS,
    KestrelModelBuilder,
    register_kestrel_models,
)
from kestrel.models.kestrel import KestrelConfig, KestrelForCausalLM, KestrelModel
from kestrel.models.outputs import KestrelCausalLMOutput

register_kestrel_models()

__all__ = [
    "BLUEPRINT_CONFIGS",
    "KestrelCausalLMOutput",
    "KestrelConfig",
    "KestrelForCausalLM",
    "KestrelModel",
    "KestrelModelBuilder",
    "activations",
    "attention",
    "blocks",
    "cache",
    "ffn",
    "heads",
    "kestrel",
    "masking",
    "norms",
    "outputs",
    "register_kestrel_models",
    "residuals",
    "rope",
]
