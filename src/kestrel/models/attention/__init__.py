"""Attention and linear sequence-mixer blocks and registry for Kestrel."""

from kestrel.models.attention.differential_gqa import DifferentialGQA
from kestrel.models.attention.gated_deltanet import GatedDeltaNet
from kestrel.models.attention.gqa import GroupedQueryAttention

ATTENTION_REGISTRY = {
    "gqa": GroupedQueryAttention,
    "differential_gqa": DifferentialGQA,
    "gated_deltanet": GatedDeltaNet,
}

__all__ = [
    "ATTENTION_REGISTRY",
    "DifferentialGQA",
    "GatedDeltaNet",
    "GroupedQueryAttention",
]
