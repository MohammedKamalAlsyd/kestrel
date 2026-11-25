"""Feed-forward network blocks and registry for Kestrel."""

from kestrel.models.ffn.dense import ReLU2FFN, SwiGLUFFN
from kestrel.models.ffn.moe import MoESwiGLUFFN

FFN_REGISTRY = {
    "swiglu": SwiGLUFFN,
    "relu2": ReLU2FFN,
    "moe_swiglu": MoESwiGLUFFN,
}

__all__ = [
    "FFN_REGISTRY",
    "MoESwiGLUFFN",
    "ReLU2FFN",
    "SwiGLUFFN",
]
