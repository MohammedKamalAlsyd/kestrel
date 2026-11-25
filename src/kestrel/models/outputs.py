"""Model output containers compatible with Hugging Face `ModelOutput` and `train.py`."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput

from kestrel.models.cache import KVCache


@dataclass
class KestrelCausalLMOutput(ModelOutput):
    """Unified output container returned by `KestrelForCausalLM`."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    past_key_values: KVCache | None = None
    aux_loss: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
