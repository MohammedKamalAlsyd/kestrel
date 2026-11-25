"""Lightweight KV and recurrent-state caches for Kestrel attention & hybrid layers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RecurrentState:
    """Per-layer recurrent cache for linear-time mixers (e.g. Gated DeltaNet)."""

    conv_state: torch.Tensor | None = None
    ssm_state: torch.Tensor | None = None


@dataclass
class KVCache:
    """Dynamic per-layer key/value (with optional SWA trimming) and recurrent-state cache."""

    key_cache: dict[int, torch.Tensor] = field(default_factory=dict)
    value_cache: dict[int, torch.Tensor] = field(default_factory=dict)
    seen_tokens: dict[int, int] = field(default_factory=dict)
    recurrent_cache: dict[int, RecurrentState] = field(default_factory=dict)

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        sliding_window: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append `(key, value)` of shape `(B, H_kv, T, D)` and return full cached tensors."""
        new_tokens = int(key.shape[-2])
        if layer_idx in self.key_cache:
            key = torch.cat((self.key_cache[layer_idx], key), dim=-2)
            value = torch.cat((self.value_cache[layer_idx], value), dim=-2)
        if sliding_window is not None and sliding_window > 0 and key.shape[-2] > sliding_window:
            key = key[:, :, -sliding_window:, :]
            value = value[:, :, -sliding_window:, :]
        self.key_cache[layer_idx] = key
        self.value_cache[layer_idx] = value
        self.seen_tokens[layer_idx] = self.seen_tokens.get(layer_idx, 0) + new_tokens
        return key, value

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return total number of tokens seen so far by `layer_idx` (or first cached layer)."""
        if layer_idx in self.seen_tokens:
            return self.seen_tokens[layer_idx]
        if self.seen_tokens:
            return next(iter(self.seen_tokens.values()))
        return 0

    def get_cached_length(self, layer_idx: int = 0) -> int:
        """Return physical KV length currently held in memory for `layer_idx`."""
        k = self.key_cache.get(layer_idx)
        return int(k.shape[-2]) if k is not None else 0

    def get_recurrent_state(self, layer_idx: int) -> RecurrentState:
        """Return or initialize the `RecurrentState` for `layer_idx`."""
        if layer_idx not in self.recurrent_cache:
            self.recurrent_cache[layer_idx] = RecurrentState()
        return self.recurrent_cache[layer_idx]

    def reset(self) -> None:
        """Clear all cached key/value and recurrent states."""
        self.key_cache.clear()
        self.value_cache.clear()
        self.seen_tokens.clear()
        self.recurrent_cache.clear()
