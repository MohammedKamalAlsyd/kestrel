"""Causal, sliding-window, padding, and packed-document attention mask utilities."""

from __future__ import annotations

import torch


def create_causal_mask(
    batch_size: int,
    query_len: int,
    kv_len: int,
    *,
    past_len: int = 0,
    sliding_window: int | None = None,
    document_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Build a `(B, 1, Q, K)` additive attention mask (`0.0` keep, `-inf` mask) or `None`.

    Returns `None` when a standard causal mask suffices (`past_len == 0`, no
    `sliding_window`, no `document_ids`, and no `attention_mask`), letting SDPA
    use its fast `is_causal=True` kernel directly.
    """
    needs_custom = (
        past_len > 0
        or (sliding_window is not None and sliding_window > 0)
        or document_ids is not None
        or attention_mask is not None
    )
    if not needs_custom:
        return None

    q_pos = torch.arange(past_len, past_len + query_len, device=device).unsqueeze(1)  # (Q, 1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)  # (1, K)

    allowed = k_pos <= q_pos  # causal: (Q, K)
    if sliding_window is not None and sliding_window > 0:
        allowed = allowed & (k_pos > q_pos - sliding_window)

    allowed = allowed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_len, kv_len)

    if document_ids is not None:
        # Block attention across packed document boundaries: (B, T) -> (B, 1, Q, K)
        q_docs = document_ids[:, -query_len:].unsqueeze(2)  # (B, Q, 1)
        k_docs = document_ids[:, :kv_len].unsqueeze(1)  # (B, 1, K)
        allowed = allowed & (q_docs == k_docs).unsqueeze(1)

    if attention_mask is not None:
        # Standard 1=keep, 0=pad key mask of shape (B, K)
        key_valid = attention_mask[:, :kv_len].bool().unsqueeze(1).unsqueeze(2)
        allowed = allowed & key_valid

    mask = torch.zeros((batch_size, 1, query_len, kv_len), device=device, dtype=dtype)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)
