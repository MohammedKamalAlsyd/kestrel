"""Shared text normalization, n-gram extraction, and GPT-2 tokenization utilities."""

from __future__ import annotations

import re
from typing import Any

try:
    import tiktoken
except ImportError:  # pragma: no cover - installed on Kaggle / via extras
    tiktoken = None  # type: ignore[assignment]

WORD_RE = re.compile(r"[a-z]+")
ALNUM_WORD_RE = re.compile(r"[a-z0-9]+")


def get_gpt2_tokenizer() -> Any:
    """Return the canonical GPT-2 BPE tokenizer via tiktoken."""
    if tiktoken is None:
        raise ImportError(
            "tiktoken is required for GPT-2 tokenization. Install with: pip install tiktoken"
        )
    return tiktoken.get_encoding("gpt2")


def count_gpt2_tokens(text: str, enc: Any | None = None) -> int:
    """Count GPT-2 BPE tokens safely, treating literal special-token strings as normal text."""
    tokenizer = enc or get_gpt2_tokenizer()
    return len(tokenizer.encode(text, disallowed_special=()))


def normalize_words(text: str) -> list[str]:
    """Extract lowercase alphabetic word tokens."""
    return WORD_RE.findall(text.lower())


def normalize_alnum_words(text: str) -> list[str]:
    """Extract lowercase alphanumeric word tokens."""
    return ALNUM_WORD_RE.findall(text.lower())


def extract_ngrams_set(text: str, n: int = 13) -> set[str]:
    """Extract unique normalized alphanumeric n-grams from a single text string."""
    words = normalize_alnum_words(text)
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
