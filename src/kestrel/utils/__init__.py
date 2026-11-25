"""Shared utility helpers for Kestrel notebooks and text processing."""

from kestrel.utils.notebook import (
    configure_hf_cache,
    find_repo_root,
    get_hf_token,
    get_work_dir,
    is_kaggle,
)
from kestrel.utils.text import (
    WORD_RE,
    count_gpt2_tokens,
    get_gpt2_tokenizer,
    normalize_words,
)

__all__ = [
    "WORD_RE",
    "configure_hf_cache",
    "count_gpt2_tokens",
    "find_repo_root",
    "get_gpt2_tokenizer",
    "get_hf_token",
    "get_work_dir",
    "is_kaggle",
    "normalize_words",
]
