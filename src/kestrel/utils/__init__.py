"""Shared utility helpers for Kestrel notebooks, text processing, and experiments."""

from kestrel.utils.notebook import (
    DEFAULT_BRANCH,
    DEFAULT_IMAGES_SUBDIR,
    DEFAULT_RAW_BASE_URL,
    DEFAULT_REPO_URL,
    configure_hf_cache,
    ensure_kestrel_repo,
    find_repo_root,
    get_hf_token,
    get_work_dir,
    is_kaggle,
    resolve_asset_source,
    show_explainer_visual,
)
from kestrel.utils.text import (
    ALNUM_WORD_RE,
    WORD_RE,
    count_gpt2_tokens,
    extract_ngrams_set,
    get_gpt2_tokenizer,
    normalize_alnum_words,
    normalize_words,
)

__all__ = [
    "ALNUM_WORD_RE",
    "DEFAULT_BRANCH",
    "DEFAULT_IMAGES_SUBDIR",
    "DEFAULT_RAW_BASE_URL",
    "DEFAULT_REPO_URL",
    "WORD_RE",
    "configure_hf_cache",
    "count_gpt2_tokens",
    "ensure_kestrel_repo",
    "extract_ngrams_set",
    "find_repo_root",
    "get_gpt2_tokenizer",
    "get_hf_token",
    "get_work_dir",
    "is_kaggle",
    "normalize_alnum_words",
    "normalize_words",
    "resolve_asset_source",
    "show_explainer_visual",
]
