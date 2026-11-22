"""Unit tests for shared text processing utilities in kestrel.utils.text."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kestrel.utils import text as text_utils


def test_normalize_words_and_alnum_words():
    sample = "GPT-2 XL (1.5B) uses BPE tokens, e.g. 'hello_world'!"
    assert text_utils.normalize_words(sample) == [
        "gpt",
        "xl",
        "b",
        "uses",
        "bpe",
        "tokens",
        "e",
        "g",
        "hello",
        "world",
    ]
    assert text_utils.normalize_alnum_words(sample) == [
        "gpt",
        "2",
        "xl",
        "1",
        "5b",
        "uses",
        "bpe",
        "tokens",
        "e",
        "g",
        "hello",
        "world",
    ]


def test_extract_ngrams_set():
    assert text_utils.extract_ngrams_set("one two three", n=5) == set()
    ngs = text_utils.extract_ngrams_set("One two, 3 four five!", n=3)
    assert ngs == {"one two 3", "two 3 four", "3 four five"}


def test_count_gpt2_tokens_passes_disallowed_special_empty():
    fake_enc = MagicMock()
    fake_enc.encode.return_value = [1, 2, 3, 4]
    with patch.object(text_utils, "get_gpt2_tokenizer", return_value=fake_enc):
        n = text_utils.count_gpt2_tokens("hello <|endoftext|>")
    assert n == 4
    fake_enc.encode.assert_called_once_with("hello <|endoftext|>", disallowed_special=())
