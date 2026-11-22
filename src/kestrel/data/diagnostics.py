"""Reusable pretraining dataset diagnostics for Kestrel (350M-700M GPT-2 successor).

Provides multi-shard randomized streaming and constant-RAM full-stream aggregation
alongside three core non-overlapping diagnostic pillars:
1. Gopher/FineWeb within-document n-gram repetition (n=2..10) & cross-document 7-gram boilerplate.
2. GPT-2 & evaluation benchmark domain alignment (Unigram/Bigram JSD & Top-10k Jaccard).
3. 13-gram exact-overlap contamination detection against evaluation test splits.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from kestrel.utils.text import (
    ALNUM_WORD_RE,
    WORD_RE,
    count_gpt2_tokens,
    extract_ngrams_set,
    get_gpt2_tokenizer,
    normalize_alnum_words,
    normalize_words,
)

# ---------------------------------------------------------------------------
# Constants & Thresholds
# ---------------------------------------------------------------------------

# Gopher / FineWeb repetition thresholds (Rae et al., 2021; Penedo et al., 2024).
GOPHER_TOP_NGRAM_THRESHOLDS: dict[int, float] = {
    2: 0.20,
    3: 0.18,
    4: 0.16,
}
GOPHER_DUP_NGRAM_THRESHOLDS: dict[int, float] = {
    5: 0.15,
    6: 0.14,
    7: 0.13,
    8: 0.12,
    9: 0.11,
    10: 0.10,
}


# ---------------------------------------------------------------------------
# Dataset & Benchmark Specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration for streaming a candidate pretraining dataset from Hugging Face."""

    key: str
    display_name: str
    repo_id: str
    config: str | None = None
    split: str = "train"
    text_field: str = "text"
    license: str = "Unknown"
    gated: bool = False
    is_control: bool = False
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    fallbacks: tuple[dict[str, Any], ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class EvalSpec:
    """Configuration for loading a GPT-2 target evaluation benchmark."""

    key: str
    display_name: str
    repo_id: str
    config: str | None = None
    split: str = "test"
    kind: str = "text"  # one of: "hellaswag", "arc", "lambada", "wikitext"


CANDIDATE_DATASETS: dict[str, DatasetSpec] = {
    "fineweb_edu": DatasetSpec(
        key="fineweb_edu",
        display_name="FineWeb-Edu (10BT)",
        repo_id="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        split="train",
        text_field="text",
        license="ODC-By v1.0",
        gated=False,
        notes="Educational classifier-filtered Common Crawl baseline (~1.3T full / 10BT sample).",
    ),
    "dclm_baseline": DatasetSpec(
        key="dclm_baseline",
        display_name="DCLM-Baseline 1.0",
        repo_id="mlfoundations/dclm-baseline-1.0-parquet",
        config=None,
        split="train",
        text_field="text",
        license="CC-BY-4.0",
        gated=False,
        fallbacks=(
            {
                "repo_id": "json",
                "config": None,
                "split": "train",
                "load_kwargs": {
                    "data_files": "hf://datasets/mlfoundations/dclm-baseline-1.0/**/*_processed.jsonl.zst"
                },
            },
        ),
        notes="FastText-filtered Common Crawl (~3.8T tokens); uses native Parquet mirror for fast streaming.",
    ),
    "nemotron_cc": DatasetSpec(
        key="nemotron_cc",
        display_name="Nemotron-CC v2.1 (HQ)",
        repo_id="nvidia/Nemotron-CC-v2.1",
        config=None,
        split="train",
        text_field="text",
        license="NVIDIA Data Agreement for Model Training",
        gated=True,
        load_kwargs={"data_dir": "High-Quality"},
        fallbacks=(
            {
                "repo_id": "nvidia/Nemotron-CC-v2",
                "config": None,
                "split": "train",
                "load_kwargs": {"data_dir": "High-Quality"},
            },
            {
                "repo_id": "nvidia/Nemotron-Pretraining-Dataset-sample",
                "config": "Nemotron-CC-v2-High-Quality",
                "split": "train",
                "load_kwargs": {},
            },
        ),
        notes="Gated NVIDIA Common Crawl HQ + synthetic rephrasing corpus; requires accepted HF token.",
    ),
    "climbmix": DatasetSpec(
        key="climbmix",
        display_name="Nemotron-ClimbMix (400B)",
        repo_id="karpathy/climbmix-400b-shuffle",
        config=None,
        split="train",
        text_field="text",
        license="CC-BY-NC-4.0 (upstream NVIDIA; MIT mirror)",
        gated=False,
        fallbacks=(
            {
                "repo_id": "OptimalScale/ClimbMix",
                "config": None,
                "split": "train",
                "load_kwargs": {},
            },
        ),
        notes=(
            "2025 NeurIPS CLIMB cluster-bootstrapped 400B mix optimized via 62M-350M proxy models; "
            "outperforms FineWeb-Edu & DCLM on sub-1B LMs."
        ),
    ),
    "finepdfs_edu": DatasetSpec(
        key="finepdfs_edu",
        display_name="FinePDFs-Edu (eng_Latn)",
        repo_id="HuggingFaceFW/finepdfs-edu",
        config="eng_Latn",
        split="train",
        text_field="text",
        license="ODC-By v1.0",
        gated=False,
        fallbacks=(
            {
                "repo_id": "HuggingFaceFW/finepdfs",
                "config": "eng_Latn",
                "split": "train",
                "load_kwargs": {},
            },
        ),
        notes=(
            "2025-2026 high-density long-form educational PDF extraction corpus; "
            "key component of winning sub-1B mixtures alongside DCLM and FineWeb-Edu."
        ),
    ),
    "openwebtext": DatasetSpec(
        key="openwebtext",
        display_name="OpenWebText (GPT-2 Control)",
        repo_id="Skylion007/openwebtext",
        config=None,
        split="train",
        text_field="text",
        license="Public Domain / CC0 (recreation of WebText)",
        gated=False,
        is_control=True,
        fallbacks=(
            {
                "repo_id": "parquet",
                "config": None,
                "split": "train",
                "load_kwargs": {
                    "data_files": "hf://datasets/Skylion007/openwebtext@refs/convert/parquet/plain_text/train/*.parquet"
                },
            },
            {
                "repo_id": "stas/openwebtext-10k",
                "config": None,
                "split": "train",
                "load_kwargs": {},
            },
        ),
        notes="Open recreation of OpenAI WebText (Reddit >=3 karma outbound links); GPT-2 distribution control.",
    ),
}

EVAL_BENCHMARKS: dict[str, EvalSpec] = {
    "hellaswag": EvalSpec(
        key="hellaswag",
        display_name="HellaSwag (val)",
        repo_id="Rowan/hellaswag",
        config=None,
        split="validation",
        kind="hellaswag",
    ),
    "arc_easy": EvalSpec(
        key="arc_easy",
        display_name="ARC-Easy (test)",
        repo_id="allenai/ai2_arc",
        config="ARC-Easy",
        split="test",
        kind="arc",
    ),
    "lambada": EvalSpec(
        key="lambada",
        display_name="LAMBADA (test)",
        repo_id="EleutherAI/lambada_openai",
        config="en",
        split="test",
        kind="lambada",
    ),
    "wikitext103": EvalSpec(
        key="wikitext103",
        display_name="WikiText-103 (test)",
        repo_id="Salesforce/wikitext",
        config="wikitext-103-raw-v1",
        split="test",
        kind="wikitext",
    ),
}


# ---------------------------------------------------------------------------
# Multi-Shard Randomized Streaming & Caching Helpers
# ---------------------------------------------------------------------------


def _load_hf_iterable(spec: DatasetSpec, hf_token: str | None = None) -> Any:
    """Open a Hugging Face streaming IterableDataset with automatic fallback attempts."""
    from datasets import load_dataset

    attempts: list[dict[str, Any]] = [
        {
            "repo_id": spec.repo_id,
            "config": spec.config,
            "split": spec.split,
            "load_kwargs": dict(spec.load_kwargs),
        },
        *[dict(fb) for fb in spec.fallbacks],
    ]

    last_exc: Exception | None = None
    for attempt in attempts:
        kwargs: dict[str, Any] = {
            "split": attempt.get("split", spec.split),
            "streaming": True,
            **attempt.get("load_kwargs", {}),
        }
        if hf_token:
            kwargs["token"] = hf_token
        try:
            config = attempt.get("config")
            if config is not None:
                ds = load_dataset(attempt["repo_id"], config, **kwargs)
            else:
                ds = load_dataset(attempt["repo_id"], **kwargs)
            _ = next(iter(ds))
            return ds
        except Exception as exc:  # pragma: no cover - network/HF runtime dependent
            last_exc = exc
            continue

    raise RuntimeError(
        f"Failed to stream '{spec.key}' ({spec.repo_id}). "
        f"{'Check HF_TOKEN / gated repo access. ' if spec.gated else ''}"
        f"Last error: {last_exc}"
    ) from last_exc


def iter_randomized_shards(
    ds: Any,
    target_tokens: int | None = None,
    max_docs: int | None = None,
    random_sample: bool = True,
    num_shards_to_sample: int = 8,
    shuffle_buffer_size: int = 2_000,
    seed: int = 42,
) -> Iterator[tuple[int, Iterator[dict[str, Any]]]]:
    """Yield `(shard_quota_idx, row_iterator)` streams distributed across remote shards."""
    if not random_sample:
        yield (0, iter(ds))
        return

    n_shards = int(getattr(ds, "n_shards", 1) or 1)
    rng = random.Random(seed)

    if n_shards > 1 and hasattr(ds, "shard") and (target_tokens is not None or max_docs is not None):
        k = min(max(1, num_shards_to_sample), n_shards)
        chosen_indices = rng.sample(range(n_shards), k)
        for pos, shard_idx in enumerate(chosen_indices):
            shard_ds = ds.shard(num_shards=n_shards, index=shard_idx)
            if shuffle_buffer_size > 1 and hasattr(shard_ds, "shuffle"):
                shard_ds = shard_ds.shuffle(seed=seed + shard_idx, buffer_size=shuffle_buffer_size)
            yield (pos, iter(shard_ds))
        return

    shuffled = (
        ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
        if shuffle_buffer_size > 1 and hasattr(ds, "shuffle")
        else ds
    )
    yield (0, iter(shuffled))


def stream_matched_slice(
    spec: DatasetSpec,
    cache_path: str | Path,
    target_tokens: int = 3_500_000,
    max_docs: int = 5_000,
    min_chars: int = 64,
    hf_token: str | None = None,
    force_refresh: bool = False,
    random_sample: bool = True,
    num_shards_to_sample: int = 8,
    shuffle_buffer_size: int = 2_000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Stream an equal-sized, multi-shard randomly sampled GPT-2 token slice to JSONL."""
    cache_file = Path(cache_path)
    if cache_file.exists() and not force_refresh:
        return load_jsonl_slice(cache_file)

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    enc = get_gpt2_tokenizer()
    ds = _load_hf_iterable(spec, hf_token=token)

    shard_streams = list(
        iter_randomized_shards(
            ds,
            target_tokens=target_tokens,
            max_docs=max_docs,
            random_sample=random_sample,
            num_shards_to_sample=num_shards_to_sample,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
        )
    )
    n_selected = max(len(shard_streams), 1)

    records: list[dict[str, Any]] = []
    total_tokens = 0

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")

    with tmp_file.open("w", encoding="utf-8") as fh:
        for pos, row_iter in shard_streams:
            remaining_shards = n_selected - pos
            remaining_tokens = max(0, target_tokens - total_tokens)
            remaining_docs = max(0, max_docs - len(records))
            if remaining_tokens == 0 or remaining_docs == 0:
                break

            shard_token_quota = math.ceil(remaining_tokens / remaining_shards)
            shard_doc_quota = math.ceil(remaining_docs / remaining_shards)
            shard_tokens = 0
            shard_docs = 0

            for row in row_iter:
                raw = row.get(spec.text_field)
                if not isinstance(raw, str):
                    continue
                text = raw.strip()
                if len(text) < min_chars:
                    continue

                n_tokens = count_gpt2_tokens(text, enc)
                if n_tokens == 0:
                    continue

                rec = {"text": text, "n_tokens": n_tokens}
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

                total_tokens += n_tokens
                shard_tokens += n_tokens
                shard_docs += 1

                if (
                    len(records) >= max_docs
                    or total_tokens >= target_tokens
                    or shard_docs >= shard_doc_quota
                    or shard_tokens >= shard_token_quota
                ):
                    break

    tmp_file.replace(cache_file)
    return records


def load_jsonl_slice(path: str | Path) -> list[dict[str, Any]]:
    """Load a cached JSONL slice from disk."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_eval_benchmark_texts(
    spec: EvalSpec,
    cache_path: str | Path | None = None,
    force_refresh: bool = False,
) -> list[str]:
    """Load 100% of an evaluation benchmark split (with optional JSONL caching)."""
    if cache_path is not None:
        cache_file = Path(cache_path)
        if cache_file.exists() and not force_refresh:
            return [r["text"] for r in load_jsonl_slice(cache_file)]

    from datasets import load_dataset

    if spec.config is not None:
        ds = load_dataset(spec.repo_id, spec.config, split=spec.split)
    else:
        ds = load_dataset(spec.repo_id, split=spec.split)

    texts: list[str] = []
    if spec.kind == "hellaswag":
        for row in ds:
            ctx = str(row.get("ctx", "")).strip()
            endings = row.get("endings", []) or []
            label_raw = row.get("label", "")
            gold_ending = ""
            if isinstance(label_raw, int) and 0 <= label_raw < len(endings):
                gold_ending = endings[label_raw]
            elif isinstance(label_raw, str) and label_raw.isdigit():
                idx = int(label_raw)
                if 0 <= idx < len(endings):
                    gold_ending = endings[idx]
            combined = f"{ctx} {gold_ending}".strip() if gold_ending else f"{ctx} {' '.join(endings)}".strip()
            if combined:
                texts.append(combined)
    elif spec.kind == "arc":
        for row in ds:
            q = str(row.get("question", "")).strip()
            choices = row.get("choices", {}) or {}
            choice_texts = choices.get("text", []) if isinstance(choices, dict) else []
            combined = f"{q} {' '.join(str(c) for c in choice_texts)}".strip()
            if combined:
                texts.append(combined)
    elif spec.kind == "lambada":
        for row in ds:
            txt = str(row.get("text", "")).strip()
            if txt:
                texts.append(txt)
    elif spec.kind == "wikitext":
        current: list[str] = []
        for row in ds:
            line = str(row.get("text", "")).strip()
            if not line:
                continue
            if line.startswith("= ") and line.endswith(" =") and not line.startswith("= =") and current:
                texts.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            texts.append("\n".join(current))
    else:
        for row in ds:
            txt = str(row.get("text", "")).strip()
            if txt:
                texts.append(txt)

    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as fh:
            for txt in texts:
                fh.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")

    return texts


# ---------------------------------------------------------------------------
# 1. Gopher Repetition & Cross-Doc 7-Gram Boilerplate Audit
# ---------------------------------------------------------------------------


def _merged_span_char_length(
    spans: list[tuple[int, int]],
    word_char_lengths: list[int],
) -> int:
    """Compute total characters covered by a union of token index spans [start, end)."""
    if not spans:
        return 0
    spans.sort()
    merged_start, merged_end = spans[0]
    covered_chars = 0

    for start, end in spans[1:]:
        if start <= merged_end:
            if end > merged_end:
                merged_end = end
        else:
            covered_chars += sum(word_char_lengths[merged_start:merged_end]) + max(
                0, merged_end - merged_start - 1
            )
            merged_start, merged_end = start, end

    covered_chars += sum(word_char_lengths[merged_start:merged_end]) + max(
        0, merged_end - merged_start - 1
    )
    return covered_chars


def compute_gopher_repetition_rates(text: str) -> dict[str, Any]:
    """Compute Gopher / FineWeb within-document n-gram repetition metrics (n=2..10)."""
    words = normalize_alnum_words(text)
    n_words = len(words)
    if n_words < 2:
        rates: dict[str, Any] = {f"top_{n}gram": 0.0 for n in (2, 3, 4)}
        rates.update({f"dup_{n}gram": 0.0 for n in range(5, 11)})
        rates["gopher_flagged"] = False
        return rates

    word_lens = [len(w) for w in words]
    total_chars = sum(word_lens) + max(0, n_words - 1)
    if total_chars == 0:
        total_chars = 1

    metrics: dict[str, Any] = {}
    flagged = False

    for n in (2, 3, 4):
        if n_words < n:
            val = 0.0
        else:
            positions: dict[tuple[str, ...], list[int]] = defaultdict(list)
            for i in range(n_words - n + 1):
                positions[tuple(words[i : i + n])].append(i)
            repeated = {ng: pos for ng, pos in positions.items() if len(pos) >= 2}
            if not repeated:
                val = 0.0
            else:
                best_pos = max(
                    repeated.values(),
                    key=lambda pos: sum(
                        sum(word_lens[i : i + n]) + (n - 1) for i in pos
                    ),
                )
                spans = [(i, i + n) for i in best_pos]
                val = min(1.0, _merged_span_char_length(spans, word_lens) / total_chars)
        metrics[f"top_{n}gram"] = float(val)
        if val > GOPHER_TOP_NGRAM_THRESHOLDS[n]:
            flagged = True

    for n in range(5, 11):
        if n_words < n * 2:
            val = 0.0
        else:
            positions = defaultdict(list)
            for i in range(n_words - n + 1):
                positions[tuple(words[i : i + n])].append(i)
            spans = [
                (i, i + n)
                for pos in positions.values()
                if len(pos) >= 2
                for i in pos
            ]
            val = min(1.0, _merged_span_char_length(spans, word_lens) / total_chars) if spans else 0.0
        metrics[f"dup_{n}gram"] = float(val)
        if val > GOPHER_DUP_NGRAM_THRESHOLDS[n]:
            flagged = True

    metrics["gopher_flagged"] = flagged
    return metrics


def audit_cross_doc_boilerplate(
    texts: Sequence[str],
    cross_doc_min_docs: int = 10,
    min_doc_fraction: float = 0.005,
    top_k: int = 10,
) -> dict[str, Any]:
    """Detect site-wide boilerplate 7-grams repeating across distinct documents."""
    n_docs = len(texts)
    if n_docs == 0:
        return {
            "cross_doc_boilerplate_doc_rate": 0.0,
            "top_cross_doc_ngrams": [],
        }

    doc_freq_7gram: Counter[str] = Counter()
    doc_7grams_list: list[set[str]] = []

    for text in texts:
        words = normalize_words(text)
        if len(words) >= 7:
            unique_7 = {" ".join(words[i : i + 7]) for i in range(len(words) - 6)}
            doc_7grams_list.append(unique_7)
            for ng in unique_7:
                doc_freq_7gram[ng] += 1
        else:
            doc_7grams_list.append(set())

    threshold = max(cross_doc_min_docs, int(min_doc_fraction * n_docs))
    boilerplate_7grams = {ng for ng, df in doc_freq_7gram.items() if df >= threshold}
    cross_doc_hits = sum(1 for s in doc_7grams_list if s & boilerplate_7grams)

    return {
        "cross_doc_boilerplate_doc_rate": float(cross_doc_hits / n_docs),
        "top_cross_doc_ngrams": [
            (ng, int(df)) for ng, df in doc_freq_7gram.most_common(top_k) if df >= threshold
        ],
    }


def audit_repetition_and_boilerplate(
    texts: Sequence[str],
    max_rep_docs: int | None = None,
) -> dict[str, Any]:
    """Run length-normalized Gopher repetition and cross-document 7-gram boilerplate audit."""
    if not texts:
        return {
            "n_docs": 0,
            "gopher_mean_rates": {},
            "gopher_flag_rate": 0.0,
            "cross_doc_boilerplate_doc_rate": 0.0,
            "top_cross_doc_ngrams": [],
        }

    rep_subset = texts if max_rep_docs is None else texts[:max_rep_docs]
    rep_rows = [compute_gopher_repetition_rates(t) for t in rep_subset]
    ngram_keys = [f"top_{n}gram" for n in (2, 3, 4)] + [f"dup_{n}gram" for n in range(5, 11)]
    gopher_means = {k: float(np.mean([r[k] for r in rep_rows])) for k in ngram_keys}
    gopher_flag_rate = float(np.mean([1.0 if r["gopher_flagged"] else 0.0 for r in rep_rows]))

    cross_doc = audit_cross_doc_boilerplate(texts)

    return {
        "n_docs": len(texts),
        "gopher_mean_rates": gopher_means,
        "gopher_flag_rate": gopher_flag_rate,
        "cross_doc_boilerplate_doc_rate": cross_doc["cross_doc_boilerplate_doc_rate"],
        "top_cross_doc_ngrams": cross_doc["top_cross_doc_ngrams"],
    }


# ---------------------------------------------------------------------------
# 2. Domain Alignment (Unigram/Bigram JSD & Top-K Jaccard)
# ---------------------------------------------------------------------------


def build_ngram_profile(
    texts: Sequence[str],
    top_vocab_k: int = 10_000,
    top_bigram_k: int = 50_000,
) -> dict[str, Any]:
    """Build compact unigram and bigram frequency profiles for domain-alignment comparisons."""
    unigrams: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()

    for text in texts:
        words = normalize_words(text)
        unigrams.update(words)
        if len(words) >= 2:
            bigrams.update(f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1))

    top_vocab = {w for w, _ in unigrams.most_common(top_vocab_k)}
    top_bigram_counts = dict(bigrams.most_common(top_bigram_k))

    return {
        "total_words": int(sum(unigrams.values())),
        "vocab_size": len(unigrams),
        "top_vocab": top_vocab,
        "unigram_counts": dict(unigrams),
        "bigram_counts": top_bigram_counts,
    }


def compute_jsd_from_counts(counts_a: dict[str, int], counts_b: dict[str, int]) -> float:
    """Compute true Jensen-Shannon Divergence in bits ([0, 1]) from two count dictionaries."""
    if not counts_a or not counts_b:
        return 1.0

    keys = list(set(counts_a.keys()) | set(counts_b.keys()))
    p = np.array([counts_a.get(k, 0) for k in keys], dtype=np.float64)
    q = np.array([counts_b.get(k, 0) for k in keys], dtype=np.float64)

    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum == 0.0 or q_sum == 0.0:
        return 1.0

    p /= p_sum
    q /= q_sum

    js_dist = float(jensenshannon(p, q, base=2.0))
    if math.isnan(js_dist):
        return 1.0
    return float(min(1.0, max(0.0, js_dist * js_dist)))


def audit_domain_alignment(
    candidate_profile: dict[str, Any],
    reference_profile: dict[str, Any],
) -> dict[str, float]:
    """Compare a candidate dataset's lexical distribution against a reference corpus/benchmark."""
    cand_vocab: set[str] = candidate_profile["top_vocab"]
    ref_vocab: set[str] = reference_profile["top_vocab"]

    union_sz = len(cand_vocab | ref_vocab)
    jaccard = float(len(cand_vocab & ref_vocab) / union_sz) if union_sz > 0 else 0.0

    unigram_jsd = compute_jsd_from_counts(
        candidate_profile["unigram_counts"],
        reference_profile["unigram_counts"],
    )
    bigram_jsd = compute_jsd_from_counts(
        candidate_profile["bigram_counts"],
        reference_profile["bigram_counts"],
    )

    return {
        "top10k_jaccard": jaccard,
        "unigram_jsd": unigram_jsd,
        "bigram_jsd": bigram_jsd,
    }


# ---------------------------------------------------------------------------
# 3. 13-Gram Exact-Overlap Contamination Audit
# ---------------------------------------------------------------------------


def build_eval_ngram_index(
    eval_texts: Sequence[str],
    n: int = 13,
) -> tuple[set[str], dict[str, list[int]], int]:
    """Index all n-grams in an evaluation split and map each n-gram to example indices."""
    all_ngrams: set[str] = set()
    ngram_to_examples: dict[str, list[int]] = defaultdict(list)
    eligible_examples = 0

    for idx, txt in enumerate(eval_texts):
        ngs = extract_ngrams_set(txt, n=n)
        if not ngs:
            continue
        eligible_examples += 1
        all_ngrams.update(ngs)
        for ng in ngs:
            ngram_to_examples[ng].append(idx)

    return all_ngrams, ngram_to_examples, eligible_examples


def audit_contamination_13gram(
    candidate_texts: Sequence[str],
    eval_texts: Sequence[str],
    n: int = 13,
    max_sample_matches: int = 5,
    precomputed_index: tuple[set[str], dict[str, list[int]], int] | None = None,
) -> dict[str, Any]:
    """Measure exact 13-gram overlap between a candidate slice and an evaluation split."""
    if precomputed_index is not None:
        eval_ngrams, ngram_to_examples, eligible_examples = precomputed_index
    else:
        eval_ngrams, ngram_to_examples, eligible_examples = build_eval_ngram_index(
            eval_texts, n=n
        )

    n_cand_docs = len(candidate_texts)
    if n_cand_docs == 0 or not eval_ngrams:
        return {
            "n_gram": n,
            "candidate_docs_checked": n_cand_docs,
            "eligible_eval_examples": eligible_examples,
            "contaminated_candidate_docs": 0,
            "contaminated_doc_rate": 0.0,
            "contaminated_eval_examples": 0,
            "eval_contamination_rate": 0.0,
            "matched_unique_ngrams": 0,
            "sample_matches": [],
        }

    contaminated_docs = 0
    matched_ngrams: set[str] = set()
    hit_eval_examples: set[int] = set()

    for text in candidate_texts:
        words = normalize_alnum_words(text)
        if len(words) < n:
            continue
        doc_hit = False
        for i in range(len(words) - n + 1):
            ng = " ".join(words[i : i + n])
            if ng in eval_ngrams:
                doc_hit = True
                matched_ngrams.add(ng)
                hit_eval_examples.update(ngram_to_examples[ng])
        if doc_hit:
            contaminated_docs += 1

    return {
        "n_gram": n,
        "candidate_docs_checked": n_cand_docs,
        "eligible_eval_examples": eligible_examples,
        "contaminated_candidate_docs": contaminated_docs,
        "contaminated_doc_rate": float(contaminated_docs / n_cand_docs),
        "contaminated_eval_examples": len(hit_eval_examples),
        "eval_contamination_rate": (
            float(len(hit_eval_examples) / eligible_examples)
            if eligible_examples > 0
            else 0.0
        ),
        "matched_unique_ngrams": len(matched_ngrams),
        "sample_matches": sorted(matched_ngrams)[:max_sample_matches],
    }


# ---------------------------------------------------------------------------
# 4. Constant-RAM Whole-Dataset / Large-Scale Streaming Aggregator
# ---------------------------------------------------------------------------


def stream_aggregate_audit(
    spec: DatasetSpec,
    eval_ngram_indices: dict[str, tuple[set[str], dict[str, list[int]], int]] | None = None,
    max_tokens: int | None = None,
    max_docs: int | None = None,
    min_chars: int = 64,
    random_sample: bool = False,
    num_shards_to_sample: int = 8,
    shuffle_buffer_size: int = 2_000,
    seed: int = 42,
    top_vocab_k: int = 10_000,
    top_bigram_k: int = 50_000,
    bigram_prune_every: int = 25_000,
    hf_token: str | None = None,
    row_stream: Iterator[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate Gopher repetition, lexical profile, and 13-gram contamination in O(1) doc RAM."""
    enc = get_gpt2_tokenizer() if row_stream is None else None
    if row_stream is None:
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ds = _load_hf_iterable(spec, hf_token=token)
        shard_streams = iter_randomized_shards(
            ds,
            target_tokens=max_tokens,
            max_docs=max_docs,
            random_sample=random_sample,
            num_shards_to_sample=num_shards_to_sample,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
        )
        rows_iter: Iterator[dict[str, Any]] = itertools.chain.from_iterable(
            it for _, it in shard_streams
        )
    else:
        rows_iter = row_stream

    ngram_keys = [f"top_{n}gram" for n in (2, 3, 4)] + [f"dup_{n}gram" for n in range(5, 11)]

    n_docs = 0
    total_tokens = 0
    gopher_sums = {k: 0.0 for k in ngram_keys}
    gopher_flagged_docs = 0

    unigrams: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()

    eval_indices = eval_ngram_indices or {}
    contam_docs = {bkey: 0 for bkey in eval_indices}
    contam_matches: dict[str, set[str]] = {bkey: set() for bkey in eval_indices}
    contam_eval_hits: dict[str, set[int]] = {bkey: set() for bkey in eval_indices}

    for row in rows_iter:
        raw = row.get(spec.text_field)
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if len(text) < min_chars:
            continue

        n_tok = int(row["n_tokens"]) if "n_tokens" in row else count_gpt2_tokens(text, enc)
        if n_tok == 0:
            continue

        n_docs += 1
        total_tokens += n_tok

        rep = compute_gopher_repetition_rates(text)
        for k in ngram_keys:
            gopher_sums[k] += rep[k]
        if rep["gopher_flagged"]:
            gopher_flagged_docs += 1

        words = normalize_words(text)
        unigrams.update(words)
        if len(words) >= 2:
            bigrams.update(f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1))
        if bigram_prune_every > 0 and n_docs % bigram_prune_every == 0 and len(bigrams) > top_bigram_k * 4:
            bigrams = Counter(dict(bigrams.most_common(top_bigram_k * 2)))

        if eval_indices:
            alnum_words = normalize_alnum_words(text)
            if len(alnum_words) >= 13:
                doc_ngrams = [
                    " ".join(alnum_words[i : i + 13]) for i in range(len(alnum_words) - 12)
                ]
                for bkey, (eval_ngrams, ng_to_ex, _) in eval_indices.items():
                    hit = False
                    for ng in doc_ngrams:
                        if ng in eval_ngrams:
                            hit = True
                            contam_matches[bkey].add(ng)
                            contam_eval_hits[bkey].update(ng_to_ex[ng])
                    if hit:
                        contam_docs[bkey] += 1

        if (max_docs is not None and n_docs >= max_docs) or (
            max_tokens is not None and total_tokens >= max_tokens
        ):
            break

    repetition_boilerplate = {
        "n_docs": n_docs,
        "gopher_mean_rates": {
            k: float(gopher_sums[k] / n_docs) if n_docs > 0 else 0.0 for k in ngram_keys
        },
        "gopher_flag_rate": float(gopher_flagged_docs / n_docs) if n_docs > 0 else 0.0,
        "cross_doc_boilerplate_doc_rate": 0.0,
        "top_cross_doc_ngrams": [],
    }

    lexical_profile = {
        "total_words": int(sum(unigrams.values())),
        "vocab_size": len(unigrams),
        "top_vocab": {w for w, _ in unigrams.most_common(top_vocab_k)},
        "unigram_counts": dict(unigrams),
        "bigram_counts": dict(bigrams.most_common(top_bigram_k)),
    }

    contamination: dict[str, dict[str, Any]] = {}
    for bkey, (_, _, eligible) in eval_indices.items():
        contamination[bkey] = {
            "n_gram": 13,
            "candidate_docs_checked": n_docs,
            "eligible_eval_examples": eligible,
            "contaminated_candidate_docs": contam_docs[bkey],
            "contaminated_doc_rate": float(contam_docs[bkey] / n_docs) if n_docs > 0 else 0.0,
            "contaminated_eval_examples": len(contam_eval_hits[bkey]),
            "eval_contamination_rate": (
                float(len(contam_eval_hits[bkey]) / eligible) if eligible > 0 else 0.0
            ),
            "matched_unique_ngrams": len(contam_matches[bkey]),
            "sample_matches": sorted(contam_matches[bkey])[:5],
        }

    return {
        "name": spec.key,
        "n_docs": n_docs,
        "total_tokens": total_tokens,
        "mean_doc_tokens": float(total_tokens / n_docs) if n_docs > 0 else 0.0,
        "repetition_boilerplate": repetition_boilerplate,
        "lexical_profile": lexical_profile,
        "contamination": contamination,
    }


# ---------------------------------------------------------------------------
# 5. End-to-End Orchestration & Scorecard
# ---------------------------------------------------------------------------


def audit_dataset_slice(
    name: str,
    records: Sequence[dict[str, Any]],
    top_vocab_k: int = 10_000,
    top_bigram_k: int = 50_000,
) -> dict[str, Any]:
    """Run Gopher repetition, cross-doc boilerplate, and lexical profiling on one slice."""
    texts = [r["text"] for r in records]
    token_counts = [int(r.get("n_tokens", 0)) for r in records]
    total_tokens = int(sum(token_counts))
    mean_doc_tokens = float(np.mean(token_counts)) if token_counts else 0.0

    repetition_boilerplate = audit_repetition_and_boilerplate(texts)
    lexical_profile = build_ngram_profile(
        texts, top_vocab_k=top_vocab_k, top_bigram_k=top_bigram_k
    )

    return {
        "name": name,
        "n_docs": len(texts),
        "total_tokens": total_tokens,
        "mean_doc_tokens": mean_doc_tokens,
        "repetition_boilerplate": repetition_boilerplate,
        "lexical_profile": lexical_profile,
    }


def run_full_audit(
    candidate_slices: dict[str, list[dict[str, Any]]],
    eval_texts_by_benchmark: dict[str, list[str]],
    control_key: str = "openwebtext",
) -> dict[str, Any]:
    """Run all core diagnostics across every candidate dataset and evaluation benchmark."""
    dataset_audits: dict[str, dict[str, Any]] = {}
    for key, recs in candidate_slices.items():
        dataset_audits[key] = audit_dataset_slice(key, recs)

    reference_profiles: dict[str, dict[str, Any]] = {}
    if control_key in dataset_audits:
        reference_profiles[control_key] = dataset_audits[control_key]["lexical_profile"]

    eval_ngram_indices: dict[str, tuple[set[str], dict[str, list[int]], int]] = {}
    for bkey, btexts in eval_texts_by_benchmark.items():
        reference_profiles[bkey] = build_ngram_profile(btexts)
        eval_ngram_indices[bkey] = build_eval_ngram_index(btexts, n=13)

    alignment_matrix: dict[str, dict[str, dict[str, float]]] = {}
    contamination_matrix: dict[str, dict[str, dict[str, Any]]] = {}

    for ckey, recs in candidate_slices.items():
        cand_profile = dataset_audits[ckey]["lexical_profile"]
        cand_texts = [r["text"] for r in recs]

        alignment_matrix[ckey] = {
            rkey: audit_domain_alignment(cand_profile, rprof)
            for rkey, rprof in reference_profiles.items()
        }
        contamination_matrix[ckey] = {
            bkey: audit_contamination_13gram(
                cand_texts,
                btexts,
                n=13,
                precomputed_index=eval_ngram_indices[bkey],
            )
            for bkey, btexts in eval_texts_by_benchmark.items()
        }

    scorecard = build_scorecard(
        dataset_audits=dataset_audits,
        alignment_matrix=alignment_matrix,
        contamination_matrix=contamination_matrix,
        control_key=control_key,
    )

    return {
        "dataset_audits": dataset_audits,
        "alignment_matrix": alignment_matrix,
        "contamination_matrix": contamination_matrix,
        "scorecard": scorecard,
    }


def build_scorecard(
    dataset_audits: dict[str, dict[str, Any]],
    alignment_matrix: dict[str, dict[str, dict[str, float]]],
    contamination_matrix: dict[str, dict[str, dict[str, Any]]],
    control_key: str = "openwebtext",
) -> pd.DataFrame:
    """Build a ranked Pandas DataFrame from length-unbiased Cleanliness, JSD Alignment, and Contamination."""
    rows: list[dict[str, Any]] = []

    for key, audit in dataset_audits.items():
        rep = audit["repetition_boilerplate"]

        # Pure repetition + boilerplate noise rate (no length-confounded LZ4 compression)
        noise_rate = (
            0.70 * rep["gopher_flag_rate"]
            + 0.30 * rep["cross_doc_boilerplate_doc_rate"]
        )

        # Distributional proximity to GPT-2 benchmarks (excluding OpenWebText control)
        align_for_ds = alignment_matrix.get(key, {})
        eval_aligns = [v for rk, v in align_for_ds.items() if rk != control_key]
        if eval_aligns:
            mean_eval_uni_jsd = float(np.mean([a["unigram_jsd"] for a in eval_aligns]))
            mean_eval_bi_jsd = float(np.mean([a["bigram_jsd"] for a in eval_aligns]))
            mean_eval_jaccard = float(np.mean([a["top10k_jaccard"] for a in eval_aligns]))
            benchmark_alignment_score = 100.0 * (
                0.60 * (1.0 - mean_eval_uni_jsd)
                + 0.40 * (1.0 - mean_eval_bi_jsd)
            )
        else:
            mean_eval_uni_jsd = 1.0
            mean_eval_bi_jsd = 1.0
            mean_eval_jaccard = 0.0
            benchmark_alignment_score = 0.0

        owt_align = align_for_ds.get(control_key, {})
        owt_uni_jsd = float(owt_align.get("unigram_jsd", float("nan")))

        # 13-gram contamination across eval benchmarks
        contam_for_ds = contamination_matrix.get(key, {})
        if contam_for_ds:
            mean_contam_pct = 100.0 * float(
                np.mean([c["eval_contamination_rate"] for c in contam_for_ds.values()])
            )
            max_contam_pct = 100.0 * float(
                np.max([c["eval_contamination_rate"] for c in contam_for_ds.values()])
            )
        else:
            mean_contam_pct = 0.0
            max_contam_pct = 0.0

        # Composite score [0, 100]: 40% Cleanliness + 45% JSD Benchmark Alignment + 15% Decontamination
        composite_score = (
            40.0 * (1.0 - min(1.0, noise_rate))
            + 45.0 * (benchmark_alignment_score / 100.0)
            + 15.0 * max(0.0, 1.0 - max_contam_pct / 5.0)
        )

        spec = CANDIDATE_DATASETS.get(key)
        rows.append(
            {
                "Dataset": spec.display_name if spec else key,
                "Key": key,
                "Docs": audit["n_docs"],
                "Tokens": audit["total_tokens"],
                "Mean Tok/Doc": round(audit["mean_doc_tokens"], 1),
                "Gopher Rep Flag (%)": round(100.0 * rep["gopher_flag_rate"], 2),
                "Cross-Doc Boilerplate (%)": round(100.0 * rep["cross_doc_boilerplate_doc_rate"], 2),
                "Noise Rate (%)": round(100.0 * noise_rate, 2),
                "OWT Unigram JSD": round(owt_uni_jsd, 4) if not math.isnan(owt_uni_jsd) else None,
                "Eval Mean JSD (Uni)": round(mean_eval_uni_jsd, 4),
                "Eval Mean JSD (Bi)": round(mean_eval_bi_jsd, 4),
                "Eval Top-10k Jaccard": round(mean_eval_jaccard, 4),
                "GPT-2 Benchmark Alignment": round(benchmark_alignment_score, 2),
                "13-gram Contam Mean (%)": round(mean_contam_pct, 4),
                "13-gram Contam Max (%)": round(max_contam_pct, 4),
                "Composite Score": round(composite_score, 2),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Composite Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df
