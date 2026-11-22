"""Reusable pretraining dataset diagnostics for Kestrel (350M-700M GPT-2 successor).

Provides equal-token-slice streaming, stratified word-transition network & conditional
entropy analysis, LZ4 compression and Gopher/FineWeb n-gram repetition audits,
GPT-2/benchmark domain alignment (Top-K Jaccard, coverage, unigram/bigram JSD),
13-gram exact-overlap contamination detection, and comparative scorecard generation.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

try:
    import lz4.block
    import lz4.frame
except ImportError:  # pragma: no cover - installed on Kaggle / via extras
    lz4 = None  # type: ignore[assignment]

try:
    import tiktoken
except ImportError:  # pragma: no cover - installed on Kaggle / via extras
    tiktoken = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants & Probe Panels
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[a-z]+")
ALNUM_WORD_RE = re.compile(r"[a-z0-9]+")

# Fixed panel of 20 mid-frequency content words spanning STEM, economics,
# systems, nature, and general expository prose.
CONTENT_PROBE_WORDS: tuple[str, ...] = (
    "system",
    "energy",
    "market",
    "water",
    "model",
    "data",
    "process",
    "research",
    "design",
    "network",
    "health",
    "policy",
    "theory",
    "method",
    "cell",
    "force",
    "value",
    "climate",
    "language",
    "algorithm",
)

# Separate function-word control panel so grammatical regularity is never
# conflated with content-level boilerplate or SEO repetition.
FUNCTION_PROBE_WORDS: tuple[str, ...] = (
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "for",
    "that",
    "with",
    "on",
)

GOLDILOCKS_LZ4_BAND: tuple[float, float] = (0.65, 0.80)

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

# Transition-noise heuristic thresholds for content words
LEFT_TAIL_TOP1_THRESHOLD = 0.35
RIGHT_TAIL_NORM_ENTROPY_THRESHOLD = 0.92
RIGHT_TAIL_TOP3_MAX = 0.10
RIGHT_TAIL_SINGLETON_THRESHOLD = 0.72
MIN_PROBE_OCCURRENCES = 25


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
# Tokenization & Streaming Slice Helpers
# ---------------------------------------------------------------------------


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
    """Extract lowercase alphabetic word tokens for lexical/transition diagnostics."""
    return WORD_RE.findall(text.lower())


def normalize_alnum_words(text: str) -> list[str]:
    """Extract lowercase alphanumeric word tokens for 13-gram contamination matching."""
    return ALNUM_WORD_RE.findall(text.lower())


def _open_hf_stream(spec: DatasetSpec, hf_token: str | None = None) -> Iterator[dict[str, Any]]:
    """Open a Hugging Face streaming dataset with eager first-row verification and fallbacks."""
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
            # Eagerly fetch the first row so gated-auth or legacy-script failures trigger fallbacks.
            it = iter(ds)
            first_row = next(it)
            return itertools.chain([first_row], it)
        except Exception as exc:  # pragma: no cover - network/HF runtime dependent
            last_exc = exc
            continue

    raise RuntimeError(
        f"Failed to stream '{spec.key}' ({spec.repo_id}). "
        f"{'Check HF_TOKEN / gated repo access. ' if spec.gated else ''}"
        f"Last error: {last_exc}"
    ) from last_exc


def stream_matched_slice(
    spec: DatasetSpec,
    cache_path: str | Path,
    target_tokens: int = 3_500_000,
    max_docs: int = 5_000,
    min_chars: int = 64,
    hf_token: str | None = None,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Stream an equal-sized GPT-2 token slice and cache it as lightweight JSONL.

    Stops as soon as either ``target_tokens`` or ``max_docs`` is reached so memory and
    disk usage remain comfortably within Kaggle's 30 GB RAM / 20 GB disk limits.
    """
    cache_file = Path(cache_path)
    if cache_file.exists() and not force_refresh:
        return load_jsonl_slice(cache_file)

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    enc = get_gpt2_tokenizer()
    stream = _open_hf_stream(spec, hf_token=token)

    records: list[dict[str, Any]] = []
    total_tokens = 0

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")

    with tmp_file.open("w", encoding="utf-8") as fh:
        for row in stream:
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
            if len(records) >= max_docs or total_tokens >= target_tokens:
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
    """Load evaluation benchmark texts (with optional JSONL caching).

    Returns one canonical string per evaluation example (or article chunk for WikiText-103)
    suitable for both distribution alignment and 13-gram contamination checks.
    """
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
        # Group non-empty lines into article documents bounded by top-level headings (" = Title = ")
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
# 1. Stratified Word-Transition Network & Conditional Entropy
# ---------------------------------------------------------------------------


def _compute_word_transition_stats(
    counter: Counter[str],
    top_k: int = 15,
) -> dict[str, Any]:
    """Compute out-degree, top-1/top-3 mass, conditional entropy, and noise flags for one word."""
    total = int(sum(counter.values()))
    if total == 0:
        return {
            "total_occurrences": 0,
            "out_degree": 0,
            "top1_mass": 0.0,
            "top3_mass": 0.0,
            "conditional_entropy": 0.0,
            "normalized_entropy": 0.0,
            "singleton_ratio": 0.0,
            "left_tail_flag": False,
            "right_tail_flag": False,
            "top_transitions": [],
        }

    counts = np.fromiter(counter.values(), dtype=np.float64)
    probs = counts / float(total)
    out_degree = len(counter)

    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = math.log2(out_degree) if out_degree > 1 else 1.0
    norm_entropy = float(entropy / max_entropy) if out_degree > 1 else 0.0

    top_items = counter.most_common(top_k)
    top1_mass = float(top_items[0][1] / total) if top_items else 0.0
    top3_mass = float(sum(c for _, c in top_items[:3]) / total)
    singletons = int(np.sum(counts == 1.0))
    singleton_ratio = float(singletons / out_degree) if out_degree > 0 else 0.0

    # Left-tail noise: collapsed transition distribution (boilerplate / repeated phrase loops)
    left_tail_flag = total >= MIN_PROBE_OCCURRENCES and top1_mass > LEFT_TAIL_TOP1_THRESHOLD
    # Right-tail noise: abnormally flat, scattered transitions (broken scrapes / keyword spam)
    right_tail_flag = (
        total >= MIN_PROBE_OCCURRENCES
        and norm_entropy > RIGHT_TAIL_NORM_ENTROPY_THRESHOLD
        and top3_mass < RIGHT_TAIL_TOP3_MAX
        and singleton_ratio > RIGHT_TAIL_SINGLETON_THRESHOLD
    )

    return {
        "total_occurrences": total,
        "out_degree": out_degree,
        "top1_mass": top1_mass,
        "top3_mass": top3_mass,
        "conditional_entropy": entropy,
        "normalized_entropy": norm_entropy,
        "singleton_ratio": singleton_ratio,
        "left_tail_flag": left_tail_flag,
        "right_tail_flag": right_tail_flag,
        "top_transitions": [(w, float(c / total), int(c)) for w, c in top_items],
    }


def detect_boilerplate_loops(
    texts: Sequence[str],
    ngram_sizes: tuple[int, ...] = (5, 7, 10),
    min_doc_repeat: int = 3,
    cross_doc_min_docs: int = 10,
) -> dict[str, Any]:
    """Flag corpus-level & within-document repeating 5-10-gram boilerplate loops.

    Captures left-tail boilerplate artifacts such as cookie banners, navigation menus,
    and copyright footers that repeat verbatim inside or across documents.
    """
    n_docs = len(texts)
    if n_docs == 0:
        return {
            "within_doc_loop_doc_rate": 0.0,
            "cross_doc_boilerplate_doc_rate": 0.0,
            "top_cross_doc_ngrams": [],
        }

    within_loop_docs = 0
    doc_freq_7gram: Counter[str] = Counter()
    doc_7grams_list: list[set[str]] = []

    for text in texts:
        words = normalize_words(text)
        has_within_loop = False
        for n in ngram_sizes:
            if len(words) < n * min_doc_repeat:
                continue
            counts = Counter(
                " ".join(words[i : i + n]) for i in range(len(words) - n + 1)
            )
            if any(c >= min_doc_repeat for c in counts.values()):
                has_within_loop = True
                break
        if has_within_loop:
            within_loop_docs += 1

        if len(words) >= 7:
            # Sample unique 7-grams per doc for cross-doc boilerplate detection
            unique_7 = {" ".join(words[i : i + 7]) for i in range(len(words) - 6)}
            doc_7grams_list.append(unique_7)
            for ng in unique_7:
                doc_freq_7gram[ng] += 1
        else:
            doc_7grams_list.append(set())

    threshold = max(cross_doc_min_docs, int(0.005 * n_docs))
    boilerplate_7grams = {ng for ng, df in doc_freq_7gram.items() if df >= threshold}
    cross_doc_hits = sum(1 for s in doc_7grams_list if s & boilerplate_7grams)

    return {
        "within_doc_loop_doc_rate": float(within_loop_docs / n_docs),
        "cross_doc_boilerplate_doc_rate": float(cross_doc_hits / n_docs),
        "boilerplate_7gram_count": len(boilerplate_7grams),
        "top_cross_doc_ngrams": [
            (ng, int(df)) for ng, df in doc_freq_7gram.most_common(10) if df >= threshold
        ],
    }


def audit_transition_network(
    texts: Sequence[str],
    content_words: Sequence[str] = CONTENT_PROBE_WORDS,
    function_words: Sequence[str] = FUNCTION_PROBE_WORDS,
    top_k: int = 15,
) -> dict[str, Any]:
    """Run stratified word-transition network & conditional entropy audit.

    Analyzes content words and function words in separate strata so normal English
    syntax on function words is never mistaken for boilerplate.
    """
    content_set = set(content_words)
    function_set = set(function_words)
    tracked = content_set | function_set

    transitions: dict[str, Counter[str]] = {w: Counter() for w in tracked}

    for text in texts:
        words = normalize_words(text)
        for i in range(len(words) - 1):
            w = words[i]
            if w in tracked:
                transitions[w][words[i + 1]] += 1

    def _summarize_stratum(probe_list: Sequence[str]) -> dict[str, Any]:
        per_word = {
            w: _compute_word_transition_stats(transitions[w], top_k=top_k)
            for w in probe_list
        }
        active = [s for s in per_word.values() if s["total_occurrences"] >= MIN_PROBE_OCCURRENCES]
        if not active:
            active = [s for s in per_word.values() if s["total_occurrences"] > 0]

        def _mean(key: str) -> float:
            return float(np.mean([s[key] for s in active])) if active else 0.0

        left_flagged = [w for w, s in per_word.items() if s["left_tail_flag"]]
        right_flagged = [w for w, s in per_word.items() if s["right_tail_flag"]]
        denom = max(len(probe_list), 1)

        return {
            "per_word": per_word,
            "active_probe_count": len(active),
            "mean_out_degree": _mean("out_degree"),
            "mean_top1_mass": _mean("top1_mass"),
            "mean_top3_mass": _mean("top3_mass"),
            "mean_conditional_entropy": _mean("conditional_entropy"),
            "mean_normalized_entropy": _mean("normalized_entropy"),
            "mean_singleton_ratio": _mean("singleton_ratio"),
            "left_tail_flagged_words": left_flagged,
            "right_tail_flagged_words": right_flagged,
            "left_tail_flag_rate": float(len(left_flagged) / denom),
            "right_tail_flag_rate": float(len(right_flagged) / denom),
        }

    content_summary = _summarize_stratum(content_words)
    function_summary = _summarize_stratum(function_words)
    boilerplate = detect_boilerplate_loops(texts)

    return {
        "content": content_summary,
        "function": function_summary,
        "boilerplate": boilerplate,
    }


def build_transition_digraph(
    transition_audit: dict[str, Any],
    probe_word: str,
    stratum: str = "content",
    top_k: int = 10,
) -> nx.DiGraph:
    """Build a directed NetworkX transition graph for a single probe word.

    Edge attributes include ``weight`` (transition probability) and ``count``.
    """
    graph = nx.DiGraph()
    word_info = transition_audit.get(stratum, {}).get("per_word", {}).get(probe_word)
    graph.add_node(probe_word, role="probe")
    if not word_info:
        return graph

    for nxt, prob, count in word_info.get("top_transitions", [])[:top_k]:
        graph.add_node(nxt, role="target")
        graph.add_edge(probe_word, nxt, weight=prob, count=count)
    return graph


# ---------------------------------------------------------------------------
# 2. Compression (LZ4) & Gopher/FineWeb N-gram Repetition Audit
# ---------------------------------------------------------------------------


def compute_lz4_ratio(text: str) -> float:
    """Compute LZ4 compression ratio (compressed_bytes / raw_bytes) without header bloat."""
    if lz4 is None:
        raise ImportError(
            "lz4 is required for compression-ratio audits. Install with: pip install lz4"
        )
    raw = text.encode("utf-8", errors="ignore")
    if not raw:
        return 0.0
    try:
        compressed = lz4.block.compress(raw, store_size=False)
    except Exception:
        compressed = lz4.frame.compress(raw)
    return float(len(compressed) / len(raw))


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
    """Compute Gopher / FineWeb within-document n-gram repetition metrics (n=2..10).

    - For n in {2, 3, 4}: fraction of characters covered by the single most frequent n-gram.
    - For n in {5, 6, 7, 8, 9, 10}: fraction of characters covered by all n-grams occurring >= 2 times.
    - ``gopher_flagged``: True if any metric exceeds Gopher/FineWeb thresholds.
    """
    words = normalize_alnum_words(text)
    n_words = len(words)
    if n_words < 2:
        rates = {f"top_{n}gram": 0.0 for n in (2, 3, 4)}
        rates.update({f"dup_{n}gram": 0.0 for n in range(5, 11)})
        rates["gopher_flagged"] = False
        return rates

    word_lens = [len(w) for w in words]
    total_chars = sum(word_lens) + max(0, n_words - 1)
    if total_chars == 0:
        total_chars = 1

    metrics: dict[str, Any] = {}
    flagged = False

    # Top-1 n-gram character fraction for n=2, 3, 4
    for n in (2, 3, 4):
        if n_words < n:
            val = 0.0
        else:
            positions: dict[tuple[str, ...], list[int]] = defaultdict(list)
            for i in range(n_words - n + 1):
                positions[tuple(words[i : i + n])].append(i)
            # Only consider n-grams that actually repeat (>= 2 occurrences)
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

    # Duplicate n-gram character fraction for n=5..10
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


def audit_compression_and_repetition(
    texts: Sequence[str],
    goldilocks_band: tuple[float, float] = GOLDILOCKS_LZ4_BAND,
    max_rep_docs: int | None = None,
) -> dict[str, Any]:
    """Run document-level LZ4 compression and Gopher/FineWeb repetition audit."""
    if not texts:
        return {
            "n_docs": 0,
            "lz4_ratios": [],
            "lz4_mean": 0.0,
            "lz4_median": 0.0,
            "lz4_std": 0.0,
            "lz4_p05": 0.0,
            "lz4_p95": 0.0,
            "lz4_below_band_rate": 0.0,
            "lz4_in_band_rate": 0.0,
            "lz4_above_band_rate": 0.0,
            "lz4_out_of_band_rate": 0.0,
            "gopher_mean_rates": {},
            "gopher_flag_rate": 0.0,
        }

    lo, hi = goldilocks_band
    ratios = np.array([compute_lz4_ratio(t) for t in texts], dtype=np.float64)

    below = float(np.mean(ratios < lo))
    in_band = float(np.mean((ratios >= lo) & (ratios <= hi)))
    above = float(np.mean(ratios > hi))

    rep_subset = texts if max_rep_docs is None else texts[:max_rep_docs]
    rep_rows = [compute_gopher_repetition_rates(t) for t in rep_subset]
    ngram_keys = [f"top_{n}gram" for n in (2, 3, 4)] + [f"dup_{n}gram" for n in range(5, 11)]
    gopher_means = {
        k: float(np.mean([r[k] for r in rep_rows])) for k in ngram_keys
    }
    gopher_flag_rate = float(np.mean([1.0 if r["gopher_flagged"] else 0.0 for r in rep_rows]))

    return {
        "n_docs": len(texts),
        "lz4_ratios": ratios.tolist(),
        "lz4_mean": float(np.mean(ratios)),
        "lz4_median": float(np.median(ratios)),
        "lz4_std": float(np.std(ratios)),
        "lz4_p05": float(np.percentile(ratios, 5)),
        "lz4_p95": float(np.percentile(ratios, 95)),
        "lz4_below_band_rate": below,
        "lz4_in_band_rate": in_band,
        "lz4_above_band_rate": above,
        "lz4_out_of_band_rate": float(1.0 - in_band),
        "gopher_mean_rates": gopher_means,
        "gopher_flag_rate": gopher_flag_rate,
    }


# ---------------------------------------------------------------------------
# 3. Domain Alignment (Top-K Jaccard, Coverage, Unigram/Bigram JSD)
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
    """Compute true Jensen-Shannon Divergence in bits ([0, 1]) from two count dictionaries.

    Uses base-2 logarithm so JSD(P || Q) == 0 for identical distributions and 1 for
    disjoint supports. Note ``scipy.spatial.distance.jensenshannon`` returns sqrt(JSD),
    so we square it to obtain JSD itself.
    """
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

    # Token-weighted common-word coverage of the reference corpus by candidate's top-10k vocab
    ref_unigrams: dict[str, int] = reference_profile["unigram_counts"]
    ref_total = sum(ref_unigrams.values())
    covered_tokens = sum(cnt for w, cnt in ref_unigrams.items() if w in cand_vocab)
    common_word_coverage = float(covered_tokens / ref_total) if ref_total > 0 else 0.0

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
        "common_word_coverage": common_word_coverage,
        "unigram_jsd": unigram_jsd,
        "bigram_jsd": bigram_jsd,
    }


# ---------------------------------------------------------------------------
# 4. 13-Gram Exact-Overlap Contamination Audit
# ---------------------------------------------------------------------------


def extract_ngrams_set(text: str, n: int = 13) -> set[str]:
    """Extract normalized alphanumeric n-grams from a single text string."""
    words = normalize_alnum_words(text)
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


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
    """Measure exact 13-gram overlap between a candidate slice and an evaluation split.

    Distinguishes healthy domain proximity (low JSD, near-zero 13-gram leakage) from
    verbatim benchmark contamination.
    """
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
# 5. End-to-End Orchestration & Scorecard
# ---------------------------------------------------------------------------


def audit_dataset_slice(
    name: str,
    records: Sequence[dict[str, Any]],
    top_vocab_k: int = 10_000,
    top_bigram_k: int = 50_000,
) -> dict[str, Any]:
    """Run transition-network, LZ4 compression, Gopher repetition, and lexical profiling on one slice."""
    texts = [r["text"] for r in records]
    token_counts = [int(r.get("n_tokens", 0)) for r in records]
    total_tokens = int(sum(token_counts))
    mean_doc_tokens = float(np.mean(token_counts)) if token_counts else 0.0

    transition = audit_transition_network(texts)
    compression_rep = audit_compression_and_repetition(texts)
    lexical_profile = build_ngram_profile(
        texts, top_vocab_k=top_vocab_k, top_bigram_k=top_bigram_k
    )

    return {
        "name": name,
        "n_docs": len(texts),
        "total_tokens": total_tokens,
        "mean_doc_tokens": mean_doc_tokens,
        "transition": transition,
        "compression_repetition": compression_rep,
        "lexical_profile": lexical_profile,
    }


def run_full_audit(
    candidate_slices: dict[str, list[dict[str, Any]]],
    eval_texts_by_benchmark: dict[str, list[str]],
    control_key: str = "openwebtext",
) -> dict[str, Any]:
    """Run all Phase-2 diagnostics across every candidate dataset and evaluation benchmark."""
    dataset_audits: dict[str, dict[str, Any]] = {}
    for key, recs in candidate_slices.items():
        dataset_audits[key] = audit_dataset_slice(key, recs)

    # Reference lexical profiles: OpenWebText (if present) + all eval benchmarks
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
    """Build a ranked Pandas DataFrame summarizing Noise, Diversity, Alignment, and Contamination."""
    rows: list[dict[str, Any]] = []

    for key, audit in dataset_audits.items():
        comp = audit["compression_repetition"]
        trans_content = audit["transition"]["content"]
        boilerplate = audit["transition"]["boilerplate"]

        # Composite noise rate (%): weighted blend of LZ4 out-of-band, Gopher repetition,
        # boilerplate loops, and left/right-tail transition flags.
        noise_rate = (
            0.40 * comp["lz4_out_of_band_rate"]
            + 0.30 * comp["gopher_flag_rate"]
            + 0.15 * boilerplate["within_doc_loop_doc_rate"]
            + 0.15 * (trans_content["left_tail_flag_rate"] + trans_content["right_tail_flag_rate"])
        )

        # Alignment to GPT-2 benchmarks (excluding OpenWebText control)
        align_for_ds = alignment_matrix.get(key, {})
        eval_aligns = [v for rk, v in align_for_ds.items() if rk != control_key]
        if eval_aligns:
            mean_eval_uni_jsd = float(np.mean([a["unigram_jsd"] for a in eval_aligns]))
            mean_eval_bi_jsd = float(np.mean([a["bigram_jsd"] for a in eval_aligns]))
            mean_eval_cov = float(np.mean([a["common_word_coverage"] for a in eval_aligns]))
            mean_eval_jaccard = float(np.mean([a["top10k_jaccard"] for a in eval_aligns]))
            # Higher is better in [0, 100]: balances lexical coverage and distributional closeness
            benchmark_alignment_score = 100.0 * (
                0.45 * (1.0 - mean_eval_uni_jsd)
                + 0.25 * (1.0 - mean_eval_bi_jsd)
                + 0.30 * mean_eval_cov
            )
        else:
            mean_eval_uni_jsd = 1.0
            mean_eval_bi_jsd = 1.0
            mean_eval_cov = 0.0
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

        content_entropy = float(trans_content["mean_conditional_entropy"])
        # Entropy score scaled against typical high-diversity prose (~7-10 bits)
        diversity_norm = min(1.0, content_entropy / 9.0)

        # Overall composite score in [0, 100] with strong contamination penalty
        composite_score = (
            35.0 * (1.0 - min(1.0, noise_rate))
            + 25.0 * diversity_norm
            + 30.0 * (benchmark_alignment_score / 100.0)
            + 10.0 * max(0.0, 1.0 - max_contam_pct / 5.0)
        )

        spec = CANDIDATE_DATASETS.get(key)
        rows.append(
            {
                "Dataset": spec.display_name if spec else key,
                "Key": key,
                "Docs": audit["n_docs"],
                "Tokens": audit["total_tokens"],
                "LZ4 In-Band (%)": round(100.0 * comp["lz4_in_band_rate"], 2),
                "Gopher Rep Flag (%)": round(100.0 * comp["gopher_flag_rate"], 2),
                "Noise Rate (%)": round(100.0 * noise_rate, 2),
                "Content Entropy (bits)": round(content_entropy, 3),
                "Content Top-1 Mass": round(trans_content["mean_top1_mass"], 4),
                "OWT Unigram JSD": round(owt_uni_jsd, 4) if not math.isnan(owt_uni_jsd) else None,
                "Eval Mean JSD (Uni)": round(mean_eval_uni_jsd, 4),
                "Eval Top-10k Jaccard": round(mean_eval_jaccard, 4),
                "Eval Word Coverage (%)": round(100.0 * mean_eval_cov, 2),
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
