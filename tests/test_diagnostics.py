"""Unit tests for kestrel.data.diagnostics."""

from __future__ import annotations

import math
import zlib
from unittest.mock import patch

from kestrel.data import diagnostics as diag


class _FakeShardedIterable:
    """Minimal mock of a multi-shard Hugging Face IterableDataset."""

    def __init__(self, shards: list[list[dict]], active_shard: int | None = None):
        self._shards = shards
        self._active = active_shard
        self.n_shards = len(shards) if active_shard is None else 1

    def shard(self, num_shards: int, index: int, contiguous: bool = True):
        assert num_shards == len(self._shards)
        return _FakeShardedIterable(self._shards, active_shard=index)

    def shuffle(self, seed=None, buffer_size: int = 1000):
        import random

        rows = list(self)
        random.Random(seed).shuffle(rows)
        return _FakeShardedIterable([rows], active_shard=0)

    def __iter__(self):
        if self._active is not None:
            return iter(self._shards[self._active])
        return (r for s in self._shards for r in s)


def test_cross_doc_boilerplate_catches_recurring_footers_missed_by_single_doc():
    footer = "subscribe to our free daily newsletter for exclusive deals"
    docs = [
        f"Unique technical essay number {i} about turbine thermodynamics and heat transfer. {footer}"
        for i in range(25)
    ]
    res = diag.audit_cross_doc_boilerplate(docs, cross_doc_min_docs=10)
    assert res["cross_doc_boilerplate_doc_rate"] == 1.0
    assert any("subscribe to our free daily newsletter" in ng for ng, _ in res["top_cross_doc_ngrams"])

    topics = [
        "planetary atmospheres", "ocean currents", "volcanic geology", "cellular respiration",
        "bridge engineering", "quantum optics", "ancient trade routes", "compiler design",
        "battery chemistry", "coral ecosystems", "glacial erosion", "urban transit",
    ]
    verbs = ["measure", "analyze", "compare", "document", "simulate", "verify"]
    clean_docs = [
        f"Report on {topics[i % len(topics)]} where teams {verbs[i % len(verbs)]} distinct findings for cohort {chr(65 + i)}."
        for i in range(25)
    ]
    clean_res = diag.audit_cross_doc_boilerplate(clean_docs, cross_doc_min_docs=10)
    assert clean_res["cross_doc_boilerplate_doc_rate"] == 0.0


def test_gopher_repetition_rates_flags_repetitive_but_not_clean_prose():
    clean_doc = (
        "Thermodynamics governs energy transfer across physical systems. "
        "Engineers measure heat capacity, entropy production, and mechanical work "
        "to design efficient turbines and refrigeration cycles in industrial laboratories."
    )
    clean_metrics = diag.compute_gopher_repetition_rates(clean_doc)
    assert clean_metrics["gopher_flagged"] is False
    assert clean_metrics["dup_5gram"] == 0.0

    spam_phrase = "click here now for free shipping deals today "
    spam_doc = spam_phrase * 12
    spam_metrics = diag.compute_gopher_repetition_rates(spam_doc)
    assert spam_metrics["gopher_flagged"] is True
    assert spam_metrics["dup_5gram"] > 0.80
    assert spam_metrics["dup_8gram"] > 0.70


def test_jsd_and_domain_alignment_bounds():
    corpus_a = [
        "The energy system models water flow across urban climate infrastructure.",
        "Research methods evaluate market value and network reliability over time.",
    ]
    corpus_b = [
        "Quantum chromodynamics computes lattice gauge fields in curved spacetime.",
        "Superconducting qubits experience decoherence near cryogenic resonators.",
    ]
    prof_a = diag.build_ngram_profile(corpus_a)
    prof_b = diag.build_ngram_profile(corpus_b)

    self_align = diag.audit_domain_alignment(prof_a, prof_a)
    assert math.isclose(self_align["unigram_jsd"], 0.0, abs_tol=1e-9)
    assert math.isclose(self_align["bigram_jsd"], 0.0, abs_tol=1e-9)
    assert math.isclose(self_align["top10k_jaccard"], 1.0, abs_tol=1e-9)
    assert math.isclose(self_align["common_word_coverage"], 1.0, abs_tol=1e-9)

    cross_align = diag.audit_domain_alignment(prof_a, prof_b)
    assert 0.5 < cross_align["unigram_jsd"] <= 1.0
    assert cross_align["top10k_jaccard"] < 0.1


def test_13gram_contamination_distinguishes_overlap_from_clean():
    eval_examples = [
        "A student pours warm water into a graduated cylinder and measures the final temperature carefully.",
        "When plants absorb sunlight during photosynthesis they convert carbon dioxide and water into glucose.",
    ]
    clean_candidates = [
        "Graduated cylinders are standard glassware used by chemistry students to measure liquid volume.",
    ]
    leaked_candidates = [
        "Note from exam: A student pours warm water into a graduated cylinder and measures the final temperature carefully. Answer: B",
        "Another completely unrelated article about bridge engineering and steel fatigue testing.",
    ]

    clean_res = diag.audit_contamination_13gram(clean_candidates, eval_examples, n=13)
    assert clean_res["contaminated_candidate_docs"] == 0
    assert clean_res["eval_contamination_rate"] == 0.0

    leaked_res = diag.audit_contamination_13gram(leaked_candidates, eval_examples, n=13)
    assert leaked_res["contaminated_candidate_docs"] == 1
    assert math.isclose(leaked_res["contaminated_doc_rate"], 0.5)
    assert math.isclose(leaked_res["eval_contamination_rate"], 0.5)
    assert len(leaked_res["sample_matches"]) >= 1


def test_multi_shard_random_sampling_avoids_prefix_bias(tmp_path):
    # 10 shards, each tagged with its shard number; prefix-only sampling would only see shard_0!
    shards = [
        [
            {"text": f"shard_{s} document {j} with enough text to pass the minimum length threshold easily."}
            for j in range(10)
        ]
        for s in range(10)
    ]
    fake_ds = _FakeShardedIterable(shards)
    spec = diag.DatasetSpec(key="mock", display_name="Mock", repo_id="mock/repo")

    with (
        patch.object(diag, "_load_hf_iterable", return_value=fake_ds),
        patch.object(diag, "get_gpt2_tokenizer", return_value=None),
        patch.object(diag, "count_gpt2_tokens", side_effect=lambda txt, enc=None: len(txt.split())),
    ):
        recs = diag.stream_matched_slice(
            spec=spec,
            cache_path=tmp_path / "mock.jsonl",
            target_tokens=10_000,
            max_docs=20,
            random_sample=True,
            num_shards_to_sample=5,
            shuffle_buffer_size=50,
            seed=42,
        )

    seen_shards = {r["text"].split()[0] for r in recs}
    assert len(recs) == 20
    assert len(seen_shards) == 5  # sampled evenly across 5 distinct shards, not just shard_0!


def test_stream_aggregate_audit_runs_constant_ram_pass():
    def fake_lz4_ratio(text: str) -> float:
        raw = text.encode("utf-8")
        return len(zlib.compress(raw, level=1)) / max(len(raw), 1)

    eval_idx = {
        "arc_easy": diag.build_eval_ngram_index(
            ["A student pours warm water into a graduated cylinder and measures the final temperature carefully."]
        )
    }
    rows = [
        {
            "text": "Thermodynamics studies heat transfer and mechanical work across clean physical systems in laboratories.",
            "n_tokens": 18,
        },
        {
            "text": "Exam leak: A student pours warm water into a graduated cylinder and measures the final temperature carefully.",
            "n_tokens": 22,
        },
    ]
    spec = diag.DatasetSpec(key="stream_mock", display_name="Stream Mock", repo_id="mock/stream")
    with patch.object(diag, "compute_lz4_ratio", side_effect=fake_lz4_ratio):
        res = diag.stream_aggregate_audit(
            spec=spec,
            eval_ngram_indices=eval_idx,
            row_stream=iter(rows),
        )

    assert res["n_docs"] == 2
    assert res["total_tokens"] == 40
    assert res["contamination"]["arc_easy"]["contaminated_candidate_docs"] == 1
    assert res["contamination"]["arc_easy"]["eval_contamination_rate"] == 1.0


def test_run_full_audit_and_scorecard_end_to_end():
    def fake_lz4_ratio(text: str) -> float:
        raw = text.encode("utf-8")
        return len(zlib.compress(raw, level=1)) / max(len(raw), 1)

    clean_records = [
        {
            "text": (
                f"Document {i}: The solar energy system converts sunlight into electricity. "
                "Water circulation cools the turbine model while market policy guides grid research."
            ),
            "n_tokens": 32,
        }
        for i in range(15)
    ]
    noisy_records = [
        {
            "text": "buy cheap discount shoes online now click here " * 8,
            "n_tokens": 70,
        }
        for _ in range(15)
    ]
    eval_splits = {
        "arc_easy": [
            "Which energy source powers solar panels to generate electricity for a home system?"
        ],
        "wikitext103": [
            "Solar energy systems convert sunlight directly into electrical power via photovoltaic cells."
        ],
    }

    patcher = (
        patch.object(diag, "compute_lz4_ratio", side_effect=fake_lz4_ratio)
        if diag.lz4 is None
        else patch.object(diag, "compute_lz4_ratio", wraps=diag.compute_lz4_ratio)
    )
    with patcher:
        result = diag.run_full_audit(
            candidate_slices={
                "fineweb_edu": clean_records,
                "noisy_scrape": noisy_records,
                "openwebtext": clean_records,
            },
            eval_texts_by_benchmark=eval_splits,
            control_key="openwebtext",
        )

    scorecard = result["scorecard"]
    assert list(scorecard["Rank"]) == [1, 2, 3]
    assert scorecard.iloc[-1]["Key"] == "noisy_scrape"
    assert scorecard.iloc[0]["Composite Score"] > scorecard.iloc[-1]["Composite Score"]
    assert "Vocab Entropy (bits)" not in scorecard.columns
    assert "Within-Doc Loop (%)" not in scorecard.columns
