"""Unit tests for kestrel.data.diagnostics."""

from __future__ import annotations

import math
import zlib
from unittest.mock import patch

import networkx as nx
import pytest

from kestrel.data import diagnostics as diag


def test_stratified_transition_network_detects_left_and_right_tail_noise():
    # Left-tail boilerplate: "policy privacy" repeats relentlessly
    boilerplate_docs = [
        "Our policy privacy notice states that policy privacy rules protect every user."
        for _ in range(40)
    ]
    left_res = diag.audit_transition_network(boilerplate_docs)
    policy_stats = left_res["content"]["per_word"]["policy"]
    assert policy_stats["top1_mass"] > 0.9
    assert policy_stats["left_tail_flag"] is True
    assert "policy" in left_res["content"]["left_tail_flagged_words"]
    assert left_res["boilerplate"]["cross_doc_boilerplate_doc_rate"] > 0.9

    # Right-tail broken scrape: "system" followed by 40 completely distinct random tokens
    unique_followers = [f"tok{chr(97 + i // 26)}{chr(97 + i % 26)}x" for i in range(60)]
    scattered_docs = [" ".join(f"system {w}" for w in unique_followers)]
    right_res = diag.audit_transition_network(scattered_docs)
    sys_stats = right_res["content"]["per_word"]["system"]
    assert sys_stats["out_degree"] == 60
    assert sys_stats["normalized_entropy"] > 0.99
    assert sys_stats["singleton_ratio"] == 1.0
    assert sys_stats["right_tail_flag"] is True
    assert "system" in right_res["content"]["right_tail_flagged_words"]

    # Build NetworkX DiGraph check
    g = diag.build_transition_digraph(left_res, "policy", stratum="content", top_k=5)
    assert isinstance(g, nx.DiGraph)
    assert g.has_edge("policy", "privacy")
    assert g["policy"]["privacy"]["weight"] > 0.9


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


def test_run_full_audit_and_scorecard_end_to_end():
    # Mock LZ4 ratio deterministically if lz4 isn't installed in the local test env
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
