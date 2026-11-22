"""Data auditing, multi-shard streaming, and diagnostic utilities for Kestrel pretraining corpora."""

from kestrel.data.diagnostics import (
    CANDIDATE_DATASETS,
    EVAL_BENCHMARKS,
    DatasetSpec,
    EvalSpec,
    audit_contamination_13gram,
    audit_cross_doc_boilerplate,
    audit_dataset_slice,
    audit_domain_alignment,
    audit_repetition_and_boilerplate,
    build_scorecard,
    compute_gopher_repetition_rates,
    load_eval_benchmark_texts,
    load_jsonl_slice,
    run_full_audit,
    stream_aggregate_audit,
    stream_matched_slice,
)

__all__ = [
    "CANDIDATE_DATASETS",
    "EVAL_BENCHMARKS",
    "DatasetSpec",
    "EvalSpec",
    "audit_contamination_13gram",
    "audit_cross_doc_boilerplate",
    "audit_dataset_slice",
    "audit_domain_alignment",
    "audit_repetition_and_boilerplate",
    "build_scorecard",
    "compute_gopher_repetition_rates",
    "load_eval_benchmark_texts",
    "load_jsonl_slice",
    "run_full_audit",
    "stream_aggregate_audit",
    "stream_matched_slice",
]
