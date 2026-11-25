"""Metrics, scaling/plateau estimation, capacity diagnostics, and local pre-flight checks."""

from kestrel.metrics.capacity import (
    effective_rank,
    measure_model_capacity,
    stratified_token_losses,
)
from kestrel.metrics.gradients import GradientHealthMonitor, audit_layer_gradients
from kestrel.metrics.plateau import EvalPoint, PlateauTracker
from kestrel.metrics.preflight import (
    check_amp_fp16_parity,
    check_causal_and_doc_leakage,
    check_single_batch_overfit,
    run_preflight_check,
)

__all__ = [
    "EvalPoint",
    "GradientHealthMonitor",
    "PlateauTracker",
    "audit_layer_gradients",
    "check_amp_fp16_parity",
    "check_causal_and_doc_leakage",
    "check_single_batch_overfit",
    "effective_rank",
    "measure_model_capacity",
    "run_preflight_check",
    "stratified_token_losses",
]
