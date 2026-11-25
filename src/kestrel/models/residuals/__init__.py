"""Residual connection mechanisms (Value Residuals and mHC) for Kestrel."""

from kestrel.models.residuals.mhc import ManifoldHyperConnection, sinkhorn_knopp
from kestrel.models.residuals.value_residual import ValueResidualGate

__all__ = [
    "ManifoldHyperConnection",
    "ValueResidualGate",
    "sinkhorn_knopp",
]
