"""Configuration schemas and JSON/CLI loaders for Kestrel training, models, and tracking."""

from kestrel.config.model import (
    KESTREL_MODEL_METADATA_KEYS,
    extract_model_metadata,
    load_model_config,
)
from kestrel.config.tracking import DEFAULT_EXPERIMENT_NAME, TrackingConfig
from kestrel.config.train import TrainConfig, apply_overrides

__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "KESTREL_MODEL_METADATA_KEYS",
    "TrackingConfig",
    "TrainConfig",
    "apply_overrides",
    "extract_model_metadata",
    "load_model_config",
]
