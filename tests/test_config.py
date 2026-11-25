"""Tests for `kestrel.config` (TrainConfig, TrackingConfig, model config loader, overrides)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel.config import (
    TrackingConfig,
    TrainConfig,
    apply_overrides,
    extract_model_metadata,
    load_model_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_config_from_json_and_to_dict() -> None:
    cfg_path = REPO_ROOT / "configs" / "train" / "climbmix_pretrain.json"
    cfg = TrainConfig.from_json(cfg_path)
    assert cfg.max_steps == 5000
    assert cfg.adamw_betas == (0.9, 0.95)
    assert isinstance(cfg.tracking, TrackingConfig)
    assert cfg.tracking.dagshub_repo == "MohammedKamalAlsyd/kestrel"

    as_dict = cfg.to_dict()
    assert as_dict["precision"] == "fp16"
    assert as_dict["tracking"]["stage"] == "stage2_bakeoff"


def test_apply_overrides_nested_and_errors() -> None:
    cfg = TrainConfig()
    apply_overrides(
        cfg,
        [
            "max_steps=250",
            "precision=bf16",
            "tracking.push_to_hub=true",
            "tracking.hf_repo_id=me/kestrel-test",
        ],
    )
    assert cfg.max_steps == 250
    assert cfg.precision == "bf16"
    assert cfg.tracking.push_to_hub is True
    assert cfg.tracking.hf_repo_id == "me/kestrel-test"

    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(cfg, ["bad_override"])

    with pytest.raises(KeyError, match="Unknown config field"):
        apply_overrides(cfg, ["nonexistent_field=1"])


def test_load_model_config_and_metadata() -> None:
    model_path = REPO_ROOT / "configs" / "model" / "kestrel_350m_dense.json"
    raw = load_model_config(model_path)
    assert raw["vocab_size"] == 50304

    meta = extract_model_metadata(raw)
    assert meta["blueprint"] == "Kestrel-350M-Dense"
    assert meta["track"] == "track_a_dense"
    assert meta["hf_repo_id"] == "MohammedKamalAlsyd/kestrel-350m-dense"
    assert "vocab_size" not in meta

    with pytest.raises(FileNotFoundError):
        load_model_config(REPO_ROOT / "configs" / "model" / "does_not_exist.json")
