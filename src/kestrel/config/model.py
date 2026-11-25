"""Model JSON configuration loading and Kestrel metadata extraction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

KESTREL_MODEL_METADATA_KEYS: tuple[str, ...] = (
    "hf_repo_id",
    "hub_model_id",
    "blueprint",
    "track",
)


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Read and parse a Kestrel model configuration JSON file."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Model config file not found: {cfg_path}")
    return dict(json.loads(cfg_path.read_text(encoding="utf-8")))


def extract_model_metadata(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Extract Kestrel-specific metadata fields from a raw model config dict."""
    return {key: raw_config[key] for key in KESTREL_MODEL_METADATA_KEYS if key in raw_config}
