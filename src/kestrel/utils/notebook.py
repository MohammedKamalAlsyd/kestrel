"""Shared Kaggle/local environment and Hugging Face cache utilities for Kestrel."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


def is_kaggle() -> bool:
    """Return True when running inside a Kaggle notebook environment."""
    return Path("/kaggle/working").exists()


def get_work_dir() -> Path:
    """Return the writable working directory (`/kaggle/working` on Kaggle, else CWD)."""
    return Path("/kaggle/working") if is_kaggle() else Path.cwd()


def find_repo_root(extra_candidates: Sequence[str | Path] | None = None) -> Path | None:
    """Locate the Kestrel repository root containing `src/kestrel`."""
    candidates: list[Path] = []
    if extra_candidates:
        candidates.extend(Path(p) for p in extra_candidates)

    module_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            module_root,
            Path.cwd(),
            Path.cwd().parent,
            get_work_dir() / "kestrel",
            Path("/kaggle/input/kestrel"),
        ]
    )

    seen: set[Path] = set()
    for root in candidates:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "src" / "kestrel").is_dir():
            return resolved
    return None


def configure_hf_cache(cache_dir: str | Path) -> Path:
    """Configure low-disk Hugging Face streaming cache variables inside `cache_dir`."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HF_DATASETS_CACHE"] = str(cache_path / "datasets")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return cache_path


def get_hf_token(secret_name: str = "HF_TOKEN") -> str | None:
    """Retrieve a Hugging Face token from environment variables or Kaggle Secrets."""
    token = os.environ.get(secret_name) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    if is_kaggle():
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

            token = UserSecretsClient().get_secret(secret_name)
            if token:
                os.environ[secret_name] = token
                return token
        except Exception:
            return None
    return None
