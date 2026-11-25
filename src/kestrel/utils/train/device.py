"""Compute device detection, selection, and backend configuration for Kestrel."""

from __future__ import annotations

import warnings

import torch


def is_mps_available() -> bool:
    """Return True when Apple Silicon MPS backend is built and available."""
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _coerce_device_arg(device: str | torch.device | None) -> torch.device | None:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        cleaned = device.strip().lower()
        if not cleaned or cleaned == "auto":
            return None
        return torch.device(cleaned)
    raise TypeError(f"device must be str, torch.device, or None, got {type(device).__name__}")


def resolve_device(
    device: str | torch.device | None = "auto",
    *,
    local_rank: int = 0,
    fallback: bool = True,
) -> torch.device:
    """Resolve a target `torch.device` based on hardware availability and `local_rank`.

    When `device` is `'auto'` or `None`, selects `cuda:{local_rank}` -> `mps` -> `cpu`.
    If an unavailable accelerator is explicitly requested and `fallback=True`, warns
    and falls back to the best available device.
    """
    requested = _coerce_device_arg(device)
    if requested is None:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested.type == "cuda":
        if torch.cuda.is_available():
            idx = local_rank if requested.index is None else requested.index
            return torch.device(f"cuda:{idx}")
        if not fallback:
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")
        alt = torch.device("mps") if is_mps_available() else torch.device("cpu")
        warnings.warn(f"CUDA is unavailable; falling back to '{alt}'.", RuntimeWarning, stacklevel=2)
        return alt

    if requested.type == "mps":
        if is_mps_available():
            return torch.device("mps")
        if not fallback:
            raise RuntimeError("MPS device requested, but torch.backends.mps.is_available() is False.")
        warnings.warn("MPS is unavailable; falling back to 'cpu'.", RuntimeWarning, stacklevel=2)
        return torch.device("cpu")

    return requested


def configure_device_backends(device: torch.device, *, allow_tf32: bool = True) -> None:
    """Apply device-specific runtime settings (active CUDA device index and TF32 flags)."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
