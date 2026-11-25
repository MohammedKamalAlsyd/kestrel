"""Mixed-precision dtype detection, autocast contexts, and GradScaler helpers."""

from __future__ import annotations

import contextlib
import inspect
import warnings
from collections.abc import Callable
from typing import Any

import torch

from kestrel.utils.train.device import is_mps_available, resolve_device

_PRECISION_ALIASES: dict[str, tuple[str, torch.dtype]] = {
    "bf16": ("bf16", torch.bfloat16),
    "bfloat16": ("bf16", torch.bfloat16),
    "fp16": ("fp16", torch.float16),
    "float16": ("fp16", torch.float16),
    "half": ("fp16", torch.float16),
    "fp32": ("fp32", torch.float32),
    "float32": ("fp32", torch.float32),
    "float": ("fp32", torch.float32),
}

_TORCH_DTYPE_TO_NAME: dict[torch.dtype, str] = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
}


def _normalize_precision(precision: str | torch.dtype) -> str:
    """Normalize a precision string or `torch.dtype` to `'auto'`, `'bf16'`, `'fp16'`, or `'fp32'`."""
    if isinstance(precision, torch.dtype):
        if precision not in _TORCH_DTYPE_TO_NAME:
            raise ValueError(f"Unsupported torch.dtype '{precision}'; expected bfloat16, float16, or float32.")
        return _TORCH_DTYPE_TO_NAME[precision]
    if not isinstance(precision, str):
        raise TypeError(f"precision must be a str or torch.dtype, got {type(precision).__name__}")
    key = precision.strip().lower()
    if key == "auto":
        return "auto"
    if key in _PRECISION_ALIASES:
        return _PRECISION_ALIASES[key][0]
    raise ValueError(f"Unknown precision '{precision}'; expected 'auto', 'bf16', 'fp16', or 'fp32'.")


def is_bf16_supported(
    device: str | torch.device | None = None,
    *,
    including_emulation: bool = False,
) -> bool:
    """Return True if `bfloat16` compute is supported on `device`.

    Defaults to `including_emulation=False` on CUDA so GPUs without native BF16
    Tensor Cores (e.g., Turing / Kaggle T4, compute capability 7.5) report `False`
    and cleanly select FP16 instead.
    """
    dev = resolve_device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            return False
        fn = torch.cuda.is_bf16_supported
        try:
            sig = inspect.signature(fn)
            if "including_emulation" in sig.parameters:
                return bool(fn(including_emulation=including_emulation))
        except (TypeError, ValueError):
            pass
        supported = bool(fn())
        if supported and not including_emulation:
            idx = 0 if dev.index is None else dev.index
            major, _ = torch.cuda.get_device_capability(idx)
            return major >= 8
        return supported

    if dev.type == "mps":
        if not is_mps_available():
            return False
        mps = torch.backends.mps
        return bool(hasattr(mps, "is_macos_or_newer") and mps.is_macos_or_newer(14, 0))

    if dev.type == "cpu":
        return bool(torch.amp.autocast_mode.is_autocast_available("cpu"))

    return False


def is_fp16_supported(device: str | torch.device | None = None) -> bool:
    """Return True if `float16` mixed-precision compute is supported on `device`."""
    dev = resolve_device(device)
    if dev.type == "cuda":
        return bool(torch.cuda.is_available())
    if dev.type == "mps":
        return is_mps_available()
    return False


def is_dtype_supported(
    dtype: str | torch.dtype,
    device: str | torch.device | None = None,
    *,
    including_emulation: bool = False,
) -> bool:
    """Return True if the requested precision/dtype is supported on `device`."""
    name = _normalize_precision(dtype)
    if name in ("auto", "fp32"):
        return True
    if name == "bf16":
        return is_bf16_supported(device, including_emulation=including_emulation)
    if name == "fp16":
        return is_fp16_supported(device)
    return False


def resolve_dtype(
    precision: str | torch.dtype = "auto",
    device: str | torch.device | None = None,
    *,
    including_emulation: bool = False,
    fallback: bool = True,
) -> tuple[str, torch.dtype]:
    """Resolve `(precision_name, torch_dtype)` for `device` based on hardware support.

    With `precision='auto'`:
    - CUDA: `'bf16'` if natively supported (Ampere+), else `'fp16'` (e.g. T4).
    - MPS:  `'bf16'` on macOS 14+, else `'fp32'` (avoids GradScaler-less FP16 overflow).
    - CPU:  `'fp32'`.
    """
    dev = resolve_device(device)
    name = _normalize_precision(precision)

    if name == "auto":
        if dev.type == "cuda":
            if is_bf16_supported(dev, including_emulation=including_emulation):
                return "bf16", torch.bfloat16
            if is_fp16_supported(dev):
                return "fp16", torch.float16
        elif dev.type == "mps" and is_bf16_supported(dev, including_emulation=including_emulation):
            return "bf16", torch.bfloat16
        return "fp32", torch.float32

    if name == "fp32":
        return "fp32", torch.float32

    if name == "bf16":
        if is_bf16_supported(dev, including_emulation=including_emulation):
            return "bf16", torch.bfloat16
        if not fallback:
            raise RuntimeError(f"Precision 'bf16' is not natively supported on device '{dev}'.")
        alt_name, alt_dtype = ("fp16", torch.float16) if is_fp16_supported(dev) else ("fp32", torch.float32)
        warnings.warn(
            f"Precision 'bf16' is not supported on '{dev}'; falling back to '{alt_name}'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return alt_name, alt_dtype

    if name == "fp16":
        if is_fp16_supported(dev):
            return "fp16", torch.float16
        if not fallback:
            raise RuntimeError(f"Precision 'fp16' is not supported on device '{dev}'.")
        warnings.warn(
            f"Precision 'fp16' is not supported on '{dev}'; falling back to 'fp32'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "fp32", torch.float32

    return "fp32", torch.float32


def get_dtype(
    precision: str | torch.dtype = "auto",
    device: str | torch.device | None = None,
    *,
    including_emulation: bool = False,
    fallback: bool = True,
) -> torch.dtype:
    """Convenience wrapper around `resolve_dtype()` returning only the `torch.dtype`."""
    _, dtype = resolve_dtype(
        precision,
        device=device,
        including_emulation=including_emulation,
        fallback=fallback,
    )
    return dtype


def make_autocast_context(
    device: torch.device,
    dtype: torch.dtype,
) -> Callable[[], contextlib.AbstractContextManager[Any]]:
    """Return a zero-arg callable creating the appropriate autocast context manager."""
    if dtype == torch.float32:
        return contextlib.nullcontext
    return lambda: torch.amp.autocast(device_type=device.type, dtype=dtype)


def make_grad_scaler(device: torch.device, dtype: torch.dtype) -> torch.amp.GradScaler | None:
    """Return an enabled CUDA `GradScaler` when training in FP16 on CUDA, else `None`."""
    if device.type == "cuda" and dtype == torch.float16:
        return torch.amp.GradScaler("cuda", enabled=True)
    return None
