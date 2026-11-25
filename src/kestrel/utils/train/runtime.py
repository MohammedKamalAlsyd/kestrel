"""High-level runtime orchestration composing device, precision, DDP, and seeding helpers."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch

from kestrel.utils.train.device import configure_device_backends, is_mps_available, resolve_device
from kestrel.utils.train.distributed import setup_distributed
from kestrel.utils.train.precision import (
    is_bf16_supported,
    is_fp16_supported,
    make_grad_scaler,
    resolve_dtype,
)
from kestrel.utils.train.seed import seed_everything


@dataclass(frozen=True)
class RuntimeEnv:
    """Initialized compute runtime state returned by `setup_runtime()`."""

    device: torch.device
    dtype: torch.dtype
    precision: str
    autocast_enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    is_distributed: bool = False
    is_main: bool = True
    backend: str | None = None
    scaler: torch.amp.GradScaler | None = None

    def autocast(self) -> contextlib.AbstractContextManager[Any]:
        """Return an autocast context manager matching `device` and `dtype`."""
        if not self.autocast_enabled:
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type=self.device.type, dtype=self.dtype)

    def summary(self) -> str:
        """Return a concise single-line description of the active runtime."""
        info = f"Device={self.device} x{self.world_size} | Dtype={self.precision}"
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(self.device)
                info = f"Device={self.device} ({name}) x{self.world_size} | Dtype={self.precision}"
            except Exception:
                pass
        if self.scaler is not None:
            info += " (GradScaler)"
        return info


def get_device_info(device: str | torch.device | None = None) -> dict[str, Any]:
    """Collect diagnostic hardware and dtype-support metadata for logging or notebooks."""
    dev = resolve_device(device)
    auto_precision, auto_dtype = resolve_dtype("auto", dev)
    info: dict[str, Any] = {
        "device": str(dev),
        "device_type": dev.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": is_mps_available(),
        "bf16_supported": is_bf16_supported(dev, including_emulation=False),
        "fp16_supported": is_fp16_supported(dev),
        "recommended_precision": auto_precision,
        "recommended_dtype": str(auto_dtype),
    }
    if dev.type == "cuda" and torch.cuda.is_available():
        idx = 0 if dev.index is None else dev.index
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_count": torch.cuda.device_count(),
                "compute_capability": f"{props.major}.{props.minor}",
                "vram_gb": round(props.total_memory / (1024**3), 2),
            }
        )
    return info


def setup_runtime(
    seed: int = 42,
    precision: str | torch.dtype = "auto",
    *,
    device: str | torch.device | None = "auto",
    distributed: bool = True,
    backend: str | None = None,
    allow_tf32: bool = True,
    deterministic: bool = False,
    including_emulation: bool = False,
    fallback: bool = True,
) -> RuntimeEnv:
    """Initialize distributed state, device, RNG seeds, dtype, autocast, and GradScaler."""
    if distributed:
        rank, world_size, local_rank, is_dist, active_backend = setup_distributed(backend=backend)
    else:
        rank, world_size, local_rank, is_dist, active_backend = 0, 1, 0, False, None

    resolved_device = resolve_device(device, local_rank=local_rank, fallback=fallback)
    configure_device_backends(resolved_device, allow_tf32=allow_tf32)
    seed_everything(seed, rank=rank, deterministic=deterministic)

    resolved_precision, resolved_dtype = resolve_dtype(
        precision,
        device=resolved_device,
        including_emulation=including_emulation,
        fallback=fallback,
    )
    autocast_enabled = resolved_dtype != torch.float32
    scaler = make_grad_scaler(resolved_device, resolved_dtype)

    return RuntimeEnv(
        device=resolved_device,
        dtype=resolved_dtype,
        precision=resolved_precision,
        autocast_enabled=autocast_enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=is_dist,
        is_main=(rank == 0),
        backend=active_backend,
        scaler=scaler,
    )
