"""Distributed (DDP / torchrun) process-group helpers for Kestrel."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed(backend: str | None = None) -> tuple[int, int, int, bool, str | None]:
    """Initialize `torch.distributed` when launched with `torchrun` (`RANK` & `WORLD_SIZE`).

    Returns `(rank, world_size, local_rank, is_distributed, backend)`.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, False, None

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    chosen_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=chosen_backend)

    is_dist = world_size > 1 and dist.is_available() and dist.is_initialized()
    return rank, world_size, local_rank, is_dist, (chosen_backend if is_dist else None)


def cleanup_distributed() -> None:
    """Destroy the active `torch.distributed` process group if initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
