"""Deterministic RNG seeding utilities for Kestrel."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, rank: int = 0, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch RNGs with `seed + rank` and return the effective seed."""
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return effective_seed
