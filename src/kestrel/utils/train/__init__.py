"""Training utilities: device, precision/dtype, DDP, seeding, runtime, optimizers, checkpoints."""

from kestrel.utils.train import checkpoint, device, distributed, optim, precision, runtime, seed
from kestrel.utils.train.checkpoint import (
    load_checkpoint,
    push_checkpoint_to_hub,
    save_checkpoint,
    unwrap_model,
)
from kestrel.utils.train.device import (
    configure_device_backends,
    is_mps_available,
    resolve_device,
)
from kestrel.utils.train.distributed import (
    cleanup_distributed,
    setup_distributed,
)
from kestrel.utils.train.optim import (
    apply_lr_schedule,
    build_optimizers,
    compute_lr_multiplier,
)
from kestrel.utils.train.precision import (
    get_dtype,
    is_bf16_supported,
    is_dtype_supported,
    is_fp16_supported,
    make_autocast_context,
    make_grad_scaler,
    resolve_dtype,
)
from kestrel.utils.train.runtime import (
    RuntimeEnv,
    get_device_info,
    setup_runtime,
)
from kestrel.utils.train.seed import seed_everything

__all__ = [
    "RuntimeEnv",
    "apply_lr_schedule",
    "build_optimizers",
    "checkpoint",
    "cleanup_distributed",
    "compute_lr_multiplier",
    "configure_device_backends",
    "device",
    "distributed",
    "get_device_info",
    "get_dtype",
    "is_bf16_supported",
    "is_dtype_supported",
    "is_fp16_supported",
    "is_mps_available",
    "load_checkpoint",
    "make_autocast_context",
    "make_grad_scaler",
    "optim",
    "precision",
    "push_checkpoint_to_hub",
    "resolve_device",
    "resolve_dtype",
    "runtime",
    "save_checkpoint",
    "seed",
    "seed_everything",
    "setup_distributed",
    "setup_runtime",
    "unwrap_model",
]
