"""Checkpoint I/O and Hugging Face Hub export utilities.

Centralizes all model serialization logic in one place so `train.py` and `mlflow_tracker.py`
stay small and focused:
- `unwrap_model`: strips `torch.compile` and DDP/FSDP wrappers safely.
- `save_checkpoint` / `load_checkpoint`: HF `save_pretrained()` weights + optimizer/RNG resume state.
- `push_checkpoint_to_hub`: uploads checkpoints to per-model HF Hub repos.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from kestrel.utils.notebook import get_hf_token


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying `nn.Module`, stripping `torch.compile` and DDP/FSDP wrappers."""
    cur: nn.Module = model
    while True:
        if hasattr(cur, "_orig_mod") and isinstance(cur._orig_mod, nn.Module):
            cur = cur._orig_mod
        elif hasattr(cur, "module") and isinstance(cur.module, nn.Module):
            cur = cur.module
        else:
            return cur


def save_checkpoint(
    ckpt_dir: str | Path,
    model: nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    scaler: torch.amp.GradScaler | None,
    *,
    step: int,
    tokens_seen: int,
    best_val_loss: float,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
) -> Path:
    """Save HF-format weights (if supported) plus full optimizer/training state for exact resume."""
    root = Path(ckpt_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw = unwrap_model(model)

    hf_dir = root / "hf_checkpoint"
    if hasattr(raw, "save_pretrained"):
        raw.save_pretrained(str(hf_dir))
    else:
        torch.save(raw.state_dict(), root / "model.pt")

    torch.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
            "optimizers": {k: opt.state_dict() for k, opt in optimizers.items()},
            "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "model_config": model_config,
            "train_config": train_config,
        },
        root / "training_state.pt",
    )
    return hf_dir if hf_dir.is_dir() else root


def load_checkpoint(
    ckpt_dir: str | Path,
    model: nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> tuple[int, int, float]:
    """Restore model weights, optimizer states, scaler, and `(step, tokens_seen, best_val_loss)`."""
    root = Path(ckpt_dir)
    state_file = root / "training_state.pt"
    if not state_file.is_file():
        raise FileNotFoundError(f"Missing checkpoint state file: {state_file}")

    raw = unwrap_model(model)
    hf_dir = root / "hf_checkpoint"
    if hf_dir.is_dir() and hasattr(raw, "from_pretrained"):
        loaded = raw.__class__.from_pretrained(str(hf_dir))
        raw.load_state_dict(loaded.state_dict())
    elif (root / "model.pt").is_file():
        raw.load_state_dict(torch.load(root / "model.pt", map_location=device, weights_only=True))

    state = torch.load(state_file, map_location=device, weights_only=False)
    for key, opt in optimizers.items():
        if key in state.get("optimizers", {}):
            opt.load_state_dict(state["optimizers"][key])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    return (
        int(state.get("step", 0)),
        int(state.get("tokens_seen", 0)),
        float(state.get("best_val_loss", float("inf"))),
    )


def push_checkpoint_to_hub(
    model: nn.Module,
    repo_id: str,
    *,
    model_config: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
    hf_checkpoint_dir: str | Path | None = None,
    revision: str | None = None,
    path_in_repo: str | None = None,
    private: bool = False,
    commit_message: str = "Upload Kestrel checkpoint",
) -> dict[str, str]:
    """Export model weights in HF safetensors format and push them to Hugging Face Hub.

    Reads `HF_TOKEN` from environment variables or Kaggle Secrets via `get_hf_token()`.
    Returns a dict with `repo_id`, `repo_url`, `revision`, `commit_sha`, and `commit_url`.
    """
    from huggingface_hub import HfApi

    token = get_hf_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)

    if revision and revision != "main":
        api.create_branch(repo_id=repo_id, branch=revision, repo_type="model", exist_ok=True, token=token)

    raw = unwrap_model(model)
    with tempfile.TemporaryDirectory(prefix="kestrel_hub_") as tmp:
        upload_dir = Path(tmp)
        if hf_checkpoint_dir is not None and Path(hf_checkpoint_dir).is_dir():
            shutil.copytree(Path(hf_checkpoint_dir), upload_dir, dirs_exist_ok=True)
        elif hasattr(raw, "save_pretrained"):
            raw.save_pretrained(str(upload_dir))
        else:
            from safetensors.torch import save_file

            state_cpu = {k: v.detach().cpu().contiguous() for k, v in raw.state_dict().items()}
            save_file(state_cpu, str(upload_dir / "model.safetensors"))
            if model_config:
                (upload_dir / "config.json").write_text(
                    json.dumps(model_config, indent=2, default=str) + "\n", encoding="utf-8"
                )

        # Save lightweight training/MLflow provenance alongside weights
        if run_metadata:
            (upload_dir / "kestrel_run_metadata.json").write_text(
                json.dumps(run_metadata, indent=2, default=str) + "\n", encoding="utf-8"
            )

        commit_info = api.upload_folder(
            repo_id=repo_id,
            folder_path=str(upload_dir),
            path_in_repo=path_in_repo,
            revision=revision,
            commit_message=commit_message,
            repo_type="model",
            token=token,
        )

    return {
        "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/{repo_id}",
        "revision": revision or "main",
        "commit_sha": str(getattr(commit_info, "oid", "") or ""),
        "commit_url": str(getattr(commit_info, "commit_url", "") or ""),
    }
