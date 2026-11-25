"""Lean DagsHub / MLflow experiment tracker and GitHub snapshot exporter for Kestrel.

Logs directly to DagsHub hosted MLflow (`https://dagshub.com/<owner>/<repo>.mlflow`) or any
`MLFLOW_TRACKING_URI` (no local server daemon or manual syncing needed). Checkpoint I/O and
Hugging Face Hub uploads are handled by `kestrel.utils.train.checkpoint`.
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from kestrel.config.tracking import DEFAULT_EXPERIMENT_NAME, TrackingConfig
from kestrel.utils.train.checkpoint import (
    push_checkpoint_to_hub,
    unwrap_model,
)

MAX_MLFLOW_PARAM_LEN = 500


def configure_dagshub_tracking(dagshub_repo: str | None = None, token: str | None = None) -> str | None:
    """Wire DagsHub MLflow env credentials (`DAGSHUB_TOKEN`) and return the tracking URI."""
    repo = (dagshub_repo or os.environ.get("DAGSHUB_REPO") or "").strip()
    if not repo:
        owner = os.environ.get("DAGSHUB_REPO_OWNER", "").strip()
        name = os.environ.get("DAGSHUB_REPO_NAME", "").strip()
        if owner and name:
            repo = f"{owner}/{name}"

    auth_token = (token or os.environ.get("DAGSHUB_TOKEN") or os.environ.get("DAGSHUB_USER_TOKEN") or "").strip()
    if auth_token and not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = auth_token

    if not repo:
        return None
    if "/" not in repo:
        raise ValueError(f"Expected DagsHub repo in 'owner/repo' format, got '{repo}'")

    owner, repo_name = repo.split("/", 1)
    if not os.environ.get("MLFLOW_TRACKING_USERNAME"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("DAGSHUB_USERNAME", owner)
    return f"https://dagshub.com/{owner}/{repo_name}.mlflow"


def resolve_tracking_uri(
    tracking_uri: str | None = None,
    dagshub_repo: str | None = None,
    storage_dir: str | Path = "artifacts/mlflow",
) -> str:
    """Resolve MLflow URI: explicit `tracking_uri` -> DagsHub -> `MLFLOW_TRACKING_URI` -> offline SQLite."""
    dagshub_uri = configure_dagshub_tracking(dagshub_repo=dagshub_repo)
    uri = (tracking_uri or dagshub_uri or os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if uri:
        return uri

    root = Path(storage_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(root / 'mlflow.db').as_posix()}"


def ensure_experiment(
    tracking_uri: str,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    artifact_location: str | None = None,
) -> str:
    """Create or retrieve the MLflow experiment on the active tracking server."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is not None:
        return str(existing.experiment_id)
    return str(
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
            tags={"project": "kestrel"},
        )
    )


def flatten_dict(data: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, str]:
    """Flatten nested config dicts into MLflow-safe string params (<=500 chars each)."""
    flat: dict[str, str] = {}
    for key, val in data.items():
        full_key = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(val, dict):
            flat.update(flatten_dict(val, prefix=full_key, sep=sep))
        elif isinstance(val, (list, tuple)):
            flat[full_key] = json.dumps(val)[:MAX_MLFLOW_PARAM_LEN]
        elif val is None:
            flat[full_key] = "null"
        else:
            flat[full_key] = str(val)[:MAX_MLFLOW_PARAM_LEN]
    return flat


def _git_provenance() -> dict[str, str]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
        return {"git_commit": sha, "git_branch": branch, "git_dirty": str(dirty).lower()}
    except Exception:
        return {}


def _hardware_tags() -> dict[str, str]:
    tags = {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()).lower(),
    }
    if torch.cuda.is_available():
        tags["gpu_name"] = torch.cuda.get_device_name(0)
        tags["gpu_count"] = str(torch.cuda.device_count())
    return tags


def _clean_run_name(name: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name.strip())
    return cleaned.strip("_.") or "kestrel_run"


class KestrelMLflowTracker:
    """Context-managed DagsHub/MLflow logger with HF Hub cross-linking and GitHub snapshots."""

    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self.enabled = cfg.enabled
        self.tracking_uri: str | None = None
        self.experiment_id: str | None = None
        self.run_id: str | None = None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_name = _clean_run_name(cfg.run_name or f"{stamp}_{cfg.blueprint}_{cfg.track}")
        self._active_run: Any = None
        self._mlflow: Any = None
        self._history: list[dict[str, Any]] = []
        self._param_counts: dict[str, int] = {}
        self._git_info: dict[str, str] = {}
        self._hf_hub_info: dict[str, str] = {}

    def __enter__(self) -> KestrelMLflowTracker:
        if not self.enabled:
            return self

        import mlflow

        self._mlflow = mlflow
        self.tracking_uri = resolve_tracking_uri(
            tracking_uri=self.cfg.tracking_uri,
            dagshub_repo=self.cfg.dagshub_repo,
            storage_dir=self.cfg.storage_dir,
        )
        artifact_loc = None
        if self.tracking_uri.startswith("sqlite:///"):
            art_dir = Path(self.cfg.storage_dir).resolve() / "artifacts"
            art_dir.mkdir(parents=True, exist_ok=True)
            artifact_loc = art_dir.as_uri()

        self.experiment_id = ensure_experiment(
            self.tracking_uri,
            experiment_name=self.cfg.experiment_name,
            artifact_location=artifact_loc,
        )
        mlflow.set_experiment(self.cfg.experiment_name)

        self._git_info = _git_provenance()
        tags = {
            "project": "kestrel",
            "stage": self.cfg.stage,
            "track": self.cfg.track,
            "blueprint": self.cfg.blueprint,
            **_hardware_tags(),
            **self._git_info,
            **self.cfg.tags,
        }
        self._active_run = mlflow.start_run(run_name=self.run_name, tags=tags)
        self.run_id = str(self._active_run.info.run_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.enabled and self._mlflow is not None and self._active_run is not None:
            self._mlflow.end_run(status="FAILED" if exc_type is not None else "FINISHED")
            self._active_run = None

    def log_configs(self, model_config: dict[str, Any], train_config: dict[str, Any]) -> None:
        """Log flattened params and raw JSON config artifacts."""
        if not self.enabled or self._mlflow is None:
            return

        flat = {**flatten_dict(model_config, prefix="model"), **flatten_dict(train_config, prefix="train")}
        items = list(flat.items())
        for idx in range(0, len(items), 100):
            self._mlflow.log_params(dict(items[idx : idx + 100]))

        with tempfile.TemporaryDirectory(prefix="kestrel_cfg_") as tmp:
            p = Path(tmp)
            (p / "model_config.json").write_text(json.dumps(model_config, indent=2, default=str) + "\n", encoding="utf-8")
            (p / "train_config.json").write_text(json.dumps(train_config, indent=2, default=str) + "\n", encoding="utf-8")
            self._mlflow.log_artifacts(str(p), artifact_path="configs")

    def log_model_summary(self, model: nn.Module) -> dict[str, int]:
        """Compute and log unique total and active parameter counts."""
        raw = unwrap_model(model)
        total = sum(p.numel() for p in set(raw.parameters()))
        active = int(raw.count_active_parameters()) if hasattr(raw, "count_active_parameters") else total
        self._param_counts = {"total": total, "active": active}
        if self.enabled and self._mlflow is not None:
            self._mlflow.log_params({f"params.{k}": str(v) for k, v in self._param_counts.items()})
        return dict(self._param_counts)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log finite scalar metrics to MLflow and buffer them for `history.csv`."""
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}
        if not clean:
            return
        self._history.append({"step": step if step is not None else -1, **clean})
        if self.enabled and self._mlflow is not None:
            self._mlflow.log_metrics(clean, step=step)

    def log_model(
        self,
        model: nn.Module,
        *,
        model_name: str = "kestrel_model",
        step: int = 0,
        extra_metadata: dict[str, Any] | None = None,
        hf_checkpoint_dir: str | Path | None = None,
    ) -> Any:
        """Optionally log PyTorch weights directly to MLflow (off by default when pushing to HF Hub)."""
        if not self.enabled or self._mlflow is None or not self.cfg.log_model_to_mlflow:
            return None

        import mlflow.pytorch

        raw = unwrap_model(model)
        was_training = raw.training
        first_param = next(raw.parameters(), None)
        orig_device = first_param.device if first_param is not None else torch.device("cpu")
        raw.eval()

        sig = inspect.signature(mlflow.pytorch.log_model)
        kwargs: dict[str, Any] = {
            "pip_requirements": [f"torch=={torch.__version__.split('+')[0]}", "transformers>=4.40"],
            ("name" if "name" in sig.parameters else "artifact_path"): model_name,
        }
        if "step" in sig.parameters:
            kwargs["step"] = step
        if "metadata" in sig.parameters and extra_metadata:
            kwargs["metadata"] = extra_metadata
        if self.cfg.registered_model_name:
            kwargs["registered_model_name"] = self.cfg.registered_model_name

        try:
            if orig_device.type != "cpu":
                raw.to("cpu")
            model_info = mlflow.pytorch.log_model(raw, **kwargs)
            if self.cfg.log_hf_checkpoint_artifact and hasattr(raw, "save_pretrained"):
                if hf_checkpoint_dir is not None and Path(hf_checkpoint_dir).is_dir():
                    self._mlflow.log_artifacts(str(hf_checkpoint_dir), artifact_path="hf_checkpoint")
                else:
                    with tempfile.TemporaryDirectory(prefix="kestrel_hf_") as tmp:
                        raw.save_pretrained(tmp)
                        self._mlflow.log_artifacts(tmp, artifact_path="hf_checkpoint")
        finally:
            if orig_device.type != "cpu":
                raw.to(orig_device)
            if was_training:
                raw.train()

        return model_info

    def publish_to_hub(
        self,
        model: nn.Module,
        *,
        model_config: dict[str, Any],
        train_config: dict[str, Any],
        final_metrics: dict[str, Any],
        hf_checkpoint_dir: str | Path | None = None,
    ) -> dict[str, str] | None:
        """Push checkpoint to Hugging Face Hub via `kestrel.utils.train.checkpoint` and tag MLflow run."""
        if not self.cfg.push_to_hub:
            return None

        repo_id = (self.cfg.hf_repo_id or model_config.get("hf_repo_id") or "").strip()
        if not repo_id:
            raise ValueError(
                "push_to_hub=True requires `tracking.hf_repo_id` or `\"hf_repo_id\"` in the model config JSON."
            )

        info = push_checkpoint_to_hub(
            model,
            repo_id=repo_id,
            model_config=model_config,
            run_metadata={
                "run_name": self.run_name,
                "mlflow_run_id": self.run_id,
                "mlflow_tracking_uri": self.tracking_uri,
                "stage": self.cfg.stage,
                "track": self.cfg.track,
                "blueprint": self.cfg.blueprint,
                "git": self._git_info,
                "parameters": self._param_counts,
                "metrics": final_metrics,
                "train_config": train_config,
            },
            hf_checkpoint_dir=hf_checkpoint_dir,
            revision=self.cfg.hf_revision,
            path_in_repo=self.cfg.hf_path_in_repo,
            private=self.cfg.hf_private,
            commit_message=f"Upload Kestrel checkpoint ({self.run_name})",
        )
        self._hf_hub_info = info
        if self.enabled and self._mlflow is not None:
            self._mlflow.set_tags({f"hf_hub.{k}": v for k, v in info.items() if v})
        return dict(info)

    def export_github_snapshot(
        self,
        model_config: dict[str, Any],
        train_config: dict[str, Any],
        final_metrics: dict[str, Any],
    ) -> Path | None:
        """Export lightweight JSON/CSV run summary to `experiments/runs/<run_name>/` (no weights)."""
        if not self.enabled or not self.cfg.export_github_snapshot:
            return None

        exp_root = Path(self.cfg.experiments_dir)
        run_dir = exp_root / "runs" / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "model_config.json").write_text(json.dumps(model_config, indent=2, default=str) + "\n", encoding="utf-8")
        (run_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, default=str) + "\n", encoding="utf-8")
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "run_name": self.run_name,
                    "mlflow_run_id": self.run_id,
                    "mlflow_experiment": self.cfg.experiment_name,
                    "stage": self.cfg.stage,
                    "track": self.cfg.track,
                    "blueprint": self.cfg.blueprint,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "git": self._git_info,
                    "hf_hub": self._hf_hub_info or None,
                    "parameters": self._param_counts,
                    "metrics": final_metrics,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        if self._history:
            cols = list(dict.fromkeys(["step"] + [k for row in self._history for k in row if k != "step"]))
            with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=cols)
                writer.writeheader()
                writer.writerows(self._history)

        self._update_runs_index(exp_root / "runs_index.csv", final_metrics)
        return run_dir

    def _update_runs_index(self, index_path: Path, final_metrics: dict[str, Any]) -> None:
        fields = [
            "timestamp_utc", "run_name", "mlflow_run_id", "stage", "track", "blueprint",
            "git_commit", "hf_repo_id", "hf_commit_sha", "total_params", "active_params",
            "tokens_seen", "final_train_loss", "best_val_loss", "best_val_ppl",
            "mean_tokens_per_sec", "gpu_hours",
        ]
        new_row = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_name": self.run_name,
            "mlflow_run_id": self.run_id or "",
            "stage": self.cfg.stage,
            "track": self.cfg.track,
            "blueprint": self.cfg.blueprint,
            "git_commit": self._git_info.get("git_commit", ""),
            "hf_repo_id": self._hf_hub_info.get("repo_id", ""),
            "hf_commit_sha": self._hf_hub_info.get("commit_sha", ""),
            "total_params": self._param_counts.get("total", ""),
            "active_params": self._param_counts.get("active", ""),
            **{k: final_metrics.get(k, "") for k in ("tokens_seen", "final_train_loss", "best_val_loss", "best_val_ppl", "mean_tokens_per_sec", "gpu_hours")},
        }

        rows: list[dict[str, Any]] = []
        if index_path.exists():
            with index_path.open("r", encoding="utf-8", newline="") as fh:
                rows = [r for r in csv.DictReader(fh) if not (self.run_id and r.get("mlflow_run_id") == self.run_id)]
        rows.append(new_row)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
