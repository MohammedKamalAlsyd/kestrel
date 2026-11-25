"""Experiment tracking, HF Hub publishing, and GitHub snapshot configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_EXPERIMENT_NAME = "kestrel"


@dataclass
class TrackingConfig:
    """Configuration for DagsHub/MLflow tracking, HF Hub checkpoints, and GitHub snapshots."""

    enabled: bool = True
    dagshub_repo: str | None = None  # "owner/repo" -> https://dagshub.com/<owner>/<repo>.mlflow
    tracking_uri: str | None = None  # Explicit MLflow URI override (or MLFLOW_TRACKING_URI env)
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    run_name: str | None = None
    stage: str = "stage1_ablation"
    track: str = "track_a_dense"
    blueprint: str = "custom"
    tags: dict[str, str] = field(default_factory=dict)

    # Model destinations (recommended: log_model_to_mlflow=False + push_to_hub=True)
    log_model_to_mlflow: bool = False
    log_hf_checkpoint_artifact: bool = False
    registered_model_name: str | None = None
    push_to_hub: bool = False
    hf_repo_id: str | None = None  # Falls back to model_config["hf_repo_id"|"hub_model_id"]
    hf_revision: str | None = None
    hf_private: bool = False
    hf_path_in_repo: str | None = None

    # Lightweight GitHub-browsable run receipt (zero weights)
    export_github_snapshot: bool = True
    experiments_dir: str = "experiments"
    storage_dir: str = "artifacts/mlflow"  # Offline SQLite fallback when no remote URI is set
