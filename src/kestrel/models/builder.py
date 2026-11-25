"""Model builder and Hugging Face `AutoConfig` / `AutoModelForCausalLM` registration for Kestrel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM

from kestrel.config import extract_model_metadata, load_model_config
from kestrel.models.kestrel import KestrelConfig, KestrelForCausalLM

_REGISTERED = False

BLUEPRINT_CONFIGS: dict[str, str] = {
    "dense": "configs/model/kestrel_350m_dense.json",
    "hybrid_gdn": "configs/model/kestrel_350m_hybrid_gdn.json",
    "moe16": "configs/model/kestrel_700m_moe16.json",
    "diff_mhc_relu2": "configs/model/kestrel_350m_diff_mhc_relu2.json",
}


def register_kestrel_models() -> None:
    """Register `KestrelConfig` and `KestrelForCausalLM` with Hugging Face Auto classes."""
    global _REGISTERED
    if _REGISTERED:
        return
    AutoConfig.register(KestrelConfig.model_type, KestrelConfig, exist_ok=True)
    AutoModelForCausalLM.register(KestrelConfig, KestrelForCausalLM, exist_ok=True)
    _REGISTERED = True


class KestrelModelBuilder:
    """Clean factory for building any of the 4 Kestrel architectures from JSON, dict, or blueprint."""

    @staticmethod
    def build(
        config_or_path: str | Path | dict[str, Any] | KestrelConfig,
    ) -> tuple[KestrelForCausalLM, dict[str, Any]]:
        """Instantiate `(model, model_config_dict)` from a JSON path, dict, or `KestrelConfig`."""
        register_kestrel_models()

        raw_meta: dict[str, Any] = {}
        if isinstance(config_or_path, KestrelConfig):
            cfg = config_or_path
        elif isinstance(config_or_path, dict):
            raw_meta = extract_model_metadata(config_or_path)
            cfg = KestrelConfig(**config_or_path)
        else:
            raw = load_model_config(config_or_path)
            raw_meta = extract_model_metadata(raw)
            cfg = KestrelConfig(**raw)

        model = KestrelForCausalLM(cfg)
        cfg_dict = cfg.to_dict()
        for k, v in raw_meta.items():
            cfg_dict.setdefault(k, v)
        return model, cfg_dict

    @classmethod
    def from_blueprint(
        cls,
        name: str,
        *,
        repo_root: str | Path | None = None,
    ) -> tuple[KestrelForCausalLM, dict[str, Any]]:
        """Build one of the 4 registered Kestrel blueprints (`dense`, `hybrid_gdn`, `moe16`, `diff_mhc_relu2`)."""
        key = name.strip().lower()
        if key not in BLUEPRINT_CONFIGS:
            raise KeyError(f"Unknown blueprint '{name}'. Choose from: {sorted(BLUEPRINT_CONFIGS)}")
        base = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
        return cls.build(base / BLUEPRINT_CONFIGS[key])

    @staticmethod
    def summarize(model: KestrelForCausalLM) -> dict[str, Any]:
        """Return parameter breakdown (total, active, non-embedding) and architecture metadata."""
        total = sum(p.numel() for p in set(model.parameters()))
        active = model.count_active_parameters()
        emb = sum(p.numel() for p in model.get_input_embeddings().parameters())
        cfg = model.config
        return {
            "blueprint": cfg.blueprint,
            "track": cfg.track,
            "num_layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "total_params": total,
            "active_params": active,
            "non_embedding_params": total - emb,
            "non_embedding_active_params": active - emb,
        }
