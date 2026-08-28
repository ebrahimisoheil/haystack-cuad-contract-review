from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    mode: str = "deterministic"
    models_path: Path = Path(__file__).parent / "model_routing.yaml"
    max_retries: int = 2
    timeout_seconds: float = 60.0
    playbook_path: Path = Path(__file__).parent / "playbooks" / "vendor_saas.yaml"

    @classmethod
    def from_env(cls, *, mode: str | None = None) -> "Settings":
        load_dotenv()
        selected_mode = mode or os.getenv("CONTRACT_REVIEW_MODE", "deterministic")
        if selected_mode not in {"deterministic", "live"}:
            raise ValueError("CONTRACT_REVIEW_MODE must be 'deterministic' or 'live'")
        return cls(
            mode=selected_mode,
            models_path=Path(
                os.getenv(
                    "CONTRACT_REVIEW_MODELS_FILE",
                    str(Path(__file__).parent / "model_routing.yaml"),
                )
            ),
            max_retries=int(os.getenv("CONTRACT_REVIEW_MAX_RETRIES", "2")),
            timeout_seconds=float(os.getenv("CONTRACT_REVIEW_TIMEOUT_SECONDS", "60")),
        )

    def model_registry(self) -> dict[str, dict[str, object]]:
        config = yaml.safe_load(self.models_path.read_text(encoding="utf-8"))
        models = config.get("models") if isinstance(config, dict) else None
        if not isinstance(models, dict) or set(models) != {"text", "vision", "judge"}:
            raise ValueError("model routing must define exactly text, vision, and judge roles")
        result: dict[str, dict[str, object]] = {}
        for role, value in models.items():
            if not isinstance(value, dict) or value.get("provider") != "litellm" or not value.get("model"):
                raise ValueError(f"invalid LiteLLM model configuration for role {role}")
            result[role] = dict(value)
        overrides = {
            "text": os.getenv("CONTRACT_REVIEW_TEXT_MODEL"),
            "vision": os.getenv("CONTRACT_REVIEW_VISION_MODEL"),
            "judge": os.getenv("CONTRACT_REVIEW_JUDGE_MODEL"),
        }
        for role, model in overrides.items():
            if model:
                result[role]["model"] = model
        return result
