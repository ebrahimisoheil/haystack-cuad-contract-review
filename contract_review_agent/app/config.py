from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    mode: str = "deterministic"
    models_path: Path = Path(__file__).parent / "model_routing.yaml"
    max_retries: int = 2
    timeout_seconds: float = 60.0
    playbook_path: Path = Path(__file__).parent / "playbooks" / "vendor_saas.yaml"
    memory_mode: Literal["off", "shadow", "retrieve"] = "off"
    memory_uri: str = "data/contract-memory"
    memory_table: str = "contract_precedents"
    memory_tenant: str = "showcase"
    memory_allowed_groups: tuple[str, ...] = ("all",)
    memory_top_k: int = 3
    memory_candidate_k: int = 12

    @classmethod
    def from_env(cls, *, mode: str | None = None) -> "Settings":
        load_dotenv()
        selected_mode = mode or os.getenv("CONTRACT_REVIEW_MODE", "deterministic")
        if selected_mode not in {"deterministic", "live"}:
            raise ValueError("CONTRACT_REVIEW_MODE must be 'deterministic' or 'live'")
        memory_mode = os.getenv("CONTRACT_REVIEW_MEMORY_MODE", "off")
        if memory_mode not in {"off", "shadow", "retrieve"}:
            raise ValueError(
                "CONTRACT_REVIEW_MEMORY_MODE must be 'off', 'shadow', or 'retrieve'"
            )
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
            memory_mode=memory_mode,
            memory_uri=os.getenv("CONTRACT_REVIEW_MEMORY_URI", "data/contract-memory"),
            memory_table=os.getenv(
                "CONTRACT_REVIEW_MEMORY_TABLE", "contract_precedents"
            ),
            memory_tenant=os.getenv("CONTRACT_REVIEW_MEMORY_TENANT", "showcase"),
            memory_allowed_groups=tuple(
                group.strip()
                for group in os.getenv(
                    "CONTRACT_REVIEW_MEMORY_ALLOWED_GROUPS", "all"
                ).split(",")
                if group.strip()
            ),
            memory_top_k=int(os.getenv("CONTRACT_REVIEW_MEMORY_TOP_K", "3")),
            memory_candidate_k=int(
                os.getenv("CONTRACT_REVIEW_MEMORY_CANDIDATE_K", "12")
            ),
        )

    def model_registry(self) -> dict[str, dict[str, object]]:
        config = yaml.safe_load(self.models_path.read_text(encoding="utf-8"))
        models = config.get("models") if isinstance(config, dict) else None
        required_roles = {"text", "vision", "judge", "embedding"}
        if not isinstance(models, dict) or set(models) != required_roles:
            raise ValueError(
                "model routing must define exactly text, vision, judge, and embedding roles"
            )
        result: dict[str, dict[str, object]] = {}
        for role, value in models.items():
            if (
                not isinstance(value, dict)
                or value.get("provider") != "litellm"
                or not value.get("model")
            ):
                raise ValueError(f"invalid LiteLLM model configuration for role {role}")
            result[role] = dict(value)
        overrides = {
            "text": os.getenv("CONTRACT_REVIEW_TEXT_MODEL"),
            "vision": os.getenv("CONTRACT_REVIEW_VISION_MODEL"),
            "judge": os.getenv("CONTRACT_REVIEW_JUDGE_MODEL"),
            "embedding": os.getenv("CONTRACT_REVIEW_EMBEDDING_MODEL"),
        }
        for role, model in overrides.items():
            if model:
                result[role]["model"] = model
        return result
