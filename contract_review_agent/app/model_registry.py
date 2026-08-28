from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from .config import Settings


ModelRole = Literal["text", "vision", "judge"]


class ModelAccessError(RuntimeError):
    """LiteLLM failed or returned malformed structured output."""


class ModelRegistry:
    """Declarative role registry backed by Haystack's official LiteLLM generator.

    The only direct LiteLLM call is ``ocr`` because the Generator integration is
    chat-oriented and does not expose LiteLLM's document OCR endpoint.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.models = settings.model_registry()
        self._generators: dict[ModelRole, Any] = {}
        self._usage: dict[str, dict[str, float | int]] = {}

    def model_name(self, role: str) -> str:
        return str(self.models[role]["model"]) if role in self.models else role

    def consume_usage(self, role: str) -> dict[str, float | int]:
        return self._usage.pop(
            role,
            {"input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0},
        )

    def _require_key(self, role: ModelRole) -> str:
        name = str(self.models[role]["api_key_env"])
        value = os.getenv(name)
        if not value:
            raise ModelAccessError(f"{name} is required for the {role} role in live mode")
        return value

    @staticmethod
    def _decode_json(text: str, role: ModelRole) -> dict[str, Any]:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelAccessError(f"LiteLLM {role} role returned malformed JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ModelAccessError(f"LiteLLM {role} role must return a JSON object")
        return value

    def _generator(self, role: Literal["text", "judge"]) -> Any:
        if role not in self._generators:
            from haystack_integrations.components.generators.litellm import LiteLLMChatGenerator

            key_name = str(self.models[role]["api_key_env"])
            self._require_key(role)
            generation_kwargs: dict[str, Any] = {
                "response_format": {"type": "json_object"},
                "timeout": self.settings.timeout_seconds,
                "num_retries": int(self.models[role].get("provider_retries", 0)),
            }
            fallbacks = self.models[role].get("fallback_models", [])
            if fallbacks:
                generation_kwargs["fallbacks"] = fallbacks
            self._generators[role] = LiteLLMChatGenerator(
                model=self.model_name(role),
                api_key=Secret.from_env_var(key_name),
                generation_kwargs=generation_kwargs,
            )
        return self._generators[role]

    def call_json(
        self,
        role: Literal["text", "judge"],
        *,
        system: str,
        prompt: str,
    ) -> dict[str, Any] | None:
        if self.settings.mode == "deterministic":
            return None
        try:
            reply = self._generator(role).run(
                messages=[ChatMessage.from_system(system), ChatMessage.from_user(prompt)]
            )["replies"][0]
            usage = reply.meta.get("usage", {}) if reply.meta else {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            estimated_cost = 0.0
            try:
                import litellm

                prompt_cost, completion_cost = litellm.cost_per_token(
                    model=self.model_name(role),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                estimated_cost = float(prompt_cost + completion_cost)
            except Exception:
                # Some private/proxy models do not publish pricing metadata.
                estimated_cost = 0.0
            self._usage[role] = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "estimated_cost_usd": estimated_cost,
            }
            return self._decode_json(reply.text, role)
        except Exception as exc:
            if isinstance(exc, ModelAccessError):
                raise
            raise ModelAccessError(f"LiteLLM {role} request failed: {exc}") from exc

    def extract_document(self, path: Path) -> list[dict[str, Any]] | None:
        if self.settings.mode == "deterministic":
            return None
        self._require_key("vision")
        try:
            import litellm

            response = litellm.ocr(
                model=self.model_name("vision"),
                document={"type": "file", "file": path},
                timeout=self.settings.timeout_seconds,
                include_image_base64=False,
                num_retries=int(self.models["vision"].get("provider_retries", 0)),
            )
            pages = [
                {"page": int(page.index) + 1, "text": page.markdown}
                for page in response.pages
            ]
            usage = getattr(response, "usage_info", None)
            estimated_cost = 0.0
            try:
                estimated_cost = float(
                    litellm.completion_cost(
                        completion_response=response,
                        model=self.model_name("vision"),
                        call_type="ocr",
                    )
                )
            except Exception:
                estimated_cost = 0.0
            self._usage["vision"] = {
                "input_tokens": int(getattr(usage, "pages_processed", 0) or 0),
                "output_tokens": 0,
                "estimated_cost_usd": estimated_cost,
            }
            return pages
        except Exception as exc:
            raise ModelAccessError(f"LiteLLM vision/OCR request failed: {exc}") from exc
