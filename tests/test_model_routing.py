from __future__ import annotations

from contract_review_agent.app.config import Settings


def test_declarative_litellm_roles() -> None:
    models = Settings().model_registry()
    assert models["text"]["model"] == "deepseek/deepseek-chat"
    assert models["vision"]["model"] == "mistral/mistral-ocr-latest"
    assert models["judge"]["model"] == "openai/gpt-5.4"
    assert {model["provider"] for model in models.values()} == {"litellm"}
