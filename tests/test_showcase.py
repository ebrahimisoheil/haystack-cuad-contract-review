from __future__ import annotations

from contract_review_agent.app.showcase import _verification


def test_showcase_verification_requires_all_provider_and_usage_evidence() -> None:
    run = {
        "execution_id": "demo",
        "status": "completed",
        "providers": "deepseek, mistral, openai",
        "model_calls": 8,
        "total_tokens": 1200,
        "known_cost": 0.04,
    }

    result = _verification(run, ["deepseek", "mistral", "openai"])

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_showcase_verification_rejects_missing_mistral() -> None:
    run = {
        "execution_id": "demo",
        "status": "completed",
        "providers": "deepseek, openai",
        "model_calls": 7,
        "total_tokens": 900,
        "known_cost": 0.03,
    }

    result = _verification(run, ["deepseek", "mistral", "openai"])

    assert result["passed"] is False
    assert result["checks"]["providers_observed"] is False
