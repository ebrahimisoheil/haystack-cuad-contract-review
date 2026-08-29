from __future__ import annotations

from pathlib import Path

from contract_review_agent.app.components.clauses import ClauseExtractor
from contract_review_agent.app.components.routing import (
    DomainReviewDispatcher,
    FallbackGenerator,
)
from contract_review_agent.app.config import Settings
from contract_review_agent.app.model_registry import ModelRegistry
from contract_review_agent.app.pipeline import run_review

from .fixtures import DEVIATING_CONTRACT, make_native_pdf


def review(path: Path) -> dict:
    return run_review(str(path), Settings(mode="deterministic"))


def test_clean_native_pdf_path(native_contract: Path) -> None:
    result = review(native_contract)
    assert result["final_decision"] == "approved"
    assert "extraction_route:native" in result["metrics"]["branch_path"]
    assert result["outcome"]["objective_met"] is True
    assert result["metrics"]["deviation_count"] == 0


def test_scanned_pdf_uses_mistral_path(scanned_contract: Path) -> None:
    result = review(scanned_contract)
    assert result["final_decision"] == "approved"
    assert "extraction_route:vision" in result["metrics"]["branch_path"]
    assert any(
        stage["model"] == "mistral/mistral-ocr-latest"
        for stage in result["metrics"]["stages"]
    )
    assert all(clause["evidence"]["page"] for clause in result["clauses"])


def test_deviations_trigger_legal_and_finance_routes(deviating_contract: Path) -> None:
    result = review(deviating_contract)
    assert result["final_decision"] == "approved_with_exceptions"
    assert set(result["review_areas"]) == {"legal", "finance"}
    assert {item["area"] for item in result["domain_reviews"]} == {
        "legal",
        "finance",
    }
    assert all(item["decision"] == "negotiate" for item in result["domain_reviews"])
    assert "domain_fanout:legal" in result["metrics"]["branch_path"]
    assert "legal_domain_review" in result["metrics"]["branch_path"]
    assert "finance_domain_review" in result["metrics"]["branch_path"]
    assert "domain_review_aggregator" in result["metrics"]["branch_path"]
    assert {item["clause"] for item in result["deviations"]} >= {
        "termination",
        "governing_law",
        "payment",
    }
    assert all(item["recommended_fallback"] for item in result["deviations"])


def test_high_risk_policy_gate_requires_human_review(tmp_path: Path) -> None:
    source = DEVIATING_CONTRACT.replace(
        "Aggregate liability is capped at fees paid in the prior 12 months.",
        "Vendor liability is unlimited.",
    )
    path = make_native_pdf(tmp_path / "high-risk-native.pdf", source)

    result = review(path)

    assert result["final_decision"] == "manual_review_required"
    assert "high_risk_policy_gate" in result["metrics"]["branch_path"]
    assert "risk_policy:mandatory_human_review" in result["metrics"]["branch_path"]
    assert any(
        item["area"] == "legal" and item["decision"] == "escalate"
        for item in result["domain_reviews"]
    )


def test_domain_dispatcher_emits_only_relevant_branches() -> None:
    settings = Settings(mode="deterministic")
    dispatcher = DomainReviewDispatcher(settings, ModelRegistry(settings))
    context = {
        "deviations": [
            {"review_area": "legal"},
            {"review_area": "security/privacy"},
        ],
        "stage_metrics": [],
        "branch_path": [],
    }

    outputs = dispatcher.run(context)

    assert set(outputs) == {"context", "legal_contexts", "security_contexts"}
    assert outputs["context"]["review_areas"] == ["legal", "security/privacy"]


def test_missing_clause_retries_with_focused_text(retry_contract: Path) -> None:
    result = review(retry_contract)
    assert result["final_decision"] == "approved"
    assert result["metrics"]["retries"] == 1
    assert "focused_re_extractor" in result["metrics"]["branch_path"]
    assert result["metrics"]["branch_path"].count("clause_extractor") == 2


def test_invalid_input_returns_typed_processing_failure(tmp_path: Path) -> None:
    result = review(tmp_path / "missing.pdf")
    assert result["final_decision"] == "processing_failed"
    assert result["outcome"]["review_completed"] is False
    assert "does not exist" in result["errors"][0]["message"]


def test_live_clause_shape_accepts_direct_role_mapping(monkeypatch) -> None:
    settings = Settings(mode="deterministic")
    models = ModelRegistry(settings)
    monkeypatch.setattr(
        models,
        "call_json",
        lambda *args, **kwargs: {
            "term": "The initial term is 12 months.",
            "governing_law": {"source_text": "Delaware"},
        },
    )
    context = {
        "normalized_text": "unstructured contract",
        "pages": [{"page": 1, "text": "The initial term is 12 months. Delaware"}],
        "extraction_method": "live-test",
        "stage_metrics": [],
        "branch_path": [],
    }
    result = ClauseExtractor(settings, models).run(context)["context"]
    assert set(result["clause_map"]) == {"term", "governing_law"}


def test_live_fallback_shape_accepts_list(monkeypatch) -> None:
    settings = Settings(mode="deterministic")
    models = ModelRegistry(settings)
    monkeypatch.setattr(
        models,
        "call_json",
        lambda *args, **kwargs: {
            "clause_to_fallback": [
                {"clause": "payment", "fallback": "Use Net 30 payment terms."}
            ]
        },
    )
    context = {
        "deviations": [{"clause": "payment", "recommended_fallback": None}],
        "stage_metrics": [],
        "branch_path": [],
    }
    result = FallbackGenerator(settings, models).run(context)["context"]
    assert (
        result["deviations"][0]["recommended_fallback"] == "Use Net 30 payment terms."
    )
