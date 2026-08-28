from __future__ import annotations

from haystack.components.routers import ConditionalRouter

from contract_review_agent.app.pipeline import build_pipeline


def test_pipeline_exposes_required_routing_nodes() -> None:
    pipeline = build_pipeline()
    names = set(pipeline.graph.nodes)
    assert {
        "extraction_router",
        "extraction_retry_router",
        "deviation_router",
        "fallback_retry_router",
        "decision_router",
    } <= names
    routers = [
        pipeline.get_component(name)
        for name in names
        if isinstance(pipeline.get_component(name), ConditionalRouter)
    ]
    assert len(routers) >= 5


def test_pipeline_has_visible_business_stages() -> None:
    names = set(build_pipeline().graph.nodes)
    assert {
        "quality_classifier",
        "mistral_extractor",
        "clause_extractor",
        "playbook_evaluator",
        "risk_judge",
        "review_router",
        "fallback_generator",
        "final_judge",
        "obligation_extractor",
        "result_assembler",
    } <= names
