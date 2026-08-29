from __future__ import annotations

from haystack.components.routers import ConditionalRouter

from contract_review_agent.app.config import Settings
from contract_review_agent.app.pipeline import build_pipeline


def test_pipeline_exposes_required_routing_nodes() -> None:
    pipeline = build_pipeline()
    names = set(pipeline.graph.nodes)
    assert {
        "extraction_router",
        "extraction_retry_router",
        "deviation_router",
        "risk_tier_router",
        "fallback_retry_router",
        "decision_router",
    } <= names
    routers = [
        pipeline.get_component(name)
        for name in names
        if isinstance(pipeline.get_component(name), ConditionalRouter)
    ]
    assert len(routers) >= 6


def test_pipeline_has_visible_business_stages() -> None:
    names = set(build_pipeline().graph.nodes)
    assert {
        "quality_classifier",
        "mistral_extractor",
        "clause_extractor",
        "playbook_evaluator",
        "risk_judge",
        "domain_review_dispatcher",
        "legal_domain_review",
        "finance_domain_review",
        "security_domain_review",
        "business_domain_review",
        "domain_review_aggregator",
        "fallback_generator",
        "final_judge",
        "obligation_extractor",
        "result_assembler",
    } <= names


def test_domain_review_is_a_real_fan_out_and_convergence_graph() -> None:
    pipeline = build_pipeline()
    edges = {(edge[0], edge[1]) for edge in pipeline.graph.edges}
    reviewers = {
        "legal_domain_review",
        "finance_domain_review",
        "security_domain_review",
        "business_domain_review",
    }

    assert {("domain_review_dispatcher", reviewer) for reviewer in reviewers} <= edges
    assert {(reviewer, "domain_review_joiner") for reviewer in reviewers} <= edges
    assert ("domain_review_joiner", "domain_review_aggregator") in edges
    assert ("domain_review_aggregator", "fallback_input_joiner") in edges


def test_cuad_ground_truth_is_not_a_haystack_pipeline_input() -> None:
    pipeline = build_pipeline(Settings(mode="deterministic"))
    inputs = pipeline.inputs(include_components_with_connected_inputs=True)

    assert all(
        "ground_truth" not in component_inputs for component_inputs in inputs.values()
    )
