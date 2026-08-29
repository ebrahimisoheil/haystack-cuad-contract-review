from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from haystack import Pipeline
from haystack.components.joiners import BranchJoiner, ListJoiner
from haystack.components.routers import ConditionalRouter
from witdem_sdk.integrations.haystack import instrument
from witdem_sdk.integrations.litellm import install_litellm

from .components.clauses import ClauseExtractor, StructuredTermNormalizer
from .components.common import Context
from .components.document_input import DocumentQualityClassifier, InputLoader
from .components.extraction import (
    FocusedReExtractor,
    MistralDocumentExtractor,
    NativePDFExtractor,
)
from .components.judges import (
    ExtractionJudge,
    FallbackJudge,
    FinalReviewJudge,
    RiskJudge,
)
from .components.normalization import (
    AgreementClassifier,
    MetadataExtractor,
    TextNormalizer,
)
from .components.obligations import ObligationExtractor, SkipObligations
from .components.playbook import PlaybookEvaluator
from .components.result import ResultAssembler
from .components.routing import (
    DomainReviewAggregator,
    DomainReviewDispatcher,
    DomainReviewer,
    FallbackGenerator,
    HighRiskPolicyGate,
    LowRiskAcceptanceRoute,
    NoEscalationRoute,
)
from .config import Settings
from .ingestion.evaluation import (
    CUAD_CATEGORY_F1_TARGET,
    CUAD_NEGATIVE_LABEL_ACCURACY_TARGET,
    CUAD_SPAN_TOKEN_F1_TARGET,
    evaluate_cuad_ground_truth,
)
from .model_registry import ModelRegistry
from .schemas import BusinessOutcome, ContractReviewResult, RunMetrics, StageMetric


def _cuad_report_result(result: dict[str, Any]) -> dict[str, Any]:
    review = result["result_assembler"]["result"]
    evaluation = review["cuad_evaluation"]
    optional_evaluations = {
        "category_f1": evaluation.get("category_f1"),
        "span_token_f1": evaluation.get("span_token_f1"),
        "negative_label_accuracy": evaluation.get("negative_label_accuracy"),
    }
    assurance_checks = [
        *(
            [evaluation["category_f1"] >= CUAD_CATEGORY_F1_TARGET]
            if evaluation.get("category_f1") is not None
            else []
        ),
        *(
            [evaluation["span_token_f1"] >= CUAD_SPAN_TOKEN_F1_TARGET]
            if evaluation.get("span_token_f1") is not None
            else []
        ),
        *(
            [
                evaluation["negative_label_accuracy"]
                >= CUAD_NEGATIVE_LABEL_ACCURACY_TARGET
            ]
            if evaluation.get("negative_label_accuracy") is not None
            else []
        ),
    ]
    evaluation_assured = bool(assurance_checks) and all(assurance_checks)
    application_evidence_sufficient = (
        float(review["metrics"]["evidence_completeness"]) >= 0.8
    )
    return {
        "contract": "cuad_contract_review",
        "result": review["final_decision"],
        "result_valid": bool(review.get("contract_id"))
        and bool(review["outcome"]["review_completed"]),
        "decision": review["final_decision"],
        "product_goal_achieved": bool(review["outcome"]["objective_met"]),
        "evidence_sufficient": application_evidence_sufficient and evaluation_assured,
        "required_path_observed": bool(review["outcome"]["routing_complete"]),
        "closest_blocker": (
            review["final_decision"]
            if not review["outcome"]["objective_met"]
            else ("none" if evaluation_assured else "cuad_evaluation_below_target")
        ),
        "evaluations": {
            key: value
            for key, value in optional_evaluations.items()
            if value is not None
        },
        "metrics": {
            "evaluated_categories": evaluation["evaluated_categories"],
            "true_positives": evaluation["true_positives"],
            "false_positives": evaluation["false_positives"],
            "false_negatives": evaluation["false_negatives"],
            "true_negatives": evaluation["true_negatives"],
            "evaluated_spans": evaluation["evaluated_spans"],
            "workflow_retries": review["metrics"]["retries"],
        },
        "dimensions": {
            "contract_id": review["contract_id"],
            "agreement_type": review.get("agreement_type") or "unresolved",
            "final_decision": review["final_decision"],
            "evaluation_version": evaluation["evaluation_version"],
        },
    }


def _cuad_result_reporter(
    ground_truth: list[dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def report(result: dict[str, Any]) -> dict[str, Any]:
        review = result["result_assembler"]["result"]
        review["cuad_evaluation"] = evaluate_cuad_ground_truth(
            review,
            ground_truth,
        ).model_dump(mode="json")
        return _cuad_report_result(result)

    return report


def _boolean_router(
    variable: str, true_name: str, false_name: str
) -> ConditionalRouter:
    return ConditionalRouter(
        [
            {
                "condition": f"{{{{ {variable} }}}}",
                "output": "context",
                "output_passthrough": True,
                "output_name": true_name,
                "output_type": Context,
            },
            {
                "condition": "{{ True }}",
                "output": "context",
                "output_passthrough": True,
                "output_name": false_name,
                "output_type": Context,
            },
        ],
        validate_output_type=True,
    )


def build_pipeline(settings: Settings | None = None) -> Pipeline:
    settings = settings or Settings.from_env()
    models = ModelRegistry(settings)
    pipeline = Pipeline(max_runs_per_component=settings.max_retries + 3)

    components: dict[str, Any] = {
        "input_loader": InputLoader(settings, models),
        "quality_classifier": DocumentQualityClassifier(settings, models),
        "extraction_router": ConditionalRouter(
            [
                {
                    "condition": "{{ extraction_mode == 'native' }}",
                    "output": "context",
                    "output_passthrough": True,
                    "output_name": "native_context",
                    "output_type": Context,
                },
                {
                    "condition": "{{ True }}",
                    "output": "context",
                    "output_passthrough": True,
                    "output_name": "vision_context",
                    "output_type": Context,
                },
            ],
            validate_output_type=True,
        ),
        "native_extractor": NativePDFExtractor(settings, models),
        "mistral_extractor": MistralDocumentExtractor(settings, models),
        "document_joiner": BranchJoiner(Context),
        "text_normalizer": TextNormalizer(settings, models),
        "agreement_classifier": AgreementClassifier(settings, models),
        "metadata_extractor": MetadataExtractor(settings, models),
        "clause_input_joiner": BranchJoiner(Context),
        "clause_extractor": ClauseExtractor(settings, models),
        "term_normalizer": StructuredTermNormalizer(settings, models),
        "extraction_judge": ExtractionJudge(settings, models),
        "extraction_retry_router": _boolean_router(
            "needs_retry", "retry_context", "accepted_context"
        ),
        "focused_re_extractor": FocusedReExtractor(settings, models),
        "playbook_evaluator": PlaybookEvaluator(settings, models),
        "deviation_router": _boolean_router(
            "has_deviations", "deviating_context", "compliant_context"
        ),
        "straight_through": NoEscalationRoute(settings, models),
        "risk_judge": RiskJudge(settings, models),
        "risk_tier_router": ConditionalRouter(
            [
                {
                    "condition": "{{ context.highest_risk == 'low' }}",
                    "output": "{{ context }}",
                    "output_name": "low_context",
                    "output_type": Context,
                },
                {
                    "condition": "{{ context.highest_risk == 'medium' }}",
                    "output": "{{ context }}",
                    "output_name": "medium_context",
                    "output_type": Context,
                },
                {
                    "condition": "{{ True }}",
                    "output": "{{ context }}",
                    "output_name": "high_context",
                    "output_type": Context,
                },
            ],
            validate_output_type=True,
        ),
        "low_risk_acceptance": LowRiskAcceptanceRoute(settings, models),
        "high_risk_policy_gate": HighRiskPolicyGate(settings, models),
        "elevated_risk_joiner": BranchJoiner(Context),
        "domain_review_dispatcher": DomainReviewDispatcher(settings, models),
        "legal_domain_review": DomainReviewer(settings, models, "legal"),
        "finance_domain_review": DomainReviewer(settings, models, "finance"),
        "security_domain_review": DomainReviewer(settings, models, "security/privacy"),
        "business_domain_review": DomainReviewer(
            settings, models, "procurement/business_owner"
        ),
        "domain_review_joiner": ListJoiner(list[Context]),
        "domain_review_aggregator": DomainReviewAggregator(settings, models),
        "fallback_input_joiner": BranchJoiner(Context),
        "fallback_generator": FallbackGenerator(settings, models),
        "fallback_judge": FallbackJudge(settings, models),
        "fallback_retry_router": _boolean_router(
            "needs_retry", "retry_context", "accepted_context"
        ),
        "final_input_joiner": BranchJoiner(Context),
        "final_judge": FinalReviewJudge(settings, models),
        "decision_router": _boolean_router(
            "approved", "approved_context", "manual_context"
        ),
        "obligation_extractor": ObligationExtractor(settings, models),
        "skip_obligations": SkipObligations(settings, models),
        "result_input_joiner": BranchJoiner(Context),
        "result_assembler": ResultAssembler(settings, models),
    }
    for name, instance in components.items():
        pipeline.add_component(name, instance)

    pipeline.connect("input_loader.context", "quality_classifier.context")
    pipeline.connect("quality_classifier.context", "extraction_router.context")
    pipeline.connect(
        "quality_classifier.extraction_mode", "extraction_router.extraction_mode"
    )
    pipeline.connect("extraction_router.native_context", "native_extractor.context")
    pipeline.connect("extraction_router.vision_context", "mistral_extractor.context")
    pipeline.connect("native_extractor.context", "document_joiner.value")
    pipeline.connect("mistral_extractor.context", "document_joiner.value")
    pipeline.connect("document_joiner.value", "text_normalizer.context")
    pipeline.connect("text_normalizer.context", "agreement_classifier.context")
    pipeline.connect("agreement_classifier.context", "metadata_extractor.context")
    pipeline.connect("metadata_extractor.context", "clause_input_joiner.value")
    pipeline.connect("focused_re_extractor.context", "clause_input_joiner.value")
    pipeline.connect("clause_input_joiner.value", "clause_extractor.context")
    pipeline.connect("clause_extractor.context", "term_normalizer.context")
    pipeline.connect("term_normalizer.context", "extraction_judge.context")
    pipeline.connect("extraction_judge.context", "extraction_retry_router.context")
    pipeline.connect(
        "extraction_judge.needs_retry", "extraction_retry_router.needs_retry"
    )
    pipeline.connect(
        "extraction_retry_router.retry_context", "focused_re_extractor.context"
    )
    pipeline.connect(
        "extraction_retry_router.accepted_context", "playbook_evaluator.context"
    )
    pipeline.connect("playbook_evaluator.context", "deviation_router.context")
    pipeline.connect(
        "playbook_evaluator.has_deviations", "deviation_router.has_deviations"
    )
    pipeline.connect("deviation_router.compliant_context", "straight_through.context")
    pipeline.connect("straight_through.context", "final_input_joiner.value")
    pipeline.connect("deviation_router.deviating_context", "risk_judge.context")
    pipeline.connect("risk_judge.context", "risk_tier_router.context")
    pipeline.connect("risk_tier_router.low_context", "low_risk_acceptance.context")
    pipeline.connect("low_risk_acceptance.context", "final_input_joiner.value")
    pipeline.connect("risk_tier_router.medium_context", "elevated_risk_joiner.value")
    pipeline.connect("risk_tier_router.high_context", "high_risk_policy_gate.context")
    pipeline.connect("high_risk_policy_gate.context", "elevated_risk_joiner.value")
    pipeline.connect("elevated_risk_joiner.value", "domain_review_dispatcher.context")
    pipeline.connect(
        "domain_review_dispatcher.legal_contexts", "legal_domain_review.contexts"
    )
    pipeline.connect(
        "domain_review_dispatcher.finance_contexts",
        "finance_domain_review.contexts",
    )
    pipeline.connect(
        "domain_review_dispatcher.security_contexts",
        "security_domain_review.contexts",
    )
    pipeline.connect(
        "domain_review_dispatcher.business_contexts",
        "business_domain_review.contexts",
    )
    pipeline.connect("legal_domain_review.contexts", "domain_review_joiner.values")
    pipeline.connect("finance_domain_review.contexts", "domain_review_joiner.values")
    pipeline.connect("security_domain_review.contexts", "domain_review_joiner.values")
    pipeline.connect("business_domain_review.contexts", "domain_review_joiner.values")
    pipeline.connect(
        "domain_review_dispatcher.context", "domain_review_aggregator.context"
    )
    pipeline.connect(
        "domain_review_joiner.values", "domain_review_aggregator.review_contexts"
    )
    pipeline.connect("domain_review_aggregator.context", "fallback_input_joiner.value")
    pipeline.connect(
        "fallback_retry_router.retry_context", "fallback_input_joiner.value"
    )
    pipeline.connect("fallback_input_joiner.value", "fallback_generator.context")
    pipeline.connect("fallback_generator.context", "fallback_judge.context")
    pipeline.connect("fallback_judge.context", "fallback_retry_router.context")
    pipeline.connect("fallback_judge.needs_retry", "fallback_retry_router.needs_retry")
    pipeline.connect(
        "fallback_retry_router.accepted_context", "final_input_joiner.value"
    )
    pipeline.connect("final_input_joiner.value", "final_judge.context")
    pipeline.connect("final_judge.context", "decision_router.context")
    pipeline.connect("final_judge.approved", "decision_router.approved")
    pipeline.connect("decision_router.approved_context", "obligation_extractor.context")
    pipeline.connect("decision_router.manual_context", "skip_obligations.context")
    pipeline.connect("obligation_extractor.context", "result_input_joiner.value")
    pipeline.connect("skip_obligations.context", "result_input_joiner.value")
    pipeline.connect("result_input_joiner.value", "result_assembler.context")
    return pipeline


def run_review(
    source: str,
    settings: Settings | None = None,
    *,
    ground_truth: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        config_path = Path(
            os.getenv(
                "CONTRACT_REVIEW_WITDEM_CONFIG",
                str(Path(__file__).resolve().parents[2] / ".witdem" / "witdem.yaml"),
            )
        )
        pipeline = instrument(
            build_pipeline(settings),
            execution_name="contract-review",
            config_path=str(config_path),
            report_result=(
                _cuad_result_reporter(ground_truth)
                if ground_truth is not None
                else None
            ),
        )
        litellm_registration = install_litellm()
        try:
            result = pipeline.run({"input_loader": {"source": source}})
        finally:
            litellm_registration.flush()
            litellm_registration.uninstall()
        return result["result_assembler"]["result"]
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        failed = ContractReviewResult(
            contract_id=hashlib.sha256(source.encode("utf-8")).hexdigest()[:16],
            agreement_type=None,
            parties={},
            effective_date=None,
            term={},
            termination={},
            governing_law=None,
            assignment_restricted=None,
            liability={},
            indemnity={},
            security={},
            sla={},
            payment_terms=None,
            clauses=[],
            deviations=[],
            final_decision="processing_failed",
            decision_explanation="The workflow could not process the supplied input.",
            review_areas=[],
            domain_reviews=[],
            obligations=[],
            outcome=BusinessOutcome(
                review_completed=False,
                evidence_complete=False,
                playbook_evaluated=False,
                routing_complete=False,
                objective_met=False,
            ),
            metrics=RunMetrics(
                total_runtime_ms=round(elapsed, 3),
                stages=[
                    StageMetric(
                        stage="pipeline_error",
                        model="local-error-boundary",
                        latency_ms=round(elapsed, 3),
                        status="failed",
                    )
                ],
                retries=0,
                branch_path=["pipeline_error"],
                extraction_confidence=0,
                deviation_count=0,
                escalation_count=0,
                unresolved_field_count=17,
                evidence_completeness=0,
            ),
            errors=[
                {"stage": "pipeline", "type": type(exc).__name__, "message": str(exc)}
            ],
        )
        return failed.model_dump(mode="json")
