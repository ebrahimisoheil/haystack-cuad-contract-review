from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

from haystack import Pipeline
from haystack.components.joiners import BranchJoiner
from haystack.components.routers import ConditionalRouter
from witdem_sdk.integrations.haystack import instrument
from witdem_sdk.integrations.litellm import install_litellm

from .components.clauses import ClauseExtractor, StructuredTermNormalizer
from .components.common import Context
from .components.document_input import DocumentQualityClassifier, InputLoader
from .components.extraction import FocusedReExtractor, MistralDocumentExtractor, NativePDFExtractor
from .components.judges import ExtractionJudge, FallbackJudge, FinalReviewJudge, RiskJudge
from .components.normalization import AgreementClassifier, MetadataExtractor, TextNormalizer
from .components.obligations import ObligationExtractor, SkipObligations
from .components.playbook import PlaybookEvaluator
from .components.result import ResultAssembler
from .components.routing import FallbackGenerator, NoEscalationRoute, ReviewRouter
from .config import Settings
from .model_registry import ModelRegistry
from .schemas import BusinessOutcome, ContractReviewResult, RunMetrics, StageMetric


def _boolean_router(variable: str, true_name: str, false_name: str) -> ConditionalRouter:
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
        "extraction_retry_router": _boolean_router("needs_retry", "retry_context", "accepted_context"),
        "focused_re_extractor": FocusedReExtractor(settings, models),
        "playbook_evaluator": PlaybookEvaluator(settings, models),
        "deviation_router": _boolean_router("has_deviations", "deviating_context", "compliant_context"),
        "straight_through": NoEscalationRoute(settings, models),
        "risk_judge": RiskJudge(settings, models),
        "review_router": ReviewRouter(settings, models),
        "fallback_input_joiner": BranchJoiner(Context),
        "fallback_generator": FallbackGenerator(settings, models),
        "fallback_judge": FallbackJudge(settings, models),
        "fallback_retry_router": _boolean_router("needs_retry", "retry_context", "accepted_context"),
        "final_input_joiner": BranchJoiner(Context),
        "final_judge": FinalReviewJudge(settings, models),
        "decision_router": _boolean_router("approved", "approved_context", "manual_context"),
        "obligation_extractor": ObligationExtractor(settings, models),
        "skip_obligations": SkipObligations(settings, models),
        "result_input_joiner": BranchJoiner(Context),
        "result_assembler": ResultAssembler(settings, models),
    }
    for name, instance in components.items():
        pipeline.add_component(name, instance)

    pipeline.connect("input_loader.context", "quality_classifier.context")
    pipeline.connect("quality_classifier.context", "extraction_router.context")
    pipeline.connect("quality_classifier.extraction_mode", "extraction_router.extraction_mode")
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
    pipeline.connect("extraction_judge.needs_retry", "extraction_retry_router.needs_retry")
    pipeline.connect("extraction_retry_router.retry_context", "focused_re_extractor.context")
    pipeline.connect("extraction_retry_router.accepted_context", "playbook_evaluator.context")
    pipeline.connect("playbook_evaluator.context", "deviation_router.context")
    pipeline.connect("playbook_evaluator.has_deviations", "deviation_router.has_deviations")
    pipeline.connect("deviation_router.compliant_context", "straight_through.context")
    pipeline.connect("straight_through.context", "final_input_joiner.value")
    pipeline.connect("deviation_router.deviating_context", "risk_judge.context")
    pipeline.connect("risk_judge.context", "review_router.context")
    pipeline.connect("review_router.context", "fallback_input_joiner.value")
    pipeline.connect("fallback_retry_router.retry_context", "fallback_input_joiner.value")
    pipeline.connect("fallback_input_joiner.value", "fallback_generator.context")
    pipeline.connect("fallback_generator.context", "fallback_judge.context")
    pipeline.connect("fallback_judge.context", "fallback_retry_router.context")
    pipeline.connect("fallback_judge.needs_retry", "fallback_retry_router.needs_retry")
    pipeline.connect("fallback_retry_router.accepted_context", "final_input_joiner.value")
    pipeline.connect("final_input_joiner.value", "final_judge.context")
    pipeline.connect("final_judge.context", "decision_router.context")
    pipeline.connect("final_judge.approved", "decision_router.approved")
    pipeline.connect("decision_router.approved_context", "obligation_extractor.context")
    pipeline.connect("decision_router.manual_context", "skip_obligations.context")
    pipeline.connect("obligation_extractor.context", "result_input_joiner.value")
    pipeline.connect("skip_obligations.context", "result_input_joiner.value")
    pipeline.connect("result_input_joiner.value", "result_assembler.context")
    return pipeline


def run_review(source: str, settings: Settings | None = None) -> dict[str, Any]:
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
            errors=[{"stage": "pipeline", "type": type(exc).__name__, "message": str(exc)}],
        )
        return failed.model_dump(mode="json")
