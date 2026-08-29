from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Severity = Literal["low", "medium", "high"]
Decision = Literal[
    "approved",
    "approved_with_exceptions",
    "manual_review_required",
    "rejected_by_playbook",
    "processing_failed",
]
ReviewArea = Literal[
    "legal", "finance", "security/privacy", "procurement/business_owner"
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    page: int | None = None
    text: str | None = None
    clause_label: str
    extraction_method: str
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def evidence_is_not_empty(self) -> Evidence:
        if self.text is None and self.confidence > 0:
            raise ValueError("evidence without source text must have zero confidence")
        return self


class Clause(StrictModel):
    label: str
    summary: str
    evidence: Evidence


class Parties(StrictModel):
    customer: str | None = None
    vendor: str | None = None


class TermTerms(StrictModel):
    initial_months: int | None = None
    auto_renewal: bool | None = None
    renewal_months: int | None = None
    notice_days: int | None = None


class TerminationTerms(StrictModel):
    for_cause: bool | None = None
    for_convenience: bool | None = None
    notice_days: int | None = None


class LiabilityTerms(StrictModel):
    cap_present: bool | None = None
    cap_description: str | None = None
    risk: Severity | None = None


class IndemnityTerms(StrictModel):
    summary: str | None = None
    risk: Severity | None = None


class SecurityTerms(StrictModel):
    security_clause_present: bool | None = None
    dpa_language_present: bool | None = None


class SLATerms(StrictModel):
    availability: str | None = None
    remedy_present: bool | None = None


class Deviation(StrictModel):
    clause: str
    severity: Severity
    reason: str
    evidence: Evidence
    recommended_fallback: str | None = None
    fallback_accepted: bool | None = None
    review_area: ReviewArea


class DomainReview(StrictModel):
    area: ReviewArea
    decision: Literal["accept", "negotiate", "escalate"]
    highest_risk: Severity
    deviation_count: int = Field(ge=1)
    rationale: str


class Obligation(StrictModel):
    type: str
    due_date_or_rule: str
    owner: str
    evidence: Evidence


class StageMetric(StrictModel):
    stage: str
    model: str
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)
    attempt: int = Field(default=1, ge=1)
    status: Literal["ok", "retry", "failed"] = "ok"


class RunMetrics(StrictModel):
    total_runtime_ms: float = Field(ge=0)
    stages: list[StageMetric]
    retries: int = Field(ge=0)
    branch_path: list[str]
    extraction_confidence: float = Field(ge=0, le=1)
    deviation_count: int = Field(ge=0)
    escalation_count: int = Field(ge=0)
    unresolved_field_count: int = Field(ge=0)
    evidence_completeness: float = Field(ge=0, le=1)
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)


class BusinessOutcome(StrictModel):
    review_completed: bool
    evidence_complete: bool
    playbook_evaluated: bool
    routing_complete: bool
    objective_met: bool


class CuadCategoryEvaluation(StrictModel):
    category: str
    ground_truth_positive: bool
    predicted_positive: bool
    outcome: Literal[
        "true_positive", "false_positive", "false_negative", "true_negative"
    ]
    span_exact_match: float | None = Field(default=None, ge=0, le=1)
    span_token_f1: float | None = Field(default=None, ge=0, le=1)


class CuadGroundTruthEvaluation(StrictModel):
    evaluation_version: str = "cuad-ground-truth-v1"
    evaluated_categories: int = Field(ge=0)
    supported_categories: list[str]
    ground_truth_positives: int = Field(ge=0)
    predicted_positives: int = Field(ge=0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)
    category_precision: float | None = Field(default=None, ge=0, le=1)
    category_recall: float | None = Field(default=None, ge=0, le=1)
    category_f1: float | None = Field(default=None, ge=0, le=1)
    category_accuracy: float = Field(ge=0, le=1)
    negative_label_accuracy: float | None = Field(default=None, ge=0, le=1)
    evaluated_spans: int = Field(ge=0)
    span_exact_match: float | None = Field(default=None, ge=0, le=1)
    span_token_f1: float | None = Field(default=None, ge=0, le=1)
    categories: list[CuadCategoryEvaluation]


class ContractReviewResult(StrictModel):
    contract_id: str
    agreement_type: str | None
    parties: Parties
    effective_date: str | None
    term: TermTerms
    termination: TerminationTerms
    governing_law: str | None
    assignment_restricted: bool | None
    liability: LiabilityTerms
    indemnity: IndemnityTerms
    security: SecurityTerms
    sla: SLATerms
    payment_terms: str | None
    clauses: list[Clause]
    deviations: list[Deviation]
    final_decision: Decision
    decision_explanation: str
    review_areas: list[ReviewArea]
    domain_reviews: list[DomainReview]
    obligations: list[Obligation]
    outcome: BusinessOutcome
    metrics: RunMetrics
    cuad_evaluation: CuadGroundTruthEvaluation | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
