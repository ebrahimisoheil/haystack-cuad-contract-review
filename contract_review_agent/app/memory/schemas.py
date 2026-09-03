from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from ..schemas import Decision, StrictModel


class ApprovedPrecedent(StrictModel):
    """A human-approved clause and decision that may safely influence later work."""

    precedent_id: str
    tenant_id: str
    contract_id: str
    contract_version: int = Field(ge=1)
    clause_type: str
    agreement_type: str
    jurisdiction: str | None = None
    page: int = Field(ge=1)
    source_text: str
    normalized_meaning: str
    final_decision: Decision
    approved_fallback: str | None = None
    decision_rationale: str
    review_status: Literal["human_approved"] = "human_approved"
    policy_version: str
    effective_at: str
    allowed_groups: list[str] = Field(default_factory=lambda: ["all"])
    source_hash: str | None = None

    @model_validator(mode="after")
    def validate_governance(self) -> ApprovedPrecedent:
        if not self.source_text.strip() or not self.decision_rationale.strip():
            raise ValueError("approved precedents require source text and rationale")
        if not self.allowed_groups:
            raise ValueError("approved precedents require at least one allowed group")
        expected = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.source_hash is None:
            self.source_hash = expected
        elif self.source_hash != expected:
            raise ValueError("precedent source_hash does not match source_text")
        return self

    def clause_retrieval_text(self) -> str:
        return "\n".join(
            (
                f"Agreement: {self.agreement_type}",
                f"Clause: {self.clause_type}",
                f"Jurisdiction: {self.jurisdiction or 'unspecified'}",
                f"Meaning: {self.normalized_meaning}",
                f"Language: {self.source_text}",
            )
        )

    def decision_retrieval_text(self) -> str:
        return "\n".join(
            (
                f"Clause: {self.clause_type}",
                f"Decision: {self.final_decision}",
                f"Rationale: {self.decision_rationale}",
                f"Approved fallback: {self.approved_fallback or 'none'}",
            )
        )


class RetrievedPrecedent(StrictModel):
    precedent_id: str
    contract_id: str
    contract_version: int
    clause_type: str
    agreement_type: str
    jurisdiction: str | None = None
    page: int
    source_text: str
    normalized_meaning: str
    final_decision: Decision
    approved_fallback: str | None = None
    decision_rationale: str
    policy_version: str
    source_hash: str
    relevance_score: float = Field(ge=0.0)


class PrecedentCorpus(StrictModel):
    corpus_id: str
    description: str
    precedents: list[ApprovedPrecedent]
