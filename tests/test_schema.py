from __future__ import annotations

import pytest
from pydantic import ValidationError

from contract_review_agent.app.schemas import Evidence


def test_unsupported_evidence_cannot_claim_confidence() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            page=None,
            text=None,
            clause_label="liability",
            extraction_method="unknown",
            confidence=0.8,
        )
