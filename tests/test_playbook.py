from __future__ import annotations

from contract_review_agent.app.components.playbook import _severity_label


def test_severity_label_accepts_nested_live_model_shape() -> None:
    assert _severity_label({"severity": "High", "explanation": "Outside policy"}) == "high"
    assert _severity_label({"level": "medium"}) == "medium"


def test_severity_label_rejects_unknown_shapes() -> None:
    assert _severity_label({"explanation": "No severity"}) is None
    assert _severity_label(["high"]) is None
