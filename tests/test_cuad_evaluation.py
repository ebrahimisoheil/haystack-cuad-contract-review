from __future__ import annotations

import pytest

from contract_review_agent.app.ingestion.evaluation import (
    aggregate_cuad_evaluations,
    evaluate_cuad_ground_truth,
)
from contract_review_agent.app.pipeline import (
    _cuad_report_result,
    _cuad_result_reporter,
)


def _label(category: str, answer: str | None) -> dict:
    return {
        "annotation_id": f"demo__{category}",
        "category": category,
        "question": category,
        "is_impossible": answer is None,
        "answers": (
            []
            if answer is None
            else [{"text": answer, "answer_start": 0, "answer_end": len(answer)}]
        ),
    }


def _prediction() -> dict:
    return {
        "governing_law": "Delaware",
        "termination": {"for_convenience": True},
        "assignment_restricted": True,
        "liability": {"cap_present": True},
        "term": {"auto_renewal": False, "renewal_months": None, "notice_days": None},
        "clauses": [
            {
                "label": "governing_law",
                "summary": "Delaware",
                "evidence": {"text": "Delaware"},
            },
            {
                "label": "termination",
                "summary": "Customer may terminate for convenience on 30 days notice.",
                "evidence": {
                    "text": "Customer may terminate for convenience on 30 days notice."
                },
            },
            {
                "label": "assignment",
                "summary": "Assignment requires consent.",
                "evidence": {"text": "Assignment requires consent."},
            },
            {
                "label": "liability",
                "summary": "Liability is capped at fees paid in the prior 12 months.",
                "evidence": {
                    "text": "Liability is capped at fees paid in the prior 12 months."
                },
            },
        ],
    }


def test_ground_truth_evaluation_measures_presence_and_spans() -> None:
    evaluation = evaluate_cuad_ground_truth(
        _prediction(),
        [
            _label("Governing Law", "Delaware"),
            _label(
                "Termination For Convenience",
                "Customer may terminate for convenience on 30 days notice.",
            ),
            _label("Anti-Assignment", None),
            _label("Cap On Liability", None),
            _label("Uncapped Liability", None),
            _label("Renewal Term", "renews for 12 months"),
            _label("Notice Period To Terminate Renewal", None),
        ],
    )

    assert evaluation.evaluated_categories == 7
    assert evaluation.true_positives == 2
    assert evaluation.false_positives == 2
    assert evaluation.false_negatives == 1
    assert evaluation.true_negatives == 2
    assert evaluation.category_precision == pytest.approx(0.5)
    assert evaluation.category_recall == pytest.approx(2 / 3)
    assert evaluation.category_f1 == pytest.approx(4 / 7)
    assert evaluation.category_accuracy == pytest.approx(4 / 7)
    assert evaluation.negative_label_accuracy == pytest.approx(0.5)
    assert evaluation.evaluated_spans == 2
    assert evaluation.span_exact_match == pytest.approx(1.0)
    assert evaluation.span_token_f1 == pytest.approx(1.0)


def test_unmapped_annotations_are_not_claimed_as_evaluated() -> None:
    evaluation = evaluate_cuad_ground_truth(
        _prediction(),
        [_label("Most Favored Nation", "most favored terms")],
    )

    assert evaluation.evaluated_categories == 0
    assert evaluation.category_f1 is None
    assert evaluation.span_token_f1 is None
    assert evaluation.categories == []


def test_batch_aggregation_recomputes_micro_metrics() -> None:
    first = evaluate_cuad_ground_truth(
        _prediction(),
        [_label("Governing Law", "Delaware"), _label("Anti-Assignment", None)],
    ).model_dump(mode="json")
    second = evaluate_cuad_ground_truth(
        {"clauses": [], "term": {}, "termination": {}, "liability": {}},
        [_label("Governing Law", "New York"), _label("Anti-Assignment", None)],
    ).model_dump(mode="json")

    aggregate = aggregate_cuad_evaluations([first, second])

    assert aggregate["evaluated_contracts"] == 2
    assert aggregate["evaluated_categories"] == 4
    assert aggregate["true_positives"] == 1
    assert aggregate["false_positives"] == 1
    assert aggregate["false_negatives"] == 1
    assert aggregate["true_negatives"] == 1
    assert aggregate["category_precision"] == pytest.approx(0.5)
    assert aggregate["category_recall"] == pytest.approx(0.5)
    assert aggregate["category_f1"] == pytest.approx(0.5)


def test_witdem_report_uses_dedicated_contract_and_omits_missing_scores() -> None:
    evaluation = evaluate_cuad_ground_truth(
        _prediction(),
        [_label("Anti-Assignment", None)],
    ).model_dump(mode="json")
    report = _cuad_report_result(
        {
            "result_assembler": {
                "result": {
                    "contract_id": "contract-1",
                    "agreement_type": "Service Agreement",
                    "final_decision": "manual_review_required",
                    "decision_explanation": "Human review is required.",
                    "outcome": {
                        "review_completed": True,
                        "routing_complete": True,
                        "objective_met": True,
                    },
                    "metrics": {"evidence_completeness": 1.0, "retries": 2},
                    "cuad_evaluation": evaluation,
                }
            }
        }
    )

    assert report["contract"] == "cuad_contract_review"
    assert report["result"] == "manual_review_required"
    assert report["product_goal_achieved"] is True
    assert "category_f1" not in report["evaluations"]
    assert report["evaluations"]["negative_label_accuracy"] == 0
    assert "span_token_f1" not in report["evaluations"]
    assert report["evidence_sufficient"] is False
    assert report["closest_blocker"] == "cuad_evaluation_below_target"
    assert report["metrics"]["false_positives"] == 1


def test_post_run_reporter_keeps_answer_text_out_of_reported_facts() -> None:
    pipeline_result = {
        "result_assembler": {
            "result": {
                "contract_id": "contract-1",
                "agreement_type": "Service Agreement",
                "governing_law": "Delaware",
                "termination": {},
                "liability": {},
                "term": {},
                "assignment_restricted": None,
                "clauses": [],
                "final_decision": "manual_review_required",
                "decision_explanation": "Human review is required.",
                "outcome": {
                    "review_completed": True,
                    "routing_complete": True,
                    "objective_met": True,
                },
                "metrics": {"evidence_completeness": 1.0, "retries": 0},
            }
        }
    }
    reporter = _cuad_result_reporter([_label("Governing Law", "SECRET GOLD SPAN")])

    reported = reporter(pipeline_result)

    assert pipeline_result["result_assembler"]["result"]["cuad_evaluation"] is not None
    assert "SECRET GOLD SPAN" not in str(reported)
