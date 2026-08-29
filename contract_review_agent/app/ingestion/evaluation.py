from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..schemas import CuadCategoryEvaluation, CuadGroundTruthEvaluation

Result = dict[str, Any]

CUAD_CATEGORY_F1_TARGET = 0.7
CUAD_SPAN_TOKEN_F1_TARGET = 0.5
CUAD_NEGATIVE_LABEL_ACCURACY_TARGET = 0.8


def _normalized_category(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _non_empty(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _path(result: Result, *parts: str) -> Any:
    value: Any = result
    for part in parts:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _clause_texts(result: Result, label: str) -> list[str]:
    texts: list[str] = []
    for clause in result.get("clauses", []):
        if not isinstance(clause, dict) or clause.get("label") != label:
            continue
        evidence = clause.get("evidence")
        if isinstance(evidence, dict) and _non_empty(evidence.get("text")):
            texts.append(str(evidence["text"]))
        if _non_empty(clause.get("summary")):
            texts.append(str(clause["summary"]))
    return list(dict.fromkeys(texts))


def _value_and_clause(result: Result, value: Any, clause: str) -> list[str]:
    values = [str(value)] if _non_empty(value) else []
    return list(dict.fromkeys(values + _clause_texts(result, clause)))


def _uncapped_liability(result: Result) -> bool:
    text = " ".join(_clause_texts(result, "liability")).casefold()
    return any(
        term in text
        for term in ("uncapped", "unlimited", "without limit", "no liability cap")
    )


@dataclass(frozen=True)
class CategorySpec:
    name: str
    predicted: Callable[[Result], bool]
    evidence: Callable[[Result], list[str]]


CATEGORY_SPECS = (
    CategorySpec(
        "Governing Law",
        lambda result: _non_empty(result.get("governing_law")),
        lambda result: _value_and_clause(
            result, result.get("governing_law"), "governing_law"
        ),
    ),
    CategorySpec(
        "Termination For Convenience",
        lambda result: _path(result, "termination", "for_convenience") is True,
        lambda result: _clause_texts(result, "termination"),
    ),
    CategorySpec(
        "Anti-Assignment",
        lambda result: result.get("assignment_restricted") is True,
        lambda result: _clause_texts(result, "assignment"),
    ),
    CategorySpec(
        "Cap On Liability",
        lambda result: _path(result, "liability", "cap_present") is True,
        lambda result: _clause_texts(result, "liability"),
    ),
    CategorySpec(
        "Uncapped Liability",
        _uncapped_liability,
        lambda result: _clause_texts(result, "liability"),
    ),
    CategorySpec(
        "Renewal Term",
        lambda result: (
            _path(result, "term", "auto_renewal") is True
            or _non_empty(_path(result, "term", "renewal_months"))
        ),
        lambda result: _clause_texts(result, "term"),
    ),
    CategorySpec(
        "Notice Period To Terminate Renewal",
        lambda result: _non_empty(_path(result, "term", "notice_days")),
        lambda result: _clause_texts(result, "term"),
    ),
)

_SPEC_BY_CATEGORY = {_normalized_category(spec.name): spec for spec in CATEGORY_SPECS}


def _token_f1(predicted: str, expected: str) -> float:
    predicted_tokens = Counter(_tokens(predicted))
    expected_tokens = Counter(_tokens(expected))
    if not predicted_tokens or not expected_tokens:
        return 0.0
    overlap = sum((predicted_tokens & expected_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(predicted_tokens.values())
    recall = overlap / sum(expected_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _span_scores(candidates: list[str], answers: list[str]) -> tuple[float, float]:
    if not candidates or not answers:
        return 0.0, 0.0
    exact = 0.0
    best_f1 = 0.0
    for candidate in candidates:
        candidate_tokens = _tokens(candidate)
        for answer in answers:
            exact = max(
                exact,
                float(candidate_tokens == _tokens(answer) and bool(candidate_tokens)),
            )
            best_f1 = max(best_f1, _token_f1(candidate, answer))
    return exact, best_f1


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_cuad_ground_truth(
    result: Result,
    labels: list[dict[str, Any]],
) -> CuadGroundTruthEvaluation:
    """Compare a completed prediction with held-out CUAD annotations.

    This function is called only by the post-run result reporter, after all
    model stages and retry routes have completed. Labels therefore cannot
    influence extraction or judging and never become Haystack pipeline input.
    """

    by_category: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        if not isinstance(label, dict):
            continue
        category = _normalized_category(str(label.get("category", "")))
        if category in _SPEC_BY_CATEGORY:
            by_category.setdefault(category, []).append(label)

    category_results: list[CuadCategoryEvaluation] = []
    for normalized, spec in _SPEC_BY_CATEGORY.items():
        annotations = by_category.get(normalized)
        if not annotations:
            continue
        answers = [
            str(answer.get("text", ""))
            for annotation in annotations
            for answer in annotation.get("answers", [])
            if isinstance(answer, dict) and _non_empty(answer.get("text"))
        ]
        truth_positive = bool(answers) and any(
            not bool(item.get("is_impossible")) for item in annotations
        )
        predicted_positive = bool(spec.predicted(result))
        if truth_positive and predicted_positive:
            outcome = "true_positive"
        elif predicted_positive:
            outcome = "false_positive"
        elif truth_positive:
            outcome = "false_negative"
        else:
            outcome = "true_negative"
        exact: float | None = None
        token_f1: float | None = None
        if outcome == "true_positive":
            exact, token_f1 = _span_scores(spec.evidence(result), answers)
        category_results.append(
            CuadCategoryEvaluation(
                category=spec.name,
                ground_truth_positive=truth_positive,
                predicted_positive=predicted_positive,
                outcome=outcome,
                span_exact_match=exact,
                span_token_f1=token_f1,
            )
        )

    counts = Counter(item.outcome for item in category_results)
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    tn = counts["true_negative"]
    precision = _optional_ratio(tp, tp + fp)
    recall = _optional_ratio(tp, tp + fn)
    category_f1 = (
        0.0
        if tp + fn and tp == 0
        else (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
    )
    span_results = [
        item for item in category_results if item.outcome == "true_positive"
    ]
    negative_count = tn + fp
    return CuadGroundTruthEvaluation(
        evaluated_categories=len(category_results),
        supported_categories=[spec.name for spec in CATEGORY_SPECS],
        ground_truth_positives=tp + fn,
        predicted_positives=tp + fp,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        category_precision=precision,
        category_recall=recall,
        category_f1=category_f1,
        category_accuracy=_ratio(tp + tn, len(category_results)),
        negative_label_accuracy=_ratio(tn, negative_count) if negative_count else None,
        evaluated_spans=len(span_results),
        span_exact_match=(
            sum(item.span_exact_match or 0.0 for item in span_results)
            / len(span_results)
            if span_results
            else None
        ),
        span_token_f1=(
            sum(item.span_token_f1 or 0.0 for item in span_results) / len(span_results)
            if span_results
            else None
        ),
        categories=category_results,
    )


def aggregate_cuad_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "evaluated_contracts": sum(
            int(item.get("evaluated_categories", 0)) > 0 for item in evaluations
        ),
        "evaluated_categories": sum(
            int(item.get("evaluated_categories", 0)) for item in evaluations
        ),
        "ground_truth_positives": sum(
            int(item.get("ground_truth_positives", 0)) for item in evaluations
        ),
        "predicted_positives": sum(
            int(item.get("predicted_positives", 0)) for item in evaluations
        ),
        "true_positives": sum(
            int(item.get("true_positives", 0)) for item in evaluations
        ),
        "false_positives": sum(
            int(item.get("false_positives", 0)) for item in evaluations
        ),
        "false_negatives": sum(
            int(item.get("false_negatives", 0)) for item in evaluations
        ),
        "true_negatives": sum(
            int(item.get("true_negatives", 0)) for item in evaluations
        ),
        "evaluated_spans": sum(
            int(item.get("evaluated_spans", 0)) for item in evaluations
        ),
    }
    tp = totals["true_positives"]
    fp = totals["false_positives"]
    fn = totals["false_negatives"]
    tn = totals["true_negatives"]
    precision = _optional_ratio(tp, tp + fp)
    recall = _optional_ratio(tp, tp + fn)
    totals["category_precision"] = precision
    totals["category_recall"] = recall
    totals["category_f1"] = (
        0.0
        if tp + fn and tp == 0
        else (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
    )
    totals["category_accuracy"] = _ratio(tp + tn, totals["evaluated_categories"])
    totals["negative_label_accuracy"] = _ratio(tn, tn + fp) if tn + fp else None
    span_count = totals["evaluated_spans"]
    totals["span_exact_match"] = (
        sum(
            float(item.get("span_exact_match") or 0.0)
            * int(item.get("evaluated_spans", 0))
            for item in evaluations
        )
        / span_count
        if span_count
        else None
    )
    totals["span_token_f1"] = (
        sum(
            float(item.get("span_token_f1") or 0.0)
            * int(item.get("evaluated_spans", 0))
            for item in evaluations
        )
        / span_count
        if span_count
        else None
    )
    return totals
