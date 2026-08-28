from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from haystack import component

from .common import Context, Stage, evidence


def _severity_label(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("severity") or value.get("level") or value.get("risk")
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    return normalized if normalized in {"low", "medium", "high"} else None


@component
class PlaybookEvaluator(Stage):
    model_role = "judge"

    def __init__(self, settings: Any, model_registry: Any):
        Stage.__init__(self, settings, model_registry)
        self.playbook = yaml.safe_load(Path(settings.playbook_path).read_text(encoding="utf-8"))

    @component.output_types(context=Context, has_deviations=bool)
    def run(self, context: Context) -> dict[str, Any]:
        def evaluate(ctx: Context) -> None:
            terms = ctx.get("terms", {})
            clause_map = ctx.get("clause_map", {})
            deviations: list[dict[str, Any]] = []

            def add(clause: str, severity: str, reason: str, area: str) -> None:
                source = clause_map.get(clause, {})
                deviations.append(
                    {
                        "clause": clause,
                        "severity": severity,
                        "reason": reason,
                        "evidence": source.get("evidence") or evidence(ctx, clause, None, confidence=0.0),
                        "recommended_fallback": None,
                        "fallback_accepted": None,
                        "review_area": area,
                    }
                )

            term = terms.get("term", {})
            if term.get("initial_months") is None:
                add("term", "medium", "Initial term is unresolved.", "procurement/business_owner")
            elif term["initial_months"] > 12:
                add("term", "medium", "Initial term exceeds 12 months.", "procurement/business_owner")
            if term.get("auto_renewal") and (term.get("renewal_months") or 10**6) > 12:
                add("term", "medium", "Automatic renewal exceeds 12 months.", "procurement/business_owner")
            if term.get("auto_renewal") and (term.get("notice_days") is None or term["notice_days"] < 30):
                add("term", "medium", "Renewal notice is under 30 days or unresolved.", "procurement/business_owner")

            termination = terms.get("termination", {})
            if not termination.get("for_cause"):
                add("termination", "high", "Termination for cause is absent.", "legal")
            if not termination.get("for_convenience"):
                add("termination", "medium", "Termination for convenience is absent.", "legal")
            if (termination.get("notice_days") or 0) > 60:
                add("termination", "medium", "Termination notice exceeds 60 days.", "legal")

            liability = terms.get("liability", {})
            if not liability.get("cap_present"):
                add("liability", "high", "Liability cap is absent or unlimited.", "legal")
            elif "prior 12 months" not in (liability.get("cap_description") or "").lower():
                add("liability", "medium", "Cap is not tied to fees paid in the prior 12 months.", "legal")
            if terms.get("unlimited_unilateral_indemnity"):
                add("indemnity", "high", "Customer has unlimited unilateral indemnity.", "legal")
            if terms.get("governing_law") not in {"Delaware", "New York"}:
                add("governing_law", "medium", "Governing law is outside Delaware/New York.", "legal")
            if terms.get("assignment_restricted") is not True:
                add("assignment", "medium", "Assignment is unrestricted or unresolved.", "legal")
            security = terms.get("security", {})
            if not security.get("security_clause_present"):
                add("security", "high", "Security commitments are missing.", "security/privacy")
            if not security.get("dpa_language_present"):
                add("dpa", "medium", "DPA language is missing or unresolved.", "security/privacy")
            sla = terms.get("sla", {})
            if not sla.get("availability") or not sla.get("remedy_present"):
                add("sla", "medium", "SLA availability or remedy is missing.", "procurement/business_owner")
            if (terms.get("payment_net_days") or 0) < 30:
                add("payment", "medium", "Payment terms are shorter than Net 30 or unresolved.", "finance")
            if terms.get("unusual_prepayment"):
                add("payment", "medium", "Unusual prepayment requires approval.", "finance")
            response = self.models.call_json(
                "judge",
                system=(
                    "Evaluate only whether the supplied rule-engine deviations conform to the fictional "
                    "playbook. Return severity_by_clause and explanation as JSON; do not extract text."
                ),
                prompt=json.dumps({"playbook": self.playbook, "deviations": deviations}),
            )
            if response and isinstance(response.get("severity_by_clause"), dict):
                for deviation in deviations:
                    severity = _severity_label(
                        response["severity_by_clause"].get(deviation["clause"])
                    )
                    if severity is not None:
                        deviation["severity"] = severity
            ctx["deviations"] = deviations
            ctx["playbook_evaluated"] = True

        context = self.execute(context, "playbook_evaluator", evaluate)
        has_deviations = bool(context.get("deviations"))
        context["branch_path"].append(
            "playbook_route:deviations" if has_deviations else "playbook_route:compliant"
        )
        return {"context": context, "has_deviations": has_deviations}
