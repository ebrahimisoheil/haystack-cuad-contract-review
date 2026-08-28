from __future__ import annotations

import json
from typing import Any

from haystack import component

from .common import Context, Stage


@component
class ExtractionJudge(Stage):
    model_role = "judge"

    @component.output_types(context=Context, needs_retry=bool)
    def run(self, context: Context) -> dict[str, Any]:
        def judge(ctx: Context) -> None:
            missing = ctx.get("missing_clauses", [])
            local_pass = not missing and float(ctx.get("extraction_confidence", 0)) >= 0.75
            response = self.models.call_json("judge",
                system=(
                    "Judge whether extraction is evidence-supported and complete. Return pass, reason, "
                    "and unresolved_labels as JSON. Do not perform extraction."
                ),
                prompt=json.dumps(
                    {
                        "confidence": ctx.get("extraction_confidence"),
                        "missing": missing,
                        "clauses": ctx.get("clauses", []),
                    }
                ),
            )
            passed = bool(response.get("pass")) if response else local_pass
            ctx["extraction_judge_passed"] = passed
            ctx["extraction_judge_reason"] = (
                str(response.get("reason")) if response else
                ("complete and evidence-supported" if passed else f"missing clauses: {', '.join(missing)}")
            )
            ctx["needs_extraction_retry"] = (
                not passed and ctx.get("retries", 0) < self.settings.max_retries
            )

        context = self.execute(context, "extraction_judge", judge)
        route = "retry" if context["needs_extraction_retry"] else "accepted"
        context["branch_path"].append(f"extraction_judge_route:{route}")
        return {"context": context, "needs_retry": context["needs_extraction_retry"]}


@component
class RiskJudge(Stage):
    model_role = "judge"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def judge(ctx: Context) -> None:
            response = self.models.call_json("judge",
                system="Judge deviation severity only. Return JSON with clause_to_severity mapping.",
                prompt=json.dumps(ctx.get("deviations", [])),
            )
            if response and isinstance(response.get("clause_to_severity"), dict):
                for deviation in ctx.get("deviations", []):
                    value = response["clause_to_severity"].get(deviation["clause"])
                    if value in {"low", "medium", "high"}:
                        deviation["severity"] = value
            ctx["highest_risk"] = (
                "high" if any(d["severity"] == "high" for d in ctx.get("deviations", []))
                else "medium" if any(d["severity"] == "medium" for d in ctx.get("deviations", []))
                else "low"
            )
            ctx["branch_path"].append(f"risk_route:{ctx['highest_risk']}")

        return {"context": self.execute(context, "risk_judge", judge)}


@component
class FallbackJudge(Stage):
    model_role = "judge"

    @component.output_types(context=Context, needs_retry=bool)
    def run(self, context: Context) -> dict[str, Any]:
        def judge(ctx: Context) -> None:
            response = self.models.call_json("judge",
                system="Judge whether each fallback resolves its deviation. Return pass and reason as JSON.",
                prompt=json.dumps(ctx.get("deviations", [])),
            )
            local_pass = all(bool(d.get("recommended_fallback")) for d in ctx.get("deviations", []))
            if "FALLBACK_RETRY" in ctx.get("normalized_text", "") and ctx.get("fallback_attempt", 1) == 1:
                local_pass = False
            passed = bool(response.get("pass")) if response else local_pass
            for deviation in ctx.get("deviations", []):
                deviation["fallback_accepted"] = passed
            ctx["fallback_judge_passed"] = passed
            ctx["fallback_judge_reason"] = str(response.get("reason")) if response else (
                "fallbacks address deviations" if passed else "fallback needs a narrower revision"
            )
            ctx["needs_fallback_retry"] = (
                not passed and ctx.get("fallback_attempt", 1) <= self.settings.max_retries
            )

        context = self.execute(context, "fallback_judge", judge)
        route = "retry" if context["needs_fallback_retry"] else "accepted"
        context["branch_path"].append(f"fallback_judge_route:{route}")
        return {"context": context, "needs_retry": context["needs_fallback_retry"]}


@component
class FinalReviewJudge(Stage):
    model_role = "judge"

    @component.output_types(context=Context, approved=bool)
    def run(self, context: Context) -> dict[str, Any]:
        def judge(ctx: Context) -> None:
            response = self.models.call_json("judge",
                system=(
                    "Make the final playbook decision. Return decision and explanation JSON. Allowed: "
                    "approved, approved_with_exceptions, manual_review_required, "
                    "rejected_by_playbook, processing_failed."
                ),
                prompt=json.dumps(
                    {
                        "deviations": ctx.get("deviations", []),
                        "review_areas": ctx.get("review_areas", []),
                        "errors": ctx.get("errors", []),
                    }
                ),
            )
            if response and response.get("decision"):
                decision = str(response["decision"])
                explanation = str(response.get("explanation", "OpenAI judge decision"))
            elif ctx.get("processing_failed"):
                decision, explanation = "processing_failed", "A required processing stage failed."
            else:
                high = sum(d["severity"] == "high" for d in ctx.get("deviations", []))
                if high >= 2:
                    decision, explanation = "rejected_by_playbook", "Multiple high-risk deviations exceed the fictional playbook."
                elif high == 1:
                    decision, explanation = "manual_review_required", "A high-risk deviation requires a human reviewer."
                elif ctx.get("deviations"):
                    decision, explanation = "approved_with_exceptions", "Fallbacks and routed exceptions require recorded approval."
                else:
                    decision, explanation = "approved", "All evaluated clauses conform to the fictional playbook."
            ctx["final_decision"] = decision
            ctx["decision_explanation"] = explanation
            ctx["approved_for_obligations"] = decision in {"approved", "approved_with_exceptions"}

        context = self.execute(context, "final_review_judge", judge)
        route = "approved" if context["approved_for_obligations"] else "manual_or_rejected"
        context["branch_path"].append(f"final_decision_route:{route}")
        return {"context": context, "approved": context["approved_for_obligations"]}
