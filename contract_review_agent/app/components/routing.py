from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from haystack import component

from .common import Context, Stage

DOMAIN_ORDER = (
    "legal",
    "finance",
    "security/privacy",
    "procurement/business_owner",
)

DOMAIN_RUBRICS = {
    "legal": "liability, indemnity, termination, governing law, and assignment exposure",
    "finance": "payment timing, prepayment, commercial exposure, and approval thresholds",
    "security/privacy": "security commitments, data processing, incident duties, and privacy risk",
    "procurement/business_owner": "term, renewal, SLA remedies, operational fit, and vendor ownership",
}


def _domain_slug(area: str) -> str:
    return area.replace("/", "_").replace(" ", "_")


@component
class HighRiskPolicyGate(Stage):
    model_role = "business-rules"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def gate(ctx: Context) -> None:
            ctx["mandatory_human_review"] = True
            ctx["branch_path"].append("risk_policy:mandatory_human_review")

        return {"context": self.execute(context, "high_risk_policy_gate", gate)}


@component
class LowRiskAcceptanceRoute(Stage):
    model_role = "business-rules"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def accept(ctx: Context) -> None:
            ctx["review_areas"] = []
            ctx["domain_reviews"] = []
            ctx["routing_complete"] = True
            ctx["branch_path"].append("risk_policy:low_risk_auto_accept")

        return {"context": self.execute(context, "low_risk_acceptance", accept)}


@component
class DomainReviewDispatcher(Stage):
    """Fan elevated-risk deviations out to real Haystack domain branches."""

    model_role = "business-rules"

    @component.output_types(
        context=Context,
        legal_contexts=list[Context],
        finance_contexts=list[Context],
        security_contexts=list[Context],
        business_contexts=list[Context],
    )
    def run(self, context: Context) -> dict[str, Any]:
        def dispatch(ctx: Context) -> None:
            found = {
                deviation["review_area"] for deviation in ctx.get("deviations", [])
            }
            ctx["review_areas"] = [area for area in DOMAIN_ORDER if area in found]

        context = self.execute(context, "domain_review_dispatcher", dispatch)
        branches: dict[str, list[Context]] = {}
        output_for_area = {
            "legal": "legal_contexts",
            "finance": "finance_contexts",
            "security/privacy": "security_contexts",
            "procurement/business_owner": "business_contexts",
        }
        for area in context["review_areas"]:
            branch = deepcopy(context)
            branch["active_review_area"] = area
            branches[output_for_area[area]] = [branch]
            context["branch_path"].append(f"domain_fanout:{area}")
        return {"context": context, **branches}


@component
class DomainReviewer(Stage):
    """Apply a domain-specific rubric to one branch of the review graph."""

    model_role = "judge"

    def __init__(self, settings: Any, model_registry: Any, area: str):
        Stage.__init__(self, settings, model_registry)
        if area not in DOMAIN_RUBRICS:
            raise ValueError(f"unsupported review area: {area}")
        self.area = area

    def to_dict(self) -> dict[str, Any]:
        data = Stage.to_dict(self)
        data["init_parameters"] = {"area": self.area}
        return data

    @component.output_types(contexts=list[Context])
    def run(self, contexts: list[Context]) -> dict[str, list[Context]]:
        reviewed: list[Context] = []
        for context in contexts:
            deviations = [
                item
                for item in context.get("deviations", [])
                if item["review_area"] == self.area
            ]

            def review(
                ctx: Context,
                branch_deviations: tuple[dict[str, Any], ...] = (*deviations,),
            ) -> None:
                severity_order = {"low": 0, "medium": 1, "high": 2}
                highest = max(
                    (item["severity"] for item in branch_deviations),
                    key=severity_order.__getitem__,
                    default="low",
                )
                default_decision = {
                    "low": "accept",
                    "medium": "negotiate",
                    "high": "escalate",
                }[highest]
                response = self.models.call_json(
                    "judge",
                    system=(
                        f"Act as the {self.area} reviewer. Assess only {DOMAIN_RUBRICS[self.area]}. "
                        "Return decision (accept, negotiate, or escalate) and rationale as JSON."
                    ),
                    prompt=json.dumps(branch_deviations),
                )
                decision = (
                    str(response.get("decision")) if response else default_decision
                )
                if decision not in {"accept", "negotiate", "escalate"}:
                    decision = default_decision
                ctx["domain_review"] = {
                    "area": self.area,
                    "decision": decision,
                    "highest_risk": highest,
                    "deviation_count": len(branch_deviations),
                    "rationale": (
                        str(response.get("rationale"))
                        if response and response.get("rationale")
                        else f"{len(branch_deviations)} deviation(s) assessed under the {self.area} rubric."
                    ),
                }

            stage_name = f"{_domain_slug(self.area)}_domain_review"
            reviewed.append(self.execute(context, stage_name, review))
        return {"contexts": reviewed}


@component
class DomainReviewAggregator(Stage):
    """Converge independently reviewed domain branches into one contract context."""

    model_role = "business-rules"

    @component.output_types(context=Context)
    def run(
        self, context: Context, review_contexts: list[Context]
    ) -> dict[str, Context]:
        def aggregate(ctx: Context) -> None:
            baseline = len(ctx.get("stage_metrics", []))
            reviews: list[dict[str, Any]] = []
            for branch in review_contexts:
                review = branch.get("domain_review")
                if review:
                    reviews.append(review)
                    stage_name = f"{_domain_slug(review['area'])}_domain_review"
                    ctx["branch_path"].append(stage_name)
                for metric in branch.get("stage_metrics", [])[baseline:]:
                    ctx.setdefault("stage_metrics", []).append(metric)
            order = {area: index for index, area in enumerate(DOMAIN_ORDER)}
            ctx["domain_reviews"] = sorted(
                reviews, key=lambda item: order[item["area"]]
            )
            ctx["routing_complete"] = len(reviews) == len(ctx.get("review_areas", []))

        return {"context": self.execute(context, "domain_review_aggregator", aggregate)}


@component
class FallbackGenerator(Stage):
    model_role = "text"

    FALLBACKS = {
        "term": "Limit the initial and renewal terms to 12 months and require at least 30 days' non-renewal notice.",
        "termination": "Add termination for cause and customer termination for convenience on no more than 60 days' notice.",
        "liability": "Cap aggregate liability at fees paid or payable in the preceding 12 months, subject to approved carve-outs.",
        "indemnity": "Make indemnities mutual, scoped to third-party claims, and subject to the agreed liability framework.",
        "governing_law": "Replace the governing law with Delaware or New York law.",
        "assignment": "Require prior written consent for assignment, with a customary affiliate or change-of-control exception.",
        "security": "Add documented administrative, technical, and organizational security safeguards.",
        "dpa": "Incorporate the customer's approved data processing addendum where personal data is processed.",
        "sla": "Define availability and a service-credit remedy for missed service levels.",
        "payment": "Use Net 30 invoicing and remove unapproved advance prepayment.",
    }

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def generate(ctx: Context) -> None:
            ctx["fallback_attempt"] = ctx.get("fallback_attempt", 0) + 1
            response = self.models.call_json(
                "text",
                system="Draft concise fallback positions for listed deviations. Return clause_to_fallback JSON.",
                prompt=json.dumps(
                    {
                        "deviations": ctx.get("deviations", []),
                        "domain_reviews": ctx.get("domain_reviews", []),
                    }
                ),
            )
            raw_generated = response.get("clause_to_fallback", {}) if response else {}
            generated: dict[str, str] = {}
            if isinstance(raw_generated, dict):
                generated = {
                    str(key): str(value)
                    for key, value in raw_generated.items()
                    if value
                }
            elif isinstance(raw_generated, list):
                for item in raw_generated:
                    if not isinstance(item, dict):
                        continue
                    clause = item.get("clause") or item.get("clause_label")
                    fallback = (
                        item.get("fallback")
                        or item.get("recommended_fallback")
                        or item.get("text")
                    )
                    if clause and fallback:
                        generated[str(clause)] = str(fallback)
            for deviation in ctx.get("deviations", []):
                fallback = (
                    generated.get(deviation["clause"])
                    or self.FALLBACKS[deviation["clause"]]
                )
                if ctx["fallback_attempt"] > 1:
                    fallback = f"Focused revision: {fallback}"
                deviation["recommended_fallback"] = fallback

        status = "retry" if context.get("fallback_attempt", 0) else "ok"
        return {
            "context": self.execute(
                context, "fallback_generator", generate, status=status
            )
        }


@component
class NoEscalationRoute(Stage):
    model_role = "business-rules"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def route(ctx: Context) -> None:
            ctx["review_areas"] = []
            ctx["routing_complete"] = True

        return {"context": self.execute(context, "straight_through_route", route)}
