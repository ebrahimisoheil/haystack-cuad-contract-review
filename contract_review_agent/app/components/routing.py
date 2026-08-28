from __future__ import annotations

import json
from typing import Any

from haystack import component

from .common import Context, Stage


@component
class ReviewRouter(Stage):
    model_role = "business-rules"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def route(ctx: Context) -> None:
            order = ["legal", "finance", "security/privacy", "procurement/business_owner"]
            found = {deviation["review_area"] for deviation in ctx.get("deviations", [])}
            ctx["review_areas"] = [area for area in order if area in found]
            for area in ctx["review_areas"]:
                ctx["branch_path"].append(f"domain_escalation:{area}")
            ctx["routing_complete"] = True

        return {"context": self.execute(context, "review_router", route)}


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
            response = self.models.call_json("text",
                system="Draft concise fallback positions for listed deviations. Return clause_to_fallback JSON.",
                prompt=json.dumps(ctx.get("deviations", [])),
            )
            raw_generated = response.get("clause_to_fallback", {}) if response else {}
            generated: dict[str, str] = {}
            if isinstance(raw_generated, dict):
                generated = {str(key): str(value) for key, value in raw_generated.items() if value}
            elif isinstance(raw_generated, list):
                for item in raw_generated:
                    if not isinstance(item, dict):
                        continue
                    clause = item.get("clause") or item.get("clause_label")
                    fallback = item.get("fallback") or item.get("recommended_fallback") or item.get("text")
                    if clause and fallback:
                        generated[str(clause)] = str(fallback)
            for deviation in ctx.get("deviations", []):
                fallback = generated.get(deviation["clause"]) or self.FALLBACKS[deviation["clause"]]
                if ctx["fallback_attempt"] > 1:
                    fallback = f"Focused revision: {fallback}"
                deviation["recommended_fallback"] = fallback

        status = "retry" if context.get("fallback_attempt", 0) else "ok"
        return {"context": self.execute(context, "fallback_generator", generate, status=status)}


@component
class NoEscalationRoute(Stage):
    model_role = "business-rules"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def route(ctx: Context) -> None:
            ctx["review_areas"] = []
            ctx["routing_complete"] = True

        return {"context": self.execute(context, "straight_through_route", route)}
