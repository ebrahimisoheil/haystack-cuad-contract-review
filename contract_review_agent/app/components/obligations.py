from __future__ import annotations

import json
from typing import Any

from haystack import component

from .common import Context, Stage


@component
class ObligationExtractor(Stage):
    model_role = "text"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def extract(ctx: Context) -> None:
            response = self.models.call_json("text",
                system=(
                    "Extract post-signature obligations as JSON with top-level obligations list. "
                    "Every item must have exactly type, due_date_or_rule, owner, and evidence. "
                    "Evidence must contain page, text, clause_label, extraction_method, confidence."
                ),
                prompt=json.dumps(ctx.get("clauses", [])),
            )
            if response and isinstance(response.get("obligations"), list):
                normalized = []
                for item in response["obligations"]:
                    if not isinstance(item, dict):
                        continue
                    raw_evidence = item.get("evidence")
                    if not isinstance(raw_evidence, dict):
                        label = str(item.get("clause_label") or item.get("type") or "")
                        raw_evidence = ctx.get("clause_map", {}).get(label, {}).get("evidence")
                    if not isinstance(raw_evidence, dict):
                        continue
                    obligation_type = item.get("type") or item.get("obligation_type")
                    if not obligation_type:
                        obligation_type = raw_evidence.get("clause_label") or "contractual_obligation"
                    rule = item.get("due_date_or_rule") or item.get("rule") or item.get("due_date")
                    if not rule:
                        continue
                    normalized.append(
                        {
                            "type": str(obligation_type),
                            "due_date_or_rule": str(rule),
                            "owner": str(item.get("owner") or "unassigned"),
                            "evidence": {
                                "page": raw_evidence.get("page"),
                                "text": raw_evidence.get("text"),
                                "clause_label": str(raw_evidence.get("clause_label") or obligation_type),
                                "extraction_method": str(
                                    raw_evidence.get("extraction_method")
                                    or ctx.get("extraction_method", "unresolved")
                                ),
                                "confidence": float(raw_evidence.get("confidence", 0.0)),
                            },
                        }
                    )
                if normalized:
                    ctx["obligations"] = normalized
                    return
            cmap = ctx.get("clause_map", {})
            obligations: list[dict[str, Any]] = []

            def add(kind: str, rule: str, owner: str, label: str) -> None:
                clause = cmap.get(label)
                if clause:
                    obligations.append(
                        {"type": kind, "due_date_or_rule": rule, "owner": owner, "evidence": clause["evidence"]}
                    )

            term = ctx.get("terms", {}).get("term", {})
            if term.get("auto_renewal"):
                add("renewal_notice", f"Give notice at least {term.get('notice_days')} days before renewal.", "procurement", "term")
            add("payment", ctx.get("terms", {}).get("payment_terms") or "Unresolved", "accounts_payable", "payment")
            add("sla_monitoring", ctx.get("terms", {}).get("sla", {}).get("availability") or "Unresolved", "vendor_management", "sla")
            add("security", "Maintain and monitor contractual security commitments.", "security", "security")
            ctx["obligations"] = obligations

        return {"context": self.execute(context, "obligation_extractor", extract)}


@component
class SkipObligations(Stage):
    model_role = "business-rules"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def skip(ctx: Context) -> None:
            ctx["obligations"] = []

        return {"context": self.execute(context, "obligations_skipped", skip)}
