from __future__ import annotations

import time
from typing import Any

from haystack import component

from ..schemas import BusinessOutcome, ContractReviewResult, RunMetrics
from .common import Context, Stage


@component
class ResultAssembler(Stage):
    model_role = "local-validation"

    @component.output_types(result=dict)
    def run(self, context: Context) -> dict[str, dict[str, Any]]:
        def assemble(ctx: Context) -> None:
            terms = ctx.get("terms", {})
            deviations = ctx.get("deviations", [])
            deviation_risks = {item["clause"]: item["severity"] for item in deviations}
            liability = dict(terms.get("liability", {}))
            liability["risk"] = deviation_risks.get("liability", "low")
            indemnity = dict(terms.get("indemnity", {}))
            indemnity["risk"] = deviation_risks.get("indemnity", "low")

            evidence_items = [item["evidence"] for item in ctx.get("clauses", [])]
            evidence_items.extend(item["evidence"] for item in deviations)
            evidence_items.extend(item["evidence"] for item in ctx.get("obligations", []))
            complete = sum(bool(item.get("text")) and item.get("page") is not None for item in evidence_items)
            evidence_ratio = complete / len(evidence_items) if evidence_items else 0.0
            required_values = [
                ctx.get("agreement_type"),
                ctx.get("parties", {}).get("customer"),
                ctx.get("parties", {}).get("vendor"),
                ctx.get("effective_date"),
                terms.get("term", {}).get("initial_months"),
                terms.get("governing_law"),
                terms.get("payment_terms"),
            ]
            unresolved = sum(value is None for value in required_values) + len(ctx.get("missing_clauses", []))
            stages = ctx.get("stage_metrics", [])
            total_input = sum(item.get("input_tokens", 0) for item in stages)
            total_output = sum(item.get("output_tokens", 0) for item in stages)
            total_cost = sum(item.get("estimated_cost_usd", 0.0) for item in stages)
            total_runtime = (time.perf_counter() - ctx["started_at"]) * 1000
            review_completed = ctx.get("final_decision") != "processing_failed"
            evidence_complete = evidence_ratio == 1.0 and unresolved == 0
            outcome = BusinessOutcome(
                review_completed=review_completed,
                evidence_complete=evidence_complete,
                playbook_evaluated=bool(ctx.get("playbook_evaluated")),
                routing_complete=bool(ctx.get("routing_complete")),
                objective_met=(
                    review_completed
                    and bool(ctx.get("playbook_evaluated"))
                    and bool(ctx.get("routing_complete"))
                    and evidence_ratio >= 0.8
                ),
            )
            metrics = RunMetrics(
                total_runtime_ms=round(total_runtime, 3),
                stages=stages,
                retries=int(ctx.get("retries", 0)) + max(0, int(ctx.get("fallback_attempt", 1)) - 1),
                branch_path=ctx.get("branch_path", []),
                extraction_confidence=float(ctx.get("extraction_confidence", 0.0)),
                deviation_count=len(deviations),
                escalation_count=len(ctx.get("review_areas", [])),
                unresolved_field_count=unresolved,
                evidence_completeness=round(evidence_ratio, 4),
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                estimated_cost_usd=round(total_cost, 8),
            )
            result = ContractReviewResult(
                contract_id=ctx["contract_id"],
                agreement_type=ctx.get("agreement_type"),
                parties=ctx.get("parties", {}),
                effective_date=ctx.get("effective_date"),
                term=terms.get("term", {}),
                termination=terms.get("termination", {}),
                governing_law=terms.get("governing_law"),
                assignment_restricted=terms.get("assignment_restricted"),
                liability=liability,
                indemnity=indemnity,
                security=terms.get("security", {}),
                sla=terms.get("sla", {}),
                payment_terms=terms.get("payment_terms"),
                clauses=ctx.get("clauses", []),
                deviations=deviations,
                final_decision=ctx.get("final_decision", "processing_failed"),
                decision_explanation=ctx.get("decision_explanation", "Processing did not reach the final gate."),
                review_areas=ctx.get("review_areas", []),
                obligations=ctx.get("obligations", []),
                outcome=outcome,
                metrics=metrics,
                errors=ctx.get("errors", []),
            )
            ctx["result"] = result.model_dump(mode="json")

        context = self.execute(context, "result_assembler", assemble)
        if "result" not in context:
            raise ValueError(f"result validation failed: {context.get('errors', [])}")
        # Include the assembler metric in the validated result without changing runtime semantics.
        context["result"]["metrics"]["stages"] = context["stage_metrics"]
        context["result"]["metrics"]["branch_path"] = context["branch_path"]
        return {"result": context["result"]}
