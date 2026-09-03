from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from haystack import component

from ..memory.store import LanceContractMemory
from .common import Context, Stage


@component
class MemoryModeRouter(Stage):
    """Make the configured memory mode visible as a real workflow branch."""

    model_role = "business-rules"

    @component.output_types(off_context=Context, enabled_context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def select(ctx: Context) -> None:
            ctx["memory_mode"] = self.settings.memory_mode
            ctx["memory"] = {
                "mode": self.settings.memory_mode,
                "embedding_model": self.models.model_name("embedding"),
                "table": self.settings.memory_table,
                "table_version": None,
                "query_count": 0,
                "candidate_count": 0,
                "selected_count": 0,
                "retrieval_latency_ms": 0.0,
                "selected_precedents": [],
                "shadow_precedents": [],
            }

        context = self.execute(context, "memory_mode_router", select)
        if self.settings.memory_mode == "off":
            context["branch_path"].append("memory:off")
            return {"off_context": context}
        context["branch_path"].append(f"memory:{self.settings.memory_mode}")
        return {"enabled_context": context}


@component
class MemoryBypass(Stage):
    model_role = "business-rules"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        return {
            "context": self.execute(
                context, "memory_bypass", lambda ctx: ctx.update(precedents=[])
            )
        }


@component
class PrecedentQueryBuilder(Stage):
    """Build one evidence-grounded retrieval query for each deviation."""

    model_role = "business-rules"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def build(ctx: Context) -> None:
            agreement_type = ctx.get("agreement_type") or "Vendor SaaS Agreement"
            queries: list[dict[str, str]] = []
            for deviation in ctx.get("deviations", []):
                evidence = deviation.get("evidence", {})
                queries.append(
                    {
                        "clause_type": str(deviation["clause"]),
                        "query_text": "\n".join(
                            (
                                f"Agreement: {agreement_type}",
                                f"Clause: {deviation['clause']}",
                                f"Issue: {deviation['reason']}",
                                f"Language: {evidence.get('text') or 'missing clause'}",
                            )
                        ),
                    }
                )
            ctx["precedent_queries"] = queries
            ctx["memory"]["query_count"] = len(queries)

        return {"context": self.execute(context, "precedent_query_builder", build)}


@component
class PrecedentRetriever(Stage):
    """Retrieve approved precedents with native LanceDB hybrid search."""

    model_role = "embedding"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def retrieve(ctx: Context) -> None:
            started = time.perf_counter()
            queries = ctx.get("precedent_queries", [])
            if not queries:
                ctx["precedent_candidates"] = []
                self._record_operation_measurements(
                    queries=0, candidates=0, results=0, table_version=None
                )
                return
            dimensions = int(self.models.models["embedding"].get("dimensions", 1024))
            store = LanceContractMemory(
                self.settings.memory_uri,
                self.settings.memory_table,
                dimensions=dimensions,
            )
            if not store.exists():
                ctx["precedent_candidates"] = []
                ctx["memory"]["retrieval_latency_ms"] = round(
                    (time.perf_counter() - started) * 1000, 3
                )
                self._record_operation_measurements(
                    queries=len(queries),
                    candidates=0,
                    results=0,
                    table_version=None,
                )
                return
            vectors = self.models.embed_texts(
                [item["query_text"] for item in queries], input_type="query"
            )
            candidates: list[dict[str, Any]] = []
            candidate_count = 0
            table_version: int | None = None
            for query, vector in zip(queries, vectors, strict=True):
                hits, version, observed_candidates = store.retrieve(
                    query_text=query["query_text"],
                    query_vector=vector,
                    tenant_id=self.settings.memory_tenant,
                    allowed_groups=list(self.settings.memory_allowed_groups),
                    clause_type=query["clause_type"],
                    agreement_type=ctx.get("agreement_type") or "Vendor SaaS Agreement",
                    exclude_contract_id=ctx["contract_id"],
                    candidate_k=self.settings.memory_candidate_k,
                    top_k=self.settings.memory_top_k,
                )
                table_version = version if version is not None else table_version
                candidate_count += observed_candidates
                candidates.extend(hit.model_dump(mode="json") for hit in hits)
            ctx["precedent_candidates"] = candidates
            ctx["memory"]["candidate_count"] = candidate_count
            ctx["memory"]["table_version"] = table_version
            ctx["memory"]["retrieval_latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            self._record_operation_measurements(
                queries=len(queries),
                candidates=candidate_count,
                results=len(candidates),
                table_version=table_version,
            )

        return {"context": self.execute(context, "precedent_retriever", retrieve)}

    def _record_operation_measurements(
        self,
        *,
        queries: int,
        candidates: int,
        results: int,
        table_version: int | None,
    ) -> None:
        """Enrich the active Haystack component span through Witdem's public API."""
        from opentelemetry import trace
        from witdem_sdk import Operation

        span = trace.get_current_span()
        span.set_attribute("gen_ai.provider.name", "lancedb")
        span.set_attribute("witdem.implementation.id", "lancedb")
        operation = Operation(span)
        operation.measure("queries", queries, unit="query")
        operation.measure("candidates", candidates, unit="document")
        operation.measure("results", results, unit="document")
        operation.measure("documents.output", results, unit="document")
        operation.measure("top_k", self.settings.memory_top_k, unit="document")
        if table_version is not None:
            operation.measure(
                "index.version",
                table_version,
                unit="version",
                aggregation="latest",
            )


@component
class PrecedentBundleAssembler(Stage):
    """Deduplicate results and enforce shadow versus influence semantics."""

    model_role = "business-rules"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def assemble(ctx: Context) -> None:
            unique: dict[str, dict[str, Any]] = {}
            for item in ctx.get("precedent_candidates", []):
                existing = unique.get(item["precedent_id"])
                if (
                    existing is None
                    or item["relevance_score"] > existing["relevance_score"]
                ):
                    unique[item["precedent_id"]] = deepcopy(item)
            ranked = sorted(
                unique.values(), key=lambda item: item["relevance_score"], reverse=True
            )
            if self.settings.memory_mode == "retrieve":
                ctx["precedents"] = ranked
                ctx["memory"]["selected_precedents"] = ranked
                ctx["memory"]["selected_count"] = len(ranked)
            else:
                ctx["precedents"] = []
                ctx["memory"]["shadow_precedents"] = ranked
                ctx["memory"]["selected_count"] = 0

        return {
            "context": self.execute(context, "precedent_bundle_assembler", assemble)
        }
