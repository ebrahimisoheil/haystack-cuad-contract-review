from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa

from .schemas import ApprovedPrecedent, RetrievedPrecedent


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class LanceContractMemory:
    """Small, explicit LanceDB boundary for approved contract precedents."""

    def __init__(self, uri: str, table_name: str, *, dimensions: int):
        import lancedb

        self.uri = uri
        self.table_name = table_name
        self.dimensions = dimensions
        if "://" not in uri and not uri.startswith("db://"):
            Path(uri).expanduser().mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(uri)

    def _schema(self) -> pa.Schema:
        vector = pa.list_(pa.float32(), self.dimensions)
        return pa.schema(
            [
                pa.field("precedent_id", pa.string(), nullable=False),
                pa.field("tenant_id", pa.string(), nullable=False),
                pa.field("contract_id", pa.string(), nullable=False),
                pa.field("contract_version", pa.int32(), nullable=False),
                pa.field("clause_type", pa.string(), nullable=False),
                pa.field("agreement_type", pa.string(), nullable=False),
                pa.field("jurisdiction", pa.string()),
                pa.field("page", pa.int32(), nullable=False),
                pa.field("source_text", pa.string(), nullable=False),
                pa.field("normalized_meaning", pa.string(), nullable=False),
                pa.field("final_decision", pa.string(), nullable=False),
                pa.field("approved_fallback", pa.string()),
                pa.field("decision_rationale", pa.string(), nullable=False),
                pa.field("review_status", pa.string(), nullable=False),
                pa.field("policy_version", pa.string(), nullable=False),
                pa.field("effective_at", pa.string(), nullable=False),
                pa.field("allowed_groups", pa.list_(pa.string()), nullable=False),
                pa.field("source_hash", pa.string(), nullable=False),
                pa.field("retrieval_text", pa.string(), nullable=False),
                pa.field("decision_text", pa.string(), nullable=False),
                pa.field("clause_vector", vector, nullable=False),
                pa.field("decision_vector", vector, nullable=False),
            ]
        )

    def exists(self) -> bool:
        return self.table_name in self.db.list_tables().tables

    def write(
        self,
        precedents: list[ApprovedPrecedent],
        clause_vectors: list[list[float]],
        decision_vectors: list[list[float]],
        *,
        recreate: bool = False,
    ) -> dict[str, Any]:
        if not precedents:
            raise ValueError("at least one precedent is required")
        if not (len(precedents) == len(clause_vectors) == len(decision_vectors)):
            raise ValueError("precedents and embedding batches must align")
        rows = []
        for precedent, clause_vector, decision_vector in zip(
            precedents, clause_vectors, decision_vectors, strict=True
        ):
            if (
                len(clause_vector) != self.dimensions
                or len(decision_vector) != self.dimensions
            ):
                raise ValueError(
                    "precedent embedding dimensions do not match the table"
                )
            row = precedent.model_dump(mode="json")
            row.update(
                {
                    "retrieval_text": precedent.clause_retrieval_text(),
                    "decision_text": precedent.decision_retrieval_text(),
                    "clause_vector": clause_vector,
                    "decision_vector": decision_vector,
                }
            )
            rows.append(row)
        data = pa.Table.from_pylist(rows, schema=self._schema())
        if recreate or not self.exists():
            table = self.db.create_table(
                self.table_name,
                data=data,
                schema=self._schema(),
                mode="overwrite" if recreate else "create",
            )
        else:
            table = self.db.open_table(self.table_name)
            (
                table.merge_insert("precedent_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(data)
            )
        self._ensure_indexes(table)
        return {
            "table": self.table_name,
            "table_version": int(table.version),
            "precedents_written": len(rows),
        }

    @staticmethod
    def _ensure_indexes(table: Any) -> None:
        from lancedb.index import BTree, FTS, LabelList

        table.create_index("retrieval_text", config=FTS(), replace=True)
        for column in (
            "tenant_id",
            "contract_id",
            "clause_type",
            "agreement_type",
            "review_status",
            "policy_version",
        ):
            table.create_index(column, config=BTree(), replace=True)
        table.create_index("allowed_groups", config=LabelList(), replace=True)

    def retrieve(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        tenant_id: str,
        allowed_groups: list[str],
        clause_type: str,
        agreement_type: str,
        exclude_contract_id: str,
        candidate_k: int,
        top_k: int,
    ) -> tuple[list[RetrievedPrecedent], int | None, int]:
        if not self.exists():
            return [], None, 0
        if len(query_vector) != self.dimensions:
            raise ValueError("query embedding dimensions do not match the table")
        table = self.db.open_table(self.table_name)
        groups = ["all", *allowed_groups]
        group_values = ", ".join(_sql_string(group) for group in sorted(set(groups)))
        where = " AND ".join(
            (
                f"tenant_id = {_sql_string(tenant_id)}",
                "review_status = 'human_approved'",
                f"clause_type = {_sql_string(clause_type)}",
                f"agreement_type = {_sql_string(agreement_type)}",
                f"contract_id != {_sql_string(exclude_contract_id)}",
                f"array_has_any(allowed_groups, [{group_values}])",
            )
        )
        rows = (
            table.search(
                query_type="hybrid",
                vector_column_name="clause_vector",
                fts_columns="retrieval_text",
            )
            .vector(query_vector)
            .text(query_text)
            .where(where, prefilter=True)
            .limit(candidate_k)
            .to_list()
        )
        results = []
        for row in rows[:top_k]:
            payload = {
                name: row.get(name)
                for name in RetrievedPrecedent.model_fields
                if name != "relevance_score"
            }
            results.append(
                RetrievedPrecedent.model_validate(
                    {
                        **payload,
                        "relevance_score": max(
                            0.0, float(row.get("_relevance_score", 0.0))
                        ),
                    }
                )
            )
        return results, int(table.version), len(rows)
