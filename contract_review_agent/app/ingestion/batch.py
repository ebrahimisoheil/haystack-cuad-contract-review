from __future__ import annotations

import hashlib
import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import Settings
from ..pipeline import run_review
from .evaluation import aggregate_cuad_evaluations
from .schemas import CuadManifest


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def run_cuad_manifest(
    manifest_path: Path,
    *,
    settings: Settings | None = None,
    output_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Review a bounded CUAD manifest through the existing Haystack graph."""

    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    manifest_path = manifest_path.expanduser().resolve()
    manifest = CuadManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if limit is not None and limit < 1:
        raise ValueError("batch limit must be at least 1")
    contracts = manifest.contracts[:limit] if limit is not None else manifest.contracts
    if not contracts:
        raise ValueError("CUAD manifest contains no contracts to review")
    active_settings = settings or Settings.from_env()
    runs = []
    for contract in contracts:
        text_path = Path(contract.text_path)
        if not text_path.exists() or _sha256_text(text_path) != contract.context_sha256:
            raise ValueError(f"CUAD materialized text failed integrity validation: {text_path}")
        result = run_review(
            contract.review_source,
            active_settings,
            ground_truth=[label.model_dump(mode="json") for label in contract.labels],
        )
        runs.append(
            {
                "cuad_contract_id": contract.contract_id,
                "title": contract.title,
                "agreement_type_hint": contract.agreement_type_hint,
                "review_source": contract.review_source,
                "ground_truth": {
                    "annotation_count": contract.annotation_count,
                    "positive_label_count": contract.positive_label_count,
                    "positive_categories": sorted(
                        label.category for label in contract.labels if label.answers
                    ),
                },
                "review": result,
            }
        )
    evaluations = [
        run["review"]["cuad_evaluation"]
        for run in runs
        if run["review"].get("cuad_evaluation") is not None
    ]
    payload = {
        "dataset": manifest.dataset,
        "dataset_version": manifest.dataset_version,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "started_at": started_at,
        "mode": active_settings.mode,
        "summary": {
            "contracts": len(runs),
            "completed": sum(run["review"]["outcome"]["review_completed"] for run in runs),
            "objective_met": sum(run["review"]["outcome"]["objective_met"] for run in runs),
            "processing_failed": sum(run["review"]["final_decision"] == "processing_failed" for run in runs),
            "total_deviations": sum(len(run["review"]["deviations"]) for run in runs),
            "total_retries": sum(run["review"]["metrics"]["retries"] for run in runs),
            "total_runtime_ms": round((time.perf_counter() - started) * 1000, 3),
            "model_cost_usd": round(
                sum(run["review"]["metrics"]["estimated_cost_usd"] for run in runs), 8
            ),
            "ground_truth_evaluation": aggregate_cuad_evaluations(evaluations),
        },
        "runs": runs,
    }
    if output_path:
        _write_json(output_path.expanduser().resolve(), payload)
    return payload
