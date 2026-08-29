from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from contract_review_agent.app.config import Settings
from contract_review_agent.app.pipeline import run_review
from tests.fixtures import (
    COMPLIANT_CONTRACT,
    DEVIATING_CONTRACT,
    make_native_pdf,
    make_scanned_pdf,
)


def main() -> None:
    settings = Settings(mode="deterministic")
    with TemporaryDirectory(prefix="contract-review-demo-") as directory:
        root = Path(directory)
        inputs = {
            "clean_native": make_native_pdf(root / "clean-native.pdf", COMPLIANT_CONTRACT),
            "clean_scanned": make_scanned_pdf(root / "clean-scanned.pdf", COMPLIANT_CONTRACT),
            "deviating_native": make_native_pdf(root / "deviating-native.pdf", DEVIATING_CONTRACT),
            "high_risk_native": make_native_pdf(
                root / "high-risk-native.pdf",
                DEVIATING_CONTRACT.replace(
                    "Aggregate liability is capped at fees paid in the prior 12 months.",
                    "Vendor liability is unlimited.",
                ),
            ),
        }
        retry_text = COMPLIANT_CONTRACT.replace(
            "[DPA]\nThe parties incorporate the Customer Data Processing Addendum when personal data is processed.\n",
            "",
        )
        retry_path = make_native_pdf(root / "retry-native.pdf", retry_text)
        retry_path.with_suffix(retry_path.suffix + ".retry.txt").write_text(
            "[DPA]\nThe parties incorporate the Customer Data Processing Addendum when personal data is processed.",
            encoding="utf-8",
        )
        inputs["focused_retry"] = retry_path
        summaries = {}
        for name, source in inputs.items():
            result = run_review(str(source), settings)
            summaries[name] = {
                "success": result["outcome"]["review_completed"],
                "decision": result["final_decision"],
                "deviations": [item["clause"] for item in result["deviations"]],
                "review_areas": result["review_areas"],
                "domain_reviews": result["domain_reviews"],
                "retries": result["metrics"]["retries"],
                "latency_ms": result["metrics"]["total_runtime_ms"],
                "tokens": result["metrics"]["total_input_tokens"] + result["metrics"]["total_output_tokens"],
                "cost_usd": result["metrics"]["estimated_cost_usd"],
                "branch_path": result["metrics"]["branch_path"],
            }
        print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
