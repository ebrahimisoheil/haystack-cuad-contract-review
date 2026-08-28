from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import Settings
from .pipeline import run_review


def _json_get(url: str) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=5) as response:  # noqa: S310 - local URL is user-configurable
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Witdem dashboard is not reachable at {url}: {exc}") from exc


def _listed_runs(dashboard_url: str) -> list[dict[str, Any]]:
    payload = _json_get(f"{dashboard_url.rstrip('/')}/api/v1/runs?page=1&page_size=50")
    items = payload.get("items")
    if not isinstance(items, list):
        raise RuntimeError("Witdem runs API returned an unexpected response")
    return [item for item in items if isinstance(item, dict)]


def _wait_for_new_run(
    dashboard_url: str,
    existing_ids: set[str],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    candidate_id: str | None = None
    while time.monotonic() < deadline:
        for run in _listed_runs(dashboard_url):
            execution_id = str(run.get("execution_id") or "")
            if execution_id and execution_id not in existing_ids:
                candidate_id = candidate_id or execution_id
                if execution_id == candidate_id and str(run.get("status") or "").casefold() != "running":
                    return run
        time.sleep(1)
    if candidate_id:
        raise RuntimeError(
            f"Witdem observed {candidate_id}, but it did not reach a terminal state "
            f"within {timeout_seconds:.0f} seconds"
        )
    raise RuntimeError(f"Witdem did not expose a new run within {timeout_seconds:.0f} seconds")


def _provider_set(run: dict[str, Any]) -> set[str]:
    raw = run.get("providers") or run.get("provider") or ""
    return {item.strip().casefold() for item in str(raw).split(",") if item.strip()}


def _verification(run: dict[str, Any], expected_providers: Iterable[str]) -> dict[str, Any]:
    observed = _provider_set(run)
    expected = {provider.casefold() for provider in expected_providers}
    checks = {
        "run_observed": bool(run.get("execution_id")),
        "terminal_status_observed": str(run.get("status") or "").casefold() in {"completed", "failed"},
        "providers_observed": expected <= observed,
        "model_calls_observed": int(run.get("model_calls") or 0) > 0,
        "tokens_observed": float(run.get("total_tokens") or 0) > 0,
        "cost_observed": run.get("known_cost") is not None or run.get("measured_cost") is not None,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_providers": sorted(expected),
        "observed_providers": sorted(observed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one live contract review and prove that Witdem observed the complete showcase"
    )
    parser.add_argument("source", help="Scanned PDF to review; use a scan to exercise Mistral OCR")
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8501")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--expect-provider",
        action="append",
        dest="expected_providers",
        help="Provider required in Witdem; repeat as needed",
    )
    args = parser.parse_args()

    expected = args.expected_providers or ["deepseek", "mistral", "openai"]
    before = {str(run.get("execution_id")) for run in _listed_runs(args.dashboard_url)}
    result = run_review(args.source, Settings.from_env(mode="live"))
    observed = _wait_for_new_run(args.dashboard_url, before, timeout_seconds=args.timeout)
    verification = _verification(observed, expected)
    execution_id = str(observed.get("execution_id"))
    report = {
        "review_decision": result.get("final_decision"),
        "witdem_run_url": f"{args.dashboard_url.rstrip('/')}/runs/{execution_id}",
        "witdem": {
            "execution_id": execution_id,
            "status": observed.get("status"),
            "providers": observed.get("providers") or observed.get("provider"),
            "models": observed.get("models") or observed.get("model"),
            "model_calls": observed.get("model_calls"),
            "total_tokens": observed.get("total_tokens"),
            "measured_cost": observed.get("known_cost") or observed.get("measured_cost"),
        },
        "verification": verification,
    }
    print(json.dumps(report, indent=2))
    if not verification["passed"]:
        failed = [name for name, passed in verification["checks"].items() if not passed]
        raise SystemExit(f"Witdem showcase verification failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
