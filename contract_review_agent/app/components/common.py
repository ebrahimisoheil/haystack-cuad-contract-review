from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from typing import Any

from ..config import Settings
from ..model_registry import ModelRegistry


Context = dict[str, Any]


class Stage:
    model_role = "deterministic"

    def __init__(self, settings: Settings, model_registry: ModelRegistry):
        self.settings = settings
        # Haystack's default component serializer resolves constructor arguments
        # by their original names. Witdem inspects this metadata to identify model
        # boundaries, so retain the registry under both the public constructor name
        # and the shorter internal alias used by the stages.
        self.model_registry = model_registry
        self.models = model_registry

    def to_dict(self) -> dict[str, Any]:
        """Expose safe component metadata without serializing runtime clients or secrets."""
        return {
            "type": f"{type(self).__module__}.{type(self).__name__}",
            "init_parameters": {},
        }

    def execute(
        self,
        context: Context,
        name: str,
        operation: Callable[[Context], None],
        *,
        status: str = "ok",
        attempt: int | None = None,
    ) -> Context:
        started = time.perf_counter()
        try:
            operation(context)
        except Exception as exc:
            context.setdefault("errors", []).append(
                {"stage": name, "type": type(exc).__name__, "message": str(exc)}
            )
            context["processing_failed"] = True
            status = "failed"
        elapsed = (time.perf_counter() - started) * 1000
        usage = self.models.consume_usage(self.model_role)
        context.setdefault("stage_metrics", []).append(
            {
                "stage": name,
                "model": self.models.model_name(self.model_role),
                "latency_ms": round(elapsed, 3),
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "estimated_cost_usd": usage["estimated_cost_usd"],
                "attempt": attempt or context.get("extraction_attempt", 1),
                "status": status,
            }
        )
        context.setdefault("branch_path", []).append(name)
        return context


def evidence(
    context: Context,
    label: str,
    text: str | None,
    *,
    page: int | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    if page is None and text:
        page = page_for_text(context.get("pages", []), text)
    return {
        "page": page,
        "text": text,
        "clause_label": label,
        "extraction_method": context.get("extraction_method", "unresolved"),
        "confidence": confidence if confidence is not None else (0.92 if text else 0.0),
    }


def page_for_text(pages: list[dict[str, Any]], snippet: str) -> int | None:
    needle = " ".join(snippet.lower().split())[:80]
    for page in pages:
        haystack = " ".join(str(page.get("text", "")).lower().split())
        if needle and needle in haystack:
            return int(page["page"])
    snippet_tokens = re_tokens(snippet)
    if len(snippet_tokens) >= 8:
        expected = Counter(snippet_tokens)
        best_page: int | None = None
        best_coverage = 0.0
        for page in pages:
            available = Counter(re_tokens(str(page.get("text", ""))))
            overlap = sum((expected & available).values())
            coverage = overlap / max(1, sum(expected.values()))
            if coverage > best_coverage:
                best_coverage = coverage
                best_page = int(page["page"])
        if best_coverage >= 0.78:
            return best_page
    return None


def re_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())
