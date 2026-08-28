from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from haystack import component
from pypdf import PdfReader

from .common import Context, Stage


@component
class InputLoader(Stage):
    model_role = "local-input"

    @component.output_types(context=Context)
    def run(self, source: str) -> dict[str, Context]:
        path = Path(source).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"contract source does not exist: {path}")
        context: Context = {
            "source": str(path),
            "contract_id": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            "started_at": time.perf_counter(),
            "branch_path": [],
            "stage_metrics": [],
            "errors": [],
            "retries": 0,
            "extraction_attempt": 1,
        }
        return {"context": self.execute(context, "input_loader", lambda _: None)}


@component
class DocumentQualityClassifier(Stage):
    model_role = "local-quality-detection"

    @component.output_types(context=Context, extraction_mode=str)
    def run(self, context: Context) -> dict[str, Any]:
        def classify(ctx: Context) -> None:
            path = Path(ctx["source"])
            if path.suffix.lower() in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8")
                ctx["quality"] = {"native_characters": len(text), "confidence": 1.0}
                ctx["extraction_mode"] = "native"
                return
            if path.suffix.lower() != ".pdf":
                ctx["quality"] = {"native_characters": 0, "confidence": 0.0}
                ctx["extraction_mode"] = "vision"
                return
            reader = PdfReader(path)
            page_text = [(page.extract_text() or "").strip() for page in reader.pages[:5]]
            chars = sum(len(text) for text in page_text)
            populated = sum(bool(text) for text in page_text)
            coverage = populated / max(1, min(5, len(reader.pages)))
            confidence = min(1.0, chars / 600.0) * coverage
            ctx["quality"] = {
                "native_characters": chars,
                "page_coverage": coverage,
                "confidence": round(confidence, 3),
            }
            ctx["extraction_mode"] = "native" if chars >= 120 and coverage >= 0.6 else "vision"

        context = self.execute(context, "document_quality_classifier", classify)
        context["branch_path"].append(f"extraction_route:{context['extraction_mode']}")
        return {"context": context, "extraction_mode": context["extraction_mode"]}
