from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from haystack import component
from pypdf import PdfReader

from .common import Context, Stage


@component
class NativePDFExtractor(Stage):
    model_role = "local-native-pdf"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def extract(ctx: Context) -> None:
            path = Path(ctx["source"])
            if path.suffix.lower() in {".txt", ".md"}:
                pages = [{"page": 1, "text": path.read_text(encoding="utf-8")}]
            else:
                reader = PdfReader(path)
                pages = [
                    {"page": index, "text": (page.extract_text() or "").strip()}
                    for index, page in enumerate(reader.pages, 1)
                ]
            ctx["pages"] = pages
            ctx["raw_text"] = "\n\n".join(page["text"] for page in pages if page["text"])
            ctx["extraction_method"] = "native_pdf_text"
            ctx["extraction_confidence"] = ctx.get("quality", {}).get("confidence", 1.0)

        return {"context": self.execute(context, "native_pdf_extractor", extract)}


@component
class MistralDocumentExtractor(Stage):
    model_role = "vision"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def extract(ctx: Context) -> None:
            path = Path(ctx["source"])
            pages = self.models.extract_document(path)
            method = "mistral_document_vision"
            if pages is None:
                sidecar = path.with_suffix(path.suffix + ".ocr.json")
                if not sidecar.exists():
                    raise FileNotFoundError(
                        f"deterministic vision mode requires page sidecar: {sidecar}"
                    )
                pages = json.loads(sidecar.read_text(encoding="utf-8"))
                method = "mistral_document_vision_deterministic_fixture"
            if not isinstance(pages, list) or not pages:
                raise ValueError("Mistral extraction produced no pages")
            ctx["pages"] = pages
            ctx["raw_text"] = "\n\n".join(str(page["text"]) for page in pages)
            ctx["extraction_method"] = method
            ctx["extraction_confidence"] = 0.9

        return {"context": self.execute(context, "mistral_document_extractor", extract)}


@component
class FocusedReExtractor(Stage):
    model_role = "text"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def retry(ctx: Context) -> None:
            ctx["retries"] += 1
            ctx["extraction_attempt"] += 1
            path = Path(ctx["source"])
            retry_sidecar = path.with_suffix(path.suffix + ".retry.txt")
            if retry_sidecar.exists():
                focused = retry_sidecar.read_text(encoding="utf-8")
                ctx["raw_text"] = f"{ctx.get('raw_text', '')}\n\n{focused}".strip()
                ctx.setdefault("pages", []).append(
                    {"page": len(ctx.get("pages", [])) + 1, "text": focused}
                )
            ctx["extraction_confidence"] = min(
                0.98, float(ctx.get("extraction_confidence", 0.0)) + 0.12
            )

        return {
            "context": self.execute(
                context,
                "focused_re_extractor",
                retry,
                status="retry",
                attempt=context.get("extraction_attempt", 1) + 1,
            )
        }
