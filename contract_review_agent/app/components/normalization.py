from __future__ import annotations

import json
import re
from haystack import component

from .common import Context, Stage


@component
class TextNormalizer(Stage):
    model_role = "text"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def normalize(ctx: Context) -> None:
            response = self.models.call_json("text",
                system="Normalize contract text without changing meaning. Return JSON.",
                prompt=json.dumps({"text": ctx.get("raw_text", "")[:120_000]}),
            )
            if response and isinstance(response.get("normalized_text"), str):
                ctx["normalized_text"] = response["normalized_text"]
            else:
                lines = [re.sub(r"[ \t]+", " ", line).strip() for line in ctx.get("raw_text", "").splitlines()]
                ctx["normalized_text"] = "\n".join(line for line in lines if line)

        return {"context": self.execute(context, "text_normalizer", normalize)}


@component
class AgreementClassifier(Stage):
    model_role = "text"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def classify(ctx: Context) -> None:
            response = self.models.call_json("text",
                system="Classify the agreement type. Return agreement_type and confidence as JSON.",
                prompt=ctx.get("normalized_text", "")[:12_000],
            )
            if response and response.get("agreement_type"):
                ctx["agreement_type"] = str(response["agreement_type"])
                ctx["agreement_type_confidence"] = float(response.get("confidence", 0.8))
            else:
                text = ctx.get("normalized_text", "").lower()
                is_saas = "saas" in text or "software as a service" in text
                ctx["agreement_type"] = "Vendor SaaS Agreement" if is_saas else "Commercial Agreement"
                ctx["agreement_type_confidence"] = 0.96 if is_saas else 0.62

        return {"context": self.execute(context, "agreement_classifier", classify)}


@component
class MetadataExtractor(Stage):
    model_role = "text"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def extract(ctx: Context) -> None:
            text = ctx.get("normalized_text", "")
            response = self.models.call_json("text",
                system=(
                    "Extract customer, vendor, and effective_date. Use null when unresolved. "
                    "Return a JSON object."
                ),
                prompt=text[:20_000],
            )
            if response:
                ctx["parties"] = {
                    "customer": response.get("customer"),
                    "vendor": response.get("vendor"),
                }
                ctx["effective_date"] = response.get("effective_date")
                return

            def find(pattern: str) -> str | None:
                match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
                return match.group(1).strip() if match else None

            ctx["parties"] = {
                "customer": find(r"^CUSTOMER:\s*(.+)$"),
                "vendor": find(r"^VENDOR:\s*(.+)$"),
            }
            ctx["effective_date"] = find(r"^EFFECTIVE DATE:\s*(.+)$")

        return {"context": self.execute(context, "metadata_extractor", extract)}
