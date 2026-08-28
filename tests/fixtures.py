from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


COMPLIANT_CONTRACT = """VENDOR SAAS AGREEMENT
CUSTOMER: Acme Example Corporation
VENDOR: Nimbus Software LLC
EFFECTIVE DATE: 2026-01-15

[TERM]
The initial term is 12 months. The agreement automatically renews for 12 months unless either party gives at least 30 days notice.
[TERMINATION]
Either party may terminate for cause. Customer may terminate for convenience upon 60 days notice.
[LIABILITY]
Aggregate liability is capped at fees paid in the prior 12 months.
[INDEMNITY]
Vendor will indemnify Customer for third-party intellectual property claims; the obligation is subject to the liability framework.
[GOVERNING_LAW]
Delaware
[ASSIGNMENT]
Neither party may assign this agreement without prior written consent, except to an affiliate.
[SECURITY]
Vendor will maintain documented administrative, technical, and organizational safeguards.
[DPA]
The parties incorporate the Customer Data Processing Addendum when personal data is processed.
[SLA]
Service availability is 99.9%. Customer receives service credits if availability is missed.
[PAYMENT]
Invoices are payable Net 30. No advance prepayment is required.
"""

DEVIATING_CONTRACT = """VENDOR SAAS AGREEMENT
CUSTOMER: Acme Example Corporation
VENDOR: Risky Cloud Inc.
EFFECTIVE DATE: 2026-02-01

[TERM]
The initial term is 12 months. The agreement automatically renews for 12 months unless Customer gives 30 days notice.
[TERMINATION]
Either party may terminate for cause upon 30 days notice.
[LIABILITY]
Aggregate liability is capped at fees paid in the prior 12 months.
[INDEMNITY]
Vendor indemnifies Customer for third-party intellectual property claims subject to the liability framework.
[GOVERNING_LAW]
California
[ASSIGNMENT]
Neither party may assign without prior written consent.
[SECURITY]
Vendor maintains administrative, technical, and organizational safeguards.
[DPA]
The approved Data Processing Addendum applies to personal data.
[SLA]
Service availability is 99.9% with service credits for missed availability.
[PAYMENT]
Invoices are payable Net 15.
"""


def make_native_pdf(path: Path, text: str) -> Path:
    writer = canvas.Canvas(str(path), pagesize=letter)
    y = 760
    for original in text.splitlines():
        for line in textwrap.wrap(original, width=95) or [""]:
            if y < 50:
                writer.showPage()
                y = 760
            writer.drawString(36, y, line)
            y -= 14
    writer.save()
    return path


def make_scanned_pdf(path: Path, text: str) -> Path:
    pages: list[dict[str, object]] = []
    chunks = [text.splitlines()[index:index + 22] for index in range(0, len(text.splitlines()), 22)]
    images: list[Image.Image] = []
    for page_number, lines in enumerate(chunks, 1):
        image = Image.new("RGB", (1275, 1650), "white")
        draw = ImageDraw.Draw(image)
        draw.multiline_text((60, 60), "\n".join(lines), fill="black", spacing=12)
        images.append(image)
        pages.append({"page": page_number, "text": "\n".join(lines)})
    images[0].save(path, save_all=True, append_images=images[1:])
    path.with_suffix(path.suffix + ".ocr.json").write_text(json.dumps(pages, indent=2), encoding="utf-8")
    return path
