from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SHOWCASE_CONTRACT = """VENDOR SOFTWARE SERVICES AGREEMENT

This Vendor Software Services Agreement is between Example Customer GmbH and Example Cloud Systems Ltd.

Services. Vendor will provide the hosted analytics service described in the applicable order form.

Term and renewal. The initial term is twelve months. The agreement renews for successive twelve-month periods unless either party gives sixty days written notice.

Termination. Either party may terminate for material breach if the breach is not cured within thirty days after written notice. Customer may terminate for convenience on ninety days written notice.

Liability. Each party's aggregate liability is limited to fees paid during the preceding twelve months, except for fraud, wilful misconduct, confidentiality breaches, and data-protection obligations.

Indemnity. Vendor will defend and indemnify Customer against third-party intellectual-property claims arising from the service.

Data protection and security. The parties incorporate the Customer Data Processing Addendum. Vendor will maintain appropriate technical and organisational security measures and notify Customer of confirmed incidents without undue delay.

Service levels. Vendor targets 99.9 percent monthly availability and provides service credits for a missed target.

Payment. Invoices are payable within thirty days.

Governing law. This agreement is governed by the laws of Germany, and the courts of Berlin have exclusive jurisdiction.

Assignment. Neither party may assign this agreement without the other party's written consent, except in connection with a merger or sale of substantially all assets.
"""


def create_scan(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    y = 100
    for paragraph in SHOWCASE_CONTRACT.splitlines():
        if not paragraph:
            y += 24
            continue
        for line in textwrap.wrap(paragraph, width=92):
            draw.text((110, y), line, fill="black", font=font)
            y += 42
        y += 12
    image.save(path, "PDF", resolution=150.0)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a raster-only PDF for the live showcase")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/showcase/scanned-vendor-saas.pdf"),
    )
    args = parser.parse_args()
    print(create_scan(args.output).resolve())


if __name__ == "__main__":
    main()
