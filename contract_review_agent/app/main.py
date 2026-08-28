from __future__ import annotations

import argparse
import json

from .config import Settings
from .pipeline import run_review


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a vendor SaaS contract")
    parser.add_argument("source", help="PDF, image, or plain text contract path")
    parser.add_argument("--mode", choices=["deterministic", "live"], default=None)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    result = run_review(args.source, Settings.from_env(mode=args.mode))
    rendered = json.dumps(result, indent=2)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
