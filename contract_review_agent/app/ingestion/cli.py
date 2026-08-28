from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import Settings
from .batch import run_cuad_manifest
from .cuad import CuadIngestor, download_official_cuad


def _ingest(args: argparse.Namespace) -> None:
    if args.download:
        source = download_official_cuad(Path(args.cache_dir), force=args.force)
    else:
        source = Path(args.source)
    manifest = CuadIngestor(source).ingest(
        Path(args.output),
        limit=args.limit,
        seed=args.seed,
        agreement_type=args.agreement_type,
        require_categories=args.require_category,
        allow_large=args.allow_large,
    )
    print(
        json.dumps(
            {
                "manifest": str(Path(args.output).expanduser().resolve() / "manifest.json"),
                "totals": manifest.totals.model_dump(),
                "selection": manifest.selection,
            },
            indent=2,
        )
    )


def _review(args: argparse.Namespace) -> None:
    result = run_cuad_manifest(
        Path(args.manifest),
        settings=Settings.from_env(mode=args.mode),
        output_path=Path(args.output) if args.output else None,
        limit=args.limit,
    )
    print(json.dumps(result["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="CUAD ingestion and review tools")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Create a bounded, reproducible CUAD manifest")
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="Local CUAD JSON file or release directory")
    source.add_argument("--download", action="store_true", help="Download official annotation archive")
    ingest.add_argument("--cache-dir", default="data/cuad-cache")
    ingest.add_argument("--force", action="store_true", help="Refresh the official archive")
    ingest.add_argument("--output", required=True, help="Output directory for manifest and contract text")
    ingest.add_argument("--limit", type=int, default=20)
    ingest.add_argument("--seed", type=int, default=42)
    ingest.add_argument("--agreement-type")
    ingest.add_argument("--require-category", action="append", default=[])
    ingest.add_argument("--allow-large", action="store_true")
    ingest.set_defaults(handler=_ingest)

    review = commands.add_parser("review", help="Run a CUAD manifest through the Haystack workflow")
    review.add_argument("manifest")
    review.add_argument("--mode", choices=["deterministic", "live"], default="deterministic")
    review.add_argument("--limit", type=int)
    review.add_argument("--output", help="Write full batch audit JSON")
    review.set_defaults(handler=_review)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
