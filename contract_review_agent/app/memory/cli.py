from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..config import Settings
from ..model_registry import ModelRegistry
from .schemas import PrecedentCorpus
from .store import LanceContractMemory


def _load_corpus(path: Path) -> PrecedentCorpus:
    return PrecedentCorpus.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def seed(path: Path, *, mode: str, recreate: bool) -> dict[str, object]:
    settings = Settings.from_env(mode=mode)
    models = ModelRegistry(settings)
    corpus = _load_corpus(path)
    dimensions = int(models.models["embedding"].get("dimensions", 1024))
    store = LanceContractMemory(
        settings.memory_uri, settings.memory_table, dimensions=dimensions
    )
    clause_vectors = models.embed_texts(
        [item.clause_retrieval_text() for item in corpus.precedents],
        input_type="document",
    )
    clause_usage = models.consume_usage("embedding")
    decision_vectors = models.embed_texts(
        [item.decision_retrieval_text() for item in corpus.precedents],
        input_type="document",
    )
    decision_usage = models.consume_usage("embedding")
    result = store.write(
        corpus.precedents,
        clause_vectors,
        decision_vectors,
        recreate=recreate,
    )
    return {
        **result,
        "corpus_id": corpus.corpus_id,
        "embedding_model": models.model_name("embedding"),
        "embedding_input_tokens": int(clause_usage["input_tokens"])
        + int(decision_usage["input_tokens"]),
        "estimated_embedding_cost_usd": float(clause_usage["estimated_cost_usd"])
        + float(decision_usage["estimated_cost_usd"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the human-approved LanceDB contract memory."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed_parser = subparsers.add_parser("seed")
    seed_parser.add_argument("corpus", type=Path)
    seed_parser.add_argument(
        "--mode", choices=("deterministic", "live"), default="deterministic"
    )
    seed_parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    if args.command == "seed":
        print(
            json.dumps(
                seed(args.corpus, mode=args.mode, recreate=args.recreate), indent=2
            )
        )


if __name__ == "__main__":
    main()
