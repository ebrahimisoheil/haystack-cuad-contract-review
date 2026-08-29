from __future__ import annotations

import json
from pathlib import Path

import pytest

from contract_review_agent.app.config import Settings
from contract_review_agent.app.ingestion.batch import run_cuad_manifest
from contract_review_agent.app.ingestion.cuad import CuadIngestor
from contract_review_agent.app.ingestion.schemas import CuadManifest
from tests.fixtures import COMPLIANT_CONTRACT


def _qa(title: str, context: str, category: str, answer: str | None) -> dict:
    answers = (
        []
        if answer is None
        else [{"text": answer, "answer_start": context.index(answer)}]
    )
    return {
        "id": f"{title}__{category}",
        "question": f'Highlight contract language related to "{category}".',
        "is_impossible": answer is None,
        "answers": answers,
    }


def _write_cuad(path: Path, contracts: list[tuple[str, str]]) -> Path:
    data = []
    for title, context in contracts:
        data.append(
            {
                "title": title,
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            _qa(
                                title,
                                context,
                                "Governing Law",
                                "Delaware" if "Delaware" in context else None,
                            ),
                            _qa(
                                title,
                                context,
                                "Document Name",
                                "VENDOR SAAS AGREEMENT"
                                if "VENDOR SAAS AGREEMENT" in context
                                else None,
                            ),
                        ],
                    }
                ],
            }
        )
    path.write_text(json.dumps({"version": "aok_v1.0", "data": data}), encoding="utf-8")
    return path


def test_ingestion_is_bounded_reproducible_and_span_preserving(tmp_path: Path) -> None:
    source = _write_cuad(
        tmp_path / "CUADv1.json",
        [
            ("ALPHA-SERVICE AGREEMENT", "VENDOR SAAS AGREEMENT\nDelaware"),
            ("BETA-LICENSE AGREEMENT", "LICENSE AGREEMENT\nCalifornia"),
            ("GAMMA-HOSTING AGREEMENT", "VENDOR SAAS AGREEMENT\nDelaware"),
        ],
    )
    first = CuadIngestor(source).ingest(tmp_path / "subset-a", limit=2, seed=7)
    second = CuadIngestor(source).ingest(tmp_path / "subset-b", limit=2, seed=7)
    assert [item.title for item in first.contracts] == [
        item.title for item in second.contracts
    ]
    assert first.totals.available_contracts == 3
    assert first.totals.selected_contracts == 2
    for contract in first.contracts:
        context = Path(contract.text_path).read_text(encoding="utf-8")
        assert contract.context_sha256
        for label in contract.labels:
            for answer in label.answers:
                assert context[answer.answer_start : answer.answer_end] == answer.text
    persisted = CuadManifest.model_validate_json(
        (tmp_path / "subset-a" / "manifest.json").read_text(encoding="utf-8")
    )
    assert persisted.source_sha256 == first.source_sha256


def test_filters_on_agreement_type_and_positive_category(tmp_path: Path) -> None:
    source = _write_cuad(
        tmp_path / "CUAD_v1.json",
        [
            ("ALPHA-SERVICE AGREEMENT", "VENDOR SAAS AGREEMENT\nDelaware"),
            ("BETA-LICENSE AGREEMENT", "LICENSE AGREEMENT\nCalifornia"),
        ],
    )
    manifest = CuadIngestor(source).ingest(
        tmp_path / "filtered",
        agreement_type="service agreement",
        require_categories=["Governing Law"],
        limit=20,
    )
    assert [item.title for item in manifest.contracts] == ["ALPHA-SERVICE AGREEMENT"]


def test_local_full_release_pdf_is_matched(tmp_path: Path) -> None:
    title = "ALPHA-SERVICE AGREEMENT"
    source = _write_cuad(tmp_path / "CUAD_v1.json", [(title, "Delaware")])
    pdf = tmp_path / "full_contract_pdf" / "Part_I" / f"{title}.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4\n% fixture only")
    manifest = CuadIngestor(source).ingest(tmp_path / "matched", limit=1)
    assert manifest.contracts[0].pdf_path == str(pdf.resolve())
    assert manifest.contracts[0].review_source == str(pdf.resolve())
    assert manifest.totals.matched_pdfs == 1


def test_invalid_cuad_answer_offset_is_rejected(tmp_path: Path) -> None:
    payload = {
        "version": "aok_v1.0",
        "data": [
            {
                "title": "BROKEN-SERVICE AGREEMENT",
                "paragraphs": [
                    {
                        "context": "Delaware",
                        "qas": [
                            {
                                "id": "broken__Governing Law",
                                "question": "Governing law",
                                "is_impossible": False,
                                "answers": [{"text": "Delaware", "answer_start": 2}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    source = tmp_path / "CUADv1.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid answer span"):
        CuadIngestor(source).ingest(tmp_path / "broken", limit=1)


def test_ingested_manifest_runs_through_haystack_workflow(tmp_path: Path) -> None:
    source = _write_cuad(
        tmp_path / "CUADv1.json",
        [("DEMO-SERVICE AGREEMENT", COMPLIANT_CONTRACT)],
    )
    CuadIngestor(source).ingest(tmp_path / "reviewable", limit=1)
    batch = run_cuad_manifest(
        tmp_path / "reviewable" / "manifest.json",
        settings=Settings(mode="deterministic"),
    )
    assert batch["summary"]["contracts"] == 1
    assert batch["summary"]["completed"] == 1
    assert batch["runs"][0]["review"]["final_decision"] == "approved"
    assert batch["runs"][0]["ground_truth"]["positive_label_count"] == 2
    evaluation = batch["runs"][0]["review"]["cuad_evaluation"]
    assert evaluation["evaluated_categories"] == 1
    assert evaluation["true_positives"] == 1
    assert batch["summary"]["ground_truth_evaluation"]["category_recall"] == 1
