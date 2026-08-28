from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schemas import CuadAnswer, CuadContract, CuadLabel, CuadManifest, CuadTotals


OFFICIAL_ARCHIVE_URL = "https://raw.githubusercontent.com/The-Atticus-Project/cuad/main/data.zip"
OFFICIAL_PROJECT_URL = "https://github.com/TheAtticusProject/cuad"
ATTRIBUTION = (
    "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review, "
    "Hendrycks, Burns, Chen, and Ball (NeurIPS 2021)."
)
JSON_NAMES = ("CUADv1.json", "CUAD_v1.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug[:120] or "contract"


def _match_key(value: str) -> str:
    name = Path(value).name
    if Path(name).suffix.lower() in {".pdf", ".txt"}:
        name = Path(name).stem
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _agreement_type_hint(title: str) -> str | None:
    pieces = re.split(r"[-_]", title)
    candidates = [piece.strip() for piece in pieces if "agreement" in piece.lower() or "contract" in piece.lower()]
    return candidates[-1].title() if candidates else None


def _category_from_qa(qa: dict[str, Any]) -> str:
    annotation_id = str(qa.get("id", ""))
    if "__" in annotation_id:
        return annotation_id.rsplit("__", 1)[1].strip()
    question = str(qa.get("question", ""))
    quoted = re.search(r'"([^"]+)"', question)
    return quoted.group(1).strip() if quoted else "Uncategorized"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def download_official_cuad(cache_dir: Path, *, force: bool = False, timeout: float = 120.0) -> Path:
    """Download and safely extract the official CUAD annotation archive.

    The archive is opt-in, cached, hashed, and only the known JSON payloads are
    extracted. Original PDFs can separately be supplied through a local full
    CUAD release directory.
    """

    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "data.zip"
    if force or not archive.exists():
        part = cache_dir / "data.zip.part"
        request = urllib.request.Request(OFFICIAL_ARCHIVE_URL, headers={"User-Agent": "cuad-ingestor/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as target:
                shutil.copyfileobj(response, target)
            if not zipfile.is_zipfile(part):
                raise ValueError("downloaded CUAD archive is not a valid ZIP file")
            part.replace(archive)
        finally:
            part.unlink(missing_ok=True)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"cached CUAD archive is invalid: {archive}")
    extracted: Path | None = None
    with zipfile.ZipFile(archive) as bundle:
        for name in JSON_NAMES:
            if name in bundle.namelist():
                target = cache_dir / name
                with bundle.open(name) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                extracted = target
                break
    if extracted is None:
        raise ValueError("official CUAD archive does not contain CUADv1.json")
    _atomic_json(
        cache_dir / "download-metadata.json",
        {
            "source_url": OFFICIAL_ARCHIVE_URL,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "archive_sha256": _sha256_file(archive),
            "annotation_sha256": _sha256_file(extracted),
        },
    )
    return extracted


class CuadIngestor:
    """Normalize an official or local CUAD release into a bounded manifest."""

    def __init__(self, source: Path):
        self.source = source.expanduser().resolve()
        self.annotation_path, self.release_root = self._discover(self.source)

    @staticmethod
    def _discover(source: Path) -> tuple[Path, Path]:
        if source.is_file():
            if source.suffix.lower() != ".json":
                raise ValueError("CUAD source file must be CUADv1.json or CUAD_v1.json")
            return source, source.parent
        if not source.is_dir():
            raise FileNotFoundError(f"CUAD source does not exist: {source}")
        candidates: list[Path] = []
        for name in JSON_NAMES:
            candidates.extend(source.rglob(name))
        if not candidates:
            raise FileNotFoundError(f"no CUADv1.json or CUAD_v1.json found under {source}")
        annotation = sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]
        return annotation, annotation.parent

    def _load(self) -> tuple[str, list[dict[str, Any]]]:
        payload = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("CUAD annotation file must use SQuAD-style {version, data} structure")
        version = str(payload.get("version", "CUAD v1"))
        records: list[dict[str, Any]] = []
        for item in payload["data"]:
            title = str(item.get("title", "")).strip()
            paragraphs = item.get("paragraphs")
            if not title or not isinstance(paragraphs, list) or not paragraphs:
                raise ValueError("each CUAD contract requires a title and at least one paragraph")
            contexts = []
            labels: list[CuadLabel] = []
            offset_base = 0
            for paragraph in paragraphs:
                context = str(paragraph.get("context", ""))
                contexts.append(context)
                for qa in paragraph.get("qas", []):
                    answers = []
                    for raw_answer in qa.get("answers", []):
                        answer_text = str(raw_answer.get("text", ""))
                        local_start = int(raw_answer.get("answer_start", -1))
                        if local_start < 0 or context[local_start:local_start + len(answer_text)] != answer_text:
                            raise ValueError(f"invalid answer span in {title}: {qa.get('id')}")
                        start = offset_base + local_start
                        answers.append(
                            CuadAnswer(text=answer_text, answer_start=start, answer_end=start + len(answer_text))
                        )
                    labels.append(
                        CuadLabel(
                            annotation_id=str(qa.get("id", "")),
                            category=_category_from_qa(qa),
                            question=str(qa.get("question", "")),
                            is_impossible=bool(qa.get("is_impossible", not answers)),
                            answers=answers,
                        )
                    )
                offset_base += len(context) + 2
            records.append({"title": title, "context": "\n\n".join(contexts), "labels": labels})
        return version, records

    def _document_index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for suffix in ("*.pdf", "*.PDF"):
            for path in sorted(self.release_root.rglob(suffix)):
                index.setdefault(_match_key(path.name), path.resolve())
        return index

    def ingest(
        self,
        output_dir: Path,
        *,
        limit: int = 20,
        seed: int = 42,
        agreement_type: str | None = None,
        require_categories: list[str] | None = None,
        allow_large: bool = False,
    ) -> CuadManifest:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if limit > 100 and not allow_large:
            raise ValueError("limit above 100 requires allow_large=True")
        version, records = self._load()
        available = len(records)
        categories = {value.casefold() for value in (require_categories or [])}
        if agreement_type:
            needle = agreement_type.casefold()
            records = [record for record in records if needle in record["title"].casefold()]
        if categories:
            records = [
                record
                for record in records
                if categories <= {label.category.casefold() for label in record["labels"] if label.answers}
            ]
        records.sort(key=lambda item: item["title"])
        random.Random(seed).shuffle(records)
        selected = records[:limit]
        if not selected:
            raise ValueError("CUAD filters selected no contracts")

        output_dir = output_dir.expanduser().resolve()
        texts_dir = output_dir / "contracts"
        texts_dir.mkdir(parents=True, exist_ok=True)
        pdf_index = self._document_index()
        contracts: list[CuadContract] = []
        for record in selected:
            context = record["context"]
            digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
            contract_id = digest[:16]
            text_path = texts_dir / f"{_safe_name(record['title'])}-{contract_id[:8]}.txt"
            text_path.write_text(context, encoding="utf-8")
            pdf_path = pdf_index.get(_match_key(record["title"]))
            labels: list[CuadLabel] = record["labels"]
            contracts.append(
                CuadContract(
                    contract_id=contract_id,
                    title=record["title"],
                    agreement_type_hint=_agreement_type_hint(record["title"]),
                    context_sha256=digest,
                    character_count=len(context),
                    annotation_count=len(labels),
                    positive_label_count=sum(bool(label.answers) for label in labels),
                    labels=labels,
                    text_path=str(text_path),
                    pdf_path=str(pdf_path) if pdf_path else None,
                    review_source=str(pdf_path or text_path),
                )
            )

        manifest = CuadManifest(
            dataset_version=version,
            source_url=OFFICIAL_PROJECT_URL,
            source_path=str(self.annotation_path),
            source_sha256=_sha256_file(self.annotation_path),
            attribution=ATTRIBUTION,
            created_at=datetime.now(UTC),
            random_seed=seed,
            selection={
                "limit": limit,
                "agreement_type": agreement_type,
                "require_categories": sorted(require_categories or []),
                "allow_large": allow_large,
            },
            totals=CuadTotals(
                available_contracts=available,
                selected_contracts=len(contracts),
                annotations=sum(contract.annotation_count for contract in contracts),
                positive_labels=sum(contract.positive_label_count for contract in contracts),
                matched_pdfs=sum(contract.pdf_path is not None for contract in contracts),
            ),
            contracts=contracts,
        )
        _atomic_json(output_dir / "manifest.json", manifest.model_dump(mode="json"))
        return manifest
