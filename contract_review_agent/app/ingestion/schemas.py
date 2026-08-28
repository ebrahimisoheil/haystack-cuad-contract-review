from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class IngestionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CuadAnswer(IngestionModel):
    text: str
    answer_start: int = Field(ge=0)
    answer_end: int = Field(gt=0)

    @model_validator(mode="after")
    def end_follows_start(self) -> "CuadAnswer":
        if self.answer_end <= self.answer_start:
            raise ValueError("answer_end must be greater than answer_start")
        if self.answer_end - self.answer_start != len(self.text):
            raise ValueError("answer offsets must exactly span answer text")
        return self


class CuadLabel(IngestionModel):
    annotation_id: str
    category: str
    question: str
    is_impossible: bool
    answers: list[CuadAnswer]

    @model_validator(mode="after")
    def impossible_has_no_answers(self) -> "CuadLabel":
        if self.is_impossible and self.answers:
            raise ValueError("impossible CUAD labels cannot contain answers")
        return self


class CuadContract(IngestionModel):
    contract_id: str
    title: str
    agreement_type_hint: str | None
    context_sha256: str
    character_count: int = Field(ge=0)
    annotation_count: int = Field(ge=0)
    positive_label_count: int = Field(ge=0)
    labels: list[CuadLabel]
    text_path: str
    pdf_path: str | None = None
    review_source: str


class CuadTotals(IngestionModel):
    available_contracts: int = Field(ge=0)
    selected_contracts: int = Field(ge=0)
    annotations: int = Field(ge=0)
    positive_labels: int = Field(ge=0)
    matched_pdfs: int = Field(ge=0)


class CuadManifest(IngestionModel):
    manifest_version: str = "1.0"
    dataset: str = "Contract Understanding Atticus Dataset (CUAD)"
    dataset_version: str
    source_url: str
    source_path: str
    source_sha256: str
    license: str = "CC BY 4.0"
    attribution: str
    created_at: datetime
    random_seed: int
    selection: dict[str, Any]
    totals: CuadTotals
    contracts: list[CuadContract]
