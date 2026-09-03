from __future__ import annotations

from pathlib import Path

import yaml

from contract_review_agent.app.config import Settings
from contract_review_agent.app.memory.schemas import PrecedentCorpus
from contract_review_agent.app.memory.store import LanceContractMemory
from contract_review_agent.app.model_registry import ModelRegistry
from contract_review_agent.app.pipeline import run_review


def _seed(memory_uri: Path) -> tuple[LanceContractMemory, ModelRegistry]:
    settings = Settings(mode="deterministic", memory_uri=str(memory_uri))
    models = ModelRegistry(settings)
    corpus_path = (
        Path(__file__).parents[1]
        / "examples"
        / "precedents"
        / "vendor_saas_approved.yaml"
    )
    corpus = PrecedentCorpus.model_validate(
        yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
    )
    clause_vectors = models.embed_texts(
        [item.clause_retrieval_text() for item in corpus.precedents],
        input_type="document",
    )
    decision_vectors = models.embed_texts(
        [item.decision_retrieval_text() for item in corpus.precedents],
        input_type="document",
    )
    store = LanceContractMemory(
        str(memory_uri), "contract_precedents", dimensions=1024
    )
    store.write(corpus.precedents, clause_vectors, decision_vectors, recreate=True)
    return store, models


def test_lancedb_retrieval_enforces_scope_and_same_contract_exclusion(
    tmp_path: Path,
) -> None:
    store, models = _seed(tmp_path / "memory")
    query = "California law is required, outside the preferred jurisdictions."
    vector = models.embed_texts([query], input_type="query")[0]

    hits, version, candidates = store.retrieve(
        query_text=query,
        query_vector=vector,
        tenant_id="showcase",
        allowed_groups=["all"],
        clause_type="governing_law",
        agreement_type="Vendor SaaS Agreement",
        exclude_contract_id="current-contract",
        candidate_k=10,
        top_k=3,
    )
    excluded, _, excluded_candidates = store.retrieve(
        query_text=query,
        query_vector=vector,
        tenant_id="showcase",
        allowed_groups=["all"],
        clause_type="governing_law",
        agreement_type="Vendor SaaS Agreement",
        exclude_contract_id="approved-data-msa-004",
        candidate_k=10,
        top_k=3,
    )
    wrong_tenant, _, wrong_tenant_candidates = store.retrieve(
        query_text=query,
        query_vector=vector,
        tenant_id="another-tenant",
        allowed_groups=["all"],
        clause_type="governing_law",
        agreement_type="Vendor SaaS Agreement",
        exclude_contract_id="current-contract",
        candidate_k=10,
        top_k=3,
    )

    assert version is not None
    assert candidates == 1
    assert [item.precedent_id for item in hits] == ["saas-governing-law-001"]
    assert excluded == []
    assert excluded_candidates == 0
    assert wrong_tenant == []
    assert wrong_tenant_candidates == 0


def test_shadow_observes_but_retrieve_exposes_precedent_evidence(
    deviating_contract: Path, tmp_path: Path
) -> None:
    memory_uri = tmp_path / "memory"
    _seed(memory_uri)

    shadow = run_review(
        str(deviating_contract),
        Settings(
            mode="deterministic", memory_mode="shadow", memory_uri=str(memory_uri)
        ),
    )
    retrieve = run_review(
        str(deviating_contract),
        Settings(
            mode="deterministic",
            memory_mode="retrieve",
            memory_uri=str(memory_uri),
        ),
    )

    assert shadow["memory"]["selected_precedents"] == []
    assert shadow["memory"]["shadow_precedents"]
    assert shadow["metrics"]["memory_selected_count"] == 0
    assert retrieve["memory"]["selected_precedents"]
    assert retrieve["memory"]["shadow_precedents"] == []
    assert retrieve["metrics"]["memory_selected_count"] >= 3
    assert retrieve["metrics"]["memory_query_count"] >= 3
    assert "precedent_retriever" in retrieve["metrics"]["branch_path"]
