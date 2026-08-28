from __future__ import annotations

from pathlib import Path

import pytest

from .fixtures import COMPLIANT_CONTRACT, DEVIATING_CONTRACT, make_native_pdf, make_scanned_pdf


@pytest.fixture(autouse=True)
def isolate_test_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests from creating runs in a developer's live Witdem instance."""

    config = Path(__file__).parent / "fixtures" / "witdem.test.yaml"
    monkeypatch.setenv("CONTRACT_REVIEW_WITDEM_CONFIG", str(config))


@pytest.fixture
def native_contract(tmp_path: Path) -> Path:
    return make_native_pdf(tmp_path / "clean-native.pdf", COMPLIANT_CONTRACT)


@pytest.fixture
def scanned_contract(tmp_path: Path) -> Path:
    return make_scanned_pdf(tmp_path / "clean-scanned.pdf", COMPLIANT_CONTRACT)


@pytest.fixture
def deviating_contract(tmp_path: Path) -> Path:
    return make_native_pdf(tmp_path / "deviating-native.pdf", DEVIATING_CONTRACT)


@pytest.fixture
def retry_contract(tmp_path: Path) -> Path:
    base = COMPLIANT_CONTRACT.replace(
        "[DPA]\nThe parties incorporate the Customer Data Processing Addendum when personal data is processed.\n",
        "",
    )
    path = make_native_pdf(tmp_path / "retry-native.pdf", base)
    path.with_suffix(path.suffix + ".retry.txt").write_text(
        "[DPA]\nThe parties incorporate the Customer Data Processing Addendum when personal data is processed.",
        encoding="utf-8",
    )
    return path
