"""CUAD dataset discovery, normalization, and batch handoff."""

from .cuad import CuadIngestor, download_official_cuad
from .schemas import CuadAnswer, CuadContract, CuadLabel, CuadManifest

__all__ = [
    "CuadAnswer",
    "CuadContract",
    "CuadIngestor",
    "CuadLabel",
    "CuadManifest",
    "download_official_cuad",
]
