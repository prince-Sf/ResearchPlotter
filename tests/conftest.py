"""Shared optional source fixtures for tests that need local research PDFs."""

from pathlib import Path

import pytest


@pytest.fixture
def paper_1_pdf() -> Path:
    path = Path(__file__).parents[1] / "Papers" / "333.pdf"
    if not path.is_file():
        pytest.skip("Local Paper 1 PDF is optional and is not distributed with this checkout.")
    return path


@pytest.fixture
def paper_2_pdf() -> Path:
    path = Path(__file__).parents[1] / "Papers" / "fninf-20-1748481.pdf"
    if not path.is_file():
        pytest.skip("Local Paper 2 PDF is optional and is not distributed with this checkout.")
    return path
