"""The src-layout package must be importable through its installation metadata."""

from pathlib import Path

import research_plotter


def test_research_plotter_import_resolves_package_source():
    source = Path(research_plotter.__file__).resolve()
    assert source.name == "__init__.py"
    assert source.parent.name == "research_plotter"
    assert source.parent.parent.name == "src"
