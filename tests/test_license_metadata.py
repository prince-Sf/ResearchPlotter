from importlib.metadata import metadata
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_mit_license_is_shipped_and_metadata_agrees():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    package_metadata = metadata("research-plotter")

    assert license_text.startswith("MIT License\n\nCopyright (c) 2026 Safwaan\n")
    assert "Permission is hereby granted, free of charge" in license_text
    assert "The above copyright notice and this permission notice" in license_text
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in license_text
    assert package_metadata.get("License-Expression") == "MIT"
    assert "License-File: LICENSE" in package_metadata.as_string()


def test_readme_license_notes_do_not_extend_to_external_research_material():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    papers_readme = ROOT / "Papers" / "README.md"

    assert "[MIT License](LICENSE)" in readme
    assert "third-party dependencies retain their respective licenses" in readme.lower()
    assert "does not override those terms" in readme
    assert "Papers/README.md" in readme
    assert papers_readme.is_file()
    assert "redistribution terms" in papers_readme.read_text(encoding="utf-8").lower()
