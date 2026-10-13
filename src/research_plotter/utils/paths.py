from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def project_root(start: str | Path | None = None) -> Path:
    current = Path(start or __file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate project root (pyproject.toml)")


def output_path(*parts: str, root: str | Path | None = None) -> Path:
    destination = (Path(root) if root else project_root()) / "outputs"
    destination = destination.joinpath(*parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def persist_uploaded_file(upload: Any, category: str, *, output_root: str | Path = "outputs") -> Path:
    """Save an uploaded file under a content-addressed name without overwriting.

    Streamlit's UploadedFile and small test doubles both expose ``getbuffer``.
    The category is intentionally a single directory name, not an arbitrary
    path, so an upload cannot choose its destination.
    """
    if not category or Path(category).name != category or category in {".", ".."}:
        raise ValueError("Upload category must be a single safe directory name")
    raw_name = str(getattr(upload, "name", "upload.bin")).replace("\\", "/")
    filename = Path(raw_name).name or "upload.bin"
    content = bytes(upload.getbuffer())
    digest = hashlib.sha256(content).hexdigest()[:16]
    destination = Path(output_root) / "uploaded" / category / f"{digest}_{filename}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != content:
        # A truncated digest collision should preserve both inputs.
        destination = destination.with_name(f"{digest}_{hashlib.sha256(content).hexdigest()}_{filename}")
    if not destination.exists():
        destination.write_bytes(content)
    return destination
