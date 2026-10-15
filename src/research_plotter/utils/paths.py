from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any


def project_root(start: str | Path | None = None) -> Path:
    current = Path(start or __file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate project root (pyproject.toml)")


def contained_path(root: str | Path, *parts: str | Path) -> Path:
    """Resolve a child path and reject traversal or symlinks escaping ``root``."""
    base = Path(root).expanduser().resolve()
    destination = base.joinpath(*parts).resolve()
    if destination != base and base not in destination.parents:
        raise ValueError("Path must remain inside the selected output directory")
    return destination


def validate_contained_tree(root: str | Path) -> Path:
    """Reject existing symlinks below an output root that resolve outside it."""
    base = Path(root).expanduser().resolve()
    if not base.exists():
        return base
    for entry in base.rglob("*"):
        is_junction = getattr(entry, "is_junction", None)
        if entry.is_symlink() or (is_junction is not None and is_junction()):
            contained_path(base, entry.relative_to(base))
    return base


def output_path(*parts: str | Path, root: str | Path | None = None) -> Path:
    project = Path(root).expanduser().resolve() if root else project_root()
    unresolved_base = project / "outputs"
    base = unresolved_base.resolve()
    is_junction = getattr(unresolved_base, "is_junction", None)
    if (
        unresolved_base.is_symlink()
        or (is_junction is not None and is_junction())
        or base.parent != project
        or base.name.lower() != "outputs"
    ):
        raise ValueError("The outputs directory must remain a real directory inside the project root")
    try:
        destination = contained_path(base, *parts)
    except ValueError as error:
        raise ValueError("Output path must remain inside the outputs directory") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Re-resolve after creating parents so a pre-existing or newly observed
    # symlink cannot silently redirect the returned path outside outputs/.
    try:
        return contained_path(base, *parts)
    except ValueError as error:
        raise ValueError("Output path must remain inside the outputs directory") from error


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
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename).rstrip(" .") or "upload.bin"
    stem, suffix = Path(filename).stem, Path(filename).suffix
    if stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        stem = f"_{stem}"
    filename = f"{stem[:100]}{suffix[:20]}"

    content = bytes(upload.getbuffer())
    full_digest = hashlib.sha256(content).hexdigest()
    root = Path(output_root).expanduser().resolve()
    directory = contained_path(root, "uploaded", category)
    directory.mkdir(parents=True, exist_ok=True)
    directory = contained_path(root, "uploaded", category)

    # Prefer the short content address, falling back to the full digest if
    # there is a truncated-digest collision. Exclusive creation avoids
    # clobbering a file that appeared after the existence check.
    for digest in (full_digest[:16], full_digest):
        destination = directory / f"{digest}_{filename}"
        if destination.is_symlink():
            raise ValueError("Refusing to read or overwrite a symlink upload destination")
        contained_path(root, "uploaded", category, destination.name)
        try:
            with destination.open("xb") as handle:
                handle.write(content)
            return destination
        except FileExistsError:
            if destination.is_symlink():
                raise ValueError("Refusing to read or overwrite a symlink upload destination")
            if destination.is_file() and destination.read_bytes() == content:
                return destination
    raise FileExistsError("Could not persist upload without overwriting existing data")
