from __future__ import annotations

import os
from pathlib import Path
import tempfile

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "research_plotter_matplotlib"))
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from research_plotter.plotting.theme import scientific_style


def save_figure(figure, path: str | Path, formats: list[str] | None = None, dpi: int = 180) -> list[Path]:
    destination = Path(path)
    formats = formats or [destination.suffix.lstrip(".") or "png"]
    stem = destination.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for extension in formats:
        output = stem.with_suffix(f".{extension.lstrip('.')}")
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        saved.append(output)
    plt.close(figure)
    return saved


def finalize(figure, output: str | Path | None = None, formats: list[str] | None = None, dpi: int = 180):
    if output:
        save_figure(figure, output, formats, dpi)
    return figure
