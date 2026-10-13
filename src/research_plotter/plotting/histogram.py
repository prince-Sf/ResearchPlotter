from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from research_plotter.plotting.core import finalize
from research_plotter.plotting.theme import scientific_style


def histogram(values, *, bins=30, xlabel="value", ylabel="count", title=None, output: str | Path | None = None, formats=None, dpi=180, figsize=(6.4, 4.2), ax=None):
    own = ax is None
    with scientific_style(dpi):
        if own: figure, ax = plt.subplots(figsize=figsize)
        else: figure = ax.figure
        ax.hist(values, bins=bins)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        return finalize(figure, output if own else None, formats, dpi)


hist_plot = histogram
