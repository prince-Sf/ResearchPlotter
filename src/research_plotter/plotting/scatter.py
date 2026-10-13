from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from research_plotter.plotting.core import finalize
from research_plotter.plotting.theme import scientific_style


def scatter_plot(x, y, *, xlabel="x", ylabel="y", title=None, c=None, cmap="viridis", output: str | Path | None = None, formats=None, dpi=180, figsize=(6.4, 4.2), ax=None):
    own = ax is None
    with scientific_style(dpi):
        if own: figure, ax = plt.subplots(figsize=figsize)
        else: figure = ax.figure
        scatter_options = {"c": c}
        if c is not None: scatter_options["cmap"] = cmap
        artist = ax.scatter(np.asarray(x), np.asarray(y), **scatter_options)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if c is not None: figure.colorbar(artist, ax=ax)
        return finalize(figure, output if own else None, formats, dpi)


scatter = scatter_plot
