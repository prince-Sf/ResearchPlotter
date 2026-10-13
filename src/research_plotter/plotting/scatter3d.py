from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from research_plotter.plotting.core import finalize
from research_plotter.plotting.theme import scientific_style


def scatter3d_plot(x, y, z, *, xlabel="x", ylabel="y", zlabel="z", title=None, c=None, cmap="viridis", output: str | Path | None = None, formats=None, dpi=180, figsize=(6.4, 4.8), ax=None):
    """Render an arbitrary 3-D point cloud using Matplotlib's 3-D axes."""
    own = ax is None
    with scientific_style(dpi):
        if own:
            figure = plt.figure(figsize=figsize)
            ax = figure.add_subplot(111, projection="3d")
        else:
            figure = ax.figure
        options = {"c": c}
        if c is not None:
            options["cmap"] = cmap
        artist = ax.scatter(np.asarray(x), np.asarray(y), np.asarray(z), **options)
        ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title)
        if c is not None:
            figure.colorbar(artist, ax=ax, shrink=.7)
        return finalize(figure, output if own else None, formats, dpi)


scatter3d = scatter3d_plot
