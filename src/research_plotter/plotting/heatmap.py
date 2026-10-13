from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from research_plotter.plotting.core import finalize
from research_plotter.plotting.theme import scientific_style


def heatmap(matrix, *, xlabels=None, ylabels=None, title=None, cmap="coolwarm", annotate=False, output: str | Path | None = None, formats=None, dpi=180, figsize=(6.4, 5.0), ax=None):
    own = ax is None
    with scientific_style(dpi):
        if own: figure, ax = plt.subplots(figsize=figsize)
        else: figure = ax.figure
        image = ax.imshow(np.asarray(matrix), aspect="auto", cmap=cmap)
        figure.colorbar(image, ax=ax)
        if xlabels is not None: ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
        if ylabels is not None: ax.set_yticks(range(len(ylabels)), ylabels)
        ax.set_title(title)
        if annotate:
            values = np.asarray(matrix)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]): ax.text(j, i, f"{values[i, j]:.2g}", ha="center", va="center")
        return finalize(figure, output if own else None, formats, dpi)


heatmap_plot = heatmap
