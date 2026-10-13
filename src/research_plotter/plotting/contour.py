from __future__ import annotations

import matplotlib.pyplot as plt
from research_plotter.plotting.core import finalize
from research_plotter.plotting.theme import scientific_style


def contour_plot(x, y, z, *, xlabel="x", ylabel="y", title=None, levels=20, output=None, formats=None, dpi=180, figsize=(6.4, 4.8), ax=None):
    own = ax is None
    with scientific_style(dpi):
        if own: figure, ax = plt.subplots(figsize=figsize)
        else: figure = ax.figure
        image = ax.contourf(x, y, z, levels=levels)
        figure.colorbar(image, ax=ax)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        return finalize(figure, output if own else None, formats, dpi)


contour = contour_plot
