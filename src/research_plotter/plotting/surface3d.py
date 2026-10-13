from __future__ import annotations

import matplotlib.pyplot as plt
from research_plotter.plotting.core import finalize
from research_plotter.plotting.theme import scientific_style


def surface_plot(x, y, z, *, xlabel="x", ylabel="y", zlabel="z", title=None, output=None, formats=None, dpi=180, figsize=(6.4, 4.8)):
    with scientific_style(dpi):
        figure = plt.figure(figsize=figsize)
        ax = figure.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, cmap="viridis")
        ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title)
        return finalize(figure, output, formats, dpi)


surface3d = surface_plot
