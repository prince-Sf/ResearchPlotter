from __future__ import annotations

import matplotlib.pyplot as plt

from research_plotter.plotting.core import finalize
from research_plotter.plotting.theme import scientific_style


def multipanel_plot(plotters, *, ncols=2, output=None, formats=None, dpi=180, figsize=None):
    nrows = (len(plotters) + ncols - 1) // ncols
    with scientific_style(dpi):
        figure, axes = plt.subplots(nrows, ncols, figsize=figsize or (6.4 * ncols, 4.2 * nrows), squeeze=False)
        for index, plotter in enumerate(plotters): plotter(axes.flat[index])
        for axis in axes.flat[len(plotters):]: axis.remove()
        return finalize(figure, output, formats, dpi)


multi_panel_plot = multipanel_plot
