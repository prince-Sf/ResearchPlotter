from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from research_plotter.plotting.core import finalize
from research_plotter.plotting.theme import scientific_style


def radar_chart(labels, values, *, title=None, output=None, formats=None, dpi=180, figsize=(5, 5)):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.append(angles, angles[0])
    data = np.append(np.asarray(values, dtype=float), values[0])
    with scientific_style(dpi):
        figure = plt.figure(figsize=figsize)
        axis = figure.add_subplot(111, polar=True)
        axis.plot(angles, data)
        axis.fill(angles, data, alpha=.15)
        axis.set_xticks(angles[:-1], labels)
        axis.set_title(title)
        return finalize(figure, output, formats, dpi)


radar_plot = radar_chart
radar = radar_chart
