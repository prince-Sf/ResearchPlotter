from pathlib import Path

import numpy as np

from research_plotter.plotting.line import line_plot

time = np.linspace(0, 10, 101)
figure = line_plot(time, np.sin(time), xlabel="time", ylabel="sin(time)", title="Basic scientific line plot")
figure.savefig(Path("outputs/figures/basic_plot.png"), bbox_inches="tight")
