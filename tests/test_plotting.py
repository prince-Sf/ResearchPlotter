from pathlib import Path

import numpy as np

from research_plotter.plotting.core import save_figure
from research_plotter.plotting.heatmap import heatmap
from research_plotter.plotting.line import line_plot
from research_plotter.plotting.scatter import scatter_plot
from research_plotter.plotting.scatter3d import scatter3d_plot


def test_plotters_write_files(tmp_path: Path):
    x = np.linspace(0, 1, 5)
    figure = line_plot(x, x**2)
    assert save_figure(figure, tmp_path / "line.png", formats=["png", "svg"])[0].exists()
    assert scatter_plot(x, x**2)
    assert heatmap(np.eye(3))
    assert scatter3d_plot(x, x**2, x**3)
