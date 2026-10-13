import os
import tempfile
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "research_plotter_matplotlib"))
matplotlib.use("Agg", force=True)

from research_plotter.plotting.comparison import comparison_plot
from research_plotter.plotting.convergence import convergence_plot
from research_plotter.plotting.core import save_figure
from research_plotter.plotting.heatmap import heatmap
from research_plotter.plotting.line import line, line_plot, multi_line, multi_line_plot
from research_plotter.plotting.scatter import scatter, scatter_plot
from research_plotter.plotting.scatter3d import scatter3d, scatter3d_plot
from research_plotter.plotting.sensitivity import sensitivity_plot
from research_plotter.plotting.graph_image import plot_digitized_comparison, plot_digitized_data, plot_fit_overlay, plot_fit_residuals, plot_pixel_overlay
from research_plotter.plotting.theme import scientific_style

__all__ = ["comparison_plot", "convergence_plot", "heatmap", "line", "line_plot", "multi_line", "multi_line_plot", "save_figure", "scatter", "scatter_plot", "scatter3d", "scatter3d_plot", "sensitivity_plot", "scientific_style", "plot_digitized_comparison", "plot_digitized_data", "plot_fit_overlay", "plot_fit_residuals", "plot_pixel_overlay"]
