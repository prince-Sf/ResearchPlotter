from __future__ import annotations

from research_plotter.plotting.comparison import comparison_plot


def parameter_sweep_plot(time, trajectories, *, parameter="parameter", output=None, formats=None, dpi=180, **kwargs):
    return comparison_plot(time, trajectories, title=f"Parameter sweep: {parameter}", output=output, formats=formats, dpi=dpi, **kwargs)


sweep_plot = parameter_sweep_plot
