from __future__ import annotations

from collections.abc import Mapping

from research_plotter.plotting.line import multi_line_plot


def comparison_plot(time, trajectories: Mapping[str, object], *, xlabel="time", ylabel="state", title=None, output=None, formats=None, dpi=180, **kwargs):
    return multi_line_plot(time, trajectories, xlabel=xlabel, ylabel=ylabel, title=title, output=output, formats=formats, dpi=dpi, **kwargs)
