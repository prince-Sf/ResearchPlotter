from __future__ import annotations

from research_plotter.plotting.line import line_plot


def sensitivity_plot(time, sensitivity, *, parameter="parameter", output=None, formats=None, dpi=180, **kwargs):
    return line_plot(time, sensitivity, xlabel="time", ylabel=f"log sensitivity ({parameter})", output=output, formats=formats, dpi=dpi, **kwargs)
