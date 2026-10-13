from __future__ import annotations

from research_plotter.plotting.line import multi_line_plot


def convergence_plot(epoch, losses, *, output=None, formats=None, dpi=180, **kwargs):
    return multi_line_plot(epoch, losses, xlabel="epoch", ylabel="loss", title="Training convergence", output=output, formats=formats, dpi=dpi, **kwargs)
