from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl


@contextmanager
def scientific_style(dpi: int = 180, font_size: float = 10):
    params = {"figure.dpi": dpi, "savefig.dpi": dpi, "font.size": font_size, "axes.grid": True, "grid.alpha": .25,
              "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}
    with mpl.rc_context(params):
        yield
