from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_plotter.plotting.core import finalize
from research_plotter.plotting.theme import scientific_style


def line_plot(x: Iterable[float], y: Iterable[float], *, xlabel: str = "x", ylabel: str = "y", title: str | None = None,
              label: str | None = None, output: str | Path | None = None, formats: list[str] | None = None, dpi: int = 180,
              figsize: tuple[float, float] = (6.4, 4.2), logx: bool = False, logy: bool = False, xlim=None, ylim=None, ax=None):
    own = ax is None
    with scientific_style(dpi, 10):
        if own:
            figure, ax = plt.subplots(figsize=figsize)
        else:
            figure = ax.figure
        ax.plot(np.asarray(x), np.asarray(y), label=label)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if logx: ax.set_xscale("log")
        if logy: ax.set_yscale("log")
        if xlim is not None: ax.set_xlim(*xlim)
        if ylim is not None: ax.set_ylim(*ylim)
        if label: ax.legend()
        return finalize(figure, output if own else None, formats, dpi)


def multi_line_plot(x: Iterable[float], series: Mapping[str, Iterable[float]] | pd.DataFrame, *, xlabel: str = "x", ylabel: str = "y", title: str | None = None, output: str | Path | None = None, formats=None, dpi: int = 180, figsize=(6.4, 4.2), ax=None):
    frame = series if isinstance(series, pd.DataFrame) else pd.DataFrame(series)
    own = ax is None
    with scientific_style(dpi, 10):
        if own: figure, ax = plt.subplots(figsize=figsize)
        else: figure = ax.figure
        for name in frame.columns: ax.plot(x, frame[name], label=str(name))
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend()
        return finalize(figure, output if own else None, formats, dpi)


line = line_plot
multi_line = multi_line_plot
