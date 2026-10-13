from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt


def add_inset(figure, plotter: Callable, *, bounds=(.58, .55, .35, .35), **kwargs):
    """Add an inset axis and let a caller render arbitrary data into it."""
    axis = figure.add_axes(bounds)
    plotter(axis, **kwargs)
    return axis
