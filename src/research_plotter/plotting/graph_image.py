"""Matplotlib figures for graph digitization and curve-fit review."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from research_plotter.analysis.equation_fitting import EquationFitResult
from research_plotter.analysis.graph_digitization import AxisCalibration
from research_plotter.plotting.core import finalize
from research_plotter.plotting.theme import scientific_style


def plot_digitized_data(data: pd.DataFrame, *, xlabel: str = "x", ylabel: str = "y", title: str = "Digitized graph data", output: str | Path | None = None, formats=None, dpi: int = 180):
    """Plot one or more extracted curves as point sequences."""
    if not {"curve_name", "x", "y"}.issubset(data.columns):
        raise ValueError("Digitized plot data needs curve_name, x, and y columns")
    with scientific_style(dpi, 10):
        figure, ax = plt.subplots(figsize=(6.4, 4.2))
        for name, frame in data.groupby("curve_name", sort=False):
            frame = frame.sort_values("x", kind="stable")
            # A scatter view preserves gaps caused by exclusions or missing
            # pixels; joining the points would visually invent observations.
            ax.scatter(frame["x"], frame["y"], s=10, label=str(name))
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if data["curve_name"].nunique() > 1:
            ax.legend()
        return finalize(figure, output, formats, dpi)


def plot_fit_overlay(
    data: pd.DataFrame,
    fit: EquationFitResult,
    *,
    curve_name: str,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "Digitized data and fitted candidate",
    output: str | Path | None = None,
    formats=None,
    dpi: int = 180,
):
    """Compare the digitized observations with the fitted empirical candidate."""
    selected = data[data["curve_name"] == curve_name].sort_values("x", kind="stable")
    if selected.empty:
        raise ValueError(f"No digitized rows found for curve {curve_name!r}")
    if len(selected) != len(fit.x):
        raise ValueError("Fit result and selected digitized curve have different point counts")
    with scientific_style(dpi, 10):
        figure, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.scatter(selected["x"], selected["y"], s=18, alpha=0.8, label="Digitized pixels")
        _plot_fit_segments(ax, selected["pixel_x"].to_numpy(dtype=float), fit.x, fit.fitted_y)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend()
        return finalize(figure, output, formats, dpi)


def plot_pixel_overlay(
    crop_image: Image.Image,
    data: pd.DataFrame,
    *,
    crop_origin: tuple[int, int],
    calibration: AxisCalibration,
    fit: EquationFitResult | None = None,
    curve_name: str | None = None,
    output: str | Path | None = None,
    formats=None,
    dpi: int = 180,
):
    """Show extracted pixels and a fitted curve mapped back over the source crop."""
    left, top = crop_origin
    selected = data if curve_name is None else data[data["curve_name"] == curve_name]
    with scientific_style(dpi, 10):
        figure, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.imshow(crop_image.convert("RGB"), origin="upper")
        ax.scatter(selected["pixel_x"] - left, selected["pixel_y"] - top, s=8, color="#d62728", label="Digitized pixels")
        if fit is not None:
            px, py = calibration.data_to_pixel(fit.x, fit.fitted_y)
            _plot_fit_segments(ax, selected.sort_values("x", kind="stable")["pixel_x"].to_numpy(dtype=float), px - left, py - top, color="#2ca02c", label="Fitted candidate mapped to pixels")
        ax.set(xlabel="Image x pixel", ylabel="Image y pixel", title="Digitization review against source crop")
        ax.set_xlim(0, crop_image.width)
        ax.set_ylim(crop_image.height, 0)
        ax.legend()
        return finalize(figure, output, formats, dpi)


def _plot_fit_segments(ax, source_pixel_x, x_values, y_values, *, color=None, label="Fitted candidate") -> None:
    """Plot predicted values without drawing a line across digitization gaps."""
    source_pixel_x = np.asarray(source_pixel_x, dtype=float)
    x_values, y_values = np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)
    if not (len(source_pixel_x) == len(x_values) == len(y_values)):
        raise ValueError("Fit predictions and digitized pixel coordinates must have matching lengths")
    if len(x_values) == 0:
        return
    # A skipped source-image column may be a crop boundary, mask, or exclusion;
    # never bridge a larger gap here. Duplicate columns can be retained when
    # the raster resolves distinct vertical trace clusters.
    segment_ids = np.r_[0, np.cumsum(np.abs(np.diff(source_pixel_x)) > 1.5)]
    first = True
    for segment_id in np.unique(segment_ids):
        selected = segment_ids == segment_id
        if selected.sum() >= 2:
            ax.plot(x_values[selected], y_values[selected], linewidth=1.8 if color is None else 1.5, color=color, label=label if first else None)
            first = False
        elif selected.sum() == 1:
            ax.scatter(x_values[selected], y_values[selected], s=18, color=color, label=label if first else None)
            first = False


def plot_fit_residuals(fit: EquationFitResult, *, xlabel: str = "x", ylabel: str = "Residual", output: str | Path | None = None, formats=None, dpi: int = 180):
    """Plot residuals in data coordinates with a zero reference line."""
    with scientific_style(dpi, 10):
        figure, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.scatter(fit.x, fit.residuals, s=16)
        ax.set(xlabel=xlabel, ylabel=ylabel, title="Fit residuals")
        return finalize(figure, output, formats, dpi)


def plot_digitized_comparison(data: pd.DataFrame, simulated_x, simulated_y, *, curve_name: str, xlabel: str = "x", ylabel: str = "y", output: str | Path | None = None, formats=None, dpi: int = 180):
    """Overlay a simulated model trajectory with a digitized paper-figure curve."""
    selected = data[data["curve_name"] == curve_name].sort_values("x", kind="stable")
    with scientific_style(dpi, 10):
        figure, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.scatter(selected["x"], selected["y"], s=18, label="Digitized published figure")
        ax.plot(np.asarray(simulated_x), np.asarray(simulated_y), linewidth=1.7, label="Simulated model")
        ax.set(xlabel=xlabel, ylabel=ylabel, title="Model vs digitized reference · not ground truth")
        ax.legend()
        return finalize(figure, output, formats, dpi)

