"""Reproducible artifact helpers for graph digitization and fit experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_plotter.analysis.equation_fitting import EquationFitResult
from research_plotter.analysis.graph_digitization import AxisCalibration, DigitizationResult, GraphImage, long_to_wide, validate_crop
from research_plotter.plotting.core import save_figure
from research_plotter.plotting.graph_image import plot_digitized_comparison, plot_digitized_data, plot_fit_overlay, plot_fit_residuals, plot_pixel_overlay
from research_plotter.utils.numerics import error_metrics


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")
    return path


def save_digitization_artifacts(
    source: GraphImage,
    result: DigitizationResult,
    *,
    calibration: AxisCalibration,
    crop: tuple[int, int, int, int],
    plot_region: tuple[int, int, int, int] | None = None,
    output_dir: str | Path,
    experiment_id: str | None = None,
    extraction_settings: dict[str, object] | None = None,
    source_context: dict[str, object] | None = None,
    exclusion_regions: tuple[tuple[int, int, int, int], ...] = (),
    calibration_provenance: str = "USER_PROVIDED",
    calibration_detection: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Save long/wide CSV, reviewed crop, figures, and provenance metadata."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    left, top, right, bottom = crop
    extraction_region = validate_crop(source.image.size, plot_region or result.plot_region_pixels or crop)
    region_image = source.image.crop(extraction_region)
    crop_path = destination / "plot_region.png"
    region_image.save(crop_path, format="PNG")
    long_path = destination / "digitized_data_long.csv"
    wide_path = destination / "digitized_data_wide.csv"
    result.data.to_csv(long_path, index=False)
    long_to_wide(result.data).to_csv(wide_path, index=False)
    figure_dir = destination / "figures"
    figure_dir.mkdir(exist_ok=True)
    data_figure = plot_digitized_data(result.data)
    data_figures = save_figure(data_figure, figure_dir / "digitized_data.png", formats=["png", "svg", "pdf"])
    overlay_figure = plot_pixel_overlay(source.image.crop(crop), result.data, crop_origin=(left, top), calibration=calibration)
    overlay_figures = save_figure(overlay_figure, figure_dir / "pixel_overlay.png", formats=["png", "svg", "pdf"])
    metadata = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "DIGITIZED",
        "source": {
            "name": source.source_name,
            "format": source.source_format,
            "page_number": source.page_number,
            "sha256": source.source_sha256,
            "dimensions_pixels": list(source.image.size),
            "provenance": "USER_PROVIDED",
        },
        "source_context": source_context or {},
        "crop_coordinates_pixels": {"left": left, "top": top, "right": right, "bottom": bottom},
        "crop_dimensions_pixels": list(source.image.crop(crop).size),
        "extraction_plot_region_pixels": {
            "left": extraction_region[0], "top": extraction_region[1],
            "right": extraction_region[2], "bottom": extraction_region[3],
        },
        "extraction_plot_region_dimensions_pixels": list(region_image.size),
        "exclusion_regions_pixels": [
            {"left": left, "top": top, "right": right, "bottom": bottom}
            for left, top, right, bottom in exclusion_regions
        ],
        "exclusion_policy": "matching pixels are removed; gaps are preserved and are not interpolated",
        "axis_calibration": calibration.to_dict(),
        "axis_calibration_provenance": calibration_provenance,
        "axis_calibration_detection": calibration_detection or {},
        "extraction_method": result.extraction_method,
        "extraction_settings": extraction_settings or {},
        "curves": [{"name": name, "point_count": int((result.data["curve_name"] == name).sum())} for name in result.curve_names],
        "point_count": int(len(result.data)),
        "data_structure": "long form CSV: one row per curve point with curve_name, pixel_x, pixel_y, x, y",
        "warnings": list(result.warnings),
        "figures": [str(path) for path in [*data_figures, *overlay_figures]],
    }
    metadata_path = _write_json(destination / "digitization_metadata.json", metadata)
    return {
        "crop": crop_path,
        "long_csv": long_path,
        "wide_csv": wide_path,
        "metadata": metadata_path,
        "data_figure": data_figures[0],
        "data_svg": data_figures[1],
        "data_pdf": data_figures[2],
        "pixel_overlay": overlay_figures[0],
        "pixel_overlay_svg": overlay_figures[1],
        "pixel_overlay_pdf": overlay_figures[2],
    }


def save_fit_artifacts(
    data: pd.DataFrame,
    fit: EquationFitResult,
    *,
    curve_name: str,
    output_dir: str | Path,
    xlabel: str = "x",
    ylabel: str = "y",
    crop_image=None,
    crop_origin: tuple[int, int] = (0, 0),
    calibration: AxisCalibration | None = None,
    fit_configuration: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Persist fitted values, residuals, equation, metrics, and plot files."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected = data[data["curve_name"] == curve_name].sort_values("x", kind="stable")
    if len(selected) != len(fit.x):
        raise ValueError("Fit result does not align with the selected digitized curve")
    selected_x = selected["x"].to_numpy(dtype=float)
    if not np.allclose(selected_x, fit.x, rtol=1e-10, atol=1e-12):
        raise ValueError("Fit result x coordinates do not match the selected digitized curve")
    reconstructed_y = np.asarray(fit.fitted_y, dtype=float) + np.asarray(fit.residuals, dtype=float)
    if reconstructed_y.shape != (len(selected),) or not np.allclose(
        selected["y"].to_numpy(dtype=float), reconstructed_y, rtol=1e-8, atol=1e-12,
    ):
        raise ValueError("Fit result residuals do not match the selected digitized observations")
    fit_data = selected[["curve_name", "pixel_x", "pixel_y", "x", "y"]].copy()
    fit_data["fitted_y"] = fit.fitted_y
    fit_data["residual"] = fit.residuals
    data_path = destination / "fit_data.csv"
    fit_data.to_csv(data_path, index=False)
    fit_json = _write_json(destination / "fit_result.json", {
        "experiment_id": (fit_configuration or {}).get("experiment_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": fit.status,
        "curve_name": curve_name,
        **fit.to_dict(include_arrays=False),
        "warnings": list(fit.warnings),
        "fitting_provenance": "ESTIMATED_FROM_DIGITIZED_IMAGE",
        "parameter_provenance": {name: ("ESTIMATED_FROM_DIGITIZED_IMAGE" if name in fit.uncertainties else "USER_PROVIDED_OR_FIXED") for name in fit.parameters},
        "fit_configuration": fit_configuration or {},
    })
    figure_dir = destination / "figures"
    figure_dir.mkdir(exist_ok=True)
    overlay = plot_fit_overlay(data, fit, curve_name=curve_name, xlabel=xlabel, ylabel=ylabel)
    overlay_paths = save_figure(overlay, figure_dir / "fit_overlay.png", formats=["png", "svg", "pdf"])
    residual = plot_fit_residuals(fit, xlabel=xlabel)
    residual_paths = save_figure(residual, figure_dir / "fit_residuals.png", formats=["png", "svg", "pdf"])
    pixel_paths: list[Path] = []
    if crop_image is not None and calibration is not None:
        pixel = plot_pixel_overlay(crop_image, data, crop_origin=crop_origin, calibration=calibration, fit=fit, curve_name=curve_name)
        pixel_paths = save_figure(pixel, figure_dir / "fit_pixel_overlay.png", formats=["png", "svg", "pdf"])
    _write_json(destination / "fit_figures.json", {"figures": [str(path) for path in [*overlay_paths, *residual_paths, *pixel_paths]], "curve_name": curve_name, "equation": fit.equation, "configuration": {"xlabel": xlabel, "ylabel": ylabel}, "fit_configuration": fit_configuration or {}})
    return {"data": data_path, "result": fit_json, "overlay_png": overlay_paths[0], "overlay_svg": overlay_paths[1], "overlay_pdf": overlay_paths[2], "residual_png": residual_paths[0], "residual_svg": residual_paths[1], "residual_pdf": residual_paths[2]}


def compare_digitized_reference(
    digitized: pd.DataFrame,
    simulation: pd.DataFrame,
    *,
    curve_name: str,
    state_name: str,
    reference_x_column: str = "x",
    reference_y_column: str = "y",
    simulation_x_column: str = "time",
) -> dict[str, object]:
    """Align a simulation with a digitized curve and compute cautious metrics."""
    selected = digitized[digitized["curve_name"] == curve_name].copy()
    if selected.empty:
        raise ValueError(f"No digitized points found for curve {curve_name!r}")
    if simulation_x_column not in simulation or state_name not in simulation:
        raise ValueError(f"Simulation must contain {simulation_x_column!r} and state {state_name!r}")
    x_ref = np.asarray(selected[reference_x_column], dtype=float)
    y_ref = np.asarray(selected[reference_y_column], dtype=float)
    x_sim = np.asarray(simulation[simulation_x_column], dtype=float)
    y_sim = np.asarray(simulation[state_name], dtype=float)
    finite_sim = np.isfinite(x_sim) & np.isfinite(y_sim)
    x_sim, y_sim = x_sim[finite_sim], y_sim[finite_sim]
    order = np.argsort(x_sim, kind="stable")
    x_sim, y_sim = x_sim[order], y_sim[order]
    if len(x_sim) < 2 or np.unique(x_sim).size < 2:
        raise ValueError("Simulation must contain at least two distinct finite time values")
    finite_ref = np.isfinite(x_ref) & np.isfinite(y_ref) & (x_ref >= x_sim[0]) & (x_ref <= x_sim[-1])
    x_ref, y_ref = x_ref[finite_ref], y_ref[finite_ref]
    if len(x_ref) < 2:
        raise ValueError("The simulation and digitized curve have fewer than two overlapping points")
    order = np.argsort(x_ref, kind="stable")
    x_ref, y_ref = x_ref[order], y_ref[order]
    simulated_at_reference = np.interp(x_ref, x_sim, y_sim)
    metrics = error_metrics(y_ref, simulated_at_reference)
    if not np.isfinite(metrics["correlation"]):
        metrics["correlation"] = None
    return {
        "status": "UNVERIFIED",
        "comparison_type": "DIGITIZED_REFERENCE_COMPARISON",
        "reference_source": "digitized published graph image",
        "curve_name": curve_name,
        "state_name": state_name,
        "interpolation": "linear interpolation of simulation at digitized x coordinates over common range",
        "compared_point_count": int(len(x_ref)),
        "x": x_ref.tolist(),
        "reference_y": y_ref.tolist(),
        "simulation_y": simulated_at_reference.tolist(),
        "metrics": metrics,
        "warnings": [
            "Digitized graph pixels are estimates, not original experimental/model data.",
            "Image resolution, axis calibration, curve segmentation, and interpolation contribute uncertainty.",
            "Agreement with a digitized figure is not exact-ground-truth validation or proof of reproduction.",
        ],
    }


def save_digitized_comparison(
    comparison: dict[str, object],
    digitized: pd.DataFrame,
    *,
    output_dir: str | Path,
    xlabel: str = "x",
    ylabel: str = "y",
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = _write_json(destination / "digitized_reference_comparison.json", comparison)
    data_path = destination / "digitized_reference_comparison.csv"
    pd.DataFrame({"x": comparison["x"], "reference_y": comparison["reference_y"], "simulation_y": comparison["simulation_y"]}).to_csv(data_path, index=False)
    figure = plot_digitized_comparison(digitized, comparison["x"], comparison["simulation_y"], curve_name=str(comparison["curve_name"]), xlabel=xlabel, ylabel=ylabel)
    paths = save_figure(figure, destination / "figures" / "digitized_reference_comparison.png", formats=["png", "svg", "pdf"])
    _write_json(destination / "digitized_reference_comparison_figure.json", {"figure_files": [str(path) for path in paths], "status": comparison["status"], "comparison_type": comparison["comparison_type"]})
    return {"report": result_path, "data": data_path, "png": paths[0], "svg": paths[1], "pdf": paths[2]}

