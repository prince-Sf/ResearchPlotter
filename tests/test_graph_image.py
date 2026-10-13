from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from research_plotter.analysis.equation_fitting import discover_equation_candidates, fit_equation
from research_plotter.analysis.graph_digitization import AxisCalibration, detect_curve_colors, digitize_color_curves, load_graph_image, long_to_wide, render_pdf_page
from research_plotter.models.formula_spec import parse_equation_text
from research_plotter.pipelines.graph_image_analysis import compare_digitized_reference, save_digitization_artifacts, save_fit_artifacts
from research_plotter.plotting.graph_image import plot_digitized_data, plot_fit_overlay
from research_plotter.ui.graph_image_page import equation_for_formula_plotter, handoff_to_dataset_analyzer, handoff_to_formula_plotter
from research_plotter.ui.paper_reproduction_page import _handoff_to_graph_analyzer


def _synthetic_chart(curves: list[tuple[str, object]], *, y_limits=(0.0, 10.0)):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    x = np.linspace(0.2, 9.8, 400)
    for color, function in curves:
        ax.plot(x, function(x), color=color, linewidth=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(*y_limits)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    bounds = ax.get_window_extent()
    width, height = fig.canvas.get_width_height()
    image = Image.frombytes("RGBA", (width, height), fig.canvas.buffer_rgba()).convert("RGB")
    crop = (int(np.floor(bounds.x0)), int(np.floor(height - bounds.y1)), int(np.ceil(bounds.x1)), int(np.ceil(height - bounds.y0)))
    calibration = AxisCalibration(
        x_left_px=float(bounds.x0), x_right_px=float(bounds.x1), x_min=0, x_max=10,
        y_top_px=float(height - bounds.y1), y_bottom_px=float(height - bounds.y0),
        y_min=float(y_limits[0]), y_max=float(y_limits[1]),
    )
    plt.close(fig)
    return image, crop, calibration


def test_image_loading_accepts_supported_format_and_rejects_invalid_payload():
    source = Image.new("RGB", (32, 24), "white")
    encoded = BytesIO()
    source.save(encoded, format="PNG")
    loaded = load_graph_image(encoded.getvalue(), source_name="chart.png")
    assert loaded.image.size == (32, 24)
    assert loaded.source_format == "PNG"
    assert loaded.source_sha256
    with pytest.raises(ValueError, match="could not be decoded"):
        load_graph_image(b"not an image")


@pytest.mark.parametrize("image_format", ["JPEG", "WEBP", "TIFF"])
def test_image_loading_accepts_additional_documented_formats(image_format):
    source = Image.new("RGB", (16, 12), "white")
    encoded = BytesIO()
    source.save(encoded, format=image_format)
    assert load_graph_image(encoded.getvalue()).source_format == image_format


def test_pdf_page_can_be_rendered_as_graph_source():
    pdf = Path("Papers/333.pdf")
    if not pdf.is_file():
        pytest.skip("Bundled validation PDF is not present in this checkout")
    rendered = render_pdf_page(pdf, 1, dpi=72)
    assert rendered.source_format == "PDF"
    assert rendered.page_number == 1
    assert rendered.image.width > 0 and rendered.image.height > 0
    with pytest.raises(ValueError, match="does not exist"):
        render_pdf_page(pdf, 10000, dpi=72)


@pytest.mark.parametrize(
    "scale,minimum,maximum,expected",
    [("linear", 0.0, 100.0, [0.0, 50.0, 100.0]), ("log", 1.0, 1000.0, [1.0, np.sqrt(1000), 1000.0])],
)
def test_axis_calibration_maps_linear_and_log_coordinates(scale, minimum, maximum, expected):
    calibration = AxisCalibration(10, 110, minimum, maximum, 20, 120, minimum, maximum, scale, scale)
    x, y = calibration.pixel_to_data(np.array([10, 60, 110]), np.array([120, 70, 20]))
    assert np.allclose(x, expected)
    assert np.allclose(y, expected)
    px, py = calibration.data_to_pixel(x, y)
    assert np.allclose(px, [10, 60, 110])
    assert np.allclose(py, [120, 70, 20])


def test_log_calibration_rejects_nonpositive_axis_and_bad_crop():
    with pytest.raises(ValueError, match="positive"):
        AxisCalibration(0, 10, 0, 100, 0, 10, 1, 10, "log", "linear")
    from research_plotter.analysis.graph_digitization import validate_crop

    with pytest.raises(ValueError, match="crop"):
        validate_crop((100, 80), (40, 10, 20, 70))


def test_calibration_validates_distinct_anchors_image_bounds_and_descending_data_ranges():
    calibration = AxisCalibration(10, 90, 8, 0, 10, 50, 1, -1)
    calibration.validate_image_bounds((100, 60), (5, 5, 95, 55))
    x, y = calibration.pixel_to_data([10, 90], [10, 50])
    assert np.allclose(x, [8, 0])
    assert np.allclose(y, [-1, 1])
    with pytest.raises(ValueError, match="image bounds"):
        calibration.validate_image_bounds((80, 60))
    with pytest.raises(ValueError, match="inside the selected plot crop"):
        calibration.validate_image_bounds((100, 60), (20, 5, 95, 55))
    with pytest.raises(ValueError, match="distinct"):
        AxisCalibration(10, 90, 0, 0, 10, 50, 0, 1)


def test_digitization_excludes_matching_pixels_outside_calibrated_axes():
    image = Image.new("RGB", (100, 50), "white")
    from PIL import ImageDraw

    ImageDraw.Draw(image).line((0, 25, 99, 25), fill=(214, 39, 40), width=3)
    calibration = AxisCalibration(10, 90, 0, 8, 10, 40, -1, 1)
    result = digitize_color_curves(
        image,
        crop=(0, 0, 100, 50),
        calibration=calibration,
        colors=[("line", (214, 39, 40))],
        color_tolerance=1,
    )
    assert result.data["pixel_x"].min() >= 10
    assert result.data["pixel_x"].max() <= 90
    assert result.data["x"].min() >= 0
    assert result.data["x"].max() <= 8


def test_exclusion_rectangle_removes_pixels_preserves_gap_and_data_plot_does_not_bridge_it():
    image = Image.new("RGB", (100, 60), "white")
    from PIL import ImageDraw

    ImageDraw.Draw(image).line((10, 30, 90, 30), fill=(214, 39, 40), width=3)
    calibration = AxisCalibration(10, 90, 0, 8, 10, 50, -1, 1)
    result = digitize_color_curves(
        image,
        crop=(0, 0, 100, 60),
        calibration=calibration,
        colors=[("curve", (214, 39, 40))],
        color_tolerance=1,
        exclusion_regions=((40, 20, 60, 40),),
    )
    pixel_x = result.data["pixel_x"].to_numpy()
    assert len(pixel_x) == 61
    assert not np.isin(pixel_x, np.arange(40, 60)).any()
    assert np.any(pixel_x < 40) and np.any(pixel_x >= 60)
    assert any("data gap" in warning and "not interpolated" in warning for warning in result.warnings)
    figure = plot_digitized_data(result.data)
    assert len(figure.axes[0].lines) == 0
    assert len(figure.axes[0].collections) == 1
    plt.close(figure)
    fitted = fit_equation(result.data, "y=a*x+b", initial_guesses={"a": 0.25, "b": 0.0})
    overlay = plot_fit_overlay(result.data, fitted, curve_name="curve")
    assert len(overlay.axes[0].lines) == 2
    plt.close(overlay)


def test_legend_or_inset_like_same_color_marks_warn_and_can_be_excluded():
    image = Image.new("RGB", (100, 60), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.line((10, 30, 90, 30), fill=(214, 39, 40), width=3)
    draw.line((45, 10, 60, 10), fill=(214, 39, 40), width=3)
    calibration = AxisCalibration(10, 90, 0, 8, 5, 55, -1, 1)
    contaminated = digitize_color_curves(
        image, crop=(0, 0, 100, 60), calibration=calibration,
        colors=[("curve", (214, 39, 40))], color_tolerance=1,
    )
    assert any("legend sample" in warning or "inset may contaminate" in warning for warning in contaminated.warnings)
    cleaned = digitize_color_curves(
        image, crop=(0, 0, 100, 60), calibration=calibration,
        colors=[("curve", (214, 39, 40))], color_tolerance=1,
        exclusion_regions=((43, 7, 63, 14),),
    )
    assert not any("vertically spread matching pixels" in warning for warning in cleaned.warnings)
    assert any("data gap" in warning for warning in cleaned.warnings)


@pytest.mark.parametrize(
    "function,tolerance",
    [(lambda x: x, 0.09), (lambda x: 0.1 * x**2, 0.09)],
)
def test_synthetic_linear_and_quadratic_graphs_digitize_with_measured_error(function, tolerance):
    image, crop, calibration = _synthetic_chart([("#d62728", function)])
    colors = detect_curve_colors(image, crop)
    assert colors
    target = next(color.rgb for color in colors if color.rgb[0] > color.rgb[2])
    result = digitize_color_curves(image, crop=crop, calibration=calibration, colors=[("curve", target)], color_tolerance=35)
    digitized = result.data
    expected = function(digitized["x"].to_numpy())
    maximum_error = float(np.max(np.abs(digitized["y"].to_numpy() - expected)))
    assert len(digitized) > 300
    assert maximum_error < tolerance


def test_multiple_colored_curves_are_separated_and_csv_views_are_supported():
    image, crop, calibration = _synthetic_chart([
        ("#d62728", lambda x: 1 + 0.5 * x),
        ("#1f77b4", lambda x: 9 - 0.5 * x),
    ])
    colors = detect_curve_colors(image, crop)
    assert len(colors) == 2
    red = next(color.rgb for color in colors if color.rgb[0] > color.rgb[2])
    blue = next(color.rgb for color in colors if color.rgb[2] > color.rgb[0])
    result = digitize_color_curves(image, crop=crop, calibration=calibration, colors=[("increasing", red), ("decreasing", blue)], color_tolerance=35)
    assert set(result.data["curve_name"]) == {"increasing", "decreasing"}
    wide = long_to_wide(result.data)
    assert {"x", "increasing", "decreasing"}.issubset(wide.columns)
    assert len(result.data) > 500


def test_digitization_point_limit_downsamples_evenly_and_warns():
    image, crop, calibration = _synthetic_chart([("#d62728", lambda x: x)])
    color = next(item.rgb for item in detect_curve_colors(image, crop) if item.rgb[0] > item.rgb[2])
    result = digitize_color_curves(image, crop=crop, calibration=calibration, colors=[("curve", color)], color_tolerance=35, max_points=100)
    assert len(result.data) == 100
    assert any("downsampled" in warning for warning in result.warnings)
    assert not any("occupies" in warning for warning in result.warnings)


def test_known_damped_sinusoid_fit_recovers_parameters_and_metrics():
    x = np.linspace(0.05, 7.0, 300)
    true = {"A": 1.7, "k": 0.22, "w": 4.1, "p": 0.3, "c": 0.1}
    y = true["A"] * np.exp(-true["k"] * x) * np.sin(true["w"] * x + true["p"]) + true["c"]
    result = fit_equation(
        pd.DataFrame({"x": x, "y": y}),
        "y = A*exp(-k*x)*sin(w*x+p)+c",
        initial_guesses={"A": 1.5, "k": 0.15, "w": 4.0, "p": 0.1, "c": 0.0},
        bounds={"A": (0, 5), "k": (0, 1), "w": (0, 10), "p": (-6, 6), "c": (-1, 1)},
    )
    assert result.status == "FIT_SUCCESSFUL"
    assert max(abs(result.parameters[name] - expected) for name, expected in true.items()) < 1e-4
    assert result.metrics["rmse"] < 1e-8
    assert len(result.residuals) == len(x)


def test_requested_damped_sinusoid_template_estimates_parameters():
    x = np.linspace(0.05, 7.0, 320)
    expected = {"A": 2.0, "k": 0.15, "w": 4.0}
    y = expected["A"] * np.exp(-expected["k"] * x) * np.sin(expected["w"] * x)
    result = fit_equation(
        pd.DataFrame({"x": x, "y": y}),
        "y= A*exp(-k*x)*sin(w*x)",
        initial_guesses={"A": 1.8, "k": 0.1, "w": 3.8},
        bounds={"A": (0, 4), "k": (0, 1), "w": (3, 5)},
    )
    assert max(abs(result.parameters[name] - value) for name, value in expected.items()) < 1e-4
    assert result.metrics["rmse"] < 1e-8
    assert result.status == "FIT_SUCCESSFUL"


def test_synthetic_damped_sinusoid_graph_digitizes_and_fits_parameters():
    function = lambda x: 2 * np.exp(-0.15 * x) * np.sin(4 * x)
    image, crop, calibration = _synthetic_chart([("#d62728", function)], y_limits=(-1.4, 1.4))
    color = next(item.rgb for item in detect_curve_colors(image, crop) if item.rgb[0] > item.rgb[2])
    digitized = digitize_color_curves(image, crop=crop, calibration=calibration, colors=[("damped", color)], color_tolerance=35).data
    fit = fit_equation(
        digitized,
        "y= A*exp(-k*x)*sin(w*x)",
        initial_guesses={"A": 1.8, "k": 0.1, "w": 3.8},
        bounds={"A": (0, 4), "k": (0, 1), "w": (3, 5)},
    )
    assert len(digitized) > 300
    assert abs(fit.parameters["A"] - 2.0) < 0.08
    assert abs(fit.parameters["k"] - 0.15) < 0.02
    assert abs(fit.parameters["w"] - 4.0) < 0.04
    assert fit.metrics["r_squared"] > 0.995


def test_fit_parser_rejects_untrusted_code_and_bad_bounds():
    data = pd.DataFrame({"x": np.arange(10, dtype=float), "y": np.arange(10, dtype=float)})
    with pytest.raises(ValueError):
        fit_equation(data, "y = __import__('os').system('whoami')")
    with pytest.raises(ValueError, match="Bounds"):
        fit_equation(data, "y=a*x+b", bounds={"a": (2, 1)})


def test_candidate_discovery_returns_ranked_validated_candidates():
    rng = np.random.default_rng(21)
    x = np.linspace(0.1, 5, 100)
    y = 0.3 * x**2 - 0.4 * x + 2 + rng.normal(0, 0.01, len(x))
    candidates = discover_equation_candidates(pd.DataFrame({"x": x, "y": y}), max_candidates=5)
    assert 1 <= len(candidates) <= 5
    holdout = [item.holdout_rmse for item in candidates]
    assert holdout == sorted(holdout)
    assert all(item.result.status == "FIT_SUCCESSFUL" and np.isfinite(item.result.metrics["rmse"]) for item in candidates)
    assert all(item.evidence_status == "EMPIRICAL_FIT" for item in candidates)
    assert all(item.holdout_metrics and np.isfinite(item.holdout_metrics["mae"]) for item in candidates)
    assert all("interpretation" in item.to_dict() for item in candidates)


def test_candidate_discovery_rejects_families_invalid_on_held_out_domain():
    x = np.linspace(0.0, 5.0, 40)
    y = 2.0 * x + 1.0
    candidates = discover_equation_candidates(pd.DataFrame({"x": x, "y": y}))
    assert "logarithmic" not in {item.family for item in candidates}
    assert "power law" not in {item.family for item in candidates}
    assert all(np.isfinite(item.result.fitted_y).all() for item in candidates)


def test_metadata_csv_figures_and_fit_exports_are_reproducible(tmp_path: Path):
    image, crop, calibration = _synthetic_chart([("#d62728", lambda x: x)])
    source = load_graph_image(image_to_png(image), source_name="linear.png")
    color = next(item.rgb for item in detect_curve_colors(source.image, crop) if item.rgb[0] > item.rgb[2])
    result = digitize_color_curves(source.image, crop=crop, calibration=calibration, colors=[("linear", color)], color_tolerance=35)
    artifacts = save_digitization_artifacts(source, result, calibration=calibration, crop=crop, output_dir=tmp_path / "digitization", experiment_id="digitization-run", extraction_settings={"color_tolerance": 35}, exclusion_regions=((int(crop[0]), int(crop[1]), int(crop[0]) + 1, int(crop[1]) + 1),))
    metadata = __import__("json").loads(artifacts["metadata"].read_text(encoding="utf-8"))
    assert metadata["status"] == "DIGITIZED"
    assert metadata["experiment_id"] == "digitization-run"
    assert metadata["axis_calibration_provenance"] == "USER_PROVIDED"
    assert metadata["exclusion_regions_pixels"][0]["right"] == int(crop[0]) + 1
    assert "not interpolated" in metadata["exclusion_policy"]
    assert artifacts["long_csv"].is_file() and artifacts["wide_csv"].is_file()
    assert artifacts["data_figure"].is_file() and artifacts["pixel_overlay_pdf"].is_file()
    fit = fit_equation(result.data, "y = a*x+b", initial_guesses={"a": 1, "b": 0})
    fit_artifacts = save_fit_artifacts(result.data, fit, curve_name="linear", output_dir=tmp_path / "fit", crop_image=source.image.crop(crop), crop_origin=(crop[0], crop[1]), calibration=calibration, fit_configuration={"experiment_id": "fit-run", "source": {"name": source.source_name}, "crop_coordinates_pixels": list(crop)})
    assert fit_artifacts["result"].is_file()
    fit_metadata = __import__("json").loads(fit_artifacts["result"].read_text(encoding="utf-8"))
    assert fit_metadata["experiment_id"] == "fit-run"
    assert fit_metadata["fit_configuration"]["source"]["name"] == "linear.png"
    assert fit_artifacts["overlay_svg"].is_file()
    assert fit_artifacts["residual_png"].is_file()
    shifted = result.data.assign(x=result.data["x"] + 0.25)
    with pytest.raises(ValueError, match="x coordinates do not match"):
        save_fit_artifacts(shifted, fit, curve_name="linear", output_dir=tmp_path / "misaligned-fit")


def image_to_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_digitized_reference_comparison_aligns_and_reports_uncertainty():
    digitized = pd.DataFrame({"curve_name": ["A"] * 3, "x": [0.5, 1.5, 2.5], "y": [1.0, 3.0, 5.0]})
    simulation = pd.DataFrame({"time": [0, 1, 2, 3], "state": [0, 2, 4, 6]})
    comparison = compare_digitized_reference(digitized, simulation, curve_name="A", state_name="state")
    assert comparison["status"] == "UNVERIFIED"
    assert comparison["comparison_type"] == "DIGITIZED_REFERENCE_COMPARISON"
    assert comparison["metrics"]["rmse"] == pytest.approx(0.0)
    assert "not original experimental/model data" in comparison["warnings"][0]


def test_dataset_formula_and_paper_figure_handoffs_reuse_existing_workflows(tmp_path: Path):
    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}

    st = FakeStreamlit()
    data = pd.DataFrame({"curve_name": ["C", "C", "C"], "pixel_x": [1, 2, 3], "pixel_y": [3, 2, 1], "x": [0.0, 0.5, 1.0], "y": [0.0, 0.25, 1.0]})
    recorded = []
    handoff_to_dataset_analyzer(st, data, lambda workflow: ("run-1", tmp_path), lambda *args, **kwargs: recorded.append((args, kwargs)))
    assert st.session_state["main_navigation"] == "Dataset Analyzer"
    assert st.session_state["dataset_frame"].equals(data)
    assert st.session_state["dataset_profile"]["rows"] == 3
    assert recorded and recorded[0][1]["status"] == "SUCCEEDED"

    x = np.linspace(0, 2, 20)
    fit = fit_equation(pd.DataFrame({"x": x, "y": 2 * x + 1}), "y=a*x+b")
    text = equation_for_formula_plotter(fit)
    handoff_to_formula_plotter(st, text, (0.0, 2.0))
    assert st.session_state["main_navigation"] == "Equation / Formula Plotter"
    assert st.session_state["formula_domain_start"] == 0.0
    assert st.session_state["formula_domain_end"] == 2.0
    assert "a = 2" in text and "b = 1" in text
    transferred = parse_equation_text(text)
    assert transferred.kind == "explicit"
    assert transferred.parameter_values == {"a": 2.0, "b": 1.0}

    _handoff_to_graph_analyzer(st, "paper.pdf", 5, "figure_02", None)
    assert st.session_state["main_navigation"] == "Graph Image Analyzer"
    assert st.session_state["graph_image_source_path"] == "paper.pdf"
    assert st.session_state["graph_image_paper_context"]["figure_id"] == "figure_02"

