from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pytest

from research_plotter.analysis.equation_fitting import discover_equation_candidates, fit_equation
from research_plotter.analysis.axis_calibration import AutomaticCalibrationResult, OCRWord, TickLabel, _fit_axis, _looks_like_lost_decimal, _settle_tick_reviews, auto_calibrate_axes, local_ocr_available, parse_tick_label, recognize_local_words, validate_auto_calibration
from research_plotter.analysis.graph_digitization import AxisCalibration, GraphImage, detect_curve_colors, digitize_color_curves, load_graph_image, long_to_wide, render_pdf_page
from research_plotter.models.formula_spec import parse_equation_text
from research_plotter.pipelines.graph_image_analysis import compare_digitized_reference, save_digitization_artifacts, save_fit_artifacts
from research_plotter.plotting.graph_image import plot_digitized_data, plot_fit_overlay
from research_plotter.ui.graph_image_page import equation_for_formula_plotter, handoff_to_dataset_analyzer, handoff_to_formula_plotter
from research_plotter.ui import graph_image_page
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


def _calibration_chart(
    *, size=(500, 360), x_values=None, x_scale="pi", y_values=None, crop=None,
    add_inset=False, axis_color=(25, 25, 25), curve_function=None,
):
    """Make a deterministic chart; its OCR boxes are supplied independently."""
    width, height = size
    crop = crop or (0, 0, width, height)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    x_axis_y = int(height * 0.50)
    y_axis_x = int(width * 0.35)
    plot_left, plot_right = int(width * 0.14), int(width * 0.91)
    plot_top, plot_bottom = int(height * 0.10), int(height * 0.86)
    draw.line((plot_left, x_axis_y, plot_right, x_axis_y), fill=axis_color, width=2)
    draw.line((y_axis_x, plot_top, y_axis_x, plot_bottom), fill=axis_color, width=2)
    if x_values is None:
        x_values = [-math.pi / 2, 0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
    x_low, x_high = float(x_values[0]), float(x_values[-1])
    x_pixels = np.linspace(plot_left + 18, plot_right - 18, len(x_values))
    y_values = list(y_values) if y_values is not None else [1.0, 0.0, -1.0]
    if len(y_values) < 3 or len(y_values) % 2 == 0:
        raise ValueError("Synthetic calibration fixture requires an odd number of at least three Y labels")
    y_pixels = np.linspace(plot_top + 28, min(plot_bottom - 13, 2 * x_axis_y - (plot_top + 28)), len(y_values)).tolist()
    labels = ["−π/2", "0", "π/2", "π", "3π/2", "2π"] if x_scale == "pi" else [f"{v:g}" for v in x_values]
    if x_scale == "log":
        labels = [f"{value:g}" for value in x_values]
    if x_scale == "degrees":
        labels = [f"{value:g}°" for value in x_values]
    for px in x_pixels:
        draw.line((int(px), x_axis_y - 4, int(px), x_axis_y + 4), fill=axis_color, width=1)
    for py in y_pixels:
        draw.line((y_axis_x - 4, py, y_axis_x + 4, py), fill=axis_color, width=1)
    if add_inset:
        # A smaller framed inset and a legend sample should not outscore main axes.
        draw.rectangle((int(width * .63), int(height * .16), int(width * .84), int(height * .35)), outline=(35, 35, 35), width=2)
        draw.line((int(width * .66), int(height * .27), int(width * .80), int(height * .22)), fill=(214, 39, 40), width=2)
        draw.line((int(width * .68), int(height * .38), int(width * .83), int(height * .38)), fill=(214, 39, 40), width=2)
    # A colored curve is deliberately distinct from the neutral axes.
    data_x = np.linspace(x_low, x_high, 500)
    if x_scale == "log":
        px = np.interp(np.log10(data_x), [np.log10(x_low), np.log10(x_high)], [x_pixels[0], x_pixels[-1]])
    else:
        px = np.interp(data_x, [x_low, x_high], [x_pixels[0], x_pixels[-1]])
    curve_y = curve_function(data_x) if curve_function is not None else np.sin(np.linspace(-math.pi / 2, 2 * math.pi, 500))
    py = y_pixels[-1] - (curve_y - y_values[-1]) / (y_values[0] - y_values[-1]) * (y_pixels[-1] - y_pixels[0])
    draw.line([(int(x), int(y)) for x, y in zip(px, py)], fill=(214, 39, 40), width=3)

    x_labels = labels
    def reader(cropped_image):
        ox, oy, _, _ = crop
        words = []
        for px_value, label in zip(x_pixels, x_labels):
            width_guess = max(8, len(label) * 7)
            center_x = float(px_value)
            center_y = float(x_axis_y + 17)
            words.append(OCRWord(label, center_x - width_guess / 2 - ox, center_y - 6 - oy, center_x + width_guess / 2 - ox, center_y + 6 - oy, 96))
        for value, py_value in zip(y_values, y_pixels):
            label = f"{value:g}"
            center_x = float(y_axis_x - 26)
            center_y = float(py_value)
            words.append(OCRWord(label, center_x - 10 - ox, center_y - 6 - oy, center_x + 10 - ox, center_y + 6 - oy, 94))
        return words

    return image, crop, reader, (plot_left, plot_top, plot_right, plot_bottom), (x_axis_y, y_axis_x), (x_pixels, y_pixels)


def _render_sine_calibration_chart(*, size=(666, 425), hide_axis_lines=False):
    """Render a labeled chart so OCR tests exercise pixels, not injected text."""
    width, height = size
    figure, axis = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    values = np.linspace(-math.pi / 2, 2 * math.pi, 600)
    axis.plot(values, np.sin(values), color="#d62728", linewidth=2)
    axis.set_xlim(-math.pi / 2 - 0.2, 2 * math.pi + 0.2)
    axis.set_ylim(-1.2, 1.2)
    axis.set_xticks([-math.pi / 2, 0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi])
    axis.set_xticklabels([r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    axis.set_yticks([-1, 0, 1])
    axis.grid(True, alpha=0.2)
    # This mirrors a common scientific layout: the vertical axis is at x=0,
    # while the horizontal axis remains along the lower plotting boundary.
    axis.spines["left"].set_position(("data", 0))
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    if hide_axis_lines:
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.tight_layout()
    figure.canvas.draw()
    canvas_width, canvas_height = figure.canvas.get_width_height()
    image = Image.frombytes("RGBA", (canvas_width, canvas_height), figure.canvas.buffer_rgba()).convert("RGB")
    plt.close(figure)
    return image


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


@pytest.mark.parametrize(
    ("label", "expected"),
    [("-π/2", -math.pi / 2), ("− pi / 2", -math.pi / 2), ("3π/2", 3 * math.pi / 2), ("3 pi / 2", 3 * math.pi / 2),
     ("0", 0.0), ("-2.75", -2.75), ("1e-3", 0.001), ("1×10⁻³", 0.001), ("-3/2", -1.5),
     ("-360°", -360.0), ("90 °", 90.0), ("10^2", 100.0), ("10⁻³", 0.001), ("2705", 2705.0),
     ("1,000", 1000.0), ("1,5", None), ("unreadable", None), ("π/0", None)],
)
def test_tick_label_parser_handles_numeric_fraction_pi_and_scientific_forms(label, expected):
    parsed = parse_tick_label(label)
    if expected is None:
        assert parsed is None
    else:
        assert parsed == pytest.approx(expected)


def test_automatic_calibration_recovers_pi_ticks_negative_ranges_and_downstream_mapping():
    image, crop, reader, region, (x_axis, y_axis), (x_pixels, y_pixels) = _calibration_chart(size=(500, 360), add_inset=True)
    result = auto_calibrate_axes(image, crop, ocr_reader=reader)
    assert result.status == "DETECTED"
    assert result.confidence == "HIGH"
    assert result.plot_region == (region[0], region[1], region[2] + 1, region[3] + 1)
    assert [tick.label for tick in result.x_ticks] == ["−π/2", "0", "π/2", "π", "3π/2", "2π"]
    assert [tick.value for tick in result.x_ticks] == pytest.approx([-math.pi / 2, 0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi])
    calibration = validate_auto_calibration(result, image.size, crop)
    assert calibration is not None
    assert calibration.x_left_px == pytest.approx(x_pixels[0])
    assert calibration.x_right_px == pytest.approx(x_pixels[-1])
    assert calibration.x_min == pytest.approx(-math.pi / 2)
    assert calibration.x_max == pytest.approx(2 * math.pi)
    assert calibration.y_min == pytest.approx(-1)
    assert calibration.y_max == pytest.approx(1)
    sample_x, sample_y = calibration.pixel_to_data([x_pixels[0], x_pixels[-1]], [y_pixels[0], y_pixels[-1]])
    assert sample_x == pytest.approx([-math.pi / 2, 2 * math.pi])
    assert sample_y == pytest.approx([1, -1])
    # Pixel-to-data results are valid inputs to the existing digitizer pipeline.
    digitized = digitize_color_curves(image, crop=crop, calibration=calibration, colors=[("sine", (214, 39, 40))], color_tolerance=35)
    assert len(digitized.data) > 250
    assert digitized.data["x"].min() >= -math.pi / 2 - 0.02
    assert digitized.data["x"].max() <= 2 * math.pi + 0.02


def test_automatic_calibration_uses_absolute_image_pixels_for_nonzero_crop_origin():
    crop = (15, 10, 490, 350)
    image, _, reader, _, _, _ = _calibration_chart(size=(500, 360), crop=crop)
    result = auto_calibrate_axes(image, crop, ocr_reader=reader)
    assert result.status == "DETECTED"
    calibration = validate_auto_calibration(result, image.size, crop)
    assert calibration is not None
    assert calibration.x_left_px > crop[0]
    assert calibration.y_top_px > crop[1]
    mapped_x, mapped_y = calibration.pixel_to_data([calibration.x_left_px, calibration.x_right_px], [calibration.y_top_px, calibration.y_bottom_px])
    assert mapped_x == pytest.approx([-math.pi / 2, 2 * math.pi])
    assert mapped_y == pytest.approx([1, -1])


def test_automatic_calibration_detects_log_axis_and_numeric_negative_ticks():
    image, crop, reader, _, _, _ = _calibration_chart(size=(640, 420), x_values=[1, 10, 100, 1000], x_scale="log")
    result = auto_calibrate_axes(image, crop, ocr_reader=reader)
    assert result.status == "DETECTED"
    assert result.x_scale == "log"
    calibration = validate_auto_calibration(result, image.size, crop)
    assert calibration is not None
    middle, _ = calibration.pixel_to_data((calibration.x_left_px + calibration.x_right_px) / 2, calibration.y_top_px)
    assert middle == pytest.approx(10 ** 1.5)
    assert [tick.value for tick in result.y_ticks] == pytest.approx([1, 0, -1])


@pytest.mark.parametrize(
    "curve_function",
    [
        lambda x: 0.16 * x - 0.2,
        lambda x: 0.025 * (x - 1.0) ** 2 - 1.0,
        lambda x: 0.001 * x**3,
        lambda x: 0.6 * np.exp(0.15 * x) - 1.5,
        lambda x: 0.8 * np.log(x + 11.0) - 1.5,
        lambda x: 1.2 * np.cos(0.35 * x + 0.2),
    ],
    ids=["linear", "quadratic", "cubic", "exponential", "logarithmic", "cosine"],
)
def test_calibration_is_independent_of_curve_family_and_accepts_colored_axes(curve_function):
    image, crop, reader, _region, (x_axis, y_axis), _pixels = _calibration_chart(
        size=(640, 420), x_values=[-10, -5, 0, 5, 10], x_scale="linear",
        y_values=[2, 1, 0, -1, -2], axis_color=(0, 112, 42),
        curve_function=curve_function, add_inset=True,
    )
    result = auto_calibrate_axes(image, crop, ocr_reader=reader)
    calibration = validate_auto_calibration(result, image.size, crop)
    assert calibration is not None
    assert result.x_axis_pixel == pytest.approx(x_axis, abs=2)
    assert result.y_axis_pixel == pytest.approx(y_axis, abs=2)
    assert calibration.x_min == pytest.approx(-10)
    assert calibration.x_max == pytest.approx(10)
    assert calibration.y_max == pytest.approx(2)
    assert calibration.y_min == pytest.approx(-2)


def test_degree_fixture_calibration_parser_does_not_reinterpret_plain_large_numbers():
    assert parse_tick_label("-360°") == pytest.approx(-360)
    assert parse_tick_label("2705") == pytest.approx(2705)
    # A large numeral is not silently corrected to a guessed missing degree
    # marker or zero. Axis-wide OCR evidence must resolve that ambiguity.


def test_log_scale_requires_three_consistent_logarithmic_tick_anchors():
    # A log line can pass exactly through any two positive points. These six
    # radian ticks form a linear pixel/value mapping, although two positive
    # ticks also define a zero-residual log candidate. The log model must not
    # win unless its own consensus contains at least three distinct ticks.
    values = [-math.pi / 2, 0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
    ticks = [
        TickLabel("x", 56 + index * 283.8, str(value), value, 90)
        for index, value in enumerate(values)
    ]
    fit = _fit_axis(ticks, "x")
    assert fit is not None
    assert fit.scale == "linear"
    assert fit.inlier_count == len(values)
    assert fit.first_value == pytest.approx(-math.pi / 2)
    assert fit.last_value == pytest.approx(2 * math.pi)


def test_automatic_calibration_detects_logarithmic_y_axis():
    image, crop, reader, _, _, (_, y_pixels) = _calibration_chart(size=(560, 400), y_values=[100, 10, 1])
    result = auto_calibrate_axes(image, crop, ocr_reader=reader)
    assert result.status == "DETECTED"
    assert result.y_scale == "log"
    calibration = validate_auto_calibration(result, image.size, crop)
    assert calibration is not None
    assert calibration.y_max == pytest.approx(100)
    assert calibration.y_min == pytest.approx(1)
    _, midpoint = calibration.pixel_to_data(calibration.x_left_px, (y_pixels[0] + y_pixels[-1]) / 2)
    assert midpoint == pytest.approx(10)


def test_automatic_calibration_supports_degree_ticks_without_radian_conversion():
    degrees = [-360, -270, -180, -90, 0, 90, 180, 270, 360]
    image, crop, reader, _region, _axes, (x_pixels, _y_pixels) = _calibration_chart(
        size=(711, 433), x_values=degrees, x_scale="degrees", y_values=[1, 0.5, 0, -0.5, -1],
        axis_color=(0, 112, 42), curve_function=lambda x: np.sin(np.deg2rad(x)),
    )
    result = auto_calibrate_axes(image, crop, ocr_reader=reader)
    calibration = validate_auto_calibration(result, image.size, crop)
    assert calibration is not None
    assert result.x_scale == "linear"
    assert calibration.x_min == pytest.approx(-360)
    assert calibration.x_max == pytest.approx(360)
    mapped, _ = calibration.pixel_to_data(x_pixels, np.full(len(x_pixels), calibration.y_top_px))
    assert mapped == pytest.approx(degrees, abs=0.05)


def test_automatic_calibration_detects_logarithmic_x_and_y_together():
    x_values = [0.01, 0.1, 1, 10, 100]
    y_values = [1000, 100, 10, 1, 0.1]
    image, crop, reader, _region, _axes, (x_pixels, y_pixels) = _calibration_chart(
        size=(680, 440), x_values=x_values, x_scale="log", y_values=y_values,
        curve_function=lambda x: np.sqrt(x),
    )
    result = auto_calibrate_axes(image, crop, ocr_reader=reader)
    calibration = validate_auto_calibration(result, image.size, crop)
    assert calibration is not None
    assert result.x_scale == "log" and result.y_scale == "log"
    midpoint_x, midpoint_y = calibration.pixel_to_data(
        [(x_pixels[0] + x_pixels[-1]) / 2], [(y_pixels[0] + y_pixels[-1]) / 2]
    )
    assert midpoint_x[0] == pytest.approx(1)
    assert midpoint_y[0] == pytest.approx(10)


def test_automatic_calibration_handles_ordinary_numeric_ticks_and_rejects_ocr_outlier():
    image, crop, reader, _, _, _ = _calibration_chart(size=(720, 480), x_values=[-5, -2.5, 0, 2.5, 5], x_scale="linear")
    result = auto_calibrate_axes(image, crop, ocr_reader=reader)
    assert result.status == "DETECTED"
    assert result.values["x_min"] == pytest.approx(-5)
    assert result.values["x_max"] == pytest.approx(5)
    assert result.x_scale == "linear"

    def inaccurate_reader(cropped):
        words = list(reader(cropped))
        return [OCRWord("99", word.left, word.top, word.right, word.bottom, word.confidence) if word.text == "0" else word for word in words]

    rejected = auto_calibrate_axes(image, crop, ocr_reader=inaccurate_reader)
    assert rejected.status == "DETECTED"
    assert len(rejected.x_ticks) < 5
    assert any("excluded" in warning for warning in rejected.warnings)


def test_automatic_calibration_does_not_guess_unreadable_ticks_and_reports_partial_axes():
    image, crop, reader, _, _, _ = _calibration_chart()
    original_reader = reader

    def x_only(cropped):
        return [word for word in original_reader(cropped) if word.text in {"−π/2", "0", "π/2", "π", "3π/2", "2π"}]

    partial = auto_calibrate_axes(image, crop, ocr_reader=x_only)
    assert partial.status == "PARTIAL"
    assert partial.values["x_left_px"] is not None
    assert partial.values["y_top_px"] is None
    assert any("one axis" in warning for warning in partial.warnings)

    def unreadable(_cropped):
        return [OCRWord("unreadable", 10, 10, 20, 20, 20)]

    uncertain = auto_calibrate_axes(image, crop, ocr_reader=unreadable)
    assert uncertain.status == "NEEDS_REVIEW"
    assert all(value is None for value in uncertain.values.values())
    assert any("cannot establish data values" in warning for warning in uncertain.warnings)


def test_automatic_calibration_failure_leaves_manual_calibration_path_available():
    blank = Image.new("RGB", (310, 190), "white")
    result = auto_calibrate_axes(blank, ocr_reader=lambda _image: [])
    assert result.status == "NEEDS_REVIEW"
    assert validate_auto_calibration(result, blank.size, (0, 0, blank.width, blank.height)) is None
    assert all(value is None for value in result.values.values())
    assert any("manually" in warning for warning in result.warnings)


def test_ocr_layout_fallback_calibrates_crop_when_axis_strokes_do_not_intersect():
    image = Image.new("RGB", (700, 520), "white")
    crop = (30, 25, 670, 495)
    x_pixels = [70, 190, 310, 430, 550]
    x_values = [-5, -2.5, 0, 2.5, 5]
    y_pixels = [60, 250, 440]
    y_values = [2, 0, -2]

    def ocr_reader(_crop):
        words = []
        for pixel, value in zip(x_pixels, x_values):
            words.append(OCRWord(str(value), pixel - 12, 410, pixel + 12, 425, 96))
        for pixel, value in zip(y_pixels, y_values):
            words.append(OCRWord(str(value), 5, pixel - 7, 25, pixel + 7, 95))
        return words

    result = auto_calibrate_axes(image, crop, ocr_reader=ocr_reader)
    assert result.status == "DETECTED"
    assert result.confidence == "LOW"
    assert result.x_axis_pixel is None and result.y_axis_pixel is None
    assert any("aligned tick-label positions" in warning for warning in result.warnings)
    calibration = validate_auto_calibration(result, image.size, crop)
    assert calibration is not None
    mapped_x, _ = calibration.pixel_to_data(np.asarray(x_pixels) + crop[0], np.full(len(x_pixels), 200))
    _, mapped_y = calibration.pixel_to_data(np.full(len(y_pixels), 300), np.asarray(y_pixels) + crop[1])
    assert mapped_x == pytest.approx(x_values, abs=1e-8)
    assert mapped_y == pytest.approx(y_values, abs=1e-8)
    assert calibration.x_left_px >= crop[0] and calibration.y_top_px >= crop[1]


def test_tick_label_support_disambiguates_plot_axes_from_decorative_image_border():
    image, crop, reader, plot_region, (expected_x_axis, expected_y_axis), _ = _calibration_chart(size=(500, 360))
    ImageDraw.Draw(image).rectangle((1, 1, image.width - 2, image.height - 2), outline=(15, 15, 15), width=2)
    result = auto_calibrate_axes(image, crop, ocr_reader=reader)
    assert result.status == "DETECTED"
    assert result.x_axis_pixel == pytest.approx(expected_x_axis)
    assert result.y_axis_pixel == pytest.approx(expected_y_axis)
    assert result.plot_region == (plot_region[0], plot_region[1], plot_region[2] + 1, plot_region[3] + 1)
    assert result.values["x_left_px"] > 1
    assert result.values["x_right_px"] < image.width - 2


@pytest.mark.skipif(not local_ocr_available()[0], reason="Local Tesseract OCR is an optional integration dependency")
def test_local_tesseract_calibrates_rendered_sine_graph_and_feeds_digitization_and_fitting():
    image = _render_sine_calibration_chart()
    crop = (0, 0, image.width, image.height)
    result = auto_calibrate_axes(image, crop)
    assert result.status == "DETECTED"
    assert result.ocr_available
    assert len(result.x_ticks) >= 5
    assert len(result.y_ticks) >= 2

    calibration = validate_auto_calibration(result, image.size, crop)
    assert calibration is not None
    tick_x, _ = calibration.pixel_to_data(
        np.asarray([tick.pixel for tick in result.x_ticks]),
        np.full(len(result.x_ticks), calibration.y_top_px),
    )
    _, tick_y = calibration.pixel_to_data(
        np.full(len(result.y_ticks), calibration.x_left_px),
        np.asarray([tick.pixel for tick in result.y_ticks]),
    )
    assert np.max(np.abs(tick_x - np.asarray([tick.value for tick in result.x_ticks]))) < 0.03
    assert np.max(np.abs(tick_y - np.asarray([tick.value for tick in result.y_ticks]))) < 0.03

    colors = detect_curve_colors(image, crop)
    red = next(color.rgb for color in colors if color.rgb[0] > color.rgb[2])
    digitized = digitize_color_curves(
        image, crop=crop, calibration=calibration, colors=[("sine", red)], color_tolerance=62
    )
    assert len(digitized.data) > 300
    fit = fit_equation(
        digitized.data,
        "y=a*sin(b*x+c)+d",
        initial_guesses={"a": 1, "b": 1, "c": 0, "d": 0},
        bounds={"a": (0.5, 1.5), "b": (0.7, 1.3), "c": (-0.5, 0.5), "d": (-0.5, 0.5)},
    )
    assert fit.status == "FIT_SUCCESSFUL"
    assert abs(fit.parameters["a"] - 1) < 0.08
    assert abs(fit.parameters["b"] - 1) < 0.04
    assert fit.metrics["r_squared"] > 0.99


@pytest.mark.skipif(not (Path(__file__).parent / "fixtures" / "real_sine_graph.png").is_file(), reason="The user-provided real graph fixture is unavailable")
@pytest.mark.skipif(not local_ocr_available()[0], reason="Local Tesseract OCR is an optional integration dependency")
def test_real_sine_graph_fixture_calibrates_both_axes_and_reaches_digitization_and_fit():
    """Exercise the exact raster that exposed the missed π labels in production."""
    image = Image.open(Path(__file__).parent / "fixtures" / "real_sine_graph.png").convert("RGB")
    crop = (0, 0, image.width, image.height)
    result = auto_calibrate_axes(image, crop)

    assert result.status == "DETECTED"
    assert result.ocr_available
    expected_x = np.asarray([-math.pi / 2, 0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi])
    expected_y = np.asarray([1, 0.5, 0, -0.5, -1])
    x_ticks = sorted((tick for tick in result.review_ticks if tick.axis == "x" and tick.review_status == "ACCEPTED"), key=lambda tick: tick.pixel)
    y_ticks = sorted((tick for tick in result.review_ticks if tick.axis == "y" and tick.review_status == "ACCEPTED"), key=lambda tick: tick.pixel)
    assert [tick.value for tick in x_ticks] == pytest.approx(expected_x, abs=0.01)
    assert [tick.value for tick in y_ticks] == pytest.approx(expected_y, abs=0.01)
    assert all(first.pixel < second.pixel for first, second in zip(x_ticks, x_ticks[1:]))
    assert all(first.pixel < second.pixel for first, second in zip(y_ticks, y_ticks[1:]))

    calibration = validate_auto_calibration(result, image.size, crop)
    assert calibration is not None
    assert result.x_scale == "linear"
    assert calibration.x_left_px == pytest.approx(x_ticks[0].pixel, abs=2)
    assert calibration.x_right_px == pytest.approx(x_ticks[-1].pixel, abs=2)
    assert calibration.x_min == pytest.approx(expected_x[0])
    assert calibration.x_max == pytest.approx(expected_x[-1])
    mapped_x, _ = calibration.pixel_to_data(
        [tick.pixel for tick in x_ticks], np.full(len(x_ticks), calibration.y_top_px)
    )
    _, mapped_y = calibration.pixel_to_data(
        np.full(len(y_ticks), calibration.x_left_px), [tick.pixel for tick in y_ticks]
    )
    assert mapped_x == pytest.approx(expected_x, abs=0.025)
    assert mapped_y == pytest.approx(expected_y, abs=0.025)

    # The validated calibration must be the calibration consumed by the
    # existing digitizer and equation fitter, not merely populated UI fields.
    colors = detect_curve_colors(image, crop)
    red = next(color.rgb for color in colors if color.rgb[0] > color.rgb[1] * 1.5 and color.rgb[0] > color.rgb[2] * 1.5)
    digitized = digitize_color_curves(image, crop=crop, calibration=calibration, colors=[("sine", red)], color_tolerance=50)
    assert len(digitized.data) > 500
    assert digitized.data["x"].min() >= -0.01
    assert digitized.data["x"].max() == pytest.approx(2 * math.pi, abs=0.02)
    fit = fit_equation(
        digitized.data,
        "y=a*sin(b*x+c)+d",
        initial_guesses={"a": 1, "b": 1, "c": 0, "d": 0},
        bounds={"a": (0.5, 1.5), "b": (0.7, 1.3), "c": (-0.5, 0.5), "d": (-0.5, 0.5)},
    )
    assert fit.status == "FIT_SUCCESSFUL"
    assert fit.parameters["a"] == pytest.approx(1, abs=0.03)
    assert fit.parameters["b"] == pytest.approx(1, abs=0.03)
    assert fit.metrics["r_squared"] > 0.999


def test_leading_zero_ocr_is_not_trusted_as_a_tick_without_axis_corroboration():
    # float("06") is a legitimate parser result, but in OCR context a leading
    # zero is a known decimal-loss warning and must not become an axis anchor
    # unless other tick labels corroborate the mapping.
    assert parse_tick_label("06") == 6
    assert _looks_like_lost_decimal("06")
    ticks = (
        TickLabel("y", 10, "1", 1, 96),
        TickLabel("y", 30, "06", 6, 77, "REVIEW_REQUIRED", "06", "Leading zero may hide a decimal point."),
        TickLabel("y", 50, "0", 0, 96),
        TickLabel("y", 70, "-0.5", -0.5, 96),
        TickLabel("y", 90, "-1", -1, 96),
    )
    trusted, reviewed, fit = _settle_tick_reviews(ticks, "y")
    assert fit is not None
    assert [tick.value for tick in trusted] == pytest.approx([1, 0, -0.5, -1])
    ambiguous = next(tick for tick in reviewed if tick.label == "06")
    assert ambiguous.review_status == "REJECTED"
    assert ambiguous.value == 6
    assert ambiguous.pixel == 30  # intermediate tick, never the bottom anchor


def test_two_tick_ocr_alternatives_remain_unresolved_without_a_third_anchor():
    ticks = (
        TickLabel("y", 10, "1", 1, 96),
        TickLabel("y", 90, "-1", -1, 70, "REVIEW_REQUIRED", "-1", "Ambiguous OCR"),
        TickLabel("y", 90, "4", 4, 95, "REVIEW_REQUIRED", "4", "Ambiguous OCR"),
    )
    trusted, reviewed, fit = _settle_tick_reviews(ticks, "y")
    assert fit is None
    assert [tick.value for tick in trusted] == [1]
    assert all(tick.review_status != "ACCEPTED" for tick in reviewed if tick.pixel == 90)
    assert any("Only two tick anchors" in (tick.note or "") for tick in reviewed if tick.pixel == 90)


def test_auto_calibration_retries_06_and_keeps_the_intermediate_tick_out_of_axis_endpoints(monkeypatch):
    import research_plotter.analysis.axis_calibration as calibration_module

    image, crop, reader, _region, (_x_axis, _y_axis), (_x_pixels, y_pixels) = _calibration_chart(
        size=(500, 360), y_values=[1, 0.5, 0, -0.5, -1]
    )
    original_words = reader(image.crop(crop))
    # Model the real OCR failure deterministically: an intermediate 0.5 label
    # is read as "06". The local retry remains ambiguous, so other y ticks
    # must prevent it from defining the mapping as 1 -> 6.
    words = [
        OCRWord(
        "06" if item.text == "0.5" else item.text,
            item.left, item.top, item.right, item.bottom, item.confidence,
        )
        for item in original_words
    ]
    monkeypatch.setattr(calibration_module, "local_ocr_available", lambda: (True, None))
    monkeypatch.setattr(calibration_module, "recognize_local_words", lambda _image: words)
    def retry_label(_image, box):
        y_center = (box[1] + box[3]) / 2
        return [("06", 6.0, 77.0)] if abs(y_center - y_pixels[1]) < 20 else []

    monkeypatch.setattr(calibration_module, "_read_local_tick_label", retry_label)

    result = auto_calibrate_axes(image, crop)
    y_ticks = sorted((tick for tick in result.review_ticks if tick.axis == "y"), key=lambda tick: tick.pixel)
    ambiguous = next(tick for tick in y_ticks if tick.label == "06")
    assert ambiguous.value == 6
    assert ambiguous.review_status == "REJECTED"
    assert ambiguous.pixel == pytest.approx(y_pixels[1])
    assert result.values["y_top_px"] == pytest.approx(y_pixels[0])
    assert result.values["y_bottom_px"] == pytest.approx(y_pixels[-1])
    assert result.values["y_max"] == pytest.approx(1)
    assert result.values["y_min"] == pytest.approx(-1)
    assert result.values["y_bottom_px"] != pytest.approx(ambiguous.pixel)


def test_local_ocr_adapter_is_optional_and_reports_unavailable_backend():
    available, reason = local_ocr_available()
    if not available:
        assert reason
        with pytest.raises(RuntimeError, match="OCR is unavailable"):
            recognize_local_words(Image.new("RGB", (100, 50), "white"))
    else:
        words = recognize_local_words(Image.new("RGB", (100, 50), "white"))
        assert isinstance(words, list)


def test_calibration_ui_seeds_once_preserves_edits_and_refreshes_for_crop_source_and_button(monkeypatch):
    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}
            self.clicks = set()

        def button(self, label, *, key=None, **_kwargs):
            clicked = key in self.clicks
            self.clicks.discard(key)
            return clicked

        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

        def columns(self, count):
            return [self] * count

        def number_input(self, _label, *, value=None, key, **_kwargs):
            self.session_state.setdefault(key, value)
            return self.session_state[key]

        def selectbox(self, _label, options, *, key, **_kwargs):
            self.session_state.setdefault(key, options[0])
            return self.session_state[key]

        @staticmethod
        def expander(*_args, **_kwargs):
            from contextlib import nullcontext
            return nullcontext()

    def result_for(image, crop):
        x_left = 20.0 if image.width == 200 else 30.0
        values = {
            "x_left_px": x_left, "x_right_px": float(image.width - 10), "x_min": -2.0, "x_max": 8.0,
            "y_top_px": 15.0, "y_bottom_px": float(image.height - 10), "y_min": -1.0, "y_max": 1.0,
        }
        return AutomaticCalibrationResult(values, "DETECTED", "HIGH", "linear", "linear", crop, image.height / 2, x_left, (), (), True, ())

    detections = []
    def detector(image, crop):
        detections.append((image.size, crop))
        return result_for(image, crop)

    monkeypatch.setattr(graph_image_page, "auto_calibrate_axes", detector)
    st = FakeStreamlit()
    prefix = "graph_state_test"
    source = GraphImage(Image.new("RGB", (200, 150), "white"), "first.png", "PNG", source_sha256="first")
    crop = (0, 0, 200, 150)
    calibration = graph_image_page._axis_calibration_controls(st, source, crop, prefix)
    assert calibration is not None and calibration.x_left_px == 20
    assert len(detections) == 1

    # A page rerun and unrelated interaction must not overwrite a manual edit.
    st.session_state[f"{prefix}_cal_x_left"] = 25.0
    rerun = graph_image_page._axis_calibration_controls(st, source, crop, prefix)
    assert rerun is not None and rerun.x_left_px == 25
    assert st.session_state[f"{prefix}_calibration_provenance"] == "USER_MODIFIED_AUTO_DETECTION"
    assert len(detections) == 1

    # Crop changes and an explicit re-detect happen before the fields render.
    st.session_state[f"{prefix}_cal_x_left"] = 35.0
    cropped = (5, 0, 200, 150)
    recalibrated = graph_image_page._axis_calibration_controls(st, source, cropped, prefix)
    assert recalibrated is not None and recalibrated.x_left_px == 20
    assert len(detections) == 2
    st.session_state[f"{prefix}_cal_x_left"] = 36.0
    st.clicks.add(f"{prefix}_auto_calibrate")
    clicked = graph_image_page._axis_calibration_controls(st, source, cropped, prefix)
    assert clicked is not None and clicked.x_left_px == 20
    assert len(detections) == 3

    # Replacing source content under the same page prefix invalidates old edits.
    replacement = GraphImage(Image.new("RGB", (220, 170), "white"), "second.png", "PNG", source_sha256="second")
    replaced = graph_image_page._axis_calibration_controls(st, replacement, (0, 0, 220, 170), prefix)
    assert replaced is not None and replaced.x_left_px == 30
    assert len(detections) == 4


def test_failed_auto_retry_preserves_manual_mapping_and_reports_it_as_active(monkeypatch):
    from contextlib import nullcontext

    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}
            self.clicks = set()
            self.messages = []

        def button(self, _label, *, key=None, **_kwargs):
            clicked = key in self.clicks
            self.clicks.discard(key)
            return clicked

        def __getattr__(self, name):
            if name in {"info", "warning", "success", "caption"}:
                return lambda message, **_kwargs: self.messages.append((name, str(message)))
            return lambda *_args, **_kwargs: None

        def columns(self, count):
            return [self] * count

        def number_input(self, _label, *, value=None, key, **_kwargs):
            self.session_state.setdefault(key, value)
            return self.session_state[key]

        def selectbox(self, _label, options, *, key, **_kwargs):
            self.session_state.setdefault(key, options[0])
            return self.session_state[key]

        @staticmethod
        def expander(*_args, **_kwargs):
            return nullcontext()

    failed = AutomaticCalibrationResult(
        {key: None for key in ("x_left_px", "x_right_px", "x_min", "x_max", "y_top_px", "y_bottom_px", "y_min", "y_max")},
        "NEEDS_REVIEW", "NONE", "linear", "linear", None, None, None, (), (), True,
        ("Could not establish a mapping from this image.",),
    )
    calls = []

    def detector(_image, _crop):
        calls.append(True)
        return failed

    monkeypatch.setattr(graph_image_page, "auto_calibrate_axes", detector)
    st = FakeStreamlit()
    prefix = "manual_after_auto_failure"
    source = GraphImage(Image.new("RGB", (200, 150), "white"), "axis.png", "PNG", source_sha256="source-1")
    crop = (0, 0, 200, 150)
    assert graph_image_page._axis_calibration_controls(st, source, crop, prefix) is None

    manual = {
        "cal_x_left": 10.0, "cal_x_right": 190.0, "cal_x_min": -math.pi / 2, "cal_x_max": 2 * math.pi,
        "cal_y_top": 10.0, "cal_y_bottom": 140.0, "cal_y_min": -1.0, "cal_y_max": 1.0,
    }
    for suffix, value in manual.items():
        st.session_state[f"{prefix}_{suffix}"] = value
    st.messages.clear()
    active = graph_image_page._axis_calibration_controls(st, source, crop, prefix)
    assert active is not None and active.x_left_px == 10
    assert any(name == "info" and "manual calibration is active" in message for name, message in st.messages)
    assert not any(name == "warning" and "could not determine a reliable mapping" in message for name, message in st.messages)

    st.clicks.add(f"{prefix}_auto_calibrate")
    st.messages.clear()
    after_retry = graph_image_page._axis_calibration_controls(st, source, crop, prefix)
    assert after_retry is not None
    assert after_retry.to_dict() == active.to_dict()
    assert all(st.session_state[f"{prefix}_{suffix}"] == value for suffix, value in manual.items())
    assert any(name == "info" and "manual calibration is active" in message for name, message in st.messages)
    assert len(calls) == 2


def test_invalid_auto_candidate_is_rejected_without_changing_manual_edits(monkeypatch):
    from contextlib import nullcontext

    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}
            self.clicked = False
            self.messages = []

        def button(self, _label, *, key=None, **_kwargs):
            if self.clicked:
                self.clicked = False
                return True
            return False

        def __getattr__(self, name):
            if name == "warning":
                return lambda message, **_kwargs: self.messages.append(str(message))
            return lambda *_args, **_kwargs: None

        def columns(self, count):
            return [self] * count

        def number_input(self, _label, *, value=None, key, **_kwargs):
            self.session_state.setdefault(key, value)
            return self.session_state[key]

        def selectbox(self, _label, options, *, key, **_kwargs):
            self.session_state.setdefault(key, options[0])
            return self.session_state[key]

        @staticmethod
        def expander(*_args, **_kwargs):
            return nullcontext()

    valid_values = {
        "x_left_px": 20.0, "x_right_px": 190.0, "x_min": -2.0, "x_max": 8.0,
        "y_top_px": 10.0, "y_bottom_px": 140.0, "y_min": -1.0, "y_max": 1.0,
    }
    invalid_values = {**valid_values, "x_right_px": 20.0}
    results = [
        AutomaticCalibrationResult(valid_values, "DETECTED", "HIGH", "linear", "linear", (0, 0, 200, 150), 75, 20, (), (), True, ()),
        AutomaticCalibrationResult(invalid_values, "DETECTED", "HIGH", "linear", "linear", (0, 0, 200, 150), 75, 20, (), (), True, ()),
    ]
    monkeypatch.setattr(graph_image_page, "auto_calibrate_axes", lambda *_args: results.pop(0))
    st = FakeStreamlit()
    prefix = "invalid_auto_candidate"
    source = GraphImage(Image.new("RGB", (200, 150), "white"), "axis.png", "PNG", source_sha256="source-2")
    crop = (0, 0, 200, 150)
    assert graph_image_page._axis_calibration_controls(st, source, crop, prefix) is not None
    st.session_state[f"{prefix}_cal_x_left"] = 15.0
    st.clicked = True
    calibration = graph_image_page._axis_calibration_controls(st, source, crop, prefix)
    assert calibration is not None and calibration.x_left_px == 15.0
    assert st.session_state[f"{prefix}_cal_x_left"] == 15.0
    assert any("candidate was rejected" in message for message in st.messages)


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
    auto_artifacts = save_digitization_artifacts(
        source, result, calibration=calibration, crop=crop, output_dir=tmp_path / "auto-digitization",
        experiment_id="auto-calibrated-run", calibration_provenance="AUTO_DETECTED",
        calibration_detection={"status": "DETECTED", "confidence": "HIGH", "warnings": []},
    )
    auto_metadata = __import__("json").loads(auto_artifacts["metadata"].read_text(encoding="utf-8"))
    assert auto_metadata["axis_calibration_provenance"] == "AUTO_DETECTED"
    assert auto_metadata["axis_calibration_detection"]["confidence"] == "HIGH"
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

