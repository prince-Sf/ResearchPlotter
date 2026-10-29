"""Conservative pixel-to-data extraction for scientific graph images.

Digitized values are estimates from pixels, not recovered source observations.
The selected plot region bounds extraction; independent axis calibration
anchors define the coordinate transform and may be interior ticks. Colored
curves are segmented independently and resolvable vertical clusters per image
column are retained rather than collapsed into a potentially false median.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal, Sequence

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


ImageSource = str | Path | bytes | bytearray | BinaryIO
ImageScale = Literal["linear", "log"]
SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "TIFF"}
MAX_IMAGE_PIXELS = 20_000_000


@dataclass(frozen=True)
class GraphImage:
    image: Image.Image
    source_name: str
    source_format: str
    page_number: int | None = None
    source_sha256: str | None = None


@dataclass(frozen=True)
class AxisCalibration:
    """Two-point calibration for a cropped or full-image plot region.

    Pixel coordinates refer to the original rendered image, with origin at the
    upper-left. Y values are mapped with the expected inverted pixel direction.
    """

    x_left_px: float
    x_right_px: float
    x_min: float
    x_max: float
    y_top_px: float
    y_bottom_px: float
    y_min: float
    y_max: float
    x_scale: ImageScale = "linear"
    y_scale: ImageScale = "linear"

    def __post_init__(self) -> None:
        pixel_values = (self.x_left_px, self.x_right_px, self.y_top_px, self.y_bottom_px)
        data_values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if not np.isfinite(pixel_values).all() or not np.isfinite(data_values).all():
            raise ValueError("Axis calibration values must all be finite")
        if self.x_right_px <= self.x_left_px:
            raise ValueError("The right x-axis pixel must be greater than the left pixel")
        if self.y_bottom_px <= self.y_top_px:
            raise ValueError("The bottom y-axis pixel must be greater than the top pixel")
        if self.x_max == self.x_min or self.y_max == self.y_min:
            raise ValueError("Axis calibration values at the two anchors must be distinct")
        if self.x_scale not in {"linear", "log"} or self.y_scale not in {"linear", "log"}:
            raise ValueError("Axis scales must be 'linear' or 'log'")
        if self.x_scale == "log" and min(self.x_min, self.x_max) <= 0:
            raise ValueError("A logarithmic x-axis requires positive calibration values")
        if self.y_scale == "log" and min(self.y_min, self.y_max) <= 0:
            raise ValueError("A logarithmic y-axis requires positive calibration values")

    def validate_image_bounds(
        self,
        image_size: tuple[int, int],
        crop: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Require calibration anchors to lie within the source image/crop."""
        width, height = image_size
        if not (0 <= self.x_left_px < self.x_right_px <= width):
            raise ValueError(f"X calibration pixels must lie within image bounds 0–{width}")
        if not (0 <= self.y_top_px < self.y_bottom_px <= height):
            raise ValueError(f"Y calibration pixels must lie within image bounds 0–{height}")
        if crop is not None:
            left, top, right, bottom = validate_crop(image_size, crop)
            if not (left <= self.x_left_px < self.x_right_px <= right):
                raise ValueError("X calibration anchors must lie inside the selected plot crop")
            if not (top <= self.y_top_px < self.y_bottom_px <= bottom):
                raise ValueError("Y calibration anchors must lie inside the selected plot crop")

    def pixel_to_data(self, pixel_x: np.ndarray | float, pixel_y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        px = np.asarray(pixel_x, dtype=float)
        py = np.asarray(pixel_y, dtype=float)
        x_fraction = (px - self.x_left_px) / (self.x_right_px - self.x_left_px)
        y_fraction = (self.y_bottom_px - py) / (self.y_bottom_px - self.y_top_px)
        x = _interpolate_axis(x_fraction, self.x_min, self.x_max, self.x_scale)
        y = _interpolate_axis(y_fraction, self.y_min, self.y_max, self.y_scale)
        return x, y

    def data_to_pixel(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        x_values, y_values = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        x_fraction = _inverse_axis(x_values, self.x_min, self.x_max, self.x_scale)
        y_fraction = _inverse_axis(y_values, self.y_min, self.y_max, self.y_scale)
        px = self.x_left_px + x_fraction * (self.x_right_px - self.x_left_px)
        py = self.y_bottom_px - y_fraction * (self.y_bottom_px - self.y_top_px)
        return px, py

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CurveColor:
    rgb: tuple[int, int, int]
    pixel_count: int

    @property
    def label(self) -> str:
        return f"RGB {self.rgb[0]}, {self.rgb[1]}, {self.rgb[2]} · {self.pixel_count}px"


@dataclass(frozen=True)
class DigitizationResult:
    data: pd.DataFrame
    curve_names: tuple[str, ...]
    warnings: tuple[str, ...]
    extraction_method: str = "RGB color-distance segmentation; separate vertical pixel clusters per x column"
    plot_region_pixels: tuple[int, int, int, int] | None = None


def _interpolate_axis(fraction: np.ndarray, minimum: float, maximum: float, scale: ImageScale) -> np.ndarray:
    if scale == "log":
        return np.power(10.0, np.log10(minimum) + fraction * (np.log10(maximum) - np.log10(minimum)))
    return minimum + fraction * (maximum - minimum)


def _inverse_axis(values: np.ndarray, minimum: float, maximum: float, scale: ImageScale) -> np.ndarray:
    if scale == "log":
        if np.any(values <= 0):
            raise ValueError("Values mapped to a logarithmic axis must be positive")
        return (np.log10(values) - np.log10(minimum)) / (np.log10(maximum) - np.log10(minimum))
    return (values - minimum) / (maximum - minimum)


def _read_source_bytes(source: ImageSource) -> tuple[bytes, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.read_bytes(), path.name
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), "uploaded-image"
    if hasattr(source, "read"):
        name = str(getattr(source, "name", "uploaded-image"))
        payload = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        return bytes(payload), Path(name).name
    raise ValueError("Image source must be a path, bytes, or a binary file object")


def load_graph_image(source: ImageSource, *, source_name: str | None = None) -> GraphImage:
    """Validate and decode PNG/JPEG/WEBP/TIFF input without changing its source."""
    payload, inferred_name = _read_source_bytes(source)
    if not payload:
        raise ValueError("The uploaded image is empty")
    try:
        with Image.open(BytesIO(payload)) as opened:
            image_format = (opened.format or "").upper()
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported image format {image_format or 'unknown'}; use PNG, JPG, WEBP, or TIFF")
            if opened.width <= 0 or opened.height <= 0 or opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"Image dimensions must be positive and no larger than {MAX_IMAGE_PIXELS:,} pixels total")
            opened.seek(0)
            image = opened.convert("RGB").copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The file could not be decoded as a supported image") from error
    import hashlib

    return GraphImage(
        image=image,
        source_name=source_name or inferred_name,
        source_format=image_format,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


def render_pdf_page(source: ImageSource, page_number: int, *, dpi: int = 150, max_pixels: int = MAX_IMAGE_PIXELS) -> GraphImage:
    """Render a chosen 1-based PDF page using the project's PyMuPDF dependency."""
    if page_number < 1:
        raise ValueError("PDF page numbers are 1-based and must be positive")
    if dpi < 36 or dpi > 600:
        raise ValueError("PDF render DPI must be between 36 and 600")
    payload, inferred_name = _read_source_bytes(source)
    import pymupdf as fitz

    try:
        document = fitz.open(stream=payload, filetype="pdf")
    except Exception as error:
        raise ValueError(f"The PDF could not be opened: {error}") from error
    try:
        if page_number > document.page_count:
            raise ValueError(f"PDF has {document.page_count} page(s); page {page_number} does not exist")
        page = document.load_page(page_number - 1)
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        estimated_width = int(page.rect.width * scale)
        estimated_height = int(page.rect.height * scale)
        if estimated_width * estimated_height > max_pixels:
            scale *= (max_pixels / (estimated_width * estimated_height)) ** 0.5
            matrix = fitz.Matrix(scale, scale)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    finally:
        document.close()
    import hashlib

    return GraphImage(
        image=image,
        source_name=inferred_name,
        source_format="PDF",
        page_number=page_number,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


def validate_crop(image_size: tuple[int, int], crop: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Validate ``(left, top, right, bottom)`` image coordinates."""
    width, height = image_size
    left, top, right, bottom = crop
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"Plot crop must satisfy 0 ≤ left < right ≤ {width} and 0 ≤ top < bottom ≤ {height}")
    return crop


def validate_exclusion_regions(
    image_size: tuple[int, int],
    crop: tuple[int, int, int, int],
    regions: Sequence[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], ...]:
    """Validate absolute-pixel exclusion rectangles contained by the plot crop.

    Bounds use the same half-open convention as Pillow crops:
    ``(left, top, right, bottom)``.
    """
    left, top, right, bottom = validate_crop(image_size, crop)
    validated: list[tuple[int, int, int, int]] = []
    for index, region in enumerate(regions, start=1):
        if len(region) != 4:
            raise ValueError(f"Exclusion region {index} must contain left, top, right, and bottom pixels")
        ex_left, ex_top, ex_right, ex_bottom = region
        if not (left <= ex_left < ex_right <= right and top <= ex_top < ex_bottom <= bottom):
            raise ValueError(
                f"Exclusion region {index} must be a non-empty rectangle inside the selected plot crop "
                f"({left}, {top}, {right}, {bottom})"
            )
        validated.append((int(ex_left), int(ex_top), int(ex_right), int(ex_bottom)))
    return tuple(validated)


def detect_curve_colors(
    image: Image.Image,
    crop: tuple[int, int, int, int],
    *,
    max_colors: int = 8,
    minimum_pixels: int = 5,
    exclusion_regions: Sequence[tuple[int, int, int, int]] = (),
) -> list[CurveColor]:
    """Return dominant non-background stroke colors for user confirmation.

    Gray grid/axis pixels are ignored when chromatic colors exist. If the region
    is monochrome, a few dark colors are exposed and the caller should review
    the crop carefully because axes/text can share the curve color.
    """
    left, top, right, bottom = validate_crop(image.size, crop)
    exclusion_regions = validate_exclusion_regions(image.size, crop, exclusion_regions)
    region_pixels = np.asarray(image.crop(crop).convert("RGB"))
    keep = np.ones(region_pixels.shape[:2], dtype=bool)
    for ex_left, ex_top, ex_right, ex_bottom in exclusion_regions:
        keep[ex_top - top:ex_bottom - top, ex_left - left:ex_right - left] = False
    pixels = region_pixels[keep]
    pixels = pixels[np.max(pixels, axis=1) - np.min(pixels, axis=1) > 0]
    if pixels.size == 0:
        return []
    channel_range = pixels.max(axis=1) - pixels.min(axis=1)
    colorful = channel_range >= 28
    if colorful.any():
        colored_pixels = pixels[colorful]
        normalized = colored_pixels.astype(float) / 255.0
        maximum = normalized.max(axis=1)
        minimum = normalized.min(axis=1)
        saturation = np.divide(maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0)
        # Favor line cores over low-saturation anti-aliased fringes, which
        # otherwise appear as separate false curve colors in the palette.
        vivid = saturation >= 0.68
        pixels = colored_pixels[vivid] if vivid.any() else colored_pixels
    else:
        brightness = pixels.mean(axis=1)
        pixels = pixels[brightness <= 115]
    if pixels.size == 0:
        return []
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    cutoff = max(minimum_pixels, int(pixels.shape[0] * 0.00015))
    candidates: list[CurveColor] = []
    for index in order:
        count = int(counts[index])
        if count < cutoff:
            break
        rgb = tuple(int(value) for value in colors[index])
        if any(np.linalg.norm(np.asarray(rgb) - np.asarray(item.rgb)) < 42 for item in candidates):
            continue
        candidates.append(CurveColor(rgb=rgb, pixel_count=count))
        if len(candidates) >= max_colors:
            break
    return candidates


def digitize_color_curves(
    image: Image.Image,
    *,
    crop: tuple[int, int, int, int],
    plot_region: tuple[int, int, int, int] | None = None,
    calibration: AxisCalibration,
    colors: list[tuple[str, tuple[int, int, int]]],
    color_tolerance: float = 62.0,
    max_points: int | None = None,
    exclusion_regions: Sequence[tuple[int, int, int, int]] = (),
    axis_pixels: tuple[float | None, float | None] | None = None,
    axis_mask_half_width_px: int = 2,
) -> DigitizationResult:
    """Digitize selected colors, optionally masking same-color detected axes.

    ``crop`` is the image area being inspected; ``plot_region`` is the selected
    extraction rectangle within that area. If omitted, the crop is the plot
    region. Calibration anchors map pixels to values but never clip the region.

    ``axis_pixels`` contains the horizontal-axis y coordinate and vertical-axis
    x coordinate in original-image pixels. A mask is applied only when the
    selected curve color has substantial support along that detected line, so
    differently colored data curves keep their axis-crossing points.
    """
    left, top, right, bottom = validate_crop(image.size, crop)
    # ``crop`` selects the source pixels to inspect. ``plot_region`` is the
    # explicit rectangle inside that crop from which curve observations may
    # be extracted. Calibration anchors are references for a coordinate
    # transform; they are deliberately not extraction bounds.
    plot_region = validate_crop(image.size, plot_region or crop)
    plot_left, plot_top, plot_right, plot_bottom = plot_region
    if not (left <= plot_left < plot_right <= right and top <= plot_top < plot_bottom <= bottom):
        raise ValueError("The selected plot region must lie inside the image-analysis crop")
    exclusion_regions = validate_exclusion_regions(image.size, crop, exclusion_regions)
    calibration.validate_image_bounds(image.size, plot_region)
    if not colors:
        raise ValueError("Select at least one curve color before digitizing")
    names = [name.strip() for name, _ in colors]
    if any(not name for name in names):
        raise ValueError("Curve names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("Curve names must be unique")
    if not np.isfinite(color_tolerance) or color_tolerance <= 0 or color_tolerance > 255:
        raise ValueError("Color tolerance must be greater than 0 and no more than 255")
    if max_points is not None and max_points < 3:
        raise ValueError("A point limit must be at least 3")
    if not isinstance(axis_mask_half_width_px, int) or not 0 <= axis_mask_half_width_px <= 10:
        raise ValueError("Axis-mask half width must be an integer from 0 to 10 pixels")
    if axis_pixels is not None:
        if len(axis_pixels) != 2 or any(value is not None and not np.isfinite(value) for value in axis_pixels):
            raise ValueError("Detected axis coordinates must be finite original-image pixels or None")
        horizontal_axis_y, vertical_axis_x = axis_pixels
        if horizontal_axis_y is not None and not top <= horizontal_axis_y < bottom:
            raise ValueError("Detected horizontal axis must lie inside the selected crop")
        if vertical_axis_x is not None and not left <= vertical_axis_x < right:
            raise ValueError("Detected vertical axis must lie inside the selected crop")
    for name, color in colors:
        if len(color) != 3 or any(not isinstance(channel, (int, np.integer)) or not 0 <= channel <= 255 for channel in color):
            raise ValueError(f"Curve color for {name!r} must be an RGB triplet with channels from 0 to 255")
    region_pixels = np.asarray(image.crop(crop).convert("RGB"), dtype=np.float32)
    rows: list[pd.DataFrame] = []
    warnings: list[str] = []
    names: list[str] = []
    for name, color in colors:
        name = name.strip()
        target = np.asarray(color, dtype=np.float32)
        region = region_pixels
        distance = np.sqrt(np.sum((region - target) ** 2, axis=2))
        mask = distance <= color_tolerance
        # Only the user's/detector's selected plot-region rectangle restricts
        # extraction. Calibration may use interior ticks and can be evaluated
        # outside those anchors, but never outside this selected image region.
        x_indices = np.arange(left, right)
        y_indices = np.arange(top, bottom)
        mask &= (x_indices[None, :] >= plot_left) & (x_indices[None, :] < plot_right)
        mask &= (y_indices[:, None] >= plot_top) & (y_indices[:, None] < plot_bottom)
        removed_pixels = 0
        masked_axes: list[str] = []
        if axis_pixels is not None:
            horizontal_axis_y, vertical_axis_x = axis_pixels
            if horizontal_axis_y is not None and axis_mask_half_width_px:
                local_axis_y = int(round(horizontal_axis_y - top))
                search_radius = max(axis_mask_half_width_px + 2, 4)
                plot_local_top, plot_local_bottom = plot_top - top, plot_bottom - top
                candidates = range(max(plot_local_top, local_axis_y - search_radius), min(plot_local_bottom, local_axis_y + search_radius + 1))
                best_row = max(candidates, key=lambda row: int(mask[row, :].sum()), default=local_axis_y)
                required_support = max(8, int((plot_right - plot_left) * 0.04))
                if 0 <= best_row < mask.shape[0] and int(mask[best_row, :].sum()) >= required_support:
                    row_slice = slice(max(0, best_row - axis_mask_half_width_px), min(mask.shape[0], best_row + axis_mask_half_width_px + 1))
                    removed_pixels += int(mask[row_slice, :].sum())
                    mask[row_slice, :] = False
                    masked_axes.append("horizontal")
            if vertical_axis_x is not None and axis_mask_half_width_px:
                local_axis_x = int(round(vertical_axis_x - left))
                search_radius = max(axis_mask_half_width_px + 2, 4)
                plot_local_left, plot_local_right = plot_left - left, plot_right - left
                candidates = range(max(plot_local_left, local_axis_x - search_radius), min(plot_local_right, local_axis_x + search_radius + 1))
                best_column = max(candidates, key=lambda column: int(mask[:, column].sum()), default=local_axis_x)
                required_support = max(8, int((plot_bottom - plot_top) * 0.04))
                if 0 <= best_column < mask.shape[1] and int(mask[:, best_column].sum()) >= required_support:
                    column_slice = slice(max(0, best_column - axis_mask_half_width_px), min(mask.shape[1], best_column + axis_mask_half_width_px + 1))
                    removed_pixels += int(mask[:, column_slice].sum())
                    mask[:, column_slice] = False
                    masked_axes.append("vertical")
        if masked_axes:
            warnings.append(
                f"{name}: the selected color also matches the detected {' and '.join(masked_axes)} axis; its stroke was masked to avoid contaminating the curve. "
                "Curve crossings may become gaps, and missing points are not interpolated."
            )
        for ex_left, ex_top, ex_right, ex_bottom in exclusion_regions:
            x_slice = slice(ex_left - left, ex_right - left)
            y_slice = slice(ex_top - top, ex_bottom - top)
            excluded_pixels = mask[y_slice, x_slice]
            removed_pixels += int(excluded_pixels.sum())
            mask[y_slice, x_slice] = False
        if removed_pixels and exclusion_regions:
            warnings.append(
                f"{name}: exclusion region(s) removed {removed_pixels} matching pixels. "
                "If an exclusion crosses this curve it creates a data gap; missing points are not interpolated."
            )
        # Avoid including pale background pixels when a light color was chosen.
        if np.min(target) > 225 and np.max(target) - np.min(target) < 25:
            warnings.append(f"{name}: selected color is close to the plot background; inspect the extracted points")
        local_y, local_x = np.nonzero(mask)
        if local_x.size == 0:
            warnings.append(f"{name}: no pixels matched RGB {tuple(int(v) for v in color)}")
            continue
        absolute_x = local_x + left
        absolute_y = local_y + top
        order = np.argsort(absolute_x, kind="stable")
        sorted_x, sorted_y = absolute_x[order], absolute_y[order]
        unique_x, group_starts = np.unique(sorted_x, return_index=True)
        group_ends = np.r_[group_starts[1:], len(sorted_x)]
        vertical_spread = np.asarray(
            [np.max(sorted_y[start:end]) - np.min(sorted_y[start:end]) for start, end in zip(group_starts, group_ends)],
            dtype=float,
        )
        suspicious_columns = int(np.count_nonzero(vertical_spread >= max(7.0, region_pixels.shape[0] * 0.02)))
        if suspicious_columns:
            warnings.append(
                f"{name}: {suspicious_columns} x-column(s) contain vertically spread matching pixels. "
                "The digitizer separates distinct vertical pixel clusters, but overlapping branches, same-color legend/inset marks, or unrelated annotations can still be ambiguous; review the pixel overlay or add an exclusion region before fitting."
            )
        # A fast oscillatory trace can cross the same one-pixel-wide x column
        # at multiple y positions. Reducing all matching pixels in that
        # column to one median creates a fictitious midline and destroys the
        # transient. Preserve separate vertical clusters as separate samples;
        # close/merged strokes remain subject to raster-resolution limits.
        sample_x: list[float] = []
        sample_y: list[float] = []
        cluster_gap_px = 2
        for x_pixel, start, end in zip(unique_x, group_starts, group_ends):
            column_y = np.sort(sorted_y[start:end])
            cluster_starts = np.r_[0, np.flatnonzero(np.diff(column_y) > cluster_gap_px) + 1]
            cluster_ends = np.r_[cluster_starts[1:], len(column_y)]
            for cluster_start, cluster_end in zip(cluster_starts, cluster_ends):
                cluster = column_y[cluster_start:cluster_end]
                sample_x.append(float(x_pixel))
                sample_y.append(float(np.median(cluster)))
        sample_x_array = np.asarray(sample_x, dtype=float)
        sample_y_array = np.asarray(sample_y, dtype=float)
        if len(sample_x_array) > len(unique_x):
            warnings.append(
                f"{name}: separate branches or overlapping marks occur in the same image columns; all resolvable vertical clusters were retained as separate observations. "
                "Their exact sub-pixel x positions cannot be recovered from this raster."
            )
        column_coverage = len(unique_x) / max(1.0, plot_right - plot_left)
        data_x, data_y = calibration.pixel_to_data(sample_x_array, sample_y_array)
        valid = np.isfinite(data_x) & np.isfinite(data_y)
        if not valid.all():
            warnings.append(f"{name}: non-finite calibrated points were discarded")
        if not valid.any():
            warnings.append(f"{name}: all matched pixels mapped to non-finite coordinates and were discarded")
            continue
        if (
            np.min(sample_x_array[valid]) < calibration.x_left_px
            or np.max(sample_x_array[valid]) > calibration.x_right_px
            or np.min(sample_y_array[valid]) < calibration.y_top_px
            or np.max(sample_y_array[valid]) > calibration.y_bottom_px
        ):
            warnings.append(
                f"{name}: some extracted points lie outside the calibration-anchor rectangle. Their coordinates use the calibrated linear/log mapping extrapolated within the selected plot region; review edge values."
            )
        curve = pd.DataFrame({
            "curve_name": name,
            "pixel_x": sample_x_array[valid],
            "pixel_y": sample_y_array[valid],
            "x": np.asarray(data_x)[valid],
            "y": np.asarray(data_y)[valid],
        }).sort_values("x", kind="stable")
        if max_points is not None and len(curve) > max_points:
            selected_indices = np.unique(np.linspace(0, len(curve) - 1, max_points).round().astype(int))
            curve = curve.iloc[selected_indices].reset_index(drop=True)
            warnings.append(f"{name}: evenly downsampled to the configured limit of {len(curve)} points")
        if len(curve) < 3:
            warnings.append(f"{name}: only {len(curve)} points were extracted; curve shape is weakly supported")
        if column_coverage < 0.5:
            warnings.append(
                f"{name}: detected curve occupies {column_coverage:.0%} of the selected plot width. "
                "Fits are supported only over the observed x-span and may be unstable or misleading outside it."
            )
        rows.append(curve)
        names.append(name)
    if not rows:
        raise ValueError("No curve pixels were extracted. Adjust the crop, selected color, or color tolerance.")
    frame = pd.concat(rows, ignore_index=True)
    return DigitizationResult(data=frame, curve_names=tuple(names), warnings=tuple(warnings), plot_region_pixels=plot_region)


def long_to_wide(data: pd.DataFrame) -> pd.DataFrame:
    """Return a documented wide CSV view with one value column per curve."""
    required = {"curve_name", "x", "y"}
    if not required.issubset(data.columns):
        raise ValueError(f"Digitized data must contain columns: {', '.join(sorted(required))}")
    return data.pivot_table(index="x", columns="curve_name", values="y", aggfunc="median").reset_index()

