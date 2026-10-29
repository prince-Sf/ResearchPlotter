"""Automatic, reviewable axis calibration for graph images.

The detector uses local image geometry and, when installed, local Tesseract OCR.
It never infers a curve equation. OCR and geometry are estimates that should be
reviewed before digitizing research figures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import math
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from research_plotter.analysis.graph_digitization import AxisCalibration, ImageScale, validate_crop


@dataclass(frozen=True)
class OCRWord:
    """A locally recognized word and its bounding box in image pixels."""

    text: str
    left: float
    top: float
    right: float
    bottom: float
    confidence: float


@dataclass(frozen=True)
class TickLabel:
    """A numeric tick label associated with a pixel position."""

    axis: Literal["x", "y"]
    pixel: float
    label: str
    value: float | None
    ocr_confidence: float
    review_status: Literal["ACCEPTED", "REVIEW_REQUIRED", "REJECTED"] = "ACCEPTED"
    recognized_label: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class AxisLine:
    """Detected line geometry in original-image coordinates."""

    pixel: float
    start: float
    end: float
    strength: float


@dataclass(frozen=True)
class AutomaticCalibrationResult:
    """Automatic calibration suggestions and their review diagnostics."""

    values: dict[str, float | None]
    status: Literal["DETECTED", "PARTIAL", "NEEDS_REVIEW"]
    confidence: Literal["HIGH", "MEDIUM", "LOW", "NONE"]
    x_scale: ImageScale
    y_scale: ImageScale
    plot_region: tuple[int, int, int, int] | None
    x_axis_pixel: float | None
    y_axis_pixel: float | None
    x_ticks: tuple[TickLabel, ...]
    y_ticks: tuple[TickLabel, ...]
    ocr_available: bool
    warnings: tuple[str, ...]
    review_ticks: tuple[TickLabel, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "values": self.values,
            "status": self.status,
            "confidence": self.confidence,
            "x_scale": self.x_scale,
            "y_scale": self.y_scale,
            "plot_region": self.plot_region,
            "x_axis_pixel": self.x_axis_pixel,
            "y_axis_pixel": self.y_axis_pixel,
            "x_ticks": [asdict(tick) for tick in self.x_ticks],
            "y_ticks": [asdict(tick) for tick in self.y_ticks],
            "ocr_available": self.ocr_available,
            "warnings": list(self.warnings),
            "review_ticks": [asdict(tick) for tick in self.review_ticks],
        }


@dataclass(frozen=True)
class _LineCandidate:
    pixel: int
    start: int
    end: int
    strength: float


@dataclass(frozen=True)
class _AxisFit:
    first_pixel: float
    first_value: float
    last_pixel: float
    last_value: float
    scale: ImageScale
    ticks: tuple[TickLabel, ...]
    rms_pixels: float
    inlier_count: int


@dataclass(frozen=True)
class _LocalTickRead:
    """One OCR interpretation with a crop-relative text bounding box."""

    text: str
    value: float | None
    confidence: float
    box: tuple[float, float, float, float] | None = None


_PI_LABEL = re.compile(r"^(?P<sign>[+-]?)(?P<num>\d*(?:\.\d+)?)?(?:π|pi)(?:/(?P<den>\d+(?:\.\d+)?))?$", re.IGNORECASE)
_FRACTION = re.compile(r"^(?P<num>[+-]?\d+)\s*/\s*(?P<den>\d+)$")
_POWER_OF_TEN = re.compile(r"^(?P<base>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:×|x|\*)?\s*10\s*\^\s*(?P<exp>[+-]?\d+)$", re.IGNORECASE)
_BARE_POWER_OF_TEN = re.compile(r"^10\s*\^\s*(?P<exp>[+-]?\d+)$", re.IGNORECASE)


def parse_tick_label(text: str) -> float | None:
    """Parse unambiguous numeric, scientific, fractional, or pi tick labels."""
    raw = str(text).strip().lower()
    has_superscript = any(character in raw for character in "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
    normalized = raw
    normalized = normalized.replace("−", "-").replace("–", "-").replace("—", "-").replace("﹣", "-")
    normalized = normalized.replace("\\pi", "pi")
    # Greek-script OCR occasionally reads a fraction slash as a vertical bar.
    # Normalize it only in an unambiguous pi-fraction context.
    normalized = re.sub(r"(π|pi)\s*[|:]\s*(?=\d)", r"\1/", normalized)
    normalized = normalized.replace("−", "-")
    normalized = re.sub(r"[\s$\\{}]", "", normalized)
    # OCR sometimes emits superscript exponents for scientific notation.
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    normalized = normalized.translate(superscripts)
    if has_superscript:
        normalized = re.sub(r"(10)([+-]?\d+)$", r"\1^\2", normalized)
    # A trailing degree glyph is a unit marker, not part of the numeric value.
    # Do not strip arbitrary trailing letters: strings such as 2705 can be real
    # values and must remain numerically distinct from a possibly lost degree
    # sign until independent tick evidence resolves them.
    normalized = re.sub(r"(?:°|º)$", "", normalized)
    if not normalized:
        return None

    pi_match = _PI_LABEL.fullmatch(normalized)
    if pi_match:
        sign = -1.0 if pi_match.group("sign") == "-" else 1.0
        numerator_text = pi_match.group("num")
        numerator = float(numerator_text) if numerator_text else 1.0
        denominator = float(pi_match.group("den") or 1.0)
        if denominator == 0:
            return None
        return sign * numerator * math.pi / denominator

    fraction = _FRACTION.fullmatch(normalized)
    if fraction:
        denominator = int(fraction.group("den"))
        return float(Fraction(int(fraction.group("num")), denominator)) if denominator else None

    # Accept 1×10^-3, 1x10-3 only when the exponent separator is explicit as ^,
    # and do not reinterpret ordinary labels such as 10-3 as scientific form.
    power = _POWER_OF_TEN.fullmatch(normalized)
    if power:
        try:
            return float(power.group("base")) * 10.0 ** int(power.group("exp"))
        except OverflowError:
            return None
    bare_power = _BARE_POWER_OF_TEN.fullmatch(normalized)
    if bare_power:
        try:
            return 10.0 ** int(bare_power.group("exp"))
        except OverflowError:
            return None

    if "," in normalized and not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", normalized):
        return None
    try:
        value = float(normalized.replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


_PI_OCR_ALIAS = re.compile(r"^(?P<sign>[+-]?)(?P<num>\d*(?:\.\d+)?)(?P<glyph>n|m|tt)(?:/(?P<den>\d+(?:\.\d+)?))?$", re.IGNORECASE)


def _looks_like_pi_ocr(text: str) -> bool:
    """Return whether OCR produced a common glyph-confusion candidate for π.

    Aliases are *not* accepted by :func:`parse_tick_label` alone: ``2m`` can
    be a real unit label. They are only resolved in an axis-label group when
    neighboring labels provide corroborating π notation.
    """
    normalized = str(text).strip().lower().replace("−", "-").replace("–", "-").replace("\ufffd", "")
    normalized = re.sub(r"[\s$\\{}]", "", normalized)
    normalized = re.sub(r"(n|m|tt)[|:]", r"\1/", normalized)
    return _PI_OCR_ALIAS.fullmatch(normalized) is not None


def _parse_axis_tick_label(text: str, *, allow_pi_ocr_alias: bool = False) -> tuple[float | None, bool]:
    """Parse OCR text, optionally resolving π glyph confusions with context."""
    value = parse_tick_label(text)
    if value is not None or not allow_pi_ocr_alias:
        return value, False
    normalized = str(text).strip().lower().replace("−", "-").replace("–", "-").replace("\ufffd", "")
    normalized = re.sub(r"[\s$\\{}]", "", normalized)
    normalized = re.sub(r"(n|m|tt)[|:]", r"\1/", normalized)
    match = _PI_OCR_ALIAS.fullmatch(normalized)
    if match is None:
        return None, False
    numerator = float(match.group("num") or 1.0)
    denominator = float(match.group("den") or 1.0)
    if denominator == 0:
        return None, False
    sign = -1.0 if match.group("sign") == "-" else 1.0
    return sign * numerator * math.pi / denominator, True


def local_ocr_available() -> tuple[bool, str | None]:
    """Return whether the optional local pytesseract + Tesseract stack exists."""
    try:
        import pytesseract
    except ImportError:
        return False, "Optional OCR is unavailable (pytesseract is not installed)."
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        # On Windows, python launched from an IDE can inherit a PATH that does
        # not contain the standard per-machine installation directory.
        executable = shutil.which("tesseract")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        standard_install = Path(program_files) / "Tesseract-OCR" / "tesseract.exe"
        if executable is None and standard_install.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(standard_install)
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                return False, "Optional OCR is unavailable (the local Tesseract executable could not be started)."
        else:
            return False, "Optional OCR is unavailable (the local Tesseract executable was not found)."
    return True, None


def recognize_local_words(image: Image.Image) -> list[OCRWord]:
    """Recognize local labels with structured boxes and OCR confidence.

    Tesseract language discovery on Windows often omits nested script models
    (for example ``script/Greek``) from ``get_languages`` even though the
    model is installed. Try that model opportunistically, then fall back to
    installed English/equation models. Every attempt is local and isolated so
    a missing optional language pack cannot disable ordinary OCR.
    """
    available, reason = local_ocr_available()
    if not available:
        raise RuntimeError(reason or "Local OCR is unavailable")
    import pytesseract
    from pytesseract import Output

    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    gray = ImageEnhance.Contrast(gray).enhance(1.6)
    # Small scientific figures usually need more than a 2x enlargement for
    # decimal points, minus signs, and superscripts to survive rasterization.
    # The previous 2x enlargement was insufficient for the small stacked
    # fractions in ordinary paper figures. A 3x floor improved π/fraction
    # recognition on the user's 1570px source without requiring OCR services;
    # smaller charts are enlarged to 4x.
    scale = min(4, max(3, int(round(4000 / max(image.size)))))
    enlarged = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.LANCZOS)
    try:
        installed = set(pytesseract.get_languages(config=""))
    except Exception:
        installed = {"eng"}

    language_options = ["eng+script/Greek"]  # nested script packs may not be listed by pytesseract
    if {"eng", "ell"}.issubset(installed):
        language_options.append("eng+ell")
    language_options.append("eng")

    attempts: list[list[OCRWord]] = []
    errors: list[str] = []
    for language in dict.fromkeys(language_options):
        # Block mode preserves stacked fractions as separate lines; sparse
        # mode complements it for isolated numeric labels such as 0.5 that
        # block mode may classify as punctuation. Both passes remain local.
        page_modes = (6, 11) if language == "eng+script/Greek" else (6,)
        for page_mode in page_modes:
            try:
                data = pytesseract.image_to_data(
                    enlarged, lang=language, config=f"--psm {page_mode}", output_type=Output.DICT
                )
            except Exception as error:
                errors.append(f"{language}/psm{page_mode}: {error}")
                continue
            words: list[OCRWord] = []
            for index, raw_text in enumerate(data.get("text", [])):
                text = str(raw_text).strip()
                try:
                    confidence = float(data["conf"][index])
                    left = float(data["left"][index]) / scale
                    top = float(data["top"][index]) / scale
                    right = left + float(data["width"][index]) / scale
                    bottom = top + float(data["height"][index]) / scale
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
                if text and confidence >= 0 and right > left and bottom > top:
                    words.append(OCRWord(text, left, top, right, bottom, confidence))
            attempts.append(words)
    if not attempts:
        reason = errors[0] if errors else "Tesseract returned no OCR result"
        raise RuntimeError(f"Local OCR failed: {reason}")

    # Merge overlapping reads from Greek/math and English models. The Greek
    # script model recovers π well, while English often reads a small Greek
    # iota as the digit 1 (e.g. -1 rather than -ι). Prefer an unambiguous
    # numeric/pi parse over a slightly higher confidence glyph confusion.
    clusters: list[list[OCRWord]] = []
    for word in sorted((word for attempt in attempts for word in attempt), key=lambda item: (item.top, item.left)):
        center_x, center_y = (word.left + word.right) / 2, (word.top + word.bottom) / 2
        match: list[OCRWord] | None = None
        for cluster in clusters:
            existing = cluster[0]
            other_x = (existing.left + existing.right) / 2
            other_y = (existing.top + existing.bottom) / 2
            if abs(center_x - other_x) <= max(8.0, (word.right - word.left + existing.right - existing.left) * 0.3) and abs(center_y - other_y) <= max(5.0, (word.bottom - word.top + existing.bottom - existing.top) * 0.3):
                match = cluster
                break
        if match is None:
            clusters.append([word])
        else:
            match.append(word)

    def preference(word: OCRWord) -> tuple[int, int, int, float]:
        direct_value = parse_tick_label(word.text) is not None
        direct_pi = bool(re.search(r"π|\\pi|\bpi\b", word.text, re.IGNORECASE))
        replacement = "\ufffd" in word.text
        return int(direct_value), int(direct_pi), int(not replacement), word.confidence

    merged: list[OCRWord] = []
    for cluster in clusters:
        unique: dict[tuple[str, float, float], OCRWord] = {}
        for word in cluster:
            unique[(word.text, round(word.left, 1), round(word.top, 1))] = word
        merged.append(max(unique.values(), key=preference))
    return merged


def _longest_run(line: np.ndarray, *, bridge_gap: int = 0) -> tuple[int, int, int]:
    indices = np.flatnonzero(line)
    if not len(indices):
        return 0, 0, 0
    breaks = np.flatnonzero(np.diff(indices) > bridge_gap + 1) + 1
    groups = np.split(indices, breaks)
    group = max(groups, key=len)
    return int(group[-1] - group[0] + 1), int(group[0]), int(group[-1] + 1)


def _line_candidates(mask: np.ndarray, *, horizontal: bool) -> list[_LineCandidate]:
    height, width = mask.shape
    axis_length = width if horizontal else height
    minimum = max(24, int(axis_length * 0.34))
    candidates: list[_LineCandidate] = []
    count = height if horizontal else width
    for position in range(count):
        line = mask[position, :] if horizontal else mask[:, position]
        # A colored curve or antialiasing can cover short sections of a dark
        # axis. Bridge only small interruptions; unrelated inset edges remain
        # too short to qualify as primary axes.
        # A 2.5% bridge can fuse a nearby axis title (notably an italic y)
        # into the vertical axis and fabricate tick-like support at the top.
        # Real axis interruptions from curve crossings are only a few pixels;
        # keep the bridge local and proportional instead.
        # Use a small absolute floor so the same short interruption can be
        # bridged in modest-size charts as in large rasters. Cap the relative
        # allowance to avoid joining nearby titles/annotations to long axes.
        bridge_gap = max(9, min(10, int(round(axis_length * 0.012))))
        length, start, end = _longest_run(line, bridge_gap=bridge_gap)
        if length >= minimum:
            candidates.append(_LineCandidate(position, start, end, length / max(1, axis_length)))
    # Collapse anti-aliased/thick lines to the strongest row or column.
    merged: list[_LineCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.pixel):
        if merged and candidate.pixel - merged[-1].pixel <= 2:
            if candidate.strength > merged[-1].strength:
                merged[-1] = candidate
        else:
            merged.append(candidate)
    return merged


def _ocr_label_bands(
    crop: tuple[int, int, int, int], words: Sequence[OCRWord]
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return OCR x-tick row centers and y-tick column centers for axis scoring."""
    left, top, right, bottom = crop
    width, height = right - left, bottom - top
    items = [
        item for item in _group_words(words)
        if parse_tick_label(item[0]) is not None or _looks_like_pi_ocr(item[0])
    ]

    def best_band(axis: Literal["x", "y"]) -> list[tuple[float, float]]:
        dimension = width if axis == "x" else height
        clusters = _baseline_clusters(
            items, center_index=1 if axis == "x" else 0, tolerance=max(4.0, dimension * 0.045)
        )
        candidates: list[tuple[tuple[int, float], list[tuple[float, float]]]] = []
        for cluster in clusters:
            axis_centers = [
                (item[1] + item[3]) / 2 if axis == "x" else (item[2] + item[4]) / 2
                for item in cluster
            ]
            span = float(np.ptp(axis_centers)) if len(axis_centers) > 1 else 0.0
            if len(axis_centers) >= 2 and span >= dimension * 0.18:
                centers = [((item[1] + item[3]) / 2, (item[2] + item[4]) / 2) for item in cluster]
                candidates.append(((len(centers), span), centers))
        return max(candidates, key=lambda item: item[0])[1] if candidates else []

    return best_band("x"), best_band("y")


def _detect_axes(
    image: Image.Image,
    crop: tuple[int, int, int, int],
    words: Sequence[OCRWord] = (),
) -> tuple[AxisLine | None, AxisLine | None, tuple[int, int, int, int] | None]:
    left, top, right, bottom = validate_crop(image.size, crop)
    region = image.crop(crop).convert("RGB")
    pixels = np.asarray(region, dtype=np.uint8)
    gray = np.asarray(ImageOps.grayscale(region), dtype=np.uint8)
    chroma = pixels.max(axis=2).astype(np.int16) - pixels.min(axis=2).astype(np.int16)
    corner = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    light_background = float(np.median(corner)) >= 127
    neutral = chroma < 90
    neutral_mask = ((gray < 145) & neutral) if light_background else ((gray > 190) & neutral)
    # Scientific plots often draw their axes in the same accent colour as a
    # series.  Do not require neutral RGB pixels: a second contrast mask
    # admits those lines while the long-line geometry and tick-label alignment
    # below still distinguish axes from curves and text.
    background = np.median(np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]), axis=0), axis=0)
    contrast = np.linalg.norm(pixels.astype(float) - background.astype(float), axis=2)
    color_agnostic_mask = contrast >= 48.0
    mask = neutral_mask | color_agnostic_mask
    horizontal = _line_candidates(mask, horizontal=True)
    vertical = _line_candidates(mask, horizontal=False)
    if not horizontal or not vertical:
        return None, None, None

    height, width = mask.shape
    x_label_centers, y_label_centers = _ocr_label_bands(crop, words) if words else ([], [])

    def support_x(line: _LineCandidate) -> float:
        if not x_label_centers:
            return 0.0
        tolerance = max(24.0, height * 0.09)
        aligned = sum(
            line.start - 3 <= center_x - left <= line.end + 3
            and abs((top + line.pixel) - center_y) <= tolerance
            for center_x, center_y in x_label_centers
        )
        return aligned / len(x_label_centers)

    def support_y(line: _LineCandidate) -> float:
        if not y_label_centers:
            return 0.0
        tolerance = max(30.0, width * 0.09)
        aligned = sum(
            line.start - 3 <= center_y - top <= line.end + 3
            and abs((left + line.pixel) - center_x) <= tolerance
            for center_x, center_y in y_label_centers
        )
        return aligned / len(y_label_centers)

    pairs: list[tuple[float, _LineCandidate, _LineCandidate]] = []
    for hline in horizontal:
        for vline in vertical:
            intersects = hline.start <= vline.pixel < hline.end and vline.start <= hline.pixel < vline.end
            x_support = support_x(hline)
            y_support = support_y(vline)
            if not intersects and not (x_support >= 0.5 and y_support >= 0.5):
                continue
            # Text/tick alignment outweighs a decorative frame or gridline.
            # Without readable OCR, retain the geometry-only tie-breaker.
            preference = 0.035 * (hline.pixel / max(1, height - 1)) + 0.025 * (1 - vline.pixel / max(1, width - 1))
            gap_x = max(0, hline.start - vline.pixel, vline.pixel - (hline.end - 1)) / max(1, width)
            gap_y = max(0, vline.start - hline.pixel, hline.pixel - (vline.end - 1)) / max(1, height)
            score = hline.strength * vline.strength + preference + 2.0 * (x_support + y_support) - gap_x - gap_y
            pairs.append((score, hline, vline))
    if not pairs:
        return None, None, None
    _, hline, vline = max(pairs, key=lambda item: item[0])
    horizontal_line = AxisLine(float(top + hline.pixel), float(left + hline.start), float(left + hline.end - 1), hline.strength)
    vertical_line = AxisLine(float(left + vline.pixel), float(top + vline.start), float(top + vline.end - 1), vline.strength)
    plot_region = (
        max(left, int(round(horizontal_line.start))),
        max(top, int(round(vertical_line.start))),
        min(right, int(round(horizontal_line.end)) + 1),
        min(bottom, int(round(vertical_line.end)) + 1),
    )
    return horizontal_line, vertical_line, plot_region


def _axis_band_side(
    words: Sequence[OCRWord], axis: Literal["x", "y"], line: AxisLine,
    crop: tuple[int, int, int, int], *, cross_axis: AxisLine | None = None,
) -> int:
    """Choose the side carrying likely tick text, with a conventional fallback."""
    left, top, right, bottom = crop
    span = bottom - top if axis == "x" else right - left
    scores = {-1: 0, 1: 0}
    for word in words:
        if not _is_tick_text(word.text):
            continue
        center_x, center_y = (word.left + word.right) / 2, (word.top + word.bottom) / 2
        if axis == "y" and cross_axis is not None and abs(center_y - cross_axis.pixel) <= (bottom - top) * 0.22:
            continue
        along = center_x if axis == "x" else center_y
        across = center_y - line.pixel if axis == "x" else center_x - line.pixel
        if line.start - 4 <= along <= line.end + 4 and 2 <= abs(across) <= span * 0.24:
            scores[1 if across > 0 else -1] += 1
    default_side = 1 if axis == "x" else -1
    return max(scores, key=lambda value: (scores[value], value == default_side))


def _axis_label_band(
    image_size: tuple[int, int], crop: tuple[int, int, int, int],
    axis: Literal["x", "y"], line: AxisLine, side: int,
    *, words: Sequence[OCRWord] = (), cross_axis: AxisLine | None = None,
    cross_axis_side: int = 1,
) -> list[tuple[int, int, int, int]]:
    """Build local OCR crop(s) around one axis label band in source pixels."""
    left, top, right, bottom = crop
    width, _height = image_size
    if axis == "x":
        x0, x1 = max(left, int(math.floor(line.start))), min(right, int(math.ceil(line.end)) + 1)
        selected = []
        for word in words:
            if not _is_tick_text(word.text):
                continue
            center_x = (word.left + word.right) / 2
            center_y = (word.top + word.bottom) / 2
            across = center_y - line.pixel
            if line.start - 4 <= center_x <= line.end + 4 and 2 <= side * across <= (bottom - top) * 0.24:
                selected.append(word)
        if selected:
            pad = max(3, int(round(np.median([word.bottom - word.top for word in selected]) * 0.5)))
            label_top = min(word.top for word in selected)
            label_bottom = max(word.bottom for word in selected)
            if side > 0:
                y0 = max(top, int(math.floor(label_top - pad)), int(math.ceil(line.pixel + 3)))
                y1 = min(bottom, int(math.ceil(label_bottom + pad)))
            else:
                y0 = max(top, int(math.floor(label_top - pad)))
                y1 = min(bottom, int(math.ceil(label_bottom + pad)), int(math.floor(line.pixel - 3)))
        else:
            extent = max(28, int(round((bottom - top) * 0.12)))
            if side > 0:
                y0, y1 = max(top, int(math.ceil(line.pixel + 3))), min(bottom, int(math.ceil(line.pixel + extent)))
            else:
                y0, y1 = max(top, int(math.floor(line.pixel - extent))), min(bottom, int(math.floor(line.pixel - 3)))
        return [validate_crop(image_size, (x0, y0, x1, y1))] if x1 > x0 and y1 > y0 else []
    else:
        y0, y1 = max(top, int(math.floor(line.start))), min(bottom, int(math.ceil(line.end)) + 1)
        selected = []
        for word in words:
            if not _is_tick_text(word.text):
                continue
            center_x = (word.left + word.right) / 2
            center_y = (word.top + word.bottom) / 2
            across = center_x - line.pixel
            if not (line.start - 4 <= center_y <= line.end + 4 and 2 <= side * across <= (right - left) * 0.24):
                continue
            if cross_axis is not None and abs(center_y - cross_axis.pixel) <= (bottom - top) * 0.22:
                continue
            selected.append(word)
        if selected:
            pad = max(3, int(round(np.median([word.bottom - word.top for word in selected]) * 0.75)))
            label_left = min(word.left for word in selected)
            label_right = max(word.right for word in selected)
            if side < 0:
                x0 = max(left, int(math.floor(label_left - pad)))
                x1 = min(right, int(math.ceil(label_right + pad)), int(math.floor(line.pixel - 3)))
            else:
                x0 = max(left, int(math.floor(label_left - pad)), int(math.ceil(line.pixel + 3)))
                x1 = min(right, int(math.ceil(label_right + pad)))
        else:
            extent = max(48, int(round((right - left) * 0.09)))
            if side < 0:
                x0, x1 = max(left, int(math.floor(line.pixel - extent))), min(right, int(math.floor(line.pixel - 3)))
            else:
                x0, x1 = max(left, int(math.ceil(line.pixel + 3))), min(right, int(math.ceil(line.pixel + extent)))
        if x1 <= x0 or y1 <= y0:
            return []
        # The vertical label strip crosses the X tick-label row in many plots.
        # OCR that row as X labels, not as a pile of false Y ticks. Keep the
        # remaining upper/lower portions as independent crops so off-origin
        # positive and negative Y labels are still visible.
        if cross_axis is None:
            return [validate_crop(image_size, (x0, y0, x1, y1))]
        cross_extent = max(24, int(round((bottom - top) * 0.22)))
        if cross_axis_side > 0:
            excluded_top = max(top, int(math.floor(cross_axis.pixel + 3)))
            excluded_bottom = min(bottom, int(math.ceil(cross_axis.pixel + cross_extent)))
        else:
            excluded_top = max(top, int(math.floor(cross_axis.pixel - cross_extent)))
            excluded_bottom = min(bottom, int(math.ceil(cross_axis.pixel - 3)))
        intervals = [(y0, min(y1, excluded_top)), (max(y0, excluded_bottom), y1)]
        return [validate_crop(image_size, (x0, start, x1, end)) for start, end in intervals if end > start]


def _recognize_axis_label_bands(
    image: Image.Image, crop: tuple[int, int, int, int],
    horizontal: AxisLine, vertical: AxisLine, words: Sequence[OCRWord],
) -> dict[Literal["x", "y"], list[OCRWord]]:
    """Locally OCR the X/Y label strips, retaining original-image coordinates.

    Full-page OCR commonly joins adjacent degree labels or fragments short Y
    labels. The crop is derived from the detected axis geometry and label side,
    not from a particular image size or curve shape.
    """
    localized: dict[Literal["x", "y"], list[OCRWord]] = {"x": [], "y": []}
    x_side = _axis_band_side(words, "x", horizontal, crop)
    for axis, line in (("x", horizontal), ("y", vertical)):
        side = x_side if axis == "x" else _axis_band_side(words, axis, line, crop, cross_axis=horizontal)
        boxes = _axis_label_band(
            image.size, crop, axis, line, side,
            words=words,
            cross_axis=horizontal if axis == "y" else None,
            cross_axis_side=x_side,
        )
        for box in boxes:
            try:
                detected = recognize_local_words(image.crop(box))
            except Exception:
                continue
            left, top, _, _ = box
            localized[axis].extend(
                OCRWord(item.text, item.left + left, item.top + top, item.right + left, item.bottom + top, item.confidence)
                for item in detected
            )
    return localized


def _axis_label_word_count(
    words: Sequence[OCRWord], axis: Literal["x", "y"], line: AxisLine,
    crop: tuple[int, int, int, int],
) -> int:
    left, top, right, bottom = crop
    tolerance = (bottom - top) * 0.24 if axis == "x" else (right - left) * 0.24
    count = 0
    for word in words:
        if not _is_tick_text(word.text):
            continue
        center_x, center_y = (word.left + word.right) / 2, (word.top + word.bottom) / 2
        along = center_x if axis == "x" else center_y
        across = abs(center_y - line.pixel) if axis == "x" else abs(center_x - line.pixel)
        count += int(line.start - 4 <= along <= line.end + 4 and 2 <= across <= tolerance)
    return count


def _group_words(words: Sequence[OCRWord]) -> list[tuple[str, float, float, float, float, float]]:
    """Merge adjacent OCR tokens on one baseline, for split pi fractions."""
    clean = [word for word in words if word.text.strip() and word.confidence >= 20 and word.right > word.left and word.bottom > word.top]
    if not clean:
        return []
    median_height = float(np.median([word.bottom - word.top for word in clean]))
    groups: list[list[OCRWord]] = []
    for word in sorted(clean, key=lambda item: (item.top + item.bottom, item.left)):
        center = (word.top + word.bottom) / 2
        match = None
        for group in reversed(groups[-8:]):
            group_center = float(np.mean([(item.top + item.bottom) / 2 for item in group]))
            group_right = max(item.right for item in group)
            group_text = "".join(item.text for item in group).lower()
            current_text = word.text.lower()
            sign_prefix = group_text in {"-", "+", "−", "–", "﹣"}
            sign_suffix = current_text in {"-", "+", "−", "–", "﹣"}
            # A detached minus belongs before the numeric token. Do not attach
            # a following tick stroke or minus-shaped grid mark to a completed
            # number, which can turn a valid OCR token "-1" into "-1−".
            math_fragment = bool(re.search(r"[π/\\.^×]|pi|[eE]", group_text + current_text)) or sign_prefix
            if sign_suffix:
                math_fragment = False
            max_gap = max(4.0, median_height * (1.15 if math_fragment else 0.5))
            if math_fragment and abs(center - group_center) <= max(3.0, median_height * 0.75) and -median_height * 0.25 <= word.left - group_right <= max_gap:
                match = group
                break
        if match is None:
            groups.append([word])
        else:
            match.append(word)
    combined = []
    for group in groups:
        group = sorted(group, key=lambda item: item.left)
        text = "".join(item.text for item in group)
        combined.append((
            text,
            min(item.left for item in group),
            min(item.top for item in group),
            max(item.right for item in group),
            max(item.bottom for item in group),
            float(np.mean([item.confidence for item in group])),
        ))
    return combined


def _baseline_cluster(items: list[tuple[str, float, float, float, float, float]], *, center_index: int, tolerance: float) -> list[tuple[str, float, float, float, float, float]]:
    if not items:
        return []
    ordered = sorted(
        items,
        key=lambda item: (item[1] + item[3]) / 2 if center_index == 0 else (item[2] + item[4]) / 2,
    )
    clusters: list[list[tuple[str, float, float, float, float, float]]] = []
    for item in ordered:
        coordinate = (item[1] + item[3]) / 2 if center_index == 0 else (item[2] + item[4]) / 2
        if not clusters:
            clusters.append([item])
            continue
        previous = clusters[-1]
        previous_center = np.mean([
            (value[1] + value[3]) / 2 if center_index == 0 else (value[2] + value[4]) / 2
            for value in previous
        ])
        if abs(coordinate - previous_center) <= tolerance:
            previous.append(item)
        else:
            clusters.append([item])
    # Largest baseline cluster is most likely a row/column of tick labels.
    return max(clusters, key=lambda cluster: (len(cluster), -float(np.std([
        (value[1] + value[3]) / 2 if center_index == 0 else (value[2] + value[4]) / 2
        for value in cluster
    ]))))


def _baseline_clusters(
    items: Sequence[tuple[str, float, float, float, float, float]], *, center_index: int, tolerance: float
) -> list[list[tuple[str, float, float, float, float, float]]]:
    """Return all OCR rows/columns instead of assuming the largest is an axis."""
    ordered = sorted(
        items,
        key=lambda item: (item[1] + item[3]) / 2 if center_index == 0 else (item[2] + item[4]) / 2,
    )
    clusters: list[list[tuple[str, float, float, float, float, float]]] = []
    for item in ordered:
        coordinate = (item[1] + item[3]) / 2 if center_index == 0 else (item[2] + item[4]) / 2
        if not clusters:
            clusters.append([item])
            continue
        previous = clusters[-1]
        previous_center = float(np.mean([
            (value[1] + value[3]) / 2 if center_index == 0 else (value[2] + value[4]) / 2
            for value in previous
        ]))
        if abs(coordinate - previous_center) <= tolerance:
            previous.append(item)
        else:
            clusters.append([item])
    return clusters


def _has_context_for_pi_alias(items: Sequence[tuple[str, float, float, float, float, float]]) -> bool:
    aliases = [item[0] for item in items if _looks_like_pi_ocr(item[0])]
    explicit_pi = any(re.search(r"π|\\pi|\bpi\b", item[0], re.IGNORECASE) for item in items)
    # A single OCR result such as "2m" could be a real unit. Require multiple
    # corroborating math-looking labels, or one such label beside explicit π.
    return bool(explicit_pi and aliases or len(aliases) >= 2 and any("/" in value for value in aliases))


def _candidate_ocr_ticks(
    items: Sequence[tuple[str, float, float, float, float, float]],
    axis: Literal["x", "y"],
    *,
    allow_pi_alias: bool,
) -> tuple[TickLabel, ...]:
    ticks: list[TickLabel] = []
    for text, x0, y0, x1, y1, confidence in items:
        value, _ = _parse_axis_tick_label(text, allow_pi_ocr_alias=allow_pi_alias)
        if value is None:
            continue
        pixel = (x0 + x1) / 2 if axis == "x" else (y0 + y1) / 2
        ticks.append(TickLabel(axis, pixel, text, value, confidence))
    return tuple(sorted(ticks, key=lambda item: item.pixel))


def _tick_labels_from_text_layout(
    crop: tuple[int, int, int, int], words: Sequence[OCRWord]
) -> tuple[tuple[TickLabel, ...], tuple[TickLabel, ...], bool]:
    """Recover tick rows/columns when graph strokes do not form an axis pair.

    This fallback uses aligned OCR label boxes and their spacing, not the
    plotted curve shape. At least two separated, consistent labels are still
    required for an axis mapping; geometry-only guesses are never returned.
    """
    left, top, right, bottom = crop
    width, height = right - left, bottom - top
    grouped = [
        item for item in _group_words(words)
        if parse_tick_label(item[0]) is not None or _looks_like_pi_ocr(item[0])
    ]
    if not grouped:
        return (), (), False

    def select_axis_cluster(axis: Literal["x", "y"]) -> tuple[tuple[TickLabel, ...], bool]:
        dimension = width if axis == "x" else height
        clusters = _baseline_clusters(
            grouped,
            center_index=1 if axis == "x" else 0,
            tolerance=max(4.0, dimension * 0.045),
        )
        ranked: list[tuple[tuple[int, float, float], tuple[TickLabel, ...], bool]] = []
        for cluster in clusters:
            pixels = np.asarray([
                (item[1] + item[3]) / 2 if axis == "x" else (item[2] + item[4]) / 2
                for item in cluster
            ], dtype=float)
            if len(cluster) < 2 or float(np.ptp(pixels)) < max(24.0, dimension * 0.18):
                continue
            contextual_pi = _has_context_for_pi_alias(cluster)
            ticks = _candidate_ocr_ticks(cluster, axis, allow_pi_alias=contextual_pi)
            fit = _fit_axis(ticks, axis)
            if fit is None:
                continue
            span = float(np.ptp([tick.pixel for tick in fit.ticks])) / max(1, dimension)
            mean_confidence = float(np.mean([tick.ocr_confidence for tick in fit.ticks]))
            ranked.append(((fit.inlier_count, span, mean_confidence), fit.ticks, contextual_pi and any(_looks_like_pi_ocr(tick.label) for tick in fit.ticks)))
        if not ranked:
            return (), False
        _, ticks, used_alias = max(ranked, key=lambda item: item[0])
        return ticks, used_alias

    x_ticks, x_alias = select_axis_cluster("x")
    y_ticks, y_alias = select_axis_cluster("y")
    return x_ticks, y_ticks, x_alias or y_alias


def _foreground_mask(image: Image.Image, crop: tuple[int, int, int, int]) -> np.ndarray:
    region = image.crop(crop).convert("RGB")
    pixels = np.asarray(region, dtype=np.uint8)
    gray = np.asarray(ImageOps.grayscale(region), dtype=np.uint8)
    chroma = pixels.max(axis=2).astype(np.int16) - pixels.min(axis=2).astype(np.int16)
    corner = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    return ((gray < 145) if float(np.median(corner)) >= 127 else (gray > 190)) & (chroma < 90)


def _profile_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous pixel spans with non-zero support in a 1-D profile."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > 2) + 1
    groups = np.split(indices, breaks)
    return [(int(group[0]), int(group[-1])) for group in groups]


def _detect_axis_tick_pixels(
    image: Image.Image,
    crop: tuple[int, int, int, int],
    axis_line: AxisLine,
    axis: Literal["x", "y"],
    *,
    cross_axis_pixel: float | None = None,
    label_words: Sequence[OCRWord] = (),
) -> list[float]:
    """Detect short neutral tick strokes independently of their text labels.

    The long axis stroke is excluded from the support window. X ticks must
    extend to both sides of a horizontal axis; Y ticks may extend on either
    side of a vertical axis, since many plotting styles draw them inward only.
    Color curves are excluded by ``_foreground_mask``.
    """
    left, top, right, bottom = validate_crop(image.size, crop)
    foreground = _foreground_mask(image, crop)
    height, width = foreground.shape
    radius = max(5, min(18, int(round(min(width, height) * 0.016))))
    margin = max(2, int(round(radius * 0.15)))
    if axis == "x":
        line = int(round(axis_line.pixel - top))
        above = np.sum(foreground[max(0, line - radius):max(0, line - margin), :], axis=0)
        below = np.sum(foreground[min(height, line + margin + 1):min(height, line + radius + 1), :], axis=0)
        weaker = np.minimum(above, below)
        total = above + below
        threshold = max(2, int(round(radius * 0.12)))
        eligible = (weaker >= threshold) & (total >= max(6, int(round(radius * 0.5))))
        eligible &= weaker / np.maximum(1, np.maximum(above, below)) >= 0.3
        start = max(0, int(math.floor(axis_line.start - left)))
        end = min(width - 1, int(math.ceil(axis_line.end - left)))
        eligible[:start] = False
        eligible[end + 1:] = False
        profile = total
    else:
        line = int(round(axis_line.pixel - left))
        before = np.sum(foreground[:, max(0, line - radius):max(0, line - margin)], axis=1)
        after = np.sum(foreground[:, min(width, line + margin + 1):min(width, line + radius + 1)], axis=1)
        stronger = np.maximum(before, after)
        threshold = max(3, int(round(radius * 0.3)))
        eligible = stronger >= threshold
        weaker = np.minimum(before, after)
        local_y = np.arange(height) + top
        at_crossing = (
            np.abs(local_y - cross_axis_pixel) <= radius * 1.5
            if cross_axis_pixel is not None else np.zeros(height, dtype=bool)
        )
        # Y tick marks are usually drawn on one side of the axis. The origin
        # can have support on both sides because the horizontal axis crosses
        # there; all other positions should show a one-sided short stroke.
        eligible &= (weaker / np.maximum(1, stronger) <= 0.45) | at_crossing
        start = max(0, int(math.floor(axis_line.start - top)))
        end = min(height - 1, int(math.ceil(axis_line.end - top)))
        eligible[:start] = False
        eligible[end + 1:] = False
        profile = stronger

    candidates: list[float] = []
    for first, last in _profile_regions(eligible):
        indices = np.arange(first, last + 1)
        weights = profile[indices].astype(float)
        pixel = float(np.average(indices, weights=weights))
        candidates.append(pixel + (left if axis == "x" else top))

    # Arrowheads often create short perpendicular strokes near the ends of an
    # axis, so the raw line profile can mistake their wings for tick marks.
    # A genuine edge tick is retained when a nearby OCR label supports its
    # along-axis position. This uses text geometry, not a presumed axis range.
    span = max(1.0, axis_line.end - axis_line.start)
    endpoint_margin = min(span * 0.12, max(radius * 6.0, span * 0.05))
    relevant_labels: list[tuple[float, float]] = []
    across_limit = max(36.0, min((height if axis == "x" else width) * 0.18, 100.0))
    for word in label_words:
        if not _is_tick_text(word.text):
            continue
        center_x, center_y = (word.left + word.right) / 2, (word.top + word.bottom) / 2
        along = center_x if axis == "x" else center_y
        across = abs(center_y - axis_line.pixel) if axis == "x" else abs(center_x - axis_line.pixel)
        if axis_line.start - 4 <= along <= axis_line.end + 4 and 2 <= across <= across_limit:
            relevant_labels.append((along, max(1.0, word.bottom - word.top)))
    if relevant_labels:
        label_tolerance = max(12.0, float(np.median([height for _, height in relevant_labels])) * 1.5)
        candidates = [
            pixel for pixel in candidates
            if not (
                min(pixel - axis_line.start, axis_line.end - pixel) <= endpoint_margin
                and min(abs(pixel - label_pixel) for label_pixel, _ in relevant_labels) > label_tolerance
            )
        ]
    return candidates


def _parse_local_tick_text(text: str) -> tuple[str, float] | None:
    """Parse one OCR-isolated label, including vertically stacked fractions.

    This function receives text from a crop centered on a single detected
    tick. It may therefore safely combine numerator/denominator lines. It does
    not reinterpret a leading-zero number such as ``06`` as a decimal.
    """
    lines = [line.strip() for line in str(text).replace("\r", "").split("\n") if line.strip()]

    def clean(value: str) -> str:
        value = value.strip().replace("−", "-").replace("–", "-").replace("—", "-")
        value = re.sub(r"^[|=;:,]+", "", value)
        value = re.sub(r"[|=;:,]+$", "", value)
        # Tesseract sometimes joins the adjacent short tick stroke to a
        # numeric label (e.g. ``0.5-`` or ``1-``). Keep a leading minus.
        if value.endswith(("-", "+")) and len(value) > 1:
            value = value[:-1]
        return re.sub(r"\s+", "", value)

    if len(lines) >= 2:
        numerator, denominator = clean(lines[0]), clean(lines[-1])
        if numerator and denominator and re.fullmatch(r"\d+(?:\.\d+)?", denominator):
            combined = f"{numerator}/{denominator}"
            value = parse_tick_label(combined)
            if value is not None and re.search(r"π|\\pi|pi", numerator, re.IGNORECASE):
                return combined, value
            # A numeric denominator by itself is not a tick value when the
            # crop contains an unreadable numerator. Never turn ``?`` over 2
            # into the accepted value 2.
            return None

    candidates: list[tuple[str, float]] = []
    for line in lines:
        cleaned = clean(line)
        if not cleaned:
            continue
        parsed = parse_tick_label(cleaned)
        if parsed is not None:
            candidates.append((cleaned, parsed))
            continue
        # A local OCR crop may contain one label plus a tick/axis artifact.
        # Only accept a single explicit mathematical token from that line.
        tokens = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|[+-]?\d*(?:π|pi)(?:/\d+)?", line, re.IGNORECASE)
        for token in tokens:
            parsed = parse_tick_label(token)
            if parsed is not None:
                candidates.append((token, parsed))
    distinct: dict[float, str] = {}
    for label, value in candidates:
        distinct.setdefault(value, label)
    if len(distinct) == 1:
        value, label = next(iter(distinct.items()))
        return label, value
    return None


def _otsu_threshold(image: Image.Image) -> int:
    """Choose a two-class grayscale threshold from the local crop histogram."""
    histogram = np.asarray(image.histogram(), dtype=np.float64)
    total = float(histogram.sum())
    if total <= 0:
        return 127
    levels = np.arange(256, dtype=np.float64)
    cumulative_weight = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * levels)
    global_mean = cumulative_mean[-1]
    denominator = cumulative_weight * (total - cumulative_weight)
    variance = np.zeros(256, dtype=float)
    valid = denominator > 0
    variance[valid] = (
        global_mean * cumulative_weight[valid] - cumulative_mean[valid] * total
    ) ** 2 / denominator[valid]
    return int(np.argmax(variance))


def _read_local_tick_label(
    image: Image.Image, box: tuple[int, int, int, int]
) -> list[_LocalTickRead]:
    """Read a crop with local Tesseract modes, confidence, and text geometry."""
    available, _ = local_ocr_available()
    if not available:
        return []
    import pytesseract
    from pytesseract import Output

    left, top, right, bottom = validate_crop(image.size, box)
    crop = image.crop((left, top, right, bottom))
    gray = ImageOps.autocontrast(ImageOps.grayscale(crop))
    # Small anti-aliased Greek glyphs are damaged by aggressive contrast
    # enhancement: on real fractions π was reduced to an underscore or a
    # neighboring numeric token. Autocontrast with a mild adjustment retains
    # the strokes while still lifting low-contrast tick labels.
    gray = ImageEnhance.Contrast(gray).enhance(1.1)
    scale = min(4, max(3, int(round(360 / max(1, min(gray.size))))))
    enlarged = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.LANCZOS)
    candidates: list[_LocalTickRead] = []

    def append_ocr_candidates(data: dict[str, object], resize_scale: int) -> None:
        line_words: dict[
            tuple[int, int, int],
            list[tuple[float, str, float, tuple[float, float, float, float]]],
        ] = {}
        confidences: list[float] = []
        for index, raw_text in enumerate(data.get("text", [])):
            token = str(raw_text).strip()
            try:
                confidence = float(data["conf"][index])
                line_key = (
                    int(data.get("block_num", [0])[index]),
                    int(data.get("par_num", [0])[index]),
                    int(data.get("line_num", [index])[index]),
                )
                token_left = float(data["left"][index]) / resize_scale
                token_top = float(data["top"][index]) / resize_scale
                token_right = token_left + float(data["width"][index]) / resize_scale
                token_bottom = token_top + float(data["height"][index]) / resize_scale
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if token and confidence >= 0 and token_right > token_left and token_bottom > token_top:
                token_box = (token_left, token_top, token_right, token_bottom)
                line_words.setdefault(line_key, []).append((token_left, token, confidence, token_box))
                confidences.append(confidence)
        ordered_lines = [
            " ".join(token for _, token, _, _ in sorted(line_words[key]))
            for key in sorted(line_words)
        ]
        raw_text = "\n".join(ordered_lines)
        all_boxes = [box for line in line_words.values() for _, _, _, box in line]
        text_box = (
            (
                min(item[0] for item in all_boxes),
                min(item[1] for item in all_boxes),
                max(item[2] for item in all_boxes),
                max(item[3] for item in all_boxes),
            )
            if all_boxes else None
        )
        confidence = float(np.mean(confidences)) if confidences else 0.0
        parsed = _parse_local_tick_text(raw_text)
        candidates.append(_LocalTickRead(raw_text, parsed[1] if parsed else None, confidence, text_box))
        if parsed:
            candidates.append(_LocalTickRead(parsed[0], parsed[1], confidence, text_box))

    languages = ["eng+script/Greek", "eng"]
    for language in languages:
        # PSM 7/10 help with isolated one-line numeric labels. They are
        # especially useful when an axis arrow or curve touches the label row:
        # block/sparse modes can merge the text with that geometry, while a
        # line/single-character mode often recovers the actual printed digits.
        page_modes = (6, 8, 13, 7, 10) if language == "eng" else (6, 8, 13)
        for page_mode in page_modes:
            try:
                data = pytesseract.image_to_data(
                    enlarged, lang=language, config=f"--psm {page_mode}", output_type=Output.DICT
                )
            except Exception:
                continue
            append_ocr_candidates(data, scale)
    # Binarized single-label crops can retain a faint zero or minus sign that
    # full-color OCR loses. Keep these as additional evidence, not a forced
    # interpretation of the label.
    threshold = _otsu_threshold(gray)
    binary = gray.point(lambda value: 255 if value > threshold else 0)
    binary_scale = min(4, max(3, int(round(360 / max(1, min(binary.size))))))
    enlarged_binary = binary.resize((binary.width * binary_scale, binary.height * binary_scale), Image.Resampling.LANCZOS)
    for page_mode in (7, 11):
        try:
            data = pytesseract.image_to_data(
                enlarged_binary,
                lang="eng",
                config=f"--psm {page_mode} -c tessedit_char_whitelist=0123456789.+-°πpiPI/",
                output_type=Output.DICT,
            )
        except Exception:
            continue
        append_ocr_candidates(data, binary_scale)
    return candidates


def _snap_to_tick(
    foreground: np.ndarray,
    *,
    axis: Literal["x", "y"],
    approximate_pixel: float,
    axis_pixel: float,
    crop: tuple[int, int, int, int],
) -> float:
    """Snap an OCR box center to a nearby short tick mark when it is visible."""
    left, top, _, _ = crop
    local_axis = int(round(axis_pixel - (top if axis == "x" else left)))
    local_approx = int(round(approximate_pixel - (left if axis == "x" else top)))
    height, width = foreground.shape
    offsets = range(-7, 8)
    supports: list[tuple[int, int]] = []
    for offset in offsets:
        candidate = local_approx + offset
        if axis == "x":
            if not 0 <= candidate < width:
                continue
            low, high = max(0, local_axis - 9), min(height, local_axis + 10)
            rows = np.r_[low:max(low, local_axis - 2), min(high, local_axis + 3):high]
            support = int(foreground[rows, candidate].sum()) if len(rows) else 0
        else:
            if not 0 <= candidate < height:
                continue
            low, high = max(0, local_axis - 9), min(width, local_axis + 10)
            columns = np.r_[low:max(low, local_axis - 2), min(high, local_axis + 3):high]
            support = int(foreground[candidate, columns].sum()) if len(columns) else 0
        supports.append((support, candidate))
    if not supports:
        return approximate_pixel
    support, candidate = max(supports, key=lambda item: (item[0], -abs(item[1] - local_approx)))
    if support < 2:
        return approximate_pixel
    return float(candidate + (left if axis == "x" else top))


def _tick_labels(
    image: Image.Image,
    crop: tuple[int, int, int, int],
    words: Sequence[OCRWord],
    horizontal: AxisLine,
    vertical: AxisLine,
    plot_region: tuple[int, int, int, int],
) -> tuple[tuple[TickLabel, ...], tuple[TickLabel, ...], bool, bool]:
    if not words:
        return (), (), False, False
    left, top, right, bottom = plot_region
    width, height = max(1, right - left), max(1, bottom - top)
    groups = _group_words(words)
    x_candidates = []
    y_candidates = []
    for item in groups:
        text, x0, y0, x1, y1, confidence = item
        if parse_tick_label(text) is None and not _looks_like_pi_ocr(text):
            continue
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        if horizontal.start - 0.04 * width <= center_x <= horizontal.end + 0.04 * width and abs(center_y - horizontal.pixel) <= 0.32 * height:
            x_candidates.append(item)
        if vertical.start - 0.04 * height <= center_y <= vertical.end + 0.04 * height and abs(center_x - vertical.pixel) <= 0.32 * width:
            y_candidates.append(item)
    x_baseline = _baseline_cluster(x_candidates, center_index=1, tolerance=max(4.0, height * 0.045))
    y_baseline = _baseline_cluster(y_candidates, center_index=0, tolerance=max(4.0, width * 0.045))
    # A tick label at the axis intersection may lie close to both axes. Avoid
    # counting an X-axis label (same horizontal text baseline as the X tick
    # row) as an extra Y tick, which could otherwise corrupt log-scale tests.
    if x_baseline and y_baseline:
        x_text_baseline = float(np.median([(item[2] + item[4]) / 2 for item in x_baseline]))
        y_baseline = [
            item for item in y_baseline
            if abs((item[2] + item[4]) / 2 - x_text_baseline) > max(3.0, height * 0.025)
        ]
    x_ticks: list[TickLabel] = []
    y_ticks: list[TickLabel] = []
    corrected_pi_alias = False
    foreground = _foreground_mask(image, crop)

    x_allow_alias = _has_context_for_pi_alias(x_baseline)
    y_allow_alias = _has_context_for_pi_alias(y_baseline)
    for item in x_baseline:
        text, x0, y0, x1, y1, confidence = item
        value, corrected = _parse_axis_tick_label(text, allow_pi_ocr_alias=x_allow_alias)
        if value is not None:
            corrected_pi_alias |= corrected
            estimate = (x0 + x1) / 2
            tick_pixel = _snap_to_tick(foreground, axis="x", approximate_pixel=estimate, axis_pixel=horizontal.pixel, crop=crop)
            x_ticks.append(TickLabel("x", tick_pixel, text, value, confidence))
    for item in y_baseline:
        text, x0, y0, x1, y1, confidence = item
        value, corrected = _parse_axis_tick_label(text, allow_pi_ocr_alias=y_allow_alias)
        if value is not None:
            corrected_pi_alias |= corrected
            estimate = (y0 + y1) / 2
            tick_pixel = _snap_to_tick(foreground, axis="y", approximate_pixel=estimate, axis_pixel=vertical.pixel, crop=crop)
            y_ticks.append(TickLabel("y", tick_pixel, text, value, confidence))
    return tuple(sorted(x_ticks, key=lambda item: item.pixel)), tuple(sorted(y_ticks, key=lambda item: item.pixel)), True, corrected_pi_alias


def _is_tick_text(text: str) -> bool:
    value = str(text).strip()
    if parse_tick_label(value) is not None or _looks_like_pi_ocr(value) or re.search(r"π|\\pi|\bpi\b", value, re.IGNORECASE):
        return True
    # Allow detached minus signs so _group_words can join them to the numeric
    # token, but do not seed spurious ticks from OCR debris such as '=' or a
    # text fragment ending in a digit.
    return bool(re.fullmatch(r"[+\-−–—]", value))


def _median_tick_spacing(pixels: Sequence[float], fallback: float) -> float:
    differences = np.diff(sorted(set(round(float(pixel), 2) for pixel in pixels)))
    usable = differences[differences > 3]
    return float(np.median(usable)) if len(usable) else max(24.0, fallback)


def _merge_tick_pixels(pixels: Sequence[float], *, tolerance: float) -> list[float]:
    if not pixels:
        return []
    groups: list[list[float]] = []
    for pixel in sorted(float(value) for value in pixels):
        if not groups or pixel - float(np.mean(groups[-1])) > tolerance:
            groups.append([pixel])
        else:
            groups[-1].append(pixel)
    return [float(np.median(group)) for group in groups]


def _complete_regular_tick_pixels(pixels: Sequence[float]) -> list[float]:
    """Infer an obscured tick only when the detected geometry supports a lattice.

    Colored curves can cover a short neutral tick stroke. When several other
    detected ticks establish a regular pixel interval, fill integer-multiple
    gaps at geometric positions. This does not assign data values; OCR still
    has to recognize the label at an inferred location.
    """
    ordered = sorted(float(pixel) for pixel in pixels)
    if len(ordered) < 4:
        return ordered
    gaps = np.diff(ordered)
    spacing = float(np.median(gaps[gaps > 1])) if np.any(gaps > 1) else 0.0
    if not math.isfinite(spacing) or spacing <= 1:
        return ordered
    completed: list[float] = [ordered[0]]
    for start, end in zip(ordered, ordered[1:]):
        gap = end - start
        intervals = int(round(gap / spacing))
        if 2 <= intervals <= 4:
            interval = gap / intervals
            if abs(interval - spacing) <= max(6.0, spacing * 0.04):
                completed.extend(start + interval * step for step in range(1, intervals))
        completed.append(end)
    return completed


def _snap_text_center_to_inferred_tick(pixel: float, stroke_pixels: Sequence[float]) -> float:
    """Use a regular stroke lattice to refine a label-centered hidden tick."""
    if len(stroke_pixels) < 4:
        return float(pixel)
    lattice = _complete_regular_tick_pixels(stroke_pixels)
    spacing = _median_tick_spacing(stroke_pixels, 24.0)
    nearest = min(lattice, key=lambda item: abs(item - pixel))
    is_inferred = all(abs(nearest - detected) > 3.0 for detected in stroke_pixels)
    if is_inferred and abs(nearest - pixel) <= max(5.0, spacing * 0.06):
        return float(nearest)
    return float(pixel)


def _tick_words_for_axis(
    words: Sequence[OCRWord],
    *,
    axis: Literal["x", "y"],
    pixel: float,
    axis_line: AxisLine,
    side: int,
    along_tolerance: float,
    across_tolerance: float,
    include_nonmath: bool = False,
) -> list[OCRWord]:
    selected: list[OCRWord] = []
    for word in words:
        if not include_nonmath and not _is_tick_text(word.text):
            continue
        center_x, center_y = (word.left + word.right) / 2, (word.top + word.bottom) / 2
        if axis == "x":
            along, across = center_x, center_y - axis_line.pixel
        else:
            along, across = center_y, center_x - axis_line.pixel
        if abs(along - pixel) <= along_tolerance and 2 <= side * across <= across_tolerance:
            selected.append(word)
    return selected


def _axis_label_side(
    words: Sequence[OCRWord],
    ticks: Sequence[float],
    *,
    axis: Literal["x", "y"],
    axis_line: AxisLine,
    along_tolerance: float,
    across_tolerance: float,
) -> int:
    scores = { -1: 0, 1: 0 }
    for side in scores:
        for pixel in ticks:
            if _tick_words_for_axis(
                words, axis=axis, pixel=pixel, axis_line=axis_line, side=side,
                along_tolerance=along_tolerance, across_tolerance=across_tolerance,
            ):
                scores[side] += 1
    # Labels are commonly below X or left of Y; this is only a crop-side
    # fallback when OCR supplies no placement evidence, never a value guess.
    default_side = 1 if axis == "x" else -1
    return max(scores, key=lambda side: (scores[side], side == default_side))


def _tick_crop_boxes(
    image: Image.Image,
    crop: tuple[int, int, int, int],
    *,
    axis: Literal["x", "y"],
    pixel: float,
    axis_line: AxisLine,
    cross_axis: AxisLine,
    side: int,
    words: Sequence[OCRWord],
    spacing: float,
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int]]:
    """Build tight and isolated label crops in original-image coordinates."""
    left, top, right, bottom = crop
    along_tolerance = max(12.0, min(100.0, spacing * 0.36))
    across_tolerance = max(36.0, min((bottom - top if axis == "x" else right - left) * 0.18, spacing * 0.58))
    selected = _tick_words_for_axis(
        words, axis=axis, pixel=pixel, axis_line=axis_line, side=side,
        along_tolerance=along_tolerance, across_tolerance=across_tolerance, include_nonmath=True,
    )
    tight: tuple[int, int, int, int] | None = None
    if selected:
        x0 = min(word.left for word in selected)
        y0 = min(word.top for word in selected)
        x1 = max(word.right for word in selected)
        y1 = max(word.bottom for word in selected)
        font_height = float(np.median([max(1.0, word.bottom - word.top) for word in selected]))
        pad_along = max(4, int(round(font_height * 0.9)))
        pad_across = max(4, int(round(font_height * 0.42)))
        if axis == "x":
            pad_left = max(4, int(round(font_height * 1.05)))
            pad_right = max(4, int(round(font_height * 0.72)))
            pad_top = max(3, int(round(font_height * 0.35)))
            pad_bottom = max(3, int(round(font_height * 0.2)))
            tight = (
                max(left, int(math.floor(x0 - pad_left))),
                max(top, int(math.floor(y0 - pad_top)), int(math.floor(axis_line.pixel + 2)) if side > 0 else top),
                min(right, int(math.ceil(x1 + pad_right))),
                min(bottom, int(math.ceil(y1 + pad_bottom)), int(math.ceil(axis_line.pixel - 2)) if side < 0 else bottom),
            )
            # The origin label is often printed beside the vertical axis. Keep
            # the axis stroke out of its local OCR crop when its text is on one
            # side of the intersection.
            center = float(np.median([(word.left + word.right) / 2 for word in selected]))
            if abs(pixel - cross_axis.pixel) <= along_tolerance:
                if center < cross_axis.pixel:
                    tight = (tight[0], tight[1], min(tight[2], int(cross_axis.pixel - 3)), tight[3])
                else:
                    tight = (max(tight[0], int(cross_axis.pixel + 3)), tight[1], tight[2], tight[3])
        else:
            tight = (
                max(left, int(math.floor(x0 - pad_across)), int(math.floor(axis_line.pixel + 2)) if side > 0 else left),
                max(top, int(math.floor(y0 - pad_along))),
                min(right, int(math.ceil(x1 + pad_across)), int(math.ceil(axis_line.pixel - 8)) if side < 0 else right),
                min(bottom, int(math.ceil(y1 + pad_along))),
            )
            center_y = float(np.median([(word.top + word.bottom) / 2 for word in selected]))
            if abs(pixel - cross_axis.pixel) <= along_tolerance:
                if center_y < cross_axis.pixel:
                    tight = (tight[0], tight[1], tight[2], min(tight[3], int(cross_axis.pixel - 3)))
                else:
                    tight = (tight[0], max(tight[1], int(cross_axis.pixel + 2)), tight[2], tight[3])
        if tight[2] <= tight[0] or tight[3] <= tight[1]:
            tight = None

    if axis == "x":
        font_height = float(np.median([word.bottom - word.top for word in selected])) if selected else max(8.0, (bottom - top) * 0.035)
        width = max(int(font_height * 3.4), int(spacing * 0.34), 40)
        # Do not let a tall OCR box (often enlarged by an adjacent curve or
        # axis arrow) make the retry crop swallow neighbouring labels or the
        # axis-end annotation. The observed tick spacing is a better bound on
        # one label's width than the full image width.
        width = min(width, max(40, int(spacing * 0.9)), max(40, int((right - left) * 0.20)))
        center = float(np.median([(word.left + word.right) / 2 for word in selected])) if selected else pixel
        x0, x1 = max(left, int(round(center - width / 2))), min(right, int(round(center + width / 2)))
        label_extent = min(max(int(spacing * 0.34), int(font_height * 3.0), 40), max(40, int((bottom - top) * 0.12)))
        if side > 0:
            # Leave the axis/tick stroke outside the label crop. Including it
            # caused Tesseract to drop the numerator of the first stacked π
            # fraction in the real figure.
            y0, y1 = max(top, int(axis_line.pixel + 4)), min(bottom, int(axis_line.pixel + label_extent + 4))
        else:
            y0, y1 = max(top, int(axis_line.pixel - label_extent - 4)), min(bottom, int(axis_line.pixel - 4))
        if abs(pixel - cross_axis.pixel) <= along_tolerance and selected:
            center = float(np.median([(word.left + word.right) / 2 for word in selected]))
            if center < cross_axis.pixel:
                x1 = min(x1, int(cross_axis.pixel - 3))
            else:
                x0 = max(x0, int(cross_axis.pixel + 3))
    else:
        font_height = float(np.median([word.bottom - word.top for word in selected])) if selected else max(8.0, (bottom - top) * 0.035)
        height = max(int(font_height * 3.2), int(spacing * 0.55), 36)
        height = min(height, max(36, int((bottom - top) * 0.13)))
        center_y = float(np.median([(word.top + word.bottom) / 2 for word in selected])) if selected else pixel
        y0, y1 = max(top, int(round(center_y - height / 2))), min(bottom, int(round(center_y + height / 2)))
        width = max(int(font_height * 3.2), int(spacing * 0.58), 36)
        width = min(width, max(36, int((right - left) * 0.14)))
        if side < 0:
            x0, x1 = max(left, int(axis_line.pixel - width)), min(right, int(axis_line.pixel - 8))
        else:
            x0, x1 = max(left, int(axis_line.pixel + 8)), min(right, int(axis_line.pixel + width))
        if abs(pixel - cross_axis.pixel) <= along_tolerance and selected:
            center_y = float(np.median([(word.top + word.bottom) / 2 for word in selected]))
            if center_y < cross_axis.pixel:
                y1 = min(y1, int(cross_axis.pixel - 3))
            else:
                y0 = max(y0, int(cross_axis.pixel + 2))
    broad = (x0, y0, x1, y1)
    broad = broad if broad[2] > broad[0] and broad[3] > broad[1] else (left, top, right, bottom)
    return tight, broad


def _direct_tick_values(words: Sequence[OCRWord]) -> list[tuple[str, float, float]]:
    results: list[tuple[str, float, float]] = []
    grouped = _group_words(words)
    for text, *_box, confidence in grouped:
        parsed = parse_tick_label(text)
        if parsed is not None:
            results.append((text, parsed, confidence))
    for word in words:
        parsed = parse_tick_label(word.text)
        if parsed is not None:
            results.append((word.text, parsed, word.confidence))
    return results


def _origin_zero_candidates(
    words: Sequence[OCRWord],
    horizontal: AxisLine,
    vertical: AxisLine,
    crop: tuple[int, int, int, int],
) -> tuple[TickLabel, ...]:
    """Represent a printed O/Omicron at an axis crossing as a reviewable zero.

    Some plots label the origin with the letter ``O`` rather than the digit
    zero. The glyph alone is ambiguous, so this is only a candidate when its
    OCR box is immediately adjacent to the detected axis intersection. It is
    never parsed as zero globally and remains review-required until the other
    tick anchors support the mapping.
    """
    left, top, right, bottom = crop
    tolerance = max(12.0, min(32.0, min(right - left, bottom - top) * 0.035))
    candidates: list[OCRWord] = []
    for word in words:
        label = re.sub(r"[\s.,:;|]+", "", word.text)
        if label not in {"O", "o", "Ο", "ο"}:
            continue
        dx = max(word.left - vertical.pixel, 0.0, vertical.pixel - word.right)
        dy = max(word.top - horizontal.pixel, 0.0, horizontal.pixel - word.bottom)
        if dx <= tolerance and dy <= tolerance:
            candidates.append(word)
    if not candidates:
        return ()
    word = max(candidates, key=lambda item: item.confidence)
    note = "The OCR glyph O/omicron is adjacent to the axis intersection and is proposed as zero; confirm this interpretation against the image."
    return (
        TickLabel("x", float(vertical.pixel), word.text, 0.0, word.confidence, "REVIEW_REQUIRED", "0 (origin-label context)", note),
        TickLabel("y", float(horizontal.pixel), word.text, 0.0, word.confidence, "REVIEW_REQUIRED", "0 (origin-label context)", note),
    )


def _looks_like_lost_decimal(text: str) -> bool:
    """Flag leading-zero OCR integers for a targeted retry, not reinterpretation."""
    return bool(re.fullmatch(r"0\d+", str(text).strip()))


def _degree_suffix_candidate(text: str, *, degree_context: bool) -> tuple[str, float] | None:
    """Suggest a review-only degree reading when OCR turns ``°`` into 0/5.

    The unmodified reading is always retained. This alternative is only
    offered on axes where another printed label established degree notation;
    the axis-wide tick fit must independently corroborate it before it can be
    accepted. Thus a legitimate value such as 2705 remains available as-is.
    """
    if not degree_context:
        return None
    normalized = str(text).strip().replace("−", "-").replace("–", "-").replace("—", "-")
    normalized = re.sub(r"\s+", "", normalized)
    match = re.fullmatch(r"(?P<base>[+-]?\d+)(?P<suffix>[05])", normalized)
    if match is None:
        return None
    candidate_text = f"{match.group('base')}°"
    value = parse_tick_label(candidate_text)
    return (candidate_text, value) if value is not None else None


def _has_leading_minus_evidence(text: str) -> bool:
    """Recognize a minus glyph adjacent to digits in the isolated OCR label."""
    normalized = str(text).replace("−", "-").replace("–", "-").replace("—", "-")
    return bool(re.search(r"-\s*\d", normalized))


def _tick_labels_from_axis_geometry(
    image: Image.Image,
    crop: tuple[int, int, int, int],
    words: Sequence[OCRWord],
    horizontal: AxisLine,
    vertical: AxisLine,
    plot_region: tuple[int, int, int, int],
) -> tuple[tuple[TickLabel, ...], tuple[TickLabel, ...], tuple[TickLabel, ...], bool]:
    """Associate OCR labels with detected tick strokes, then retry ambiguous crops.

    This is the production path for figures with visible axis/tick geometry.
    Tick positions are found from the image first; the OCR labels are then
    associated to their nearest physical tick rather than used as axis
    endpoints by OCR result order.
    """
    left, top, right, bottom = crop
    plot_left, plot_top, plot_right, plot_bottom = plot_region
    width, height = plot_right - plot_left, plot_bottom - plot_top
    x_stroke_pixels = _detect_axis_tick_pixels(
        image, crop, horizontal, "x", cross_axis_pixel=vertical.pixel, label_words=words
    )
    y_stroke_pixels = _detect_axis_tick_pixels(
        image, crop, vertical, "y", cross_axis_pixel=horizontal.pixel, label_words=words
    )
    x_pixels = list(x_stroke_pixels)
    y_pixels = list(y_stroke_pixels)
    x_spacing = _median_tick_spacing(x_pixels, width / 5)
    y_spacing = _median_tick_spacing(y_pixels, height / 4)
    x_band = max(40.0, min(height * 0.18, max(80.0, x_spacing * 0.62)))
    y_band = max(40.0, min(width * 0.18, max(80.0, y_spacing * 0.62)))
    x_tolerance = max(14.0, min(100.0, x_spacing * 0.36))
    y_tolerance = max(14.0, min(100.0, y_spacing * 0.38))

    # Some real tick strokes are obscured by a colored curve. A math-like OCR
    # box on the correct axis label band can supply a pixel candidate, but it
    # is snapped to a nearby detected tick whenever one exists.
    for word in words:
        if not _is_tick_text(word.text):
            continue
        center_x, center_y = (word.left + word.right) / 2, (word.top + word.bottom) / 2
        if horizontal.start <= center_x <= horizontal.end and 2 <= abs(center_y - horizontal.pixel) <= x_band:
            if not x_pixels or min(abs(center_x - item) for item in x_pixels) > x_tolerance:
                x_pixels.append(_snap_text_center_to_inferred_tick(center_x, x_stroke_pixels))
        if vertical.start <= center_y <= vertical.end and 2 <= abs(center_x - vertical.pixel) <= y_band:
            if not y_pixels or min(abs(center_y - item) for item in y_pixels) > y_tolerance:
                y_pixels.append(_snap_text_center_to_inferred_tick(center_y, y_stroke_pixels))
    # OCR passes can return overlapping boxes for one printed label with
    # slightly different edges/centres. Merge those duplicates using a bound
    # derived from the observed local tick spacing and image size. The
    # spacing-relative cap keeps distinct nearby ticks separate on dense axes.
    x_spacing = _median_tick_spacing(x_pixels, width / 5)
    y_spacing = _median_tick_spacing(y_pixels, height / 4)
    merge_limit = min(24.0, min(width, height) * 0.06)
    x_merge_tolerance = max(4.0, min(merge_limit, x_spacing * 0.22))
    y_merge_tolerance = max(4.0, min(merge_limit, y_spacing * 0.22))
    x_pixels = _merge_tick_pixels(x_pixels, tolerance=x_merge_tolerance)
    y_pixels = _merge_tick_pixels(y_pixels, tolerance=y_merge_tolerance)
    # A colored curve may cover one short tick mark. Fill only gaps supported
    # by the repeated spacing of at least four independently detected ticks;
    # the inferred position remains a candidate until its own label is read.
    x_pixels = _complete_regular_tick_pixels(x_pixels)
    y_pixels = _complete_regular_tick_pixels(y_pixels)
    x_spacing = _median_tick_spacing(x_pixels, width / 5)
    y_spacing = _median_tick_spacing(y_pixels, height / 4)

    x_side = _axis_label_side(words, x_pixels, axis="x", axis_line=horizontal, along_tolerance=x_tolerance, across_tolerance=x_band)
    y_side = _axis_label_side(words, y_pixels, axis="y", axis_line=vertical, along_tolerance=y_tolerance, across_tolerance=y_band)
    all_reviews: list[TickLabel] = []
    corrected_pi_alias = False
    foreground = _foreground_mask(image, crop)

    def labels_for_axis(axis: Literal["x", "y"], pixels: Sequence[float], spacing: float, along_tolerance: float, across_tolerance: float, side: int, axis_line: AxisLine, cross_axis: AxisLine) -> tuple[TickLabel, ...]:
        nonlocal corrected_pi_alias
        degree_context = False
        for word in words:
            if "°" not in word.text and "º" not in word.text:
                continue
            center_x, center_y = (word.left + word.right) / 2, (word.top + word.bottom) / 2
            along = center_x if axis == "x" else center_y
            across = center_y - axis_line.pixel if axis == "x" else center_x - axis_line.pixel
            if axis_line.start - 4 <= along <= axis_line.end + 4 and 2 <= side * across <= across_tolerance:
                degree_context = True
                break
        output: list[TickLabel] = []
        for pixel in pixels:
            selected = _tick_words_for_axis(
                words, axis=axis, pixel=pixel, axis_line=axis_line, side=side,
                along_tolerance=along_tolerance, across_tolerance=across_tolerance, include_nonmath=True,
            )
            # If a label is OCR'd as a π glyph alias, resolve it only when
            # other nearby axis labels corroborate mathematical notation.
            grouped = _group_words(selected)
            raw_label = "\n".join(
                " ".join(item[0] for item in sorted(line, key=lambda item: item[1]))
                for line in _baseline_clusters(grouped, center_index=1, tolerance=max(4.0, min(height, width) * 0.025))
            ) if grouped else ""
            if not raw_label:
                raw_label = " ".join(word.text for word in sorted(selected, key=lambda item: (item.top, item.left)))
            direct = _direct_tick_values(selected)
            direct_values = {round(value, 12) for _label, value, _confidence in direct}
            unique_direct = len(direct_values) == 1
            direct_value = direct[0][1] if unique_direct and direct else None
            confidence = max((item[2] for item in direct), default=max((word.confidence for word in selected), default=0.0))
            pixel_font = float(np.median([word.bottom - word.top for word in selected])) if selected else 18.0
            selected_centers = [(word.left + word.right) / 2 if axis == "x" else (word.top + word.bottom) / 2 for word in selected]
            offset = min((abs(center - pixel) for center in selected_centers), default=0.0)
            vertical_centers = [(word.top + word.bottom) / 2 for word in selected]
            stacked = bool(vertical_centers and max(vertical_centers) - min(vertical_centers) > max(6.0, pixel_font * 0.55))
            pi_tokens = [word for word in selected if re.search(r"π|\\pi|\bpi\b", word.text, re.IGNORECASE)]
            coefficient_pi = any(re.match(r"\s*\d+\s*(?:π|pi)", word.text, re.IGNORECASE) for word in pi_tokens)
            lost_decimal = any(_looks_like_lost_decimal(word.text) for word in selected)
            shifted_pi = axis == "x" and any(abs(((word.left + word.right) / 2) - pixel) > pixel_font * 0.55 for word in pi_tokens)
            missing_degree_marker = degree_context and not any("°" in word.text or "º" in word.text for word in selected)
            low_degree_read = degree_context and confidence < 60
            needs_retry = not unique_direct or stacked or lost_decimal or shifted_pi or coefficient_pi or missing_degree_marker or low_degree_read

            local_options: list[_LocalTickRead] = []
            if needs_retry:
                tight_box, broad_box = _tick_crop_boxes(
                    image, crop, axis=axis, pixel=pixel, axis_line=axis_line,
                    cross_axis=cross_axis, side=side, words=selected, spacing=spacing,
                )
                oversized_tight_box = bool(
                    tight_box is not None and broad_box is not None and (
                        tight_box[2] - tight_box[0] > (broad_box[2] - broad_box[0]) * 1.5
                        or tight_box[3] - tight_box[1] > (broad_box[3] - broad_box[1]) * 1.5
                    )
                )
                boxes = (
                    (
                        tight_box,
                        broad_box if (
                            axis == "x" and (stacked or coefficient_pi or oversized_tight_box)
                        ) else None,
                    )
                    if selected else (broad_box,)
                )
                for box in boxes:
                    if box is not None:
                        for item in _read_local_tick_label(image, box):
                            # Retain compatibility with small injected OCR
                            # adapters used by callers/tests that return the
                            # earlier three-field tuple form.
                            if isinstance(item, _LocalTickRead):
                                label, value, local_confidence, local_box = item.text, item.value, item.confidence, item.box
                            else:
                                label, value, local_confidence = item
                                local_box = None
                            absolute_box = (
                                (
                                    local_box[0] + box[0], local_box[1] + box[1],
                                    local_box[2] + box[0], local_box[3] + box[1],
                                ) if local_box is not None else None
                            )
                            local_options.append(_LocalTickRead(label, value, local_confidence, absolute_box))

            has_pi_evidence = bool(pi_tokens or re.search(r"π|\\pi|\bpi\b", raw_label, re.IGNORECASE))
            parsed_local = [
                item for item in local_options
                if item.value is not None and not (
                    degree_context
                    and not has_pi_evidence
                    and re.search(r"π|Π|\\pi|\bpi\b", item.text, re.IGNORECASE)
                )
            ]
            chosen_label: str | None = None
            chosen_value: float | None = None
            chosen_confidence = confidence
            refined_pixel = float(pixel)
            if parsed_local:
                # A locally isolated mathematical expression outranks a
                # fragmented full-page read (e.g. π and 2 as separate lines).
                # Prefer a recognized fraction when OCR returned both a
                # compact and a multiline candidate for the same tick.
                def local_rank(item: _LocalTickRead) -> tuple[int, int, float]:
                    contains_pi = bool(re.search(r"π|\\pi|\bpi\b", item.text, re.IGNORECASE))
                    fraction = contains_pi and ("/" in item.text or "\n" in item.text)
                    return (
                        int(has_pi_evidence and contains_pi),
                        int(has_pi_evidence and fraction),
                        item.confidence,
                    )
                best = max(parsed_local, key=local_rank)
                chosen_label, chosen_value, chosen_confidence = best.text, best.value, best.confidence
                if best.box is not None:
                    text_center = (
                        (best.box[0] + best.box[2]) / 2
                        if axis == "x" else (best.box[1] + best.box[3]) / 2
                    )
                    refined_pixel = _snap_to_tick(
                        foreground, axis=axis, approximate_pixel=text_center,
                        axis_pixel=axis_line.pixel, crop=crop,
                    )
            elif unique_direct and direct:
                chosen_label, chosen_value, chosen_confidence = max(direct, key=lambda item: item[2])
            else:
                # Retain a suspicious numeric interpretation for review, but
                # never promote it into an axis fit without corroboration.
                if direct:
                    chosen_label, chosen_value, chosen_confidence = max(direct, key=lambda item: item[2])

            if chosen_value is None:
                status: Literal["ACCEPTED", "REVIEW_REQUIRED", "REJECTED"] = "REVIEW_REQUIRED"
                note = "No unambiguous numerical label was recognized in the isolated tick crop."
                output.append(TickLabel(axis, float(pixel), raw_label or "unreadable", None, confidence, status, chosen_label, note))
                continue

            conflicting = bool(direct_values and round(chosen_value, 12) not in direct_values)
            suspicious = lost_decimal or conflicting or len(direct_values) > 1
            note = None
            if lost_decimal:
                note = "A leading-zero OCR reading can hide a decimal point; the isolated label was retried and needs axis-consistency review."
            elif conflicting:
                note = f"Full-page OCR ({raw_label!r}) disagreed with isolated-crop OCR ({chosen_label!r}); check the source label."
            elif len(direct_values) > 1:
                note = "OCR split a stacked label into multiple numerical tokens; the isolated crop was used for review."
            if chosen_label and _looks_like_pi_ocr(chosen_label):
                alias_value, corrected = _parse_axis_tick_label(chosen_label, allow_pi_ocr_alias=True)
                if corrected and alias_value is not None:
                    chosen_value = alias_value
                    corrected_pi_alias = True
            label_for_review = raw_label or chosen_label or "unreadable"
            # Preserve distinct OCR readings at the same physical tick. A
            # global axis fit can then select among observed OCR alternatives
            # using the other tick positions; it never synthesizes a missing
            # digit or substitutes a value from the curve shape.
            candidate_options: dict[float, tuple[str, float, float]] = {}
            degree_alternatives: set[float] = set()
            minus_alternatives: set[float] = set()
            leading_minus_evidence = _has_leading_minus_evidence(raw_label)

            def add_candidate(candidate_label: str, candidate_value: float | None, candidate_confidence: float) -> None:
                if candidate_value is None or not math.isfinite(candidate_value):
                    return
                key = round(float(candidate_value), 12)
                previous = candidate_options.get(key)
                if previous is None or candidate_confidence > previous[1]:
                    candidate_options[key] = (candidate_label, float(candidate_confidence), refined_pixel)
                degree_candidate = _degree_suffix_candidate(candidate_label, degree_context=degree_context)
                if degree_candidate is not None:
                    alternate_label, alternate_value = degree_candidate
                    alternate_key = round(float(alternate_value), 12)
                    degree_alternatives.add(alternate_key)
                    previous_alternate = candidate_options.get(alternate_key)
                    review_label = f"{alternate_label} candidate from OCR {candidate_label!r}"
                    if previous_alternate is None or candidate_confidence > previous_alternate[1]:
                        candidate_options[alternate_key] = (review_label, float(candidate_confidence), refined_pixel)
                    # Full-page OCR may retain a leading minus while isolated
                    # OCR reads the digits but drops that glyph (for example,
                    # a local ``1800`` read beside a full-page ``-18...``).
                    # Combine those observations only as another review-only
                    # candidate; the independent axis fit must corroborate it.
                    if leading_minus_evidence and alternate_value > 0:
                        negative_alternate = -float(alternate_value)
                        negative_key = round(negative_alternate, 12)
                        degree_alternatives.add(negative_key)
                        minus_alternatives.add(negative_key)
                        previous_negative = candidate_options.get(negative_key)
                        review_label = f"-{alternate_label} candidate from the visible minus glyph and OCR {candidate_label!r}"
                        if previous_negative is None or candidate_confidence > previous_negative[1]:
                            candidate_options[negative_key] = (review_label, float(candidate_confidence), refined_pixel)
                if leading_minus_evidence and candidate_value > 0:
                    negative_value = -float(candidate_value)
                    negative_key = round(negative_value, 12)
                    minus_alternatives.add(negative_key)
                    previous_negative = candidate_options.get(negative_key)
                    review_label = f"-{candidate_label} candidate from the visible minus glyph"
                    if previous_negative is None or candidate_confidence > previous_negative[1]:
                        candidate_options[negative_key] = (review_label, float(candidate_confidence), refined_pixel)

            for candidate_label, candidate_value, candidate_confidence in direct:
                add_candidate(candidate_label, candidate_value, candidate_confidence)
            for candidate in parsed_local:
                add_candidate(candidate.text, candidate.value, candidate.confidence)
            candidate_options.setdefault(
                round(float(chosen_value), 12),
                (chosen_label or label_for_review, float(chosen_confidence), refined_pixel),
            )
            multiple_interpretations = len(candidate_options) > 1
            for candidate_value, (candidate_label, candidate_confidence, candidate_pixel) in candidate_options.items():
                differs_from_primary = not math.isclose(candidate_value, float(chosen_value), rel_tol=1e-10, abs_tol=1e-10)
                candidate_status: Literal["ACCEPTED", "REVIEW_REQUIRED", "REJECTED"] = (
                    "REVIEW_REQUIRED" if suspicious or multiple_interpretations or differs_from_primary else "ACCEPTED"
                )
                candidate_note = note
                if multiple_interpretations and not candidate_note:
                    candidate_note = "OCR returned multiple numerical readings for this tick; the axis-wide mapping must corroborate one interpretation."
                if round(candidate_value, 12) in degree_alternatives and round(candidate_value, 12) in minus_alternatives:
                    candidate_note = (
                        "The degree suffix and leading minus were recovered from separate OCR evidence; this combined reading is only an alternative and was accepted only because multiple axis anchors support it. Verify it against the image."
                    )
                elif round(candidate_value, 12) in degree_alternatives:
                    candidate_note = (
                        "The final OCR character may be a degree symbol; this is an alternative reading only and was retained because other labels establish degree notation. "
                        "It is accepted only if the independently detected tick positions support the same mapping."
                    )
                elif round(candidate_value, 12) in minus_alternatives:
                    candidate_note = (
                        "A minus glyph was present in the isolated source-label region but may have been lost by OCR. "
                        "This negative reading is an alternative only and requires corroboration from other tick anchors."
                    )
                output.append(TickLabel(
                    axis, float(candidate_pixel), label_for_review, float(candidate_value), float(candidate_confidence),
                    candidate_status, candidate_label, candidate_note,
                ))
        return tuple(output)

    x_ticks = labels_for_axis("x", x_pixels, x_spacing, x_tolerance, x_band, x_side, horizontal, vertical)
    y_ticks = labels_for_axis("y", y_pixels, y_spacing, y_tolerance, y_band, y_side, vertical, horizontal)
    origin_zeros = _origin_zero_candidates(words, horizontal, vertical, crop)
    for origin_zero in origin_zeros:
        target = x_ticks if origin_zero.axis == "x" else y_ticks
        if not any(
            tick.axis == origin_zero.axis and abs(tick.pixel - origin_zero.pixel) <= 2 and tick.value == 0
            for tick in target
        ):
            if origin_zero.axis == "x":
                x_ticks = (*x_ticks, origin_zero)
            else:
                y_ticks = (*y_ticks, origin_zero)
    all_reviews.extend(x_ticks)
    all_reviews.extend(y_ticks)
    return x_ticks, y_ticks, tuple(all_reviews), corrected_pi_alias


def _fit_axis(ticks: Sequence[TickLabel], axis: Literal["x", "y"]) -> _AxisFit | None:
    unique: dict[tuple[float, float], TickLabel] = {}
    for tick in ticks:
        if tick.value is not None and tick.review_status != "REJECTED":
            unique[(round(tick.pixel, 2), tick.value)] = tick
    points = sorted(unique.values(), key=lambda tick: tick.pixel)
    if len({round(point.pixel, 2) for point in points}) < 2:
        return None

    if np.ptp([tick.pixel for tick in points]) <= 2:
        return None

    def robust_fit(scale: ImageScale) -> tuple[float, tuple[TickLabel, ...], np.ndarray] | None:
        usable = [tick for tick in points if scale == "linear" or float(tick.value) > 0]
        if len({round(tick.pixel, 2) for tick in usable}) < 2:
            return None
        pixel = np.asarray([tick.pixel for tick in usable], dtype=float)
        values = np.asarray([tick.value for tick in usable], dtype=float)
        transformed = np.log10(values) if scale == "log" else values
        distinct_pixels = np.unique(np.round(pixel, 2))
        local_spacing = float(np.median(np.diff(distinct_pixels))) if len(distinct_pixels) >= 2 else float(np.ptp(pixel))
        # OCR label centers are not always printed directly over tick marks;
        # on small plots, font alignment can shift a label by several pixels.
        # Use a modest fraction of the observed interval (still capped by the
        # axis span) and let multi-anchor consensus reject value outliers.
        threshold = max(5.0, min(float(np.ptp(pixel)) * 0.018, local_spacing * 0.08))
        best: tuple[int, float, float, float, float] | None = None
        for first in range(len(usable) - 1):
            for second in range(first + 1, len(usable)):
                if abs(pixel[second] - pixel[first]) <= 2:
                    continue
                dx = transformed[second] - transformed[first]
                if abs(dx) < 1e-14:
                    continue
                slope = (pixel[second] - pixel[first]) / dx
                intercept = pixel[first] - slope * transformed[first]
                residuals = np.abs(pixel - (slope * transformed + intercept))
                best_by_pixel: dict[float, int] = {}
                for index, tick in enumerate(usable):
                    if residuals[index] > threshold:
                        continue
                    key = round(tick.pixel, 2)
                    previous = best_by_pixel.get(key)
                    if previous is None or (residuals[index], -tick.ocr_confidence) < (residuals[previous], -usable[previous].ocr_confidence):
                        best_by_pixel[key] = index
                chosen_indices = list(best_by_pixel.values())
                if len(chosen_indices) < 2:
                    continue
                error = float(np.median(residuals[chosen_indices]))
                mean_confidence = float(np.mean([usable[index].ocr_confidence for index in chosen_indices]))
                candidate = (len(chosen_indices), -error, mean_confidence, -abs(slope), -abs(intercept))
                if best is None or candidate > best:
                    best = candidate
                    best_indices = chosen_indices
        if best is None:
            return None
        # Keep only the best-supported OCR interpretation per distinct tick;
        # duplicate OCR alternatives must not count as extra calibration data.
        inlier_points = [usable[index] for index in best_indices]
        transformed_selected = np.asarray([
            math.log10(float(tick.value)) if scale == "log" else float(tick.value)
            for tick in inlier_points
        ])
        pixel_selected = np.asarray([tick.pixel for tick in inlier_points], dtype=float)
        design = np.column_stack((transformed_selected, np.ones(len(inlier_points))))
        slope, intercept = np.linalg.lstsq(design, pixel_selected, rcond=None)[0]
        if abs(slope) < 1e-14:
            return None
        residuals = pixel_selected - (slope * transformed_selected + intercept)
        # A final residual check protects against a pairwise model that was
        # supported only by a loose threshold at large image dimensions.
        keep = np.abs(residuals) <= threshold
        inlier_points = [tick for tick, accepted in zip(inlier_points, keep) if accepted]
        residuals = residuals[keep]
        if len(inlier_points) < 2:
            return None
        transformed_selected = np.asarray([
            math.log10(float(tick.value)) if scale == "log" else float(tick.value)
            for tick in inlier_points
        ])
        pixel_selected = np.asarray([tick.pixel for tick in inlier_points], dtype=float)
        design = np.column_stack((transformed_selected, np.ones(len(inlier_points))))
        slope, intercept = np.linalg.lstsq(design, pixel_selected, rcond=None)[0]
        residuals = pixel_selected - (slope * transformed_selected + intercept)
        return float(np.sqrt(np.mean(residuals**2))), tuple(inlier_points), residuals

    linear_fit = robust_fit("linear")
    positive_tick_count = len({round(tick.pixel, 2) for tick in points if float(tick.value) > 0})
    # A logarithmic axis cannot represent any confirmed zero/negative tick.
    # Do not let a positive subset of a mixed-sign axis win merely because a
    # few OCR alternatives happen to line up in log space.
    all_ticks_positive = all(float(tick.value) > 0 for tick in points)
    logarithmic_fit = robust_fit("log") if positive_tick_count >= 3 and all_ticks_positive else None
    selected_scale: ImageScale = "linear"
    selected = linear_fit
    # A two-point logarithmic fit has zero residual by construction and can
    # therefore beat a well-supported linear fit even when the axis is linear.
    # Require the logarithmic model itself to explain at least three distinct
    # tick positions before considering it; the candidate pool may contain
    # three positive labels while its actual consensus contains only two.
    log_is_supported = logarithmic_fit is not None and len(logarithmic_fit[1]) >= 3
    if log_is_supported and linear_fit is not None:
        # Select log only when the observed tick spacing strongly supports it.
        if logarithmic_fit[0] < max(1.0, linear_fit[0] * 0.55):
            selected_scale, selected = "log", logarithmic_fit
    elif log_is_supported:
        selected_scale, selected = "log", logarithmic_fit
    if selected is None:
        return None
    _, accepted, residuals = selected
    if len(accepted) < 2:
        return None
    first, last = accepted[0], accepted[-1]
    transformed = np.asarray([
        math.log10(float(tick.value)) if selected_scale == "log" else float(tick.value)
        for tick in accepted
    ])
    pixels = np.asarray([tick.pixel for tick in accepted], dtype=float)
    slope, intercept = np.linalg.lstsq(
        np.column_stack((transformed, np.ones(len(transformed)))), pixels, rcond=None
    )[0]
    first_coordinate = math.log10(float(first.value)) if selected_scale == "log" else float(first.value)
    last_coordinate = math.log10(float(last.value)) if selected_scale == "log" else float(last.value)
    # Preserve the multi-anchor regression in the two-point calibration type:
    # raw outer label centers can be misaligned by a few pixels, and using them
    # directly would discard the noise reduction from the accepted tick set.
    first_pixel = float(slope * first_coordinate + intercept)
    last_pixel = float(slope * last_coordinate + intercept)
    return _AxisFit(
        first_pixel=first_pixel,
        first_value=first.value,
        last_pixel=last_pixel,
        last_value=last.value,
        scale=selected_scale,
        ticks=accepted,
        rms_pixels=float(np.sqrt(np.mean(residuals**2))),
        inlier_count=len(accepted),
    )


def _settle_tick_reviews(
    ticks: Sequence[TickLabel], axis: Literal["x", "y"]
) -> tuple[tuple[TickLabel, ...], tuple[TickLabel, ...], _AxisFit | None]:
    """Accept a corrected/uncertain OCR value only when other ticks corroborate it."""
    provisional = _fit_axis(ticks, axis)
    inliers = {
        (round(tick.pixel, 2), round(float(tick.value), 12))
        for tick in (provisional.ticks if provisional else ())
        if tick.value is not None
    }
    settled: list[TickLabel] = []
    for tick in ticks:
        key = (round(tick.pixel, 2), round(float(tick.value), 12)) if tick.value is not None else None
        corroborated = key in inliers if key is not None else False
        if tick.review_status == "REVIEW_REQUIRED":
            # Two values define a line but cannot resolve an OCR ambiguity.
            # Require at least three geometrically consistent tick anchors.
            if corroborated and provisional is not None and provisional.inlier_count >= 3:
                note = tick.note
                suffix = f"The value agrees with {provisional.inlier_count} consistent tick anchors."
                note = f"{note} {suffix}" if note else suffix
                settled.append(replace(tick, review_status="ACCEPTED", note=note))
            elif corroborated:
                note = tick.note
                suffix = "Only two tick anchors support this interpretation; additional review is needed to resolve OCR ambiguity."
                note = f"{note} {suffix}" if note else suffix
                settled.append(replace(tick, note=note))
            else:
                settled.append(replace(tick, review_status="REJECTED", note=tick.note or "This reading conflicts with the other tick labels."))
        elif tick.value is not None and provisional is not None and not corroborated:
            settled.append(replace(tick, review_status="REJECTED", note="This reading was excluded because its pixel/value pair conflicts with the axis mapping."))
        else:
            settled.append(tick)
    trusted = tuple(tick for tick in settled if tick.review_status == "ACCEPTED" and tick.value is not None)
    return trusted, tuple(settled), _fit_axis(trusted, axis)


def _empty_values() -> dict[str, float | None]:
    return {
        "x_left_px": None, "x_right_px": None, "x_min": None, "x_max": None,
        "y_top_px": None, "y_bottom_px": None, "y_min": None, "y_max": None,
    }


def auto_calibrate_axes(
    image: Image.Image,
    crop: tuple[int, int, int, int] | None = None,
    *,
    ocr_reader: Callable[[Image.Image], Sequence[OCRWord]] | None = None,
) -> AutomaticCalibrationResult:
    """Estimate axis geometry and pixel/data mapping using local OCR.

    ``crop`` and all returned pixel positions use absolute coordinates in the
    original image. An injected ``ocr_reader`` is useful for deterministic tests;
    production defaults to local Tesseract when available.
    """
    if image.width <= 0 or image.height <= 0:
        raise ValueError("The source image must have positive dimensions")
    crop = validate_crop(image.size, crop or (0, 0, image.width, image.height))
    warnings: list[str] = []
    ocr_available = ocr_reader is not None
    if ocr_reader is not None:
        words = list(ocr_reader(image.crop(crop)))
        # Injected readers return crop-relative locations, matching OCR output.
        words = [OCRWord(item.text, item.left + crop[0], item.top + crop[1], item.right + crop[0], item.bottom + crop[1], item.confidence) for item in words]
    else:
        available, reason = local_ocr_available()
        if not available:
            words = []
            warnings.append(reason or "Local OCR is unavailable.")
        else:
            try:
                words = recognize_local_words(image.crop(crop))
                words = [OCRWord(item.text, item.left + crop[0], item.top + crop[1], item.right + crop[0], item.bottom + crop[1], item.confidence) for item in words]
                ocr_available = True
            except Exception as error:
                words = []
                warnings.append(str(error))

    horizontal, vertical, plot_region = _detect_axes(image, crop, words)
    geometry_available = horizontal is not None and vertical is not None and plot_region is not None
    review_ticks: tuple[TickLabel, ...] = ()
    if geometry_available:
        if ocr_reader is None:
            # Full-page OCR is fast for ordinary plots, but it can collapse
            # adjacent labels in compact figures. Re-read strips selected from
            # the detected axis geometry and use them axis-by-axis only when
            # they produce multiple likely tick labels. All OCR coordinates are
            # translated back to the original image before association.
            localized = _recognize_axis_label_bands(image, crop, horizontal, vertical, words) if ocr_available else {"x": [], "y": []}
            # Keep both full-page and localized OCR hypotheses. They may
            # disagree (for example, a dropped degree glyph); downstream
            # robust fitting resolves only alternatives supported by distinct
            # tick positions, while retaining every reading for human review.
            supplemental = [
                *(
                    localized["x"]
                    if _axis_label_word_count(localized["x"], "x", horizontal, crop) >= 1
                    else []
                ),
                *(
                    localized["y"]
                    if _axis_label_word_count(localized["y"], "y", vertical, crop) >= 1
                    else []
                ),
            ]
            tick_words = [*words, *supplemental]
            x_ticks, y_ticks, review_ticks, corrected_pi_alias = _tick_labels_from_axis_geometry(
                image, crop, tick_words, horizontal, vertical, plot_region
            )
            saw_ocr = bool(tick_words)
        else:
            x_ticks, y_ticks, saw_ocr, corrected_pi_alias = _tick_labels(image, crop, words, horizontal, vertical, plot_region)
            review_ticks = tuple((*x_ticks, *y_ticks))
        if len(x_ticks) < 2 or len(y_ticks) < 2:
            layout_x, layout_y, layout_corrected = _tick_labels_from_text_layout(crop, words)
            if len(x_ticks) < 2 and len(layout_x) >= 2:
                x_ticks = layout_x
            if len(y_ticks) < 2 and len(layout_y) >= 2:
                y_ticks = layout_y
            corrected_pi_alias |= layout_corrected
            if ocr_reader is None and (layout_x or layout_y):
                review_ticks = tuple((*x_ticks, *y_ticks))
    else:
        # Axis lines may be absent, faint, clipped, or fail to intersect inside
        # the crop. Aligned labels can still establish a candidate mapping,
        # although without line geometry it must be reviewed at low confidence.
        x_ticks, y_ticks, corrected_pi_alias = _tick_labels_from_text_layout(crop, words)
        saw_ocr = bool(words)
        review_ticks = tuple((*x_ticks, *y_ticks))
        if x_ticks or y_ticks:
            plot_region = crop
            warnings.append("Axis strokes could not be confirmed as an intersecting pair. Mapping is estimated from aligned tick-label positions; review it carefully.")
        else:
            available, reason = local_ocr_available() if ocr_reader is None else (True, None)
            if not available and reason:
                warnings.append(reason)
            warnings.append("Could not identify plot axes or aligned numeric tick-label rows/columns. Enter calibration anchors manually.")
            return AutomaticCalibrationResult(
                _empty_values(), "NEEDS_REVIEW", "NONE", "linear", "linear", plot_region,
                horizontal.pixel if horizontal else None, vertical.pixel if vertical else None,
                (), (), available, tuple(warnings),
            )
    if corrected_pi_alias:
        warnings.append("Some tick labels used a context-supported OCR correction for the π symbol. Verify those labels against the image.")
    if not saw_ocr or not x_ticks and not y_ticks:
        warnings.append("No unambiguous numeric tick labels were recognized near the detected axes. Geometry alone cannot establish data values; enter them manually.")
    elif len(x_ticks) < 2:
        warnings.append("Fewer than two usable X tick labels were recognized; X calibration needs manual anchors.")
    elif len(y_ticks) < 2:
        warnings.append("Fewer than two usable Y tick labels were recognized; Y calibration needs manual anchors.")

    trusted_x, reviewed_x, x_fit = _settle_tick_reviews(x_ticks, "x")
    trusted_y, reviewed_y, y_fit = _settle_tick_reviews(y_ticks, "y")
    if review_ticks:
        by_axis = {"x": reviewed_x, "y": reviewed_y}
        final_review_ticks = tuple((*by_axis["x"], *by_axis["y"]))
    else:
        final_review_ticks = tuple((*reviewed_x, *reviewed_y))
    if any(tick.review_status in {"REVIEW_REQUIRED", "REJECTED"} for tick in final_review_ticks):
        warnings.append("One or more tick labels remain ambiguous or were rejected; review the detected-label table before using the mapping scientifically.")
    for tick in final_review_ticks:
        if tick.note and "OCR" in tick.note and tick.review_status == "ACCEPTED":
            warnings.append("A localized OCR retry corrected a full-page label; the correction agrees with the other axis ticks but should still be visually reviewed.")
            break
    x_ticks, y_ticks = trusted_x, trusted_y
    values = _empty_values()
    if x_fit:
        values.update({
            "x_left_px": x_fit.first_pixel,
            "x_right_px": x_fit.last_pixel,
            "x_min": x_fit.first_value,
            "x_max": x_fit.last_value,
        })
    if y_fit:
        values.update({
            "y_top_px": y_fit.first_pixel,
            "y_bottom_px": y_fit.last_pixel,
            "y_max": y_fit.first_value,
            "y_min": y_fit.last_value,
        })
    x_rejected = any(tick.axis == "x" and tick.review_status == "REJECTED" for tick in final_review_ticks)
    y_rejected = any(tick.axis == "y" and tick.review_status == "REJECTED" for tick in final_review_ticks)
    if x_fit and y_fit:
        status: Literal["DETECTED", "PARTIAL", "NEEDS_REVIEW"] = "DETECTED"
        confidence = "HIGH" if x_fit.inlier_count >= 3 and y_fit.inlier_count >= 3 and max(x_fit.rms_pixels, y_fit.rms_pixels) < 3 else "MEDIUM"
        if not geometry_available:
            confidence = "LOW"
        if len(x_fit.ticks) == 2 or len(y_fit.ticks) == 2:
            confidence = "LOW"
        ocr_quality = min(
            float(np.mean([tick.ocr_confidence for tick in x_fit.ticks])),
            float(np.mean([tick.ocr_confidence for tick in y_fit.ticks])),
        )
        if ocr_quality < 45:
            confidence = "LOW"
            warnings.append("Tick-label OCR confidence was low; verify the detected values against the source image.")
        elif ocr_quality < 70 and confidence == "HIGH":
            confidence = "MEDIUM"
            warnings.append("Some tick labels had moderate OCR confidence; verify the detected values.")
    elif x_fit or y_fit:
        status, confidence = "PARTIAL", "MEDIUM"
        warnings.append("Only one axis has two or more consistent numeric ticks; complete the other axis manually.")
    else:
        status, confidence = "NEEDS_REVIEW", "LOW" if horizontal and vertical else "NONE"
    if x_fit and len(x_fit.ticks) == 2:
        warnings.append("X has only two usable ticks, so linear versus logarithmic scale cannot be inferred from spacing. Linear is selected provisionally; review the axis scale.")
    if y_fit and len(y_fit.ticks) == 2:
        warnings.append("Y has only two usable ticks, so linear versus logarithmic scale cannot be inferred from spacing. Linear is selected provisionally; review the axis scale.")
    contextual_zero_used = any(
        tick.note and "proposed as zero" in tick.note and tick.review_status == "ACCEPTED"
        for tick in final_review_ticks
    )
    if contextual_zero_used:
        confidence = "MEDIUM"
        warnings.append("An O/omicron glyph at the axis intersection supported the zero tick only in context; confirm this label against the source image.")
    if x_rejected or y_rejected:
        warnings.append("One or more tick readings did not agree with the fitted axis mapping and were excluded. Review the tick overlay.")
    return AutomaticCalibrationResult(
        values=values,
        status=status,
        confidence=confidence,
        x_scale=x_fit.scale if x_fit else "linear",
        y_scale=y_fit.scale if y_fit else "linear",
        plot_region=plot_region,
        x_axis_pixel=horizontal.pixel if horizontal else None,
        y_axis_pixel=vertical.pixel if vertical else None,
        x_ticks=x_fit.ticks if x_fit else x_ticks,
        y_ticks=y_fit.ticks if y_fit else y_ticks,
        ocr_available=ocr_available,
        warnings=tuple(warnings),
        review_ticks=final_review_ticks,
    )


def validate_auto_calibration(result: AutomaticCalibrationResult, image_size: tuple[int, int], crop: tuple[int, int, int, int]) -> AxisCalibration | None:
    """Build the existing calibration type when both axes are supported."""
    if result.status != "DETECTED" or any(value is None for value in result.values.values()):
        return None
    calibration = AxisCalibration(
        x_left_px=float(result.values["x_left_px"]),
        x_right_px=float(result.values["x_right_px"]),
        x_min=float(result.values["x_min"]),
        x_max=float(result.values["x_max"]),
        y_top_px=float(result.values["y_top_px"]),
        y_bottom_px=float(result.values["y_bottom_px"]),
        y_min=float(result.values["y_min"]),
        y_max=float(result.values["y_max"]),
        x_scale=result.x_scale,
        y_scale=result.y_scale,
    )
    calibration.validate_image_bounds(image_size, crop)
    return calibration
