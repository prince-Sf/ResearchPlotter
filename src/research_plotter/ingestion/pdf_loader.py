"""Text and layout-aware extraction from PDF documents.

The plain-text helpers are retained for existing workflows.  Research-paper
analysis should use :func:`extract_pdf_layout`, which keeps block geometry,
font cues, and image regions instead of flattening a page into one string.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from pathlib import Path
from typing import Literal


PDFContentType = Literal[
    "prose", "heading", "equation", "figure", "figure_caption", "table",
    "reference", "metadata",
]


@dataclass(frozen=True)
class PDFLayoutRegion:
    """A text or image block with its page position and conservative class."""

    page_number: int
    region_id: str
    bbox: tuple[float, float, float, float]
    content_type: PDFContentType
    text: str = ""
    font_sizes: tuple[float, ...] = ()
    font_names: tuple[str, ...] = ()
    is_bold: bool = False

    def model_dump(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class PDFPageLayout:
    """Layout summary for one PDF page."""

    page_number: int
    width: float
    height: float
    regions: tuple[PDFLayoutRegion, ...]

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for region in self.regions:
            result[region.content_type] = result.get(region.content_type, 0) + 1
        return result


_FIGURE_CAPTION = re.compile(r"^\s*(?:fig(?:ure)?\.?\s*\d+\b)", re.I)
_TABLE_CAPTION = re.compile(r"^\s*table\s+\d+\b", re.I)
_REFERENCE = re.compile(r"^\s*(?:\[\d{1,3}\]|\d{1,3}\.\s+[A-Z])")
_EQUATION_SIGNS = re.compile(r"(?:=|∫|∑|∏|√|≤|≥|↦|\b(?:d\w*\s*/\s*d\w*|D\s*\^|Gamma|Gamma\s*\())")


def _classify_text_region(
    text: str,
    *,
    font_sizes: tuple[float, ...],
    bbox: tuple[float, float, float, float],
    page_height: float,
    first_page: bool,
    references_page: bool,
    is_bold: bool,
) -> PDFContentType:
    normalized = " ".join(text.split())
    lowered = normalized.lower()
    if not normalized:
        return "prose"
    if _FIGURE_CAPTION.match(normalized):
        return "figure_caption"
    if _TABLE_CAPTION.match(normalized):
        return "table"
    if references_page or _REFERENCE.match(normalized):
        return "reference"
    if _EQUATION_SIGNS.search(normalized) and len(normalized) < 2400:
        return "equation"
    if any(key in lowered for key in ("doi:", "issn", "journal homepage", "article info", "received ", "accepted ")):
        return "metadata"
    median_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 0.0
    top = bbox[1]
    short = len(normalized) <= 140 and not normalized.endswith((".", ";", ","))
    if short and (is_bold or median_size >= 10.0):
        return "heading"
    if first_page and top < page_height * 0.2 and len(normalized) <= 180 and median_size >= 12.0:
        return "metadata"
    if re.match(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z]", normalized) and len(normalized) <= 140:
        return "heading"
    # Table body blocks are often arranged as aligned rows with short cells;
    # keep the classification cautious and use table context where possible.
    if _TABLE_CAPTION.search(lowered) or ("parameter" in lowered and "value" in lowered):
        return "table"
    return "prose"


def _open_pdf(path: str | Path):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    return fitz, fitz.open(Path(path))


def extract_pdf_pages(path: str | Path) -> list[str]:
    source = Path(path)
    try:
        try:
            import pymupdf as fitz  # current PyMuPDF namespace
        except ImportError:
            import fitz  # legacy namespace

        with fitz.open(source) as document:
            return [page.get_text("text") for page in document]
    except ImportError:
        from pypdf import PdfReader

        return [page.extract_text() or "" for page in PdfReader(source).pages]


def extract_pdf_layout(
    path: str | Path,
    page_numbers: list[int] | tuple[int, ...] | None = None,
) -> list[PDFPageLayout]:
    """Extract positioned text/image blocks and conservative content classes.

    ``page_numbers`` are one-based.  Text blocks retain their original line
    breaks; this function does not claim to reconstruct mathematical layout.
    The PyMuPDF fallback namespace is supported, but layout needs PyMuPDF.
    """
    fitz, document = _open_pdf(path)
    try:
        selected = set(page_numbers) if page_numbers is not None else None
        if selected is not None and any(number < 1 or number > len(document) for number in selected):
            raise ValueError("PDF page numbers must be within the document page range")
        result: list[PDFPageLayout] = []
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            if selected is not None and page_number not in selected:
                continue
            raw = page.get_text("dict")
            blocks = raw.get("blocks", [])
            all_text = "\n".join(
                span.get("text", "")
                for block in blocks if block.get("type") == 0
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            )
            references_page = bool(re.search(r"(?im)^\s*(?:references|bibliography)\s*$", all_text))
            regions: list[PDFLayoutRegion] = []
            for block_index, block in enumerate(blocks):
                bbox_values = block.get("bbox", (0, 0, 0, 0))
                bbox = tuple(float(value) for value in bbox_values)
                if block.get("type") == 1:
                    regions.append(PDFLayoutRegion(page_number, f"p{page_number}-b{block_index}", bbox, "figure"))
                    continue
                lines = block.get("lines", [])
                spans = [span for line in lines for span in line.get("spans", [])]
                text = "\n".join(
                    "".join(span.get("text", "") for span in line.get("spans", []))
                    for line in lines
                ).strip()
                font_sizes = tuple(float(span.get("size", 0.0)) for span in spans)
                font_names = tuple(sorted({str(span.get("font", "")) for span in spans if span.get("font")}))
                is_bold = any("bold" in str(span.get("font", "")).lower() for span in spans)
                content_type = _classify_text_region(
                    text, font_sizes=font_sizes, bbox=bbox, page_height=float(page.rect.height),
                    first_page=page_number == 1, references_page=references_page, is_bold=is_bold,
                )
                if text:
                    regions.append(PDFLayoutRegion(
                        page_number, f"p{page_number}-b{block_index}", bbox,
                        content_type, text, font_sizes, font_names, is_bold,
                    ))
            # Mark table-body text near a table caption when it shares the
            # caption's column. This avoids classifying every aligned paragraph
            # on the page as a table.
            table_captions = [region for region in regions if region.content_type == "table" and _TABLE_CAPTION.match(region.text)]
            for index, region in enumerate(regions):
                if region.content_type != "prose":
                    continue
                for caption in table_captions:
                    vertical_distance = region.bbox[1] - caption.bbox[3]
                    horizontal_overlap = min(region.bbox[2], caption.bbox[2]) - max(region.bbox[0], caption.bbox[0])
                    if 0 <= vertical_distance <= 145 and horizontal_overlap >= 10:
                        regions[index] = PDFLayoutRegion(
                            region.page_number, region.region_id, region.bbox, "table", region.text,
                            region.font_sizes, region.font_names, region.is_bold,
                        )
                        break
            # Vector charts are common in scientific papers and do not appear
            # as image blocks. Associate large drawing bounds with nearby
            # figure captions; deduplicate overlapping nested chart drawings.
            captions = [region for region in regions if region.content_type == "figure_caption"]
            page_area = float(page.rect.width) * float(page.rect.height)
            chart_bounds: list[tuple[float, float, float, float]] = []
            drawings = sorted(page.get_drawings(), key=lambda item: item["rect"].width * item["rect"].height, reverse=True)
            for drawing in drawings:
                rect = drawing["rect"]
                bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                area = max(0.0, rect.width) * max(0.0, rect.height)
                if area < page_area * 0.035:
                    continue
                matched_caption = any(
                    0 <= caption.bbox[1] - bbox[3] <= 34
                    and min(caption.bbox[2], bbox[2]) - max(caption.bbox[0], bbox[0]) >= 12
                    for caption in captions
                )
                if not matched_caption:
                    continue
                overlaps = False
                for prior in chart_bounds:
                    intersection = max(0.0, min(prior[2], bbox[2]) - max(prior[0], bbox[0])) * max(0.0, min(prior[3], bbox[3]) - max(prior[1], bbox[1]))
                    union = (prior[2] - prior[0]) * (prior[3] - prior[1]) + area - intersection
                    if union and intersection / union > 0.86:
                        overlaps = True
                        break
                if not overlaps:
                    chart_bounds.append(bbox)
                    regions.append(PDFLayoutRegion(page_number, f"p{page_number}-figure-{len(chart_bounds):02d}", bbox, "figure"))
            result.append(PDFPageLayout(page_number, float(page.rect.width), float(page.rect.height), tuple(regions)))
        return result
    finally:
        document.close()


def render_pdf_region(
    path: str | Path,
    page_number: int,
    bbox: tuple[float, float, float, float] | None = None,
    *,
    scale: float = 2.0,
) -> bytes:
    """Render a page or selected region as PNG bytes for human review."""
    if scale <= 0 or scale > 5:
        raise ValueError("PDF preview scale must be greater than 0 and no greater than 5")
    fitz, document = _open_pdf(path)
    try:
        if page_number < 1 or page_number > len(document):
            raise ValueError("PDF page number is outside the document")
        page = document[page_number - 1]
        clip = None
        if bbox is not None:
            clip = fitz.Rect(*bbox) & page.rect
            if clip.is_empty or clip.width <= 0 or clip.height <= 0:
                raise ValueError("PDF region must overlap the selected page")
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        return pixmap.tobytes("png")
    finally:
        document.close()


def extract_pdf_text(path: str | Path) -> str:
    return "\n\f\n".join(extract_pdf_pages(path))
