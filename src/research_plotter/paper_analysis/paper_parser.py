"""Evidence-preserving PDF analysis.

The parser extracts text and conservative metadata. It does not attempt to
execute or infer equations from arbitrary PDF content; uncertain fields remain
explicitly marked for review.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_plotter.ingestion.pdf_loader import extract_pdf_layout, extract_pdf_pages
from research_plotter.paper_adapters.facts import enrich_figure_facts, identify_document, known_facts
from research_plotter.paper_analysis.figure_manifest import EvidenceStatus, FigureManifest
from research_plotter.paper_analysis.equation_extractor import (
    EquationSpec,
    MathematicalType,
    extract_pdf_equations,
)


class EvidenceItem(BaseModel):
    value: Any
    status: EvidenceStatus
    source_pages: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PaperAnalysis(BaseModel):
    source_document: str
    page_count: int
    title: EvidenceItem
    authors: EvidenceItem
    sections: list[str] = Field(default_factory=list)
    equations: list[EquationSpec] = Field(default_factory=list)
    page_layout_summary: list[dict[str, Any]] = Field(default_factory=list)
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    initial_conditions: EvidenceItem
    state_variables: EvidenceItem
    numerical_methods: EvidenceItem
    fractional_orders: EvidenceItem
    fractal_dimensions: EvidenceItem
    time_ranges: EvidenceItem
    sensitivity_methodology: EvidenceItem
    noise_experiments: EvidenceItem
    neural_network_methodology: EvidenceItem
    control_variables: EvidenceItem
    validation_methodology: EvidenceItem
    figures: list[FigureManifest] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    unresolved_information: list[str] = Field(default_factory=list)


def _first_match(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, flags=re.I | re.S)
    return match.group(1).strip() if match else default


def _item(value: Any, status: EvidenceStatus, page: int | None = None, notes: list[str] | None = None) -> EvidenceItem:
    return EvidenceItem(value=value, status=status, source_pages=[page] if page else [], notes=notes or [])


def parse_paper(path: str | Path) -> PaperAnalysis:
    source = Path(path)
    pages = extract_pdf_pages(source)
    try:
        layouts = extract_pdf_layout(source)
    except (ImportError, RuntimeError, ValueError) as error:
        layouts = []
        layout_warning = f"Layout-aware PDF extraction was unavailable: {error}"
    else:
        layout_warning = ""
    text = "\n".join(pages)
    document_key = identify_document(source)
    facts = known_facts(source)
    title_text = ""
    normalized_text = " ".join(text.split())
    known_title = facts.get("title", "")
    if known_title and " ".join(known_title.lower().split()) in normalized_text.lower():
        title_text = known_title
    title_text = title_text or _first_match(r"(?:^|\n)([^\n]{15,180})\n(?:[A-Z][^\n]{2,80}\n){1,4}", pages[0] if pages else "")
    if not title_text:
        title_text = _first_match(r"(?:title|article)\s*[:\-]?\s*([^\n]{15,180})", text)
    authors = facts.get("authors", "") or _first_match(r"(?:authors?|by)\s*[:\-]?\s*([^\n]{5,300})", pages[0] if pages else "")
    if not authors:
        author_lines = [line.strip() for line in (pages[0] if pages else "").splitlines() if line.strip()]
        title_index = next((index for index, line in enumerate(author_lines) if title_text and title_text[:30].lower() in line.lower()), -1)
        if title_index >= 0 and title_index + 1 < len(author_lines):
            authors = author_lines[title_index + 1]
    sections = []
    for page in pages:
        for match in re.finditer(r"(?m)^\s*(\d+(?:\.\d+)*\s+[A-Z][^\n]{2,100})\s*$", page):
            value = match.group(1).strip()
            if value not in sections: sections.append(value)
    figures = _extract_figures(source, pages)
    combined_lower = text.lower()
    state_tokens = sorted(set(re.findall(r"\b(?:[A-Z][A-Za-z]{0,3}|[A-Za-z]{1,3}_[A-Za-z0-9]+)\s*\(t\)", text)))
    parameter_tokens = sorted(set(re.findall(r"\b(?:delta\d+|ks|gamma|mu|xi\d+|delta\d+|g|kT|k|J|alpha|beta|rho|phi\d+|kappa|d(?:b|mu)|Pi1N|beta\d+)\b", text, flags=re.I)))
    if facts:
        state_tokens = list(facts.get("state_variables", state_tokens))
        parameter_tokens = list(facts.get("parameter_tokens", parameter_tokens))
    equations = extract_pdf_equations(
        str(source), layouts=layouts, state_variables=state_tokens, parameters=parameter_tokens,
    )
    # Preserve the paper's operator family as metadata only.  The source text
    # and rendered source region remain authoritative; no formula is rebuilt
    # from these labels.
    if document_key == "paper_1":
        for candidate in equations:
            is_model_equation = candidate.page == 3 and candidate.bbox is not None and candidate.bbox[0] > 290 and 300 <= candidate.bbox[1] <= 420
            is_numerical_recurrence = candidate.page == 7 and candidate.bbox is not None and candidate.bbox[0] > 290 and candidate.bbox[1] < 660
            if is_model_equation or is_numerical_recurrence:
                candidate.parameters = sorted(set(candidate.parameters + ["alpha", "beta"]))
                if is_model_equation:
                    candidate.type = MathematicalType.FRACTIONAL_DIFFERENTIAL_EQUATION
                    candidate.initial_conditions = {
                        "S(0)": "S0 (symbolic; numerical value not supplied)",
                        "E(0)": "E0 (symbolic; numerical value not supplied)",
                        "T(0)": "T0 (symbolic; numerical value not supplied)",
                        "C(0)": "C0=0 conditional on J0=0; verify condition",
                    }
                candidate.derivative_type = "fractal_fractional"
                candidate.fractional_order = "alpha"
                candidate.fractal_dimension = "beta"
                candidate.kernel = "generalized_mittag_leffler"
                candidate.history_dependent = True
                candidate.solver_requirement = "Atangana-Baleanu Caputo fractal-fractional history recurrence; inspect Eq. (58) indexing before execution"
                candidate.notes.append("Paper-specific operator metadata is source-backed; the extracted symbols are not a verified transcription.")
                if is_numerical_recurrence:
                    candidate.type = MathematicalType.COUPLED_RECURRENCE
                    candidate.equation_numbers = ["56", "57", "58"]
                    candidate.notes.append("The rendered numerical-scheme page shows coupled S, E, T, C updates with history sums; the displayed recurrence indexing remains ambiguous.")
                elif is_model_equation:
                    candidate.equation_numbers = ["7"]
    figures = enrich_figure_facts(figures, document_key)
    known_parameter_values = facts.get("parameter_values", {})
    methods = [match.group(0).strip() for match in re.finditer(r"(?i)(?:numerical scheme|solver|discretization|Grünwald|Atangana|Caputo|NSFD|PINN)[^\.\n]{0,180}", text)]
    unresolved = []
    if not re.search(r"initial conditions?|S\s*\(0\)|F_N0", text, re.I): unresolved.append("Initial conditions were not detected by the parser.")
    if not re.search(r"step size|Δt|time step|h\s*=", text, re.I): unresolved.append("A numerical time-step specification was not detected.")
    unresolved.extend(facts.get("unresolved", []))
    initial_values = facts.get("initial_conditions", [])
    initial_status = facts.get("initial_conditions_status", EvidenceStatus.NEEDS_REVIEW)
    initial_page = facts.get("initial_conditions_page")
    initial_notes = facts.get("initial_conditions_notes", [])
    return PaperAnalysis(
        source_document=str(source), page_count=len(pages),
        title=_item(title_text or "", EvidenceStatus.PARTIAL if title_text else EvidenceStatus.NEEDS_REVIEW, 1),
        authors=_item(authors or "", EvidenceStatus.PARTIAL if authors else EvidenceStatus.NEEDS_REVIEW, 1),
        sections=sections,
        equations=equations,
        page_layout_summary=[{"page_number": page.page_number, "width": page.width, "height": page.height, "region_counts": page.counts()} for page in layouts],
        parameters=[{"name": token, "value": known_parameter_values.get(token), "status": EvidenceStatus.CONFIRMED.value if token in known_parameter_values else EvidenceStatus.PARTIAL.value, "source": "parameter table" if token in known_parameter_values else "text token"} for token in parameter_tokens],
        initial_conditions=_item(initial_values, initial_status, initial_page, initial_notes),
        state_variables=_item(state_tokens, EvidenceStatus.PARTIAL if state_tokens else EvidenceStatus.NEEDS_REVIEW),
        numerical_methods=_item(methods, EvidenceStatus.PARTIAL if methods else EvidenceStatus.NEEDS_REVIEW),
        fractional_orders=_item(sorted(set(re.findall(r"\b(?:alpha|α|sigma|σ)\s*=\s*[0-9.]+", text, flags=re.I))), EvidenceStatus.PARTIAL),
        fractal_dimensions=_item(sorted(set(re.findall(r"\b(?:beta|β)\s*=\s*[0-9.]+", text, flags=re.I))), EvidenceStatus.PARTIAL),
        time_ranges=_item(sorted(set(re.findall(r"(?:over|from)\s+\d+\s+(?:to|-)\s+\d+\s+(?:days?|years?)", text, flags=re.I))), EvidenceStatus.PARTIAL),
        sensitivity_methodology=_item([line.strip() for line in text.splitlines() if "sensitivity" in line.lower()][:10], EvidenceStatus.PARTIAL if "sensitivity" in combined_lower else EvidenceStatus.UNCERTAIN),
        noise_experiments=_item([line.strip() for line in text.splitlines() if "noise" in line.lower()][:10], EvidenceStatus.PARTIAL if "noise" in combined_lower else EvidenceStatus.UNCERTAIN),
        neural_network_methodology=_item([line.strip() for line in text.splitlines() if "PINN" in line or "neural network" in line.lower()][:15], EvidenceStatus.PARTIAL if "pinn" in combined_lower else EvidenceStatus.UNCERTAIN),
        control_variables=_item(sorted(set(re.findall(r"\bz\d+\s*\(?t?\)?", text))), EvidenceStatus.PARTIAL if re.search(r"\bz\d", text) else EvidenceStatus.UNCERTAIN),
        validation_methodology=_item([line.strip() for line in text.splitlines() if "validation" in line.lower() or "comparison" in line.lower()][:10], EvidenceStatus.PARTIAL),
        figures=figures,
        uncertainties=["Automated PDF extraction is layout-sensitive; equation text and captions require human review.", "Equation candidates are positioned source regions, not exact LaTeX or executable model expressions."],
        unresolved_information=unresolved + ([layout_warning] if layout_warning else []),
    )


def _extract_figures(source: Path, pages: list[str]) -> list[FigureManifest]:
    figures: list[FigureManifest] = []
    for page_number, page in enumerate(pages, 1):
        for match in re.finditer(r"(?is)Fig(?:ure)?\.?\s*(\d+)\.?\s*(.{10,500}?)(?=\n\s*Fig(?:ure)?\.?\s*\d+|\n\s*\d+\.|\Z)", page):
            figure_id, caption = match.group(1), " ".join(match.group(2).split())
            lowered = caption.lower()
            plot_type = "line" if "over time" in lowered or "dynamics" in lowered else "unknown"
            variables = [token for token in ("S", "E", "T", "C", "FN", "IN", "Aβ", "Tμ", "Mρ") if token.lower() in lowered]
            varying = {}
            for symbol, values in re.findall(r"([αβ])\s*=\s*([^,.;]+)", caption): varying[symbol] = [value.strip() for value in values.split("and")]
            equation_numbers = sorted(set(re.findall(r"(?:eq(?:uation)?\.?\s*)\((\d{1,3})\)", caption, flags=re.I)), key=int)
            figures.append(FigureManifest(
                figure_id=f"figure_{int(figure_id):02d}", source_document=str(source), source_page=page_number, caption=caption,
                plot_type=plot_type, variables=variables, varying_parameters=varying,
                required_equation_ids=[f"equation_{number.zfill(2)}" for number in equation_numbers],
                equation_link_status=EvidenceStatus.PARTIAL if equation_numbers else EvidenceStatus.NEEDS_REVIEW,
                required_parameters=sorted(varying),
                provenance={"caption": "SOURCE_PAPER", "equation_link": "SOURCE_PAPER" if equation_numbers else "UNKNOWN"},
                notes=[f"Caption extracted from page {page_number}; review axis values and curves."] if caption else [f"Figure reference on page {page_number} requires caption review."],
                confidence=EvidenceStatus.PARTIAL, reproducibility_status=EvidenceStatus.NEEDS_REVIEW,
            ))
    unique = {}
    for figure in figures: unique[figure.figure_id] = figure
    return list(unique.values())
