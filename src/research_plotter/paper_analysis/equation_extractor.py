"""Review-first mathematical classification and PDF equation candidates.

PDF equations are not converted to executable expressions here.  The extractor
retains source geometry and marks candidates for human review; execution still
uses the restricted parser in ``models.formula_spec`` after an explicit edit
and acceptance.
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Any, Iterable

from pydantic import BaseModel, Field

from research_plotter.ingestion.pdf_loader import PDFLayoutRegion, PDFPageLayout, extract_pdf_layout


class MathematicalType(str, Enum):
    EXPLICIT_FUNCTION = "EXPLICIT_FUNCTION"
    PARAMETERIZED_FUNCTION = "PARAMETERIZED_FUNCTION"
    PARAMETRIC_EQUATIONS = "PARAMETRIC_EQUATIONS"
    ODE_SYSTEM = "ODE_SYSTEM"
    FRACTIONAL_DIFFERENTIAL_EQUATION = "FRACTIONAL_DIFFERENTIAL_EQUATION"
    DISCRETE_RECURRENCE = "DISCRETE_RECURRENCE"
    COUPLED_RECURRENCE = "COUPLED_RECURRENCE"
    ALGEBRAIC_EQUATION = "ALGEBRAIC_EQUATION"
    UNKNOWN = "UNKNOWN"


class EquationReviewStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNRESOLVED = "UNRESOLVED"


class EquationConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class EquationProvenance(str, Enum):
    SOURCE_PAPER = "SOURCE_PAPER"
    USER_PROVIDED = "USER_PROVIDED"
    ASSUMED = "ASSUMED"
    UNKNOWN = "UNKNOWN"


class EquationSpec(BaseModel):
    """Safe intermediate representation of one equation or equation region."""

    equation_id: str
    equation_numbers: list[str] = Field(default_factory=list)
    source_document: str = ""
    page: int | None = None
    region_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    type: MathematicalType = MathematicalType.UNKNOWN
    source_text: str = ""
    lhs: str | None = None
    rhs: str | None = None
    editable_representation: str = ""
    variables: list[str] = Field(default_factory=list)
    parameters: list[str] = Field(default_factory=list)
    independent_variable: str | None = None
    derivative_type: str = "unknown"
    fractional_order: str | None = None
    fractal_dimension: str | None = None
    kernel: str | None = None
    history_dependent: bool | None = None
    solver_requirement: str | None = None
    initial_conditions: dict[str, str] = Field(default_factory=dict)
    confidence: EquationConfidence = EquationConfidence.LOW
    provenance: EquationProvenance = EquationProvenance.UNKNOWN
    review_status: EquationReviewStatus = EquationReviewStatus.REVIEW_REQUIRED
    execution_supported: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def execution_ready(self) -> bool:
        """True only after a supported structure has been explicitly accepted."""
        return self.execution_supported and self.review_status is EquationReviewStatus.ACCEPTED


_ASSIGNMENT = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*(?:\s*\([^)]*\))?)\s*=\s*(.+?)\s*$")
_PARAMETRIC = re.compile(r"(?m)^\s*[A-Za-z]\w*\s*\(\s*[A-Za-z]\w*\s*\)\s*=")
_DERIVATIVE = re.compile(r"(?i)(?:\bd\s*[A-Za-z]\w*\s*/\s*d\s*[A-Za-z]\w*\b|\\frac\s*\{\s*d)")
_FRACTIONAL = re.compile(
    r"(?i)(?:fractional|fractal.?fractional|atangana|mittag.?leffler|caputo|grunwald|"
    r"\\mathcal\s*\{?D|\bD\s*\^\s*\{?\s*(?:alpha|α)|"
    r"D\s*[_\{].{0,24}(?:alpha|α)|\bD\s*[αα])"
)
_RECURRENCE = re.compile(
    r"(?i)(?:\[\s*n\s*\+\s*1\s*\]|\{\s*n\s*\+\s*1\s*\}|"
    r"\b[A-Za-z][A-Za-z0-9]*\s*(?:\^\s*\{?\s*\(?\s*n\s*\+\s*1|_?n\s*\+\s*1))"
)
_GREEK_PARAM = re.compile(r"(?<![A-Za-z])(?:α|β|δ\d*|γ|μ|ξ\d*|κ|ρ|phi\d*)\b", re.I)
_PAPER_PARAM = re.compile(r"\b(?:delta\d+|ks|gamma|mu|xi\d+|kT|J|alpha|beta|rho|phi\d+|kappa|d(?:b|mu))\b", re.I)


def classify_mathematical_text(text: str, *, context: str = "") -> MathematicalType:
    """Classify common equation forms without executing or solving them."""
    value = text.strip()
    if not value:
        return MathematicalType.UNKNOWN
    combined = f"{value}\n{context}"
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    recurrence = bool(_RECURRENCE.search(value))
    assignment_count = len(re.findall(r"=", value))
    if recurrence:
        return MathematicalType.COUPLED_RECURRENCE if assignment_count > 1 or len(lines) > 1 else MathematicalType.DISCRETE_RECURRENCE
    if _FRACTIONAL.search(combined) and ("=" in value or _DERIVATIVE.search(value)):
        return MathematicalType.FRACTIONAL_DIFFERENTIAL_EQUATION
    if value.count("?") + value.count("�") > max(3, len(value) // 12):
        return MathematicalType.UNKNOWN
    if _DERIVATIVE.search(value) and "=" in value:
        return MathematicalType.ODE_SYSTEM
    if _PARAMETRIC.search(value):
        parametric_names = re.findall(r"(?m)^\s*[A-Za-z]\w*\s*\(\s*([A-Za-z]\w*)\s*\)\s*=", value)
        if len(parametric_names) >= 2 and len(set(parametric_names)) == 1:
            return MathematicalType.PARAMETRIC_EQUATIONS
    if "=" in value:
        assignments = [_ASSIGNMENT.match(line) for line in lines]
        matched = [match for match in assignments if match]
        if len(matched) == 1 and len(lines) == 1:
            right = matched[0].group(2)
            # Bare symbols besides a short explicit-function variable are a
            # conservative parameter cue.  The restricted formula parser does
            # the authoritative check before any expression can run.
            identifiers = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", right))
            reserved = {"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "pi", "E"}
            if identifiers - reserved - {"x", "t"}:
                return MathematicalType.PARAMETERIZED_FUNCTION
            if matched[0].group(1).strip() in {"y", "f(x)", "f(t)"}:
                return MathematicalType.EXPLICIT_FUNCTION
            return MathematicalType.ALGEBRAIC_EQUATION
        return MathematicalType.ALGEBRAIC_EQUATION
    return MathematicalType.UNKNOWN


def build_equation_spec_from_text(
    text: str,
    *,
    source_document: str = "",
    provenance: EquationProvenance = EquationProvenance.USER_PROVIDED,
    accepted: bool = False,
    independent_variable: str = "x",
    context: str = "",
) -> EquationSpec:
    """Create a safe, editable EquationSpec from manual or TXT equation text.

    The restricted FormulaSpec parser is used only to validate supported
    explicit, parametric, and ordinary ODE forms. Fractional and recurrence
    content remains non-executable even if fragments resemble assignments.
    """
    from research_plotter.models.formula_spec import parse_equation_text

    mathematical_type = classify_mathematical_text(text, context=context)
    formula = None
    parse_note = ""
    try:
        formula = parse_equation_text(text, independent_variable=independent_variable)
    except ValueError as error:
        parse_note = str(error)
    compatible = mathematical_type in {
        MathematicalType.EXPLICIT_FUNCTION,
        MathematicalType.PARAMETERIZED_FUNCTION,
        MathematicalType.PARAMETRIC_EQUATIONS,
        MathematicalType.ODE_SYSTEM,
    }
    execution_supported = formula is not None and compatible
    variables: list[str] = []
    parameter_names: list[str] = []
    initial_conditions: dict[str, str] = {}
    independent = independent_variable
    lhs = rhs = None
    if formula is not None and compatible:
        variables = list(formula.state_variables or formula.dependent_variables)
        parameter_names = list(formula.parameter_names)
        initial_conditions = {name: str(value) for name, value in (formula.initial_conditions or {}).items()}
        independent = formula.independent_variable
        if len(formula.expressions) == 1:
            lhs, rhs = next(iter(formula.expressions.items()))
    derivative_type = "ordinary" if mathematical_type == MathematicalType.ODE_SYSTEM else "unknown"
    if mathematical_type == MathematicalType.FRACTIONAL_DIFFERENTIAL_EQUATION:
        derivative_type = "unknown"
    notes = ["Editable representation supplied as text; no text was evaluated as Python."]
    if parse_note:
        notes.append(f"Safe formula parser note: {parse_note}")
    return EquationSpec(
        equation_id="user-equation-01",
        source_document=source_document,
        type=mathematical_type,
        source_text=text,
        editable_representation=text,
        lhs=lhs,
        rhs=rhs,
        variables=variables,
        parameters=parameter_names,
        independent_variable=independent,
        derivative_type=derivative_type,
        confidence=EquationConfidence.HIGH if execution_supported else EquationConfidence.MEDIUM if mathematical_type != MathematicalType.UNKNOWN else EquationConfidence.LOW,
        provenance=provenance,
        review_status=EquationReviewStatus.ACCEPTED if accepted else EquationReviewStatus.REVIEW_REQUIRED,
        execution_supported=execution_supported,
        initial_conditions=initial_conditions,
        notes=notes,
    )


def _math_block(region: PDFLayoutRegion) -> bool:
    text = region.text.strip()
    if not text:
        return False
    # Layout classification already narrows the search.  A second signal
    # avoids including prose blocks classified by an incidental comparison.
    if region.content_type != "equation" or not ("=" in text or bool(_FRACTIONAL.search(text))):
        return False
    if len(text) > 220 and not re.search(r"[∫∑∏Γ∂]|\\(?:sum|int|frac|partial)", text):
        return False
    return True


def _column(bbox: tuple[float, float, float, float], page_width: float) -> int:
    return 0 if (bbox[0] + bbox[2]) / 2 < page_width / 2 else 1


def _region_groups(layout: PDFPageLayout) -> list[list[PDFLayoutRegion]]:
    candidates = [region for region in layout.regions if _math_block(region)]
    groups: list[list[PDFLayoutRegion]] = []
    for column in (0, 1):
        column_regions = sorted((region for region in candidates if _column(region.bbox, layout.width) == column), key=lambda item: (item.bbox[1], item.bbox[0]))
        column_groups: list[list[PDFLayoutRegion]] = []
        for region in column_regions:
            if column_groups:
                previous = column_groups[-1][-1]
                vertical_gap = region.bbox[1] - previous.bbox[3]
                similar_left = abs(region.bbox[0] - previous.bbox[0]) < layout.width * 0.14
                if similar_left and -2 <= vertical_gap <= 45:
                    column_groups[-1].append(region)
                    continue
            column_groups.append([region])
        groups.extend(column_groups)
    return sorted(groups, key=lambda group: (group[0].bbox[1], group[0].bbox[0]))


def extract_pdf_equations(
    path: str,
    *,
    layouts: Iterable[PDFPageLayout] | None = None,
    state_variables: Iterable[str] = (),
    parameters: Iterable[str] = (),
) -> list[EquationSpec]:
    """Return equation-region candidates from PDF geometry, never executable code."""
    pages = list(layouts) if layouts is not None else extract_pdf_layout(path)
    output: list[EquationSpec] = []
    for layout in pages:
        for group_index, group in enumerate(_region_groups(layout), start=1):
            group_text = "\n".join(region.text for region in group).strip()
            if not group_text:
                continue
            bbox = (
                min(region.bbox[0] for region in group),
                min(region.bbox[1] for region in group),
                max(region.bbox[2] for region in group),
                max(region.bbox[3] for region in group),
            )
            local_context = " ".join(
                region.text for region in layout.regions
                if region.content_type in {"prose", "heading", "equation"}
                and _column(region.bbox, layout.width) == _column(bbox, layout.width)
                and -45 <= region.bbox[3] - bbox[1] <= 100
            )
            equation_numbers: list[str] = []
            math_type = classify_mathematical_text(group_text, context=local_context)
            if math_type == MathematicalType.UNKNOWN:
                # PDF fonts often lose Greek operators in text extraction.  A
                # nearby kernel/derivative description is useful classification
                # evidence, but not sufficient to reconstruct the equation.
                if _FRACTIONAL.search(local_context) and "=" in group_text and group_text.count("?") < max(8, len(group_text) // 3):
                    math_type = MathematicalType.FRACTIONAL_DIFFERENTIAL_EQUATION
            if math_type == MathematicalType.UNKNOWN and "=" not in group_text:
                continue
            derivative_type = "unknown"
            kernel = None
            history_dependent: bool | None = None
            solver_requirement = None
            fractional_order = None
            fractal_dimension = None
            context_lower = local_context.lower()
            if _FRACTIONAL.search(local_context):
                derivative_type = "fractal_fractional"
                fractional_order = "alpha" if re.search(r"\b(?:alpha|α)\b", local_context, re.I) else None
                fractal_dimension = "beta" if re.search(r"\b(?:beta|β)\b", local_context, re.I) else None
                kernel = "generalized_mittag_leffler" if "mittag" in context_lower else None
                history_dependent = True
                solver_requirement = "fractional_history_method; verify the paper's recurrence and indexing before execution"
            confidence = EquationConfidence.LOW
            notes = [
                "Positioned text block candidate; compare with the rendered source region.",
                "PDF text extraction may lose superscripts, subscripts, and Greek glyphs; this is not a verified transcription.",
            ]
            if "�" not in group_text and len(group) == 1 and math_type in {MathematicalType.EXPLICIT_FUNCTION, MathematicalType.PARAMETERIZED_FUNCTION, MathematicalType.ODE_SYSTEM}:
                confidence = EquationConfidence.MEDIUM
                notes[1] = "Text is structurally parseable, but the rendered source still requires human confirmation."
            output.append(EquationSpec(
                equation_id=f"p{layout.page_number:03d}-region-{group_index:02d}",
                equation_numbers=equation_numbers,
                source_document=str(path),
                page=layout.page_number,
                region_id=group[0].region_id,
                bbox=bbox,
                type=math_type,
                source_text=group_text,
                editable_representation=group_text,
                variables=list(state_variables),
                parameters=list(parameters),
                derivative_type=derivative_type,
                fractional_order=fractional_order,
                fractal_dimension=fractal_dimension,
                kernel=kernel,
                history_dependent=history_dependent,
                solver_requirement=solver_requirement,
                confidence=confidence,
                provenance=EquationProvenance.SOURCE_PAPER,
                review_status=EquationReviewStatus.REVIEW_REQUIRED,
                execution_supported=False,
            ))
    return output


def extract_equations(analysis: Any) -> list[dict[str, Any]]:
    """Compatibility serializer for structured paper equation candidates."""
    return [item.model_dump(mode="json") for item in analysis.equations]
