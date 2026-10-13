from research_plotter.ingestion.pdf_loader import (
    PDFLayoutRegion,
    PDFPageLayout,
    extract_pdf_layout,
    render_pdf_region,
)
from research_plotter.paper_analysis.equation_extractor import (
    EquationProvenance,
    EquationReviewStatus,
    MathematicalType,
    build_equation_spec_from_text,
    classify_mathematical_text,
    extract_pdf_equations,
)
from research_plotter.paper_analysis.paper_parser import parse_paper
from research_plotter.ui.paper_reproduction_page import _validate_paper1_inputs


def test_safe_text_equation_types_cover_functions_and_systems():
    explicit = build_equation_spec_from_text("y = x**2", accepted=True)
    ordinary = build_equation_spec_from_text("y = sin(x)", accepted=True)
    parameterized = build_equation_spec_from_text("y = A*exp(-k*x)*sin(w*x)")
    parametric = build_equation_spec_from_text("x(t)=cos(t)\ny(t)=sin(t)")
    ode = build_equation_spec_from_text("dx/dt = a*x-b*x*y\ndy/dt = c*x*y-d*y")

    assert explicit.type is MathematicalType.EXPLICIT_FUNCTION
    assert explicit.execution_supported and explicit.review_status is EquationReviewStatus.ACCEPTED
    assert explicit.execution_ready
    assert ordinary.type is MathematicalType.EXPLICIT_FUNCTION
    assert parameterized.type is MathematicalType.PARAMETERIZED_FUNCTION
    assert parameterized.parameters == ["A", "k", "w"]
    assert not parameterized.execution_ready
    assert parametric.type is MathematicalType.PARAMETRIC_EQUATIONS
    assert parametric.execution_supported
    assert ode.type is MathematicalType.ODE_SYSTEM
    assert ode.variables == ["x", "y"]
    assert ode.execution_supported


def test_fractional_and_recurrence_equations_are_not_forced_to_functions():
    assert classify_mathematical_text("D^alpha x = -k*x") is MathematicalType.FRACTIONAL_DIFFERENTIAL_EQUATION
    assert classify_mathematical_text("x[n+1] = x[n] + h*f(x[n])") is MathematicalType.DISCRETE_RECURRENCE
    assert classify_mathematical_text("x[n+1]=x[n]+a\ny[n+1]=y[n]+b") is MathematicalType.COUPLED_RECURRENCE

    spec = build_equation_spec_from_text("D^alpha x = -k*x", accepted=True)
    assert spec.review_status is EquationReviewStatus.ACCEPTED
    assert not spec.execution_supported


def test_malformed_pdf_equation_remains_reviewable_and_non_executable():
    layout = PDFPageLayout(
        page_number=1,
        width=595,
        height=792,
        regions=(PDFLayoutRegion(1, "p1-b1", (40, 100, 260, 145), "equation", "x[n+1] = ???"),),
    )
    candidates = extract_pdf_equations("example.pdf", layouts=[layout])

    assert len(candidates) == 1
    assert candidates[0].bbox == (40, 100, 260, 145)
    assert candidates[0].review_status is EquationReviewStatus.REVIEW_REQUIRED
    assert candidates[0].provenance is EquationProvenance.SOURCE_PAPER
    assert not candidates[0].execution_supported


def test_paper1_extracts_fractional_model_and_history_recurrence_and_links_figures(paper_1_pdf):
    analysis = parse_paper(paper_1_pdf)
    model_regions = [item for item in analysis.equations if "7" in item.equation_numbers]
    recurrence_regions = [item for item in analysis.equations if "58" in item.equation_numbers]

    assert model_regions
    assert model_regions[0].type is MathematicalType.FRACTIONAL_DIFFERENTIAL_EQUATION
    assert model_regions[0].derivative_type == "fractal_fractional"
    assert model_regions[0].fractional_order == "alpha"
    assert model_regions[0].fractal_dimension == "beta"
    assert model_regions[0].kernel == "generalized_mittag_leffler"
    assert recurrence_regions
    assert recurrence_regions[0].type is MathematicalType.COUPLED_RECURRENCE
    assert recurrence_regions[0].history_dependent is True
    assert recurrence_regions[0].review_status is EquationReviewStatus.REVIEW_REQUIRED
    assert recurrence_regions[0].confidence.value == "LOW"

    figure = next(item for item in analysis.figures if item.figure_id == "figure_02")
    assert figure.required_equation_ids == ["equation_07", "equation_58"]
    assert figure.equation_link_status.value == "PARTIAL"
    assert figure.reproducibility_status.value == "BLOCKED"
    assert "S(0)" in figure.required_initial_conditions
    assert any("step" in item.lower() for item in figure.missing_information)
    assert figure.provenance["equations"] == "SOURCE_PAPER"
    assert figure.provenance["initial_conditions"] == "UNKNOWN"
    assert model_regions[0].provenance is EquationProvenance.SOURCE_PAPER


def test_paper_layout_regions_retain_bbox_and_render_source_preview(paper_1_pdf):
    page = extract_pdf_layout(paper_1_pdf, page_numbers=[8])[0]
    assert page.page_number == 8
    assert page.counts().get("figure", 0) >= 1
    assert page.counts().get("figure_caption", 0) >= 1
    assert page.counts().get("table", 0) >= 1
    region = next(item for item in page.regions if item.content_type == "figure")
    assert 0 <= region.bbox[0] < region.bbox[2] <= page.width
    assert 0 <= region.bbox[1] < region.bbox[3] <= page.height

    png = render_pdf_region(paper_1_pdf, page.page_number, region.bbox)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_paper1_readiness_validation_keeps_missing_inputs_explicit():
    blocked = _validate_paper1_inputs(
        equations_reviewed=False, initial_text="", time_end_text="", step_text="",
        figure_id="figure_01", sweep_text="",
    )
    assert len(blocked) == 3

    complete_for_exploration = _validate_paper1_inputs(
        equations_reviewed=True, initial_text="S=1,E=0.8,T=0.7,C=0",
        time_end_text="90", step_text="0.1", figure_id="figure_01", sweep_text="",
    )
    assert complete_for_exploration == []
    missing_sweep = _validate_paper1_inputs(
        equations_reviewed=True, initial_text="S=1,E=0.8,T=0.7,C=0",
        time_end_text="90", step_text="0.1", figure_id="figure_11", sweep_text="",
    )
    assert any("paper does not list" in message for message in missing_sweep)

