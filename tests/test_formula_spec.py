from pathlib import Path

import numpy as np
import pytest

from research_plotter.models.formula_spec import (
    build_ode_model,
    evaluate_formula,
    parse_equation_text,
    parse_numeric_assignments,
    parse_safe_expression,
)
from research_plotter.solvers import ODESolver


@pytest.mark.parametrize(
    "text,expected",
    [
        ("y = x**2", np.array([0.0, 1.0, 4.0])),
        ("y = sin(x)", np.sin(np.array([0.0, 1.0, 2.0]))),
        ("y = exp(-x)", np.exp(-np.array([0.0, 1.0, 2.0]))),
    ],
)
def test_safe_explicit_formula_evaluation(text, expected):
    spec = parse_equation_text(text)
    actual = evaluate_formula(spec, np.array([0.0, 1.0, 2.0]))["y"]
    assert np.allclose(actual, expected)


def test_parameterized_formula_requires_user_values_and_evaluates():
    spec = parse_equation_text("y = A * exp(-k*x) * sin(w*x)")
    assert spec.missing_parameter_values == ["A", "k", "w"]
    with pytest.raises(ValueError, match="Supply numeric values"):
        evaluate_formula(spec, np.linspace(0, 1, 4))
    grid = np.linspace(0, 1, 5)
    actual = evaluate_formula(spec, grid, {"A": 2, "k": .1, "w": 5})["y"]
    assert np.allclose(actual, 2 * np.exp(-.1 * grid) * np.sin(5 * grid))


def test_parametric_equation_pair():
    spec = parse_equation_text("x(t) = cos(t)\ny(t) = sin(t)")
    assert spec.kind == "parametric"
    values = evaluate_formula(spec, np.array([0, np.pi / 2]))
    assert np.allclose(values["x"], [1, 0], atol=1e-12)
    assert np.allclose(values["y"], [0, 1], atol=1e-12)


def test_txt_style_ode_system_routes_through_model_spec_and_solver():
    source = """dx/dt = a*x - b*x*y
dy/dt = c*x*y - d*y
x(0) = 10
y(0) = 5
a = 0.1
b = 0.01
c = 0.02
d = 0.1
"""
    spec = parse_equation_text(source)
    assert spec.kind == "ode"
    assert spec.missing_initial_conditions == []
    model = build_ode_model(spec)
    result = ODESolver().solve(model, None, model.initial_conditions, np.linspace(0, 1, 11))
    assert result.states.shape == (11, 2)
    assert np.isfinite(result.states).all()


def test_txt_parameter_assignment_parser_is_numeric_only():
    assert parse_numeric_assignments("A=2, k=1e-2") == {"A": 2.0, "k": .01}
    with pytest.raises(ValueError, match="finite numeric literal"):
        parse_numeric_assignments("k=__import__('os')")


@pytest.mark.parametrize("source", [
    "__import__('os').system('whoami')",
    "x.__class__",
    "open('secret.txt')",
])
def test_formula_parser_rejects_python_execution_constructs(source):
    with pytest.raises(ValueError):
        parse_safe_expression(source)


def test_formula_text_from_txt_file_and_pdf_extraction_smoke(tmp_path: Path, paper_1_pdf: Path):
    from research_plotter.ingestion.pdf_loader import extract_pdf_pages
    from research_plotter.ui.formula_page import _extract_uploaded_text

    text_path = tmp_path / "formula.txt"
    text_path.write_text("y = sqrt(x) + pi", encoding="utf-8")
    spec = parse_equation_text(text_path.read_text(encoding="utf-8"))
    values = evaluate_formula(spec, np.array([0.0, 1.0]))["y"]
    assert np.allclose(values, [np.pi, 1 + np.pi])
    pages = extract_pdf_pages(paper_1_pdf)
    assert pages and any(page.strip() for page in pages)

    class UploadedText:
        name = "equation.txt"

        def getbuffer(self):
            return b"dy/dt = rate*y\ny(0)=1\nrate=0.2"

    extracted, no_pages, stored_path = _extract_uploaded_text(UploadedText(), tmp_path)
    assert no_pages is None
    assert stored_path.is_file()
    assert stored_path.is_relative_to(tmp_path.resolve())
    assert "dy/dt" in extracted
    uploaded_spec = parse_equation_text(extracted)
    assert uploaded_spec.kind == "ode"
    assert uploaded_spec.parameter_values == {"rate": .2}


def test_uploaded_equation_review_is_invalidated_when_text_changes():
    from research_plotter.ui.formula_page import _reviewed_text

    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}

        def checkbox(self, _label, *, key):
            return self.session_state.get(key, False)

    st = FakeStreamlit()
    key = "equation_review"
    assert not _reviewed_text(st, "Review equation", key, "dy/dt = a*y")
    st.session_state[key] = True
    assert _reviewed_text(st, "Review equation", key, "dy/dt = a*y")
    assert not _reviewed_text(st, "Review equation", key, "dy/dt = b*y")

