"""Equation / Formula Plotter view using the safe formula and model APIs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from research_plotter.ingestion.pdf_loader import extract_pdf_layout, extract_pdf_pages, render_pdf_region
from research_plotter.models.formula_spec import (
    build_ode_model,
    evaluate_formula,
    parse_equation_text,
    parse_numeric_assignments,
    parse_safe_expression,
)
from research_plotter.paper_analysis.equation_extractor import (
    EquationProvenance,
    MathematicalType,
    build_equation_spec_from_text,
    classify_mathematical_text,
    extract_pdf_equations,
)


def _extract_uploaded_text(upload: Any, output_root: Path) -> tuple[str, list[str] | None, Path]:
    suffix = Path(upload.name).suffix.lower()
    data = bytes(upload.getbuffer())
    if suffix not in {".txt", ".pdf"}:
        raise ValueError("Formula input supports TXT or PDF files")
    from research_plotter.utils.paths import persist_uploaded_file

    destination = persist_uploaded_file(upload, "formulas", output_root=output_root)
    if suffix == ".txt":
        try:
            return data.decode("utf-8-sig"), None, destination
        except UnicodeDecodeError as error:
            raise ValueError("TXT input must be UTF-8 encoded; save the file as UTF-8 and upload again") from error
    return "", extract_pdf_pages(destination), destination


def _format_assignments(values: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:g}" for key, value in values.items())


def _latex_preview(spec) -> list[str]:
    import sympy as sp

    output = []
    for name, raw in spec.expressions.items():
        expression = parse_safe_expression(raw)
        if spec.kind == "ode":
            output.append(rf"\frac{{d {name}}}{{d {spec.independent_variable}}} = {sp.latex(expression)}")
        elif spec.kind == "parametric":
            output.append(rf"{name}({spec.independent_variable}) = {sp.latex(expression)}")
        else:
            output.append(rf"{name} = {sp.latex(expression)}")
    return output


def _navigate_to_reproduction() -> None:
    # Streamlit invokes callbacks before reconstructing widgets on rerun.
    import streamlit as st

    st.session_state["main_navigation"] = "Figure Reproduction"


def _reviewed_text(st: Any, label: str, key: str, text: str) -> bool:
    """Require review again whenever an editable uploaded equation changes."""
    signature = hashlib.sha256(text.encode("utf-8")).hexdigest()
    signature_key = f"{key}_content_signature"
    if st.session_state.get(signature_key) != signature:
        # Set the checkbox before constructing it on this rerun. This prevents
        # an earlier acceptance from carrying over to edited equation text.
        st.session_state[key] = False
        st.session_state[signature_key] = signature
    return bool(st.checkbox(label, key=key))


def render_formula_page(
    st: Any,
    *,
    output_root: Path,
    new_run_dir: Callable[[str], tuple[str, Path]],
    save_figure: Callable[..., list[Path]],
    record_run: Callable[..., Path],
) -> None:
    st.title("Equation / Formula Plotter")
    st.write("Plot a formula or review mathematical regions extracted from a document. Paper equations are structured as candidates; they are never treated as executable curves without explicit review and a supported model type.")
    mode = st.radio("Equation source", ["Manual formula", "Upload TXT", "Upload PDF"], horizontal=True, key="formula_input_mode")
    source_text = ""
    source_document = ""
    source_provenance = "USER_PROVIDED"
    reviewed_pdf = True
    reviewed_text = True
    candidate = None
    candidate_context = ""

    if "formula_manual_text" not in st.session_state:
        # Keep a value from the older shared widget key when upgrading an
        # existing Streamlit session, but isolate the input modes going
        # forward so manual text cannot replace a PDF candidate silently.
        st.session_state["formula_manual_text"] = st.session_state.get(
            "formula_text_input", "y = exp(-0.1*x) * cos(5*x)"
        )
    if mode == "Manual formula":
        st.caption("Example: `y = exp(-0.1*x) * cos(5*x)` on x = 0 to 20.")
        source_text = st.text_area("Equation text", key="formula_manual_text", height=160)
    else:
        upload = st.file_uploader("TXT or PDF containing equations", type=["txt", "pdf"], key="formula_file_upload")
        if upload is None:
            st.info("Upload a .txt or .pdf file. You can edit the extracted text before analysis or execution.")
            return
        try:
            extracted_text, pages, uploaded_path = _extract_uploaded_text(upload, output_root)
        except Exception as error:
            st.error(f"Could not read the formula input.\n\nReason: {error}\n\nTry saving the file as UTF-8 text or uploading a text-based PDF.")
            return
        digest = hashlib.sha256(bytes(upload.getbuffer())).hexdigest()
        source_document = upload.name
        if mode == "Upload TXT":
            if Path(upload.name).suffix.lower() != ".txt":
                st.warning("Choose a .txt file for TXT mode.")
                return
            text_key = f"formula_txt_text_{digest[:16]}"
            if text_key not in st.session_state:
                st.session_state[text_key] = extracted_text
            source_text = st.text_area("Review and edit extracted text", key=text_key, height=220)
            st.caption("Plain text is treated as a candidate formula only; no embedded code is executed.")
            reviewed_text = _reviewed_text(
                st, "I reviewed the structured TXT equations",
                f"formula_txt_reviewed_{digest[:16]}", source_text,
            )
            source_provenance = "USER_PROVIDED"
        else:
            if Path(upload.name).suffix.lower() != ".pdf":
                st.warning("Choose a .pdf file for PDF mode.")
                return
            if not pages:
                st.warning("No PDF pages could be extracted.")
                return
            pdf_path = uploaded_path
            source_document = str(pdf_path)
            try:
                layouts = extract_pdf_layout(pdf_path)
            except Exception as error:
                st.error(f"Could not inspect PDF page layout.\n\nReason: {error}")
                return
            page_index = st.selectbox("PDF page", list(range(len(pages))), format_func=lambda index: f"Page {index + 1}", key="formula_pdf_page")
            page_layout = layouts[page_index]
            page_candidates = extract_pdf_equations(str(pdf_path), layouts=[page_layout])
            if not page_candidates:
                st.info("No equation-like layout regions were detected on this page. This tool will not pass whole-page prose to the formula parser.")
                st.image(render_pdf_region(pdf_path, page_index + 1), caption=f"Source page {page_index + 1}", use_container_width=True)
                st.caption("For a full research paper, use Paper Analyzer. If this page contains mathematics, select another page or enter a corrected equation manually.")
                if st.button("Open in Figure Reproduction", key="formula_open_reproduction_empty", on_click=_navigate_to_reproduction):
                    pass
                return
            candidate = st.selectbox(
                "Candidate equation region",
                page_candidates,
                format_func=lambda item: f"{item.equation_id} · {item.type.value} · {item.confidence.value}",
                key=f"formula_pdf_candidate_{page_index}",
            )
            candidate_text_key = f"formula_pdf_text_{digest[:12]}_{page_index}_{candidate.equation_id}"
            if candidate_text_key not in st.session_state:
                st.session_state[candidate_text_key] = candidate.editable_representation
            candidate_context = "fractal-fractional Atangana-Baleanu generalized Mittag-Leffler history operator" if candidate.derivative_type == "fractal_fractional" else ""
            st.image(render_pdf_region(pdf_path, candidate.page or page_index + 1, candidate.bbox), caption=f"Source PDF page {candidate.page or page_index + 1} · candidate region", use_container_width=True)
            source_text = st.text_area("Editable candidate representation", key=candidate_text_key, height=220)
            detected_pdf_type = candidate.type if source_text == candidate.source_text else classify_mathematical_text(source_text, context=candidate_context)
            status_cols = st.columns(4)
            status_cols[0].metric("Detected type", detected_pdf_type.value)
            status_cols[1].metric("Confidence", candidate.confidence.value)
            status_cols[2].metric("Review status", candidate.review_status.value)
            status_cols[3].metric("Equation number(s)", ", ".join(candidate.equation_numbers) or "unresolved")
            st.write("**Variables:**", candidate.variables or "Needs review from the rendered source")
            st.write("**Parameters:**", candidate.parameters or "Needs review from the rendered source")
            st.write("**Numerical method:**", candidate.solver_requirement or "Not detected from this region")
            st.warning("PDF equation text can lose Greek letters, subscripts, superscripts, and fraction layout. Compare the editable text with the image; confidence is not a correctness guarantee.")
            reviewed_pdf = _reviewed_text(
                st, "I reviewed and accept this candidate representation",
                f"formula_pdf_reviewed_{digest[:12]}_{page_index}_{candidate.equation_id}", source_text,
            )
            source_provenance = "USER_PROVIDED" if source_text != candidate.source_text else "SOURCE_PAPER"
            st.caption("For full paper-level model, parameter, and figure links, use Paper Analyzer.")
            st.button("Open in Figure Reproduction", key="formula_open_reproduction", on_click=_navigate_to_reproduction)

    variable_cols = st.columns(3)
    independent_variable = variable_cols[0].text_input("Independent variable", value="x", key="formula_independent")
    start = variable_cols[1].number_input("Domain start", value=0.0, key="formula_domain_start")
    end = variable_cols[2].number_input("Domain end", value=20.0, key="formula_domain_end")
    point_cols = st.columns(2)
    points = int(point_cols[0].number_input("Number of points", min_value=2, max_value=100000, value=1000, step=100, key="formula_points"))
    point_cols[1].caption("The domain and sampling grid are explicit controls for the current exploratory plot.")

    if st.button("Analyze Equation", key="analyze_formula"):
        try:
            detected_type = classify_mathematical_text(source_text, context=candidate_context)
            if detected_type in {MathematicalType.DISCRETE_RECURRENCE, MathematicalType.COUPLED_RECURRENCE, MathematicalType.FRACTIONAL_DIFFERENTIAL_EQUATION, MathematicalType.ALGEBRAIC_EQUATION, MathematicalType.UNKNOWN}:
                equation_spec = build_equation_spec_from_text(
                    source_text, source_document=source_document,
                    provenance=EquationProvenance.SOURCE_PAPER if mode == "Upload PDF" and source_text == (candidate.source_text if candidate else "") else EquationProvenance.USER_PROVIDED,
                    accepted=False, independent_variable=independent_variable, context=candidate_context,
                )
                st.session_state["formula_analysis"] = {
                    "kind": detected_type.value,
                    "mathematical_type": detected_type.value,
                    "independent_variable": independent_variable,
                    "expressions": {}, "dependent_variables": [], "state_variables": [],
                    "parameters": [], "parameter_values": {}, "initial_conditions": {},
                    "missing_parameters": [], "missing_initial_conditions": [],
                    "confidence": candidate.confidence.value if candidate is not None else "MEDIUM",
                    "review_status": "REVIEW_REQUIRED",
                    "execution_supported": equation_spec.execution_supported,
                    "equation_spec": equation_spec.model_dump(mode="json"),
                }
            else:
                spec = parse_equation_text(source_text, independent_variable=independent_variable)
                equation_spec = build_equation_spec_from_text(
                    source_text,
                    source_document=source_document,
                    provenance=EquationProvenance.SOURCE_PAPER if mode == "Upload PDF" and source_text == (candidate.source_text if candidate else "") else EquationProvenance.USER_PROVIDED,
                    accepted=(mode == "Manual formula"),
                    independent_variable=independent_variable,
                    context=candidate_context,
                )
                structured_type = {
                    "explicit": MathematicalType.PARAMETERIZED_FUNCTION if spec.parameter_names else MathematicalType.EXPLICIT_FUNCTION,
                    "parametric": MathematicalType.PARAMETRIC_EQUATIONS,
                    "ode": MathematicalType.ODE_SYSTEM,
                }[spec.kind]
                st.session_state["formula_analysis"] = {
                    "kind": spec.kind,
                    "mathematical_type": structured_type.value,
                    "independent_variable": spec.independent_variable,
                    "expressions": dict(spec.expressions),
                    "dependent_variables": list(spec.dependent_variables),
                    "state_variables": list(spec.state_variables),
                    "parameters": list(spec.parameter_names),
                    "parameter_values": dict(spec.parameter_values),
                    "initial_conditions": dict(spec.initial_conditions or {}),
                    "missing_parameters": spec.missing_parameter_values,
                    "missing_initial_conditions": spec.missing_initial_conditions,
                    "confidence": "HIGH" if mode == "Manual formula" else "MEDIUM",
                    "review_status": equation_spec.review_status.value,
                    "execution_supported": equation_spec.execution_supported,
                    "equation_spec": equation_spec.model_dump(mode="json"),
                }
            st.session_state["formula_analysis"]["source_signature"] = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        except Exception as error:
            st.session_state.pop("formula_analysis", None)
            st.error(f"Equation analysis failed.\n\nReason: {error}\n\nEdit the equation syntax and try again.")
    analysis = st.session_state.get("formula_analysis")
    if analysis and analysis.get("source_signature") != hashlib.sha256(source_text.encode("utf-8")).hexdigest():
        analysis = None
    if analysis:
        st.subheader("Detected mathematical structure")
        status_columns = st.columns(4)
        status_columns[0].metric("Type", analysis["kind"].upper())
        status_columns[1].metric("Independent variable", analysis["independent_variable"])
        status_columns[2].metric("Mathematical type", analysis.get("mathematical_type", analysis["kind"].upper()))
        status_columns[3].metric("State variables", len(analysis["state_variables"]) or "—")
        st.caption(f"Confidence: {analysis.get('confidence', 'MEDIUM')} · Review status: {analysis.get('review_status', 'REVIEW_REQUIRED')} · Execution supported: {analysis.get('execution_supported', True)}")
        st.write("**Detected parameters:**", analysis["parameters"] or "None")
        st.write("**Detected initial conditions:**", analysis["initial_conditions"] or "None supplied")
        try:
            if not analysis.get("execution_supported", True):
                raise ValueError("This equation type is reviewable but is not an executable explicit/parametric/ODE formula.")
            for expression in _latex_preview(parse_equation_text(source_text, independent_variable=independent_variable)):
                st.latex(expression)
        except Exception as error:
            if analysis.get("execution_supported", True):
                st.warning(f"The current text differs from the last analyzed version: {error}")
            else:
                st.code(source_text, language="text")
                st.info("Keep this structure in review or open Figure Reproduction. Fractional operators and recurrence regions are not converted to ordinary functions.")

    try:
        spec = parse_equation_text(source_text, independent_variable=independent_variable)
    except Exception as error:
        if source_text.strip() and mode in {"Upload PDF", "Upload TXT"}:
            st.warning(f"This input is not a supported explicit formula, parametric curve, or ODE system: {error}")
            if mode == "Upload PDF":
                st.caption("Research-paper equations should be reviewed in Paper Analyzer / Figure Reproduction; no extracted text has been executed.")
        return

    default_parameter_values = _format_assignments(dict(spec.parameter_values))
    source_signature = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if st.session_state.get("formula_parameter_source") != source_signature:
        st.session_state["formula_parameter_values"] = default_parameter_values
        st.session_state["formula_parameter_source"] = source_signature
    parameter_text = st.text_input("Parameter values (name=value; leave blank rather than guessing)", key="formula_parameter_values") if spec.parameter_names else ""
    initial_text_default = _format_assignments(dict(spec.initial_conditions or {}))
    if st.session_state.get("formula_initial_source") != source_signature:
        st.session_state["formula_initial_values"] = initial_text_default
        st.session_state["formula_initial_source"] = source_signature
    initial_text = st.text_input("Initial conditions (for ODE systems, e.g. x=10, y=5)", key="formula_initial_values") if spec.kind == "ode" else ""

    solver_name = "ode"
    fractional_order = None
    if spec.kind == "ode":
        solver_name = st.selectbox("ODE solver", ["ode", "explicit_euler", "grunwald_letnikov"], key="formula_solver")
        if solver_name == "grunwald_letnikov":
            fractional_order = st.number_input("Fractional order", min_value=0.05, max_value=1.0, value=0.9, step=0.05, key="formula_fractional_order")
            st.warning("The selected fractional solver interprets the supplied RHS using its fractional derivative formulation. Confirm that this matches your model before drawing scientific conclusions.")

    plot_clicked = st.button("Plot / Run Equation", type="primary", key="plot_formula")
    if not plot_clicked:
        return
    accepted_input = mode == "Manual formula" or (mode == "Upload TXT" and reviewed_text) or (mode == "Upload PDF" and reviewed_pdf)
    equation_spec = build_equation_spec_from_text(
        source_text,
        source_document=source_document,
        provenance=EquationProvenance.USER_PROVIDED if mode == "Manual formula" or (mode == "Upload PDF" and source_text != (candidate.source_text if candidate else "")) else EquationProvenance.SOURCE_PAPER if mode == "Upload PDF" else EquationProvenance.USER_PROVIDED,
        accepted=accepted_input,
        independent_variable=independent_variable,
        context=candidate_context,
    )
    if not accepted_input:
        st.error("Review and accept the selected equation text before execution.")
        return
    if not equation_spec.execution_ready:
        st.error(f"This structure is not accepted and/or is unsupported by the Formula Plotter ({equation_spec.type.value}). Use Model Simulation with a reviewed ModelSpec, or Figure Reproduction for a paper adapter.")
        return
    if mode == "Upload PDF" and not reviewed_pdf:
        st.error("Review and accept the selected PDF equation candidate before execution.")
        return
    if mode == "Upload TXT" and not reviewed_text:
        st.error("Review and accept the structured TXT equations before execution.")
        return
    if not np.isfinite([start, end]).all() or end <= start:
        st.error("Domain end must be finite and greater than domain start.")
        return
    run_id, run_dir = new_run_dir("formula")
    started = time.perf_counter()
    try:
        provided_parameters = parse_numeric_assignments(parameter_text, field="parameters") if parameter_text.strip() else {}
        parameter_values = {**spec.parameter_values, **provided_parameters}
        grid = np.linspace(float(start), float(end), points)
        if spec.kind == "ode":
            initial_values = parse_numeric_assignments(initial_text, field="initial conditions") if initial_text.strip() else {}
            model = build_ode_model(spec, parameter_values, initial_values, fractional_order=fractional_order)
            from research_plotter.cli import _solver
            from research_plotter.utils.numerics import uniform_time_grid

            step = (float(end) - float(start)) / (points - 1)
            time_grid = uniform_time_grid(float(start), float(end), step)
            with st.spinner("Solving the reviewed model equations…"):
                result = _solver(solver_name).solve(model, None, model.initial_conditions, time_grid, {"fractional_order": fractional_order} if fractional_order is not None else {})
            frame = result.to_frame()
            from research_plotter.plotting.line import multi_line_plot

            figure = multi_line_plot(result.time, frame.drop(columns="time"), xlabel=spec.independent_variable, ylabel="state", title="Formula ODE simulation")
            data_path = run_dir / "simulation.csv"
            frame.to_csv(data_path, index=False)
            run_details = {
                "model": model.name, "solver": solver_name, "parameters": result.parameters,
                "initial_conditions": result.initial_conditions, "time_grid": result.to_metadata()["time_grid"],
                "warnings": result.warnings, "status": "EXPLORATORY",
            }
            st.write({"Solver": result.solver_name, "Runtime (s)": result.runtime_seconds, "Parameters": result.parameters, "Initial conditions": result.initial_conditions, "Diagnostics": result.diagnostics})
        else:
            values = evaluate_formula(spec, grid, provided_parameters)
            frame = pd.DataFrame({spec.independent_variable: grid, **values})
            data_path = run_dir / "formula_data.csv"
            frame.to_csv(data_path, index=False)
            from research_plotter.plotting.line import line_plot

            if spec.kind == "explicit":
                output_name = spec.dependent_variables[0]
                figure = line_plot(grid, values[output_name], xlabel=spec.independent_variable, ylabel=output_name, title=f"{output_name}({spec.independent_variable})")
            else:
                x_name, y_name = spec.dependent_variables[:2]
                figure = line_plot(values[x_name], values[y_name], xlabel=x_name, ylabel=y_name, title="Parametric curve")
            run_details = {"model": "formula", "solver": None, "parameters": parameter_values, "initial_conditions": {}, "time_grid": {"start": float(start), "end": float(end), "count": points}, "status": "EXPLORATORY"}
        st.subheader("Generated figure")
        st.pyplot(figure)
        figure_paths = save_figure(figure, run_dir / "figures", "equation_plot", {
            "workflow": "formula", "plot_type": spec.kind, "source_text": source_text,
            "equation_spec": equation_spec.model_dump(mode="json"),
            "equation_execution_ready": equation_spec.execution_ready,
            "equations": dict(spec.expressions), "independent_variable": spec.independent_variable,
            "parameter_values": parameter_values, "parameter_provenance": "USER_PROVIDED" if parameter_values else "UNKNOWN",
            "formula_provenance": source_provenance, "domain": {"start": float(start), "end": float(end), "points": points, "provenance": "USER_PROVIDED"},
            "reproducibility_status": "EXPLORATORY", "created_at": datetime.now(timezone.utc).isoformat(),
        })
        configuration = {
            "input_kind": spec.kind, "source_text": source_text, "equations": dict(spec.expressions),
            "equation_spec": equation_spec.model_dump(mode="json"),
            "equation_execution_ready": equation_spec.execution_ready,
            "independent_variable": spec.independent_variable, "dependent_variables": list(spec.dependent_variables),
            "parameters": parameter_values, "initial_conditions": dict(spec.initial_conditions or {}) | (initial_values if spec.kind == "ode" else {}),
            "solver": solver_name, "fractional_order": fractional_order,
            "time": {"start": float(start), "end": float(end), "points": points},
            "provenance": {"formula": source_provenance, "parameters": "USER_PROVIDED", "domain": "USER_PROVIDED"},
            "status": "EXPLORATORY",
        }
        configuration_path = run_dir / "configuration.json"
        configuration_path.write_text(json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8")
        run_details.update({"configuration": str(configuration_path), "data": str(data_path), "figures": [str(path) for path in figure_paths], "equation_spec": equation_spec.model_dump(mode="json"), "warnings": ["Generated from an accepted, safely parsed formula; this is not a paper-reproduction claim."]})
        record_run(run_id, run_dir, "formula", started, status="EXPLORATORY", details=run_details)
        st.success("Formula evaluated successfully. Result is labeled EXPLORATORY.")
        st.dataframe(frame.head(100), use_container_width=True, hide_index=True)
        st.download_button("Download result data CSV", data_path.read_bytes(), file_name=data_path.name, mime="text/csv", key="formula_data_download")
        for index, path in enumerate(figure_paths):
            st.download_button(f"Download {path.suffix[1:].upper()}", path.read_bytes(), file_name=path.name, key=f"formula_figure_{index}")
        st.download_button("Download configuration JSON", configuration_path.read_bytes(), file_name=configuration_path.name, mime="application/json", key="formula_config_download")
    except Exception as error:
        record_run(run_id, run_dir, "formula", started, status="FAILED", details={"source_text": source_text, "formula_provenance": source_provenance}, error=str(error))
        st.error(f"Formula run failed.\n\nReason: {error}\n\nCheck the domain, provide every detected parameter and initial condition, and review the expression syntax.")

