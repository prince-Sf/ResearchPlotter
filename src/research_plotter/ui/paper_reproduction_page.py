"""Evidence-led selected-figure reproduction workspace."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable

import pandas as pd

from research_plotter.paper_analysis.paper_parser import parse_paper
from research_plotter.models.formula_spec import parse_numeric_assignments


def _paper_files(root: Path) -> list[Path]:
    paths = list(root.glob("Papers/*.pdf")) + list(root.glob("papers/*.pdf"))
    return sorted(set(path.resolve() for path in paths if path.is_file()))


def _validate_paper1_inputs(
    *,
    equations_reviewed: bool,
    initial_text: str,
    time_end_text: str,
    step_text: str,
    figure_id: str,
    sweep_text: str,
) -> list[str]:
    """Validate supplied controls without running a scientific simulation."""
    missing: list[str] = []
    if not equations_reviewed:
        missing.append("Review the linked source model and recurrence equations.")
    try:
        initial = parse_numeric_assignments(initial_text, field="initial conditions")
        if set(initial) != {"S", "E", "T", "C"}:
            missing.append("Enter exactly one numeric value for each of S, E, T, and C.")
    except ValueError:
        missing.append("Enter finite numeric initial conditions S, E, T, and C.")
    try:
        time_values = parse_numeric_assignments(f"end={time_end_text}, step={step_text}", field="time configuration")
        if time_values["end"] <= 0 or time_values["step"] <= 0:
            missing.append("End time and time step must be positive.")
    except (ValueError, KeyError):
        missing.append("Enter a positive finite end time and time step.")
    if figure_id in {"figure_11", "figure_12"}:
        try:
            values = [token.strip() for token in sweep_text.split(",") if token.strip()]
            if not values:
                raise ValueError("empty sweep")
            parse_numeric_assignments(", ".join(f"value{index}={value}" for index, value in enumerate(values)), field="sweep values")
        except ValueError:
            missing.append("Enter finite numeric sweep values; the paper does not list these values.")
    return missing


def _handoff_to_graph_analyzer(st: Any, source_document: str, page_number: int | None, figure_id: str, reference_image: str | None) -> None:
    """Open the selected paper figure context in the graph-digitization page."""
    st.session_state["graph_image_source_path"] = reference_image or source_document
    st.session_state["graph_image_page_number"] = None if reference_image else page_number
    st.session_state["graph_image_paper_context"] = {
        "source_document": source_document,
        "figure_id": figure_id,
        "source_page": page_number,
        "reference_image": reference_image,
        "provenance": "SOURCE_PAPER_FIGURE",
    }
    st.session_state["graph_image_source_kind"] = "Paper figure handoff"
    st.session_state["main_navigation"] = "Graph Image Analyzer"


def render_paper_reproduction_page(
    st: Any,
    *,
    root: Path,
    outputs: Path,
    new_run_dir: Callable[[str], tuple[str, Path]],
    record_run: Callable[..., Path],
) -> None:
    """Collect user-reviewed settings and call the existing reproduction API."""
    st.title("Figure Reproduction")
    st.write("Select a paper figure, review the extracted reproduction specification, resolve missing inputs explicitly, and run only what the available evidence supports.")
    bundled = _paper_files(root)
    source_kind = st.radio("Paper source", ["Bundled paper", "Upload PDF"], horizontal=True, key="repro_source_kind")
    source: Path | None = None
    if source_kind == "Bundled paper":
        if not bundled:
            st.info("No bundled PDF papers were found.")
            return
        source = st.selectbox("Select source paper", bundled, format_func=lambda path: path.name, key="repro_bundled_paper")
    else:
        upload = st.file_uploader("Upload a research paper", type=["pdf"], key="repro_paper_upload")
        if upload is not None:
            from research_plotter.utils.paths import persist_uploaded_file

            source = persist_uploaded_file(upload, "papers", output_root=outputs)
    if source is None:
        st.info("Choose or upload a PDF to inspect its figure manifests.")
        return

    try:
        analysis = parse_paper(source)
    except Exception as error:
        st.error(f"Paper analysis failed.\n\nReason: {error}\n\nTry a text-based PDF or inspect the PDF extraction diagnostics.")
        return
    st.caption(f"Source: {source.name} · {analysis.page_count} pages")
    if not analysis.figures:
        st.warning("No figure captions were detected. Reproduction cannot proceed without a reviewed figure specification.")
        return

    figure_rows = []
    figures = {figure.figure_id: figure for figure in analysis.figures}
    for figure in analysis.figures:
        figure_rows.append({
            "Figure": figure.figure_id.replace("figure_", "Figure "),
            "Caption": figure.caption,
            "Plot type": figure.plot_type,
            "Variables": ", ".join(figure.variables),
            "X": figure.x_variable,
            "Y": figure.y_variable,
            "Fixed parameters": json.dumps(figure.fixed_parameters, ensure_ascii=False, default=str),
            "Varying parameters": json.dumps(figure.varying_parameters, ensure_ascii=False, default=str),
            "Time range": json.dumps(figure.time_range, ensure_ascii=False, default=str),
            "Solver": figure.solver,
            "Source page": figure.source_page,
            "Required equations": ", ".join(figure.required_equation_ids),
            "Equation link": figure.equation_link_status.value,
            "Status": figure.reproducibility_status.value,
            "Confidence": figure.confidence.value,
        })
    st.dataframe(pd.DataFrame(figure_rows), use_container_width=True, hide_index=True)
    selected_id = st.selectbox("Figure to configure", list(figures), format_func=lambda value: value.replace("figure_", "Figure "), key="repro_selected_figure")
    selected = figures[selected_id]
    st.subheader(selected_id.replace("figure_", "Figure "))
    st.write(selected.caption or "No caption confidently extracted.")
    spec_cols = st.columns(3)
    spec_cols[0].write({"Plot type": selected.plot_type, "Variables": selected.variables, "X": selected.x_variable, "Y": selected.y_variable})
    spec_cols[1].write({"Fixed parameters": selected.fixed_parameters, "Varying parameters": selected.varying_parameters, "Values": selected.parameter_values})
    spec_cols[2].write({"Time range": selected.time_range, "Solver": selected.solver, "Confidence": selected.confidence.value, "Manifest status": selected.reproducibility_status.value})
    if selected.source_page:
        from research_plotter.ingestion.pdf_loader import render_pdf_region

        st.image(render_pdf_region(source, selected.source_page), caption=f"Published source page {selected.source_page} · figure/caption context", use_container_width=True)
    st.write("**Required equations:**", selected.required_equation_ids or "Not confidently linked")
    st.write("**Required parameters:**", selected.required_parameters or "Not detected")
    st.write("**Required initial conditions:**", selected.required_initial_conditions or "Not detected")
    st.write("**Provenance:**", selected.provenance or {"unresolved": "UNKNOWN"})
    if selected.missing_information:
        st.markdown("**Missing information**")
        for missing_item in selected.missing_information:
            st.warning(missing_item)
    linked_candidates = []
    required_numbers = {identifier.removeprefix("equation_").lstrip("0") for identifier in selected.required_equation_ids}
    for equation in analysis.equations:
        numbers = {number.lstrip("0") for number in equation.equation_numbers}
        if numbers & required_numbers:
            linked_candidates.append(equation)
    if linked_candidates:
        with st.expander("Linked source equation regions", expanded=True):
            for equation in linked_candidates:
                st.markdown(f"**{equation.equation_id} · equation(s) {', '.join(equation.equation_numbers)} · {equation.type.value} · {equation.review_status.value}**")
                if equation.page and equation.bbox:
                    from research_plotter.ingestion.pdf_loader import render_pdf_region

                    st.image(render_pdf_region(source, equation.page, equation.bbox), caption=f"Page {equation.page} · {equation.confidence.value} confidence", use_container_width=True)
                st.text_area("Candidate representation", value=equation.editable_representation, disabled=True, key=f"repro_eq_{selected_id}_{equation.equation_id}")
                for note in equation.notes:
                    st.caption(note)
    else:
        st.warning("No extracted equation region could be matched to the figure's required equation identifiers. Review the paper analysis before treating this linkage as established.")
    for note in selected.notes:
        st.warning(note)
    reference_image_path: Path | None = None
    reference_upload = st.file_uploader("Optional published figure image (visual inspection only)", type=["png", "jpg", "jpeg"], key=f"reference_image_{selected_id}")
    if reference_upload is not None:
        from research_plotter.utils.paths import persist_uploaded_file

        reference_image_path = persist_uploaded_file(reference_upload, "paper_figure_references", output_root=outputs)
    if source is not None and (selected.source_page is not None or reference_image_path is not None):
        st.button(
            "Digitize this published figure",
            on_click=_handoff_to_graph_analyzer,
            args=(st, str(source), selected.source_page, selected_id, str(reference_image_path) if reference_image_path else None),
            key=f"repro_digitize_{selected_id}",
        )

    from research_plotter.paper_adapters.facts import identify_document

    paper_key = identify_document(source)
    config_upload = st.file_uploader("Optional user-supplied Paper 2 configuration (YAML)", type=["yaml", "yml"], key="repro_paper2_config") if paper_key == "paper_2" else None
    config_path = None
    if config_upload is not None:
        from research_plotter.utils.paths import persist_uploaded_file

        config_path = persist_uploaded_file(config_upload, "configs", output_root=outputs)

    if paper_key == "paper_1":
        st.warning("Paper 1 is not an exact-reproduction-ready specification: the source leaves initial conditions and step size unclear, has recurrence/indexing ambiguity, and does not tabulate Figure 11/12 sweep values. Values below are USER_PROVIDED, not source-paper facts.")
        initial_text = st.text_input("Initial conditions S=…, E=…, T=…, C=… (required; no defaults)", key="repro_p1_initial")
        time_cols = st.columns(2)
        time_end_text = time_cols[0].text_input("End time (user supplied; required)", value="", key="repro_p1_end")
        step_text = time_cols[1].text_input("Time step (user supplied; required)", value="", key="repro_p1_step")
        varying = next(iter(selected.varying_parameters), None)
        figure_values_text = ""
        if selected_id in {"figure_11", "figure_12"}:
            figure_values_text = st.text_input(f"{varying} sweep values (comma-separated; required from reviewed source/experiment notes)", value="", key=f"repro_{selected_id}_values")
            st.caption("These values are not supplied by the paper. The run will remain blocked until you explicitly enter them.")
        st.caption("Time origin t=0 is retained by the current paper adapter; inspect the solver notes before treating an exploratory run as a reproduction.")
        equations_reviewed = st.checkbox("I reviewed the linked source equations and recurrence region above", key=f"repro_equations_reviewed_{selected_id}")
    else:
        initial_text, time_end_text, step_text, varying, figure_values_text = "", "", "", None, ""
        equations_reviewed = False
        if paper_key == "paper_2":
            st.warning("Paper 2's figure-specific controlled/PINN/NSFD settings are incomplete. A configured run below is an exploratory partial baseline, not a reproduction of the selected figure.")
        else:
            st.info("No dedicated paper adapter is registered for this document. The extracted manifest can be reviewed, but this figure cannot be simulated automatically.")

    can_run = paper_key in {"paper_1", "paper_2"} and (paper_key != "paper_1" or equations_reviewed)
    if paper_key == "paper_1" and not equations_reviewed:
        st.info("Review the linked model and recurrence regions before running the paper adapter. This acknowledgement does not resolve the source's missing numerical details.")
    if paper_key == "paper_2":
        run_label = "Run partial Paper 2 baseline (not selected-figure reproduction)"
    else:
        run_label = "Run selected figure experiment"
    action_cols = st.columns(2)
    validate_clicked = action_cols[0].button("Validate selected figure", disabled=paper_key not in {"paper_1", "paper_2"}, key="repro_validate_selected")
    run_clicked = action_cols[1].button(run_label, type="primary", disabled=not can_run, key="repro_run_selected")
    if validate_clicked:
        if paper_key == "paper_1":
            missing_inputs = _validate_paper1_inputs(
                equations_reviewed=equations_reviewed,
                initial_text=initial_text,
                time_end_text=time_end_text,
                step_text=step_text,
                figure_id=selected_id,
                sweep_text=figure_values_text,
            )
            if missing_inputs:
                st.error("Readiness: NEEDS_REVIEW · " + " ".join(missing_inputs))
            else:
                st.info("Inputs are valid for a user-configured exploratory run. Scientific reproduction remains BLOCKED because the paper's step/index details are unresolved.")
        else:
            st.warning("Paper 2 remains PARTIAL. Review controlled equations, NSFD details, and PINN settings in the user configuration before interpreting any run as a reproduction.")
    if run_clicked:
        run_id, run_dir = new_run_dir("paper-reproduction")
        started = time.perf_counter()
        try:
            from research_plotter.pipelines.paper_reproduction import reproduce_paper

            if paper_key == "paper_1":
                initial = parse_numeric_assignments(initial_text, field="initial conditions") if initial_text.strip() else {}
                if set(initial) != {"S", "E", "T", "C"}:
                    raise ValueError("Enter exactly four user-supplied initial conditions: S, E, T, and C.")
                time_values = parse_numeric_assignments(f"end={time_end_text}, step={step_text}", field="time configuration")
                time_end, step = time_values["end"], time_values["step"]
                if time_end <= 0 or step <= 0:
                    raise ValueError("End time and time step must be positive finite numeric values.")
                user_values: dict[str, list[float]] = {}
                if selected_id in {"figure_11", "figure_12"}:
                    values = [float(token.strip()) for token in figure_values_text.split(",") if token.strip()]
                    if not values:
                        raise ValueError(f"Enter explicit user-provided {varying} values; the source does not list them.")
                    user_values[str(varying)] = values
                with st.spinner("Running the existing fractal-fractional reference solver (exploratory; not an exact method claim)…"):
                    report = reproduce_paper(source, output_root=run_dir, initial_conditions=initial, time_end=float(time_end), step=float(step), figure_id=selected_id, figure_parameter_values=user_values)
                config = {"source_document": str(source), "selected_figure": selected_id, "required_equation_ids": selected.required_equation_ids, "accepted_equation_candidates": [item.equation_id for item in linked_candidates], "equation_review_provenance": "USER_PROVIDED", "initial_conditions": initial, "initial_conditions_provenance": "USER_PROVIDED", "time": {"start": 0.0, "start_provenance": "ASSUMED", "end": float(time_end), "end_provenance": "USER_PROVIDED", "step": float(step), "step_provenance": "USER_PROVIDED"}, "varying_parameter": varying, "user_sweep_values": user_values, "user_sweep_values_provenance": "USER_PROVIDED" if user_values else "SOURCE_PAPER", "solver": selected.solver, "reproducibility_status": "PARTIAL"}
            else:
                with st.spinner("Running the configured Paper 2 partial baseline…"):
                    report = reproduce_paper(source, output_root=run_dir, config_path=config_path)
                config = {"source_document": str(source), "selected_figure_reviewed": selected_id, "user_config": str(config_path) if config_path else None, "reproducibility_status": "PARTIAL", "note": "Figure-specific reproduction is not implemented; generated results are partial baseline outputs."}
            config_path_out = run_dir / "selected_figure_configuration.json"
            config_path_out.write_text(json.dumps(config, indent=2), encoding="utf-8")
            report_path = run_dir / "selected_figure_report.json"
            report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            status = str(report.get("status", "PARTIAL"))
            record_run(run_id, run_dir, "paper-reproduction", started, status=status, details={"source": str(source), "selected_figure": selected_id, "configuration": str(config_path_out), "report": str(report_path), "warnings": report.get("warnings", []), "reference_image": str(reference_image_path) if reference_image_path else None})
            st.session_state["latest_selected_reproduction"] = {"run_dir": str(run_dir), "selected_figure": selected_id, "report": report, "reference_image": str(reference_image_path) if reference_image_path else None}
        except Exception as error:
            record_run(run_id, run_dir, "paper-reproduction", started, status="FAILED", error=str(error), details={"source": str(source), "selected_figure": selected_id})
            st.error(f"Figure experiment failed.\n\nReason: {error}\n\nReview required fields, numerical ranges, and the warnings above.")

    result = st.session_state.get("latest_selected_reproduction")
    if result and result["selected_figure"] == selected_id:
        report = result["report"]
        status = report.get("status", "UNVERIFIED")
        if status == "BLOCKED":
            st.error(f"Reproduction status: BLOCKED · {'; '.join(map(str, report.get('warnings', [])))}")
        elif status == "PARTIAL":
            st.warning("Reproduction status: PARTIAL. A completed numerical run is not evidence of exact visual or scientific reproduction.")
        else:
            st.info(f"Reproduction status: {status}")
        for warning in report.get("warnings", []):
            st.caption(str(warning))
        artifacts = []
        for experiment in report.get("experiments", []):
            if experiment.get("figure_id") == selected_id:
                artifacts.extend(Path(path) for path in experiment.get("figures", []))
        if not artifacts:
            artifacts = sorted(Path(result["run_dir"]).rglob("*.png"))
        if artifacts:
            selected_png = next((path for path in artifacts if path.suffix.lower() == ".png" and path.is_file()), None)
            if selected_png:
                st.subheader("Generated figure preview")
                reference_image = result.get("reference_image")
                if reference_image and Path(reference_image).is_file():
                    preview_cols = st.columns(2)
                    preview_cols[0].image(str(reference_image), caption="Published figure image · visual only", use_container_width=True)
                    preview_cols[1].image(str(selected_png), caption="Generated result · not scientifically validated", use_container_width=True)
                    st.info("Visual inspection only: a reference image is not numerical trajectory data, so no quantitative reproduction metrics are computed.")
                else:
                    st.image(str(selected_png), caption="Generated result · not scientifically validated", use_container_width=True)
                for path in sorted(selected_png.parent.glob(selected_png.stem + ".*")):
                    if path.suffix.lower() in {".png", ".svg", ".pdf"}:
                        st.download_button(f"Download {path.suffix[1:].upper()}", path.read_bytes(), file_name=path.name, key=f"repro_download_{path.name}")
        with st.expander("Reproduction report and diagnostics", expanded=True):
            st.json(report)

