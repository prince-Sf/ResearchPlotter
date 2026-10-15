"""Streamlit Research Workbench backed by the public ResearchPlotter services.

This module owns UI state and presentation only. Models, solvers, analysis,
paper parsing, and Matplotlib figure construction are delegated to the existing
package APIs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
WORKBENCH = OUTPUTS / "reports" / "workbench"
NAVIGATION = [
    "Dashboard",
    "Dataset Analyzer",
    "Paper Analyzer",
    "Figure Reproduction",
    "Equation / Formula Plotter",
    "Graph Image Analyzer",
    "Model Simulation",
    "Parameter Sweep",
    "Sensitivity Analysis",
    "Noise / Uncertainty",
    "Convergence / Validation",
    "PINN Experiments",
    "Figure Gallery",
    "Experiment History",
    "Reports / Export",
]


def _json_default(value: Any) -> Any:
    """Convert common scientific values to JSON without lossy code execution."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")
    return path


def _save_configuration(path: Path, payload: Any) -> Path:
    """Persist the exact resolved configuration used for an experiment."""
    from research_plotter.config import dump_yaml

    return dump_yaml(payload, path)


def _new_run_dir(workflow: str) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{workflow}-{stamp}-{uuid4().hex[:8]}"
    from research_plotter.utils.paths import output_path

    destination = output_path("reports", "workbench", workflow, run_id, root=ROOT)
    destination.mkdir(parents=True, exist_ok=False)
    destination = output_path("reports", "workbench", workflow, run_id, root=ROOT)
    return run_id, destination


def _record_run(
    run_id: str,
    run_dir: Path,
    workflow: str,
    started: float,
    *,
    status: str,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> Path:
    record_path = run_dir / "experiment.json"
    previous: dict[str, Any] = {}
    if record_path.is_file():
        try:
            previous = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    combined_details = {**previous.get("details", {}), **(details or {})}
    record = {
        "experiment_id": run_id,
        "workflow": workflow,
        "created_at": previous.get("created_at", datetime.now(timezone.utc).isoformat()),
        "runtime_seconds": float(previous.get("runtime_seconds", 0.0)) + max(0.0, time.perf_counter() - started),
        "status": status,
        "details": combined_details,
        "error": error,
        "artifacts": [str(path.relative_to(ROOT)) for path in run_dir.rglob("*") if path.is_file() and path.name != "experiment.json"],
    }
    return _write_json(record_path, record)


def _save_figure(figure: Any, directory: Path, name: str, metadata: dict[str, Any]) -> list[Path]:
    """Save a displayed Matplotlib figure in the supported publication formats."""
    from research_plotter.plotting.core import save_figure

    paths = save_figure(figure, directory / f"{name}.png", formats=["png", "svg", "pdf"])
    _write_json(directory / f"{name}.json", {**metadata, "figure_files": [str(path) for path in paths]})
    return paths


def _show_downloads(st: Any, paths: list[Path], prefix: str) -> None:
    columns = st.columns(max(1, len(paths)))
    for column, path in zip(columns, paths):
        column.download_button(
            f"Download {path.suffix[1:].upper()}",
            data=path.read_bytes(),
            file_name=path.name,
            mime={".png": "image/png", ".svg": "image/svg+xml", ".pdf": "application/pdf"}.get(path.suffix, "application/octet-stream"),
            key=f"{prefix}_{path.name}",
        )


def _load_config_file(st: Any, key: str) -> Path | None:
    """Choose a bundled example, a previously used config, or uploaded YAML."""
    example_dir = ROOT / "configs" / "examples"
    examples = sorted(example_dir.glob("*.yaml")) + sorted(example_dir.glob("*.yml"))
    recent_path = st.session_state.get("last_config_path")
    choices: list[tuple[str, Path]] = [(path.name, path) for path in examples]
    if recent_path and Path(recent_path).is_file() and all(Path(recent_path) != item[1] for item in choices):
        choices.insert(0, (f"Last used: {Path(recent_path).name}", Path(recent_path)))
    labels = ["Upload YAML…"] + [label for label, _ in choices]
    selected = st.selectbox("Model configuration", labels, index=1 if choices else 0, key=f"{key}_config_choice")
    upload = st.file_uploader("Or upload a YAML configuration", type=["yaml", "yml"], key=f"{key}_config_upload")
    if upload is not None:
        from research_plotter.utils.paths import persist_uploaded_file

        destination = persist_uploaded_file(upload, "configs", output_root=OUTPUTS)
        st.session_state["last_config_path"] = str(destination)
        return destination
    if selected == "Upload YAML…":
        return None
    selected_path = next((path for label, path in choices if label == selected), None)
    if selected_path is not None:
        st.session_state["last_config_path"] = str(selected_path)
    return selected_path


def _configured_payload_ui(st: Any, payload: Any, key: str, *, show_solver: bool = True) -> Any:
    """Collect editable model/time values while retaining the validated config."""
    from research_plotter.config import SolverConfig, TimeConfig

    st.caption(f"Model: **{payload.model.get('name', 'configured model')}** · states: {', '.join(payload.model.get('states', []))}")
    time_cols = st.columns(3)
    start = time_cols[0].number_input("Start time", value=float(payload.time.start), key=f"{key}_start")
    end = time_cols[1].number_input("End time", value=float(payload.time.end), min_value=float(start) + 1e-9, key=f"{key}_end")
    step = time_cols[2].number_input("Time step", value=float(payload.time.step), min_value=1e-8, format="%.6g", key=f"{key}_step")

    parameters: dict[str, float] = {}
    if payload.parameters:
        st.markdown("**Parameters**")
        param_cols = st.columns(min(4, len(payload.parameters)))
        for index, (name, value) in enumerate(payload.parameters.items()):
            parameters[name] = param_cols[index % len(param_cols)].number_input(
                name, value=float(value), key=f"{key}_param_{name}"
            )
    initial_conditions: dict[str, float] = {}
    if payload.initial_conditions:
        st.markdown("**Initial conditions**")
        state_cols = st.columns(min(4, len(payload.initial_conditions)))
        for index, (name, value) in enumerate(payload.initial_conditions.items()):
            initial_conditions[name] = state_cols[index % len(state_cols)].number_input(
                f"{name}(0)", value=float(value), key=f"{key}_initial_{name}"
            )

    solver_name = payload.solver.type
    options = dict(payload.solver.options)
    if show_solver:
        known = ["ode", "explicit_euler", "grunwald_letnikov"]
        solver_options = list(dict.fromkeys([solver_name, *known]))
        solver_name = st.selectbox("Solver", solver_options, index=solver_options.index(payload.solver.type), key=f"{key}_solver")
    if solver_name.lower().replace("-", "_") in {"gl", "fractional", "grunwald_letnikov"}:
        fallback_order = payload.model.get("fractional_order", options.get("fractional_order", 0.9))
        order = st.number_input("GL fractional order", min_value=0.01, max_value=1.0, value=float(fallback_order), step=0.05, key=f"{key}_order")
        options["fractional_order"] = float(order)

    return payload.model_copy(update={
        "time": TimeConfig(start=float(start), end=float(end), step=float(step)),
        "parameters": parameters,
        "initial_conditions": initial_conditions,
        "solver": SolverConfig(type=solver_name, options=options),
    })


def _run_simulation(payload: Any):
    """Run the project's standard configured model/solver path."""
    from research_plotter.cli import _configured_model, _solver
    from research_plotter.utils.numerics import uniform_time_grid

    model = _configured_model(payload)
    solver = _solver(payload.solver.type)
    grid = uniform_time_grid(payload.time.start, payload.time.end, payload.time.step)
    result = solver.solve(model, None, payload.initial_conditions, grid, payload.solver.options)
    return model, solver, result


def _figure_file_map(base: Path) -> list[Path]:
    return [base.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf") if base.with_suffix(suffix).is_file()]


def _render_dashboard(st: Any) -> None:
    from research_plotter.ui.dashboard import render_dashboard

    render_dashboard(st, root=ROOT, outputs=OUTPUTS, workbench=WORKBENCH)


def _render_dataset(st: Any) -> None:
    import pandas as pd
    from research_plotter.pipelines.dataset_analysis import analyze_dataset

    st.title("Dataset Analyzer")
    st.write("Profile CSV, clearly delimited TXT tables, Excel, JSON, or Parquet data, review plot suggestions, then choose what to render.")
    selected_search_path = st.session_state.get("dataset_open_path")
    source_options = ["Bundled example", "Upload dataset"]
    if selected_search_path and Path(selected_search_path).is_file():
        source_options.insert(0, "Open selected local dataset")
    source_kind = st.selectbox("Data source", source_options, key="dataset_source")
    upload = st.file_uploader("Dataset file", type=["csv", "txt", "xlsx", "xls", "json", "parquet"], key="dataset_upload") if source_kind == "Upload dataset" else None
    example_files = sorted((ROOT / "data" / "examples").glob("*"))
    source_path: Path | None = None
    if source_kind == "Open selected local dataset":
        source_path = Path(selected_search_path)
        st.caption(f"Opened from local search: {source_path.name}")
    elif source_kind == "Bundled example":
        selected_example = st.selectbox("Example dataset", example_files, format_func=lambda path: path.name, key="dataset_example") if example_files else None
        source_path = selected_example
    elif upload is not None:
        from research_plotter.utils.paths import persist_uploaded_file

        source_path = persist_uploaded_file(upload, "datasets", output_root=OUTPUTS)
    if st.button("Analyze dataset", type="primary", disabled=source_path is None, key="analyze_dataset"):
        run_id, run_dir = _new_run_dir("dataset")
        started = time.perf_counter()
        try:
            with st.spinner("Profiling dataset…"):
                frame, profile, report = analyze_dataset(source_path, run_dir)
            st.session_state["dataset_frame"] = frame
            st.session_state["dataset_profile"] = profile.model_dump(mode="json")
            st.session_state["dataset_source_path"] = str(source_path)
            st.session_state["dataset_run_dir"] = str(run_dir)
            st.session_state["dataset_run_id"] = run_id
            _record_run(run_id, run_dir, "dataset", started, status="SUCCEEDED", details={"source": str(source_path), "rows": profile.rows, "columns": profile.columns, "report": str(report)})
        except Exception as error:
            _record_run(run_id, run_dir, "dataset", started, status="FAILED", error=str(error))
            st.error(f"Dataset analysis failed.\n\nReason: {error}")

    frame = st.session_state.get("dataset_frame")
    profile_data = st.session_state.get("dataset_profile")
    if frame is None or profile_data is None:
        return
    st.subheader("Profile")
    metric_cols = st.columns(4)
    for column, label, value in zip(metric_cols, ("Rows", "Columns", "Duplicate rows", "Missing values"), (
        profile_data["rows"], profile_data["columns"], profile_data["duplicate_rows"],
        sum(item["missing_count"] for item in profile_data["columns_detail"]),
    )):
        column.metric(label, value)
    st.write("**Time-like columns:**", ", ".join(profile_data["time_like_columns"]) or "None detected")
    info_cols = st.columns(3)
    info_cols[0].write({"Numeric": profile_data["numeric_columns"]})
    info_cols[1].write({"Categorical": profile_data["categorical_columns"]})
    info_cols[2].write({"Datetime": profile_data["datetime_columns"]})
    with st.expander("Column types, missingness, and descriptive statistics", expanded=True):
        st.dataframe(pd.DataFrame(profile_data["columns_detail"]), use_container_width=True, hide_index=True)
    st.write("**Backend recommendations**")
    for recommendation in profile_data.get("recommendations", []):
        st.write(f"• {recommendation}")
    st.dataframe(frame.head(100), use_container_width=True, hide_index=True)
    st.download_button("Download source data as CSV", frame.to_csv(index=False), file_name="dataset.csv", mime="text/csv", key="dataset_csv")

    numeric = profile_data["numeric_columns"]
    time_candidates = profile_data["time_like_columns"] or profile_data["datetime_columns"]
    suggestions = []
    if time_candidates and numeric:
        suggestions.append("Line plot")
    if len(numeric) >= 2:
        suggestions.extend(["Scatter plot", "Correlation heatmap"])
    if numeric:
        suggestions.extend(["Histogram", "Box plot"])
    if profile_data["categorical_columns"] and numeric:
        suggestions.append("Grouped box plot")
    if not suggestions:
        st.info("No obvious quantitative plot is suggested; inspect the data and choose a suitable analysis externally.")
        return
    st.subheader("Choose plots")
    chosen = st.multiselect("Suggested plot types", suggestions, default=suggestions[:1], key="dataset_plot_types")
    plot_controls: dict[str, Any] = {}
    if "Line plot" in chosen:
        x_default = time_candidates[0]
        plot_controls["line_x"] = st.selectbox("Line plot x column", frame.columns.tolist(), index=frame.columns.get_loc(x_default), key="dataset_line_x")
        y_defaults = [name for name in numeric if name != plot_controls["line_x"]][:3] or numeric[:1]
        plot_controls["line_y"] = st.multiselect("Line plot numeric series", numeric, default=y_defaults, key="dataset_line_y")
    if "Scatter plot" in chosen:
        cols = st.columns(2)
        plot_controls["scatter_x"] = cols[0].selectbox("Scatter x", numeric, key="dataset_scatter_x")
        plot_controls["scatter_y"] = cols[1].selectbox("Scatter y", numeric, index=min(1, len(numeric)-1), key="dataset_scatter_y")
    if "Histogram" in chosen or "Box plot" in chosen or "Grouped box plot" in chosen:
        plot_controls["distribution_column"] = st.selectbox("Distribution column", numeric, key="dataset_dist_column")
    if "Grouped box plot" in chosen:
        plot_controls["group_column"] = st.selectbox("Group column", profile_data["categorical_columns"], key="dataset_group_column")
    if st.button("Generate selected figures", type="primary", disabled=not chosen, key="generate_dataset_figures"):
        import pandas as pd
        from research_plotter.plotting.boxplot import boxplot
        from research_plotter.plotting.core import save_figure
        from research_plotter.plotting.heatmap import heatmap
        from research_plotter.plotting.histogram import histogram
        from research_plotter.plotting.line import multi_line_plot
        from research_plotter.plotting.scatter import scatter_plot

        directory = Path(st.session_state["dataset_run_dir"]) / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        generated: list[tuple[str, Any, list[Path]]] = []
        if "Line plot" in chosen and plot_controls.get("line_y"):
            fig = multi_line_plot(frame[plot_controls["line_x"]], frame[plot_controls["line_y"]], xlabel=plot_controls["line_x"], ylabel="value", title="Dataset trajectories")
            generated.append(("dataset_line", fig, [directory / "dataset_line.png"]))
        if "Scatter plot" in chosen:
            fig = scatter_plot(frame[plot_controls["scatter_x"]], frame[plot_controls["scatter_y"]], xlabel=plot_controls["scatter_x"], ylabel=plot_controls["scatter_y"], title="Dataset scatter")
            generated.append(("dataset_scatter", fig, [directory / "dataset_scatter.png"]))
        if "Correlation heatmap" in chosen:
            fig = heatmap(frame[numeric].corr(), xlabels=numeric, ylabels=numeric, title="Numeric correlation")
            generated.append(("dataset_correlation", fig, [directory / "dataset_correlation.png"]))
        if "Histogram" in chosen:
            name = plot_controls["distribution_column"]
            fig = histogram(frame[name].dropna(), xlabel=name, title=f"Distribution of {name}")
            generated.append(("dataset_histogram", fig, [directory / "dataset_histogram.png"]))
        if "Box plot" in chosen:
            name = plot_controls["distribution_column"]
            fig = boxplot([frame[name].dropna()], labels=[name], ylabel=name, title=f"Spread of {name}")
            generated.append(("dataset_boxplot", fig, [directory / "dataset_boxplot.png"]))
        if "Grouped box plot" in chosen:
            name, group = plot_controls["distribution_column"], plot_controls["group_column"]
            groups = [(label, part[name].dropna()) for label, part in frame.groupby(group, observed=False)]
            if groups:
                fig = boxplot([values for _, values in groups], labels=[str(label) for label, _ in groups], xlabel=group, ylabel=name, title=f"{name} by {group}")
                generated.append(("dataset_grouped_boxplot", fig, [directory / "dataset_grouped_boxplot.png"]))
        if not generated:
            st.warning("Select at least one usable plot and series.")
            return
        export_paths = []
        for name, figure, _ in generated:
            st.pyplot(figure, clear_figure=False)
            plot_configuration = {
                "selected_plot_types": chosen,
                "columns": plot_controls,
                "formats": ["png", "svg", "pdf"],
            }
            export_paths.extend(_save_figure(figure, directory, name, {
                "workflow": "dataset", "source": st.session_state["dataset_source_path"],
                "plot_type": name.removeprefix("dataset_"), "created_at": datetime.now(timezone.utc).isoformat(),
                "configuration": plot_configuration,
            }))
        st.session_state["dataset_figure_paths"] = [str(path) for path in export_paths]
        Path(st.session_state["dataset_run_dir"], "dataset.csv").write_text(frame.to_csv(index=False), encoding="utf-8")
        _record_run(st.session_state["dataset_run_id"], Path(st.session_state["dataset_run_dir"]), "dataset", time.perf_counter(), status="SUCCEEDED", details={"source": st.session_state["dataset_source_path"], "plots": chosen})
        st.success(f"Generated {len(generated)} selected figure(s).")
    for index, file_name in enumerate(st.session_state.get("dataset_figure_paths", [])):
        path = Path(file_name)
        if path.is_file():
            st.download_button(f"Download {path.suffix[1:].upper()} · {path.stem}", path.read_bytes(), file_name=path.name, key=f"dataset_export_{index}_{path.name}")


def _render_paper(st: Any) -> None:
    import pandas as pd

    st.title("Paper Analyzer")
    st.write("Extract evidence and unresolved reproduction details from a PDF. Machine extraction is reviewable, not authoritative.")
    bundled = sorted((ROOT / "Papers").glob("*.pdf"))
    search_path = st.session_state.get("paper_open_path")
    paper_sources = ["Bundled paper", "Upload PDF"]
    if search_path and Path(search_path).is_file():
        paper_sources.insert(0, "Open selected local paper")
    kind = st.selectbox("PDF source", paper_sources, key="paper_source_kind")
    source: Path | None = None
    if kind == "Open selected local paper":
        source = Path(search_path)
        st.caption(f"Opened from local search: {source.name}")
    elif kind == "Bundled paper" and bundled:
        source = st.selectbox("Select paper", bundled, format_func=lambda path: path.name, key="paper_bundled")
    elif kind == "Upload PDF":
        upload = st.file_uploader("Research paper PDF", type=["pdf"], key="paper_upload")
        if upload is not None:
            from research_plotter.utils.paths import persist_uploaded_file

            source = persist_uploaded_file(upload, "papers", output_root=OUTPUTS)
    config_upload = st.file_uploader("Optional Paper 2 user-override YAML", type=["yaml", "yml"], key="paper_override_upload")
    config_path = None
    if config_upload is not None:
        from research_plotter.utils.paths import persist_uploaded_file

        config_path = persist_uploaded_file(config_upload, "configs", output_root=OUTPUTS)
    do_analyze, do_reproduce = st.columns(2)
    analyze = do_analyze.button("Analyze paper", type="primary", disabled=source is None, key="paper_analyze")
    reproduce = do_reproduce.button("Assess reproduction", disabled=source is None, key="paper_reproduce")
    if analyze:
        from research_plotter.pipelines.paper_reproduction import analyze_paper
        run_id, run_dir = _new_run_dir("paper-analysis")
        started = time.perf_counter()
        try:
            with st.spinner("Inspecting PDF layout, equation regions, and figure captions…"):
                report_path = analyze_paper(source, run_dir)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            st.session_state["paper_analysis"] = report
            st.session_state["paper_analysis_report_path"] = str(report_path)
            _record_run(run_id, run_dir, "paper-analysis", started, status="SUCCEEDED", details={"source": str(source), "report": str(report_path), "status": "PARTIAL"})
        except Exception as error:
            _record_run(run_id, run_dir, "paper-analysis", started, status="FAILED", error=str(error))
            st.error(f"Paper analysis failed.\n\nReason: {error}")
    report = st.session_state.get("paper_analysis")
    if report and report.get("source_document") and source is not None:
        try:
            if Path(report["source_document"]).resolve() != source.resolve():
                report = None
        except OSError:
            report = None
    if report:
        st.session_state["paper_source_path"] = report.get("source_document")
        title = report.get("title", {})
        st.subheader(title.get("value") or Path(report.get("source_document", "paper")).name)
        st.write("**Authors:**", report.get("authors", {}).get("value") or "Not confidently extracted")
        st.write("**Sections:**", report.get("sections") or "None detected")
        for label, key in (("State variables", "state_variables"), ("Parameters", "parameters"), ("Initial conditions", "initial_conditions"), ("Equation candidates", "equations"), ("Numerical methods", "numerical_methods"), ("Detected figures and captions", "figures")):
            with st.expander(label, expanded=(label in {"State variables", "Numerical methods"})):
                value = report.get(key, {})
                if isinstance(value, dict) and "status" in value:
                    st.markdown(f"Status: **{value.get('status')}**")
                st.json(value)
        layout_summary = report.get("page_layout_summary", [])
        if layout_summary:
            with st.expander("PDF page layout classification", expanded=False):
                st.dataframe(pd.DataFrame(layout_summary), use_container_width=True, hide_index=True)
        equation_items = report.get("equations", [])
        if equation_items:
            st.subheader("Equation region review")
            labels = [
                f"Page {item.get('page')} · {item.get('equation_id')} · {item.get('type')} · {item.get('confidence')}"
                for item in equation_items
            ]
            selected_index = st.selectbox("Candidate equation", range(len(equation_items)), format_func=lambda index: labels[index], key="paper_equation_candidate_index")
            equation_item = equation_items[selected_index]
            equation_page = equation_item.get("page")
            bbox = equation_item.get("bbox")
            if equation_page and bbox and source and source.is_file():
                from research_plotter.ingestion.pdf_loader import render_pdf_region

                st.image(render_pdf_region(source, int(equation_page), tuple(float(value) for value in bbox)), caption=f"Original source region · page {equation_page}", use_container_width=True)
            review_cols = st.columns(3)
            review_cols[0].metric("Mathematical type", equation_item.get("type", "UNKNOWN"))
            review_cols[1].metric("Confidence", equation_item.get("confidence", "LOW"))
            review_cols[2].metric("Review status", equation_item.get("review_status", "REVIEW_REQUIRED"))
            st.write("**Derivative type:**", equation_item.get("derivative_type", "unknown"))
            st.write("**Fractional order / fractal dimension:**", equation_item.get("fractional_order") or "unknown", "/", equation_item.get("fractal_dimension") or "unknown")
            st.write("**Kernel / solver requirement:**", equation_item.get("kernel") or "unknown", "/", equation_item.get("solver_requirement") or "unknown")
            st.write("**Variables / parameters:**", equation_item.get("variables", []), "/", equation_item.get("parameters", []))
            equation_edit_key = f"paper_equation_edit_{selected_index}"
            if equation_edit_key not in st.session_state:
                st.session_state[equation_edit_key] = equation_item.get("editable_representation", equation_item.get("source_text", ""))
            candidate_text = st.text_area("Editable mathematical representation", height=180, key=equation_edit_key)
            accept_equation = st.checkbox("I reviewed this representation against the source image", key=f"paper_equation_accept_{selected_index}")
            if equation_item.get("notes"):
                for note in equation_item["notes"]:
                    st.caption(note)
            if st.button("Save equation review", key=f"paper_equation_save_{selected_index}"):
                equation_item["editable_representation"] = candidate_text
                equation_item["provenance"] = "USER_PROVIDED" if candidate_text != equation_item.get("source_text", "") else "SOURCE_PAPER"
                equation_item["review_status"] = "ACCEPTED" if accept_equation else "REVIEW_REQUIRED"
                # Acceptance preserves the review decision but never makes a
                # PDF candidate executable by itself. The Formula/Model APIs
                # must independently validate supported syntax and operators.
                equation_item["execution_supported"] = False
                report_path_value = st.session_state.get("paper_analysis_report_path")
                if report_path_value:
                    Path(report_path_value).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                st.session_state["paper_analysis"] = report
                st.success(f"Review saved · {equation_item['review_status']} · {equation_item['provenance']}")
        figure_items = report.get("figures", [])
        if figure_items:
            st.subheader("Figure reproduction specifications")
            st.dataframe([{
                "Figure": item.get("figure_id"), "Caption": item.get("caption"), "Plot type": item.get("plot_type"),
                "Variables": json.dumps(item.get("variables"), ensure_ascii=False, default=_json_default),
                "X": item.get("x_variable"), "Y": item.get("y_variable"),
                "Fixed parameters": json.dumps(item.get("fixed_parameters"), ensure_ascii=False, default=_json_default),
                "Varying parameters": json.dumps(item.get("varying_parameters"), ensure_ascii=False, default=_json_default),
                "Values": json.dumps(item.get("parameter_values"), ensure_ascii=False, default=_json_default),
                "Time range": json.dumps(item.get("time_range"), ensure_ascii=False, default=_json_default),
                "Solver": item.get("solver"),
                "Equations": json.dumps(item.get("required_equation_ids"), ensure_ascii=False, default=_json_default),
                "Equation link": item.get("equation_link_status"),
                "Status": item.get("reproducibility_status"), "Confidence": item.get("confidence"),
                "Warnings": json.dumps(item.get("notes"), ensure_ascii=False, default=_json_default),
            } for item in figure_items], use_container_width=True, hide_index=True)
            st.info("Select Figure Reproduction in the navigation to configure one of these manifests. Detected captions and values remain reviewable evidence, not verified simulation instructions.")
        unresolved = report.get("unresolved_information", [])
        uncertainties = report.get("uncertainties", [])
        st.subheader("Evidence and limitations")
        st.info(f"Analysis status: PARTIAL · {len(unresolved)} unresolved information item(s)")
        if unresolved:
            for item in unresolved:
                st.warning(item)
        if uncertainties:
            for item in uncertainties:
                st.caption(item)
        st.download_button("Download paper analysis JSON", json.dumps(report, indent=2), file_name="paper_analysis.json", mime="application/json", key="paper_analysis_json")
    if reproduce:
        from research_plotter.pipelines.paper_reproduction import reproduce_paper
        run_id, run_dir = _new_run_dir("paper-reproduction")
        started = time.perf_counter()
        try:
            with st.spinner("Checking paper reproduction readiness…"):
                result = reproduce_paper(source, output_root=run_dir, config_path=config_path)
            _write_json(run_dir / "reproduction_report.json", result)
            status = str(result.get("status", "UNVERIFIED"))
            _record_run(run_id, run_dir, "paper-reproduction", started, status=status, details={"source": str(source), "report": str(run_dir / "reproduction_report.json"), "warnings": result.get("warnings", [])})
            if status == "BLOCKED":
                st.error(f"Reproduction status: BLOCKED\n\n{'; '.join(map(str, result.get('warnings', []))) or 'Required source details remain unresolved.'}")
            elif status == "PARTIAL":
                st.warning("Reproduction status: PARTIAL. Review the unresolved source details and assumptions before interpreting generated outputs.")
            else:
                st.info(f"Reproduction status: {status}")
            st.json(result)
        except Exception as error:
            _record_run(run_id, run_dir, "paper-reproduction", started, status="FAILED", error=str(error))
            st.error(f"Reproduction assessment failed.\n\nReason: {error}")


def _render_simulation(st: Any) -> None:
    import pandas as pd
    from research_plotter.analysis.validation import validate_simulation
    from research_plotter.plotting.line import multi_line_plot

    st.title("Model Simulation")
    config_path = _load_config_file(st, "simulation")
    if config_path is None:
        st.info("Choose a bundled YAML example or upload a model configuration to start.")
        return
    from research_plotter.config import load_experiment
    try:
        payload = _configured_payload_ui(st, load_experiment(config_path), "simulation")
    except Exception as error:
        st.error(f"Configuration could not be loaded.\n\nReason: {error}")
        return
    configured_states = payload.model.get("states", [])
    visible_states = st.multiselect("States to display", configured_states, default=configured_states, key="simulation_visible_states")
    if st.button("Run simulation", type="primary", key="run_simulation"):
        run_id, run_dir = _new_run_dir("simulation")
        started = time.perf_counter()
        try:
            with st.spinner("Solving the configured model…"):
                model, solver, result = _run_simulation(payload)
            validation = validate_simulation(result, model)
            data_path = run_dir / "simulation.csv"
            result.to_frame().to_csv(data_path, index=False)
            configuration_path = _save_configuration(run_dir / "configuration.yaml", payload)
            frame = result.to_frame()
            selected_states = [name for name in visible_states if name in frame.columns]
            if not selected_states:
                selected_states = [name for name in frame.columns if name != "time"]
            figure = multi_line_plot(result.time, frame[selected_states], xlabel=model.independent_variable, ylabel="state", title=model.name)
            st.pyplot(figure)
            figure_paths = _save_figure(figure, run_dir / "figures", "simulation", {"workflow": "simulation", "source_config": str(config_path), "solver": payload.solver.type, "parameters": payload.parameters, "initial_conditions": payload.initial_conditions, "status": "UNVERIFIED"})
            _record_run(run_id, run_dir, "simulation", started, status="SUCCEEDED", details={"model": model.name, "solver": payload.solver.type, "configuration": str(configuration_path), "parameters": payload.parameters, "initial_conditions": payload.initial_conditions, "time": payload.time.model_dump(mode="json"), "validation": validation, "data": str(data_path), "figures": [str(path) for path in figure_paths]})
            st.session_state["latest_simulation"] = {"run_dir": str(run_dir), "data": str(data_path), "metadata": result.to_metadata(), "validation": validation}
            st.subheader("Simulation summary")
            summary_cols = st.columns(3)
            summary_cols[0].metric("Runtime", f"{result.runtime_seconds:.4g} s")
            summary_cols[1].metric("Solver", result.solver_name)
            summary_cols[2].metric("Validation", validation.get("status", "UNVERIFIED"))
            st.write("**Parameters**", payload.parameters)
            st.write("**Initial conditions**", payload.initial_conditions)
            if validation.get("warnings"):
                for warning in validation["warnings"]:
                    st.warning(str(warning))
            st.dataframe(frame.head(100), use_container_width=True, hide_index=True)
            st.download_button("Download simulation data CSV", data_path.read_bytes(), file_name=data_path.name, mime="text/csv", key="simulation_data_export")
            _show_downloads(st, figure_paths, "simulation_figure")
        except Exception as error:
            _record_run(run_id, run_dir, "simulation", started, status="FAILED", error=str(error))
            st.error(f"Simulation failed.\n\nReason: {error}")


def _render_sweep(st: Any) -> None:
    import pandas as pd
    from research_plotter.plotting.comparison import comparison_plot

    st.title("Parameter Sweep")
    st.write("Run the shared solver across selected parameter values; the model itself comes from the chosen YAML configuration.")
    config_path = _load_config_file(st, "sweep")
    if config_path is None:
        st.info("Select or upload a model config that defines states, equations, parameters, and initial conditions.")
        return
    from research_plotter.config import load_experiment
    try:
        payload = load_experiment(config_path)
        parameter_names = list(payload.parameters)
        if not parameter_names:
            st.error("This configuration has no model parameters to sweep.")
            return
        parameter = st.selectbox("Parameter", parameter_names, key="sweep_parameter")
        values_from_config = payload.model.get("sweep", {}).get(parameter)
        base = float(payload.parameters[parameter])
        default_values = values_from_config if values_from_config else [round(base * .8, 6), base, round(base * 1.2, 6)]
        values_text = st.text_input("Values (comma-separated)", value=", ".join(map(str, default_values)), key="sweep_values")
        output = st.selectbox("Output state", payload.model.get("states", []), key="sweep_output")
        payload = _configured_payload_ui(st, payload, "sweep", show_solver=True)
        raw_values = [float(item.strip()) for item in values_text.split(",") if item.strip()]
    except Exception as error:
        st.error(f"Sweep configuration is invalid.\n\nReason: {error}")
        return
    if st.button("Run parameter sweep", type="primary", disabled=not raw_values, key="run_sweep"):
        from research_plotter.analysis.parameter_sweep import run_parameter_sweep
        from research_plotter.cli import _configured_model, _solver
        from research_plotter.utils.numerics import uniform_time_grid

        run_id, run_dir = _new_run_dir("parameter-sweep")
        started = time.perf_counter()
        try:
            model = _configured_model(payload)
            grid = uniform_time_grid(payload.time.start, payload.time.end, payload.time.step)
            with st.spinner(f"Running {len(raw_values)} simulations…"):
                sweep = run_parameter_sweep(model, _solver(payload.solver.type), grid, {parameter: raw_values}, payload.initial_conditions, payload.solver.options, progress=False)
            sweep_config = payload.model_copy(update={"model": {**payload.model, "sweep": {parameter: raw_values}}})
            configuration_path = _save_configuration(run_dir / "configuration.yaml", sweep_config)
            trajectories = {f"{parameter}={values[parameter]:g}": simulation.states[:, model.state_names.index(output)] for values, simulation in zip(sweep.parameter_values, sweep.simulations)}
            table = pd.DataFrame([{parameter: values[parameter], f"final_{output}": simulation.states[-1, model.state_names.index(output)], "runtime_seconds": simulation.runtime_seconds} for values, simulation in zip(sweep.parameter_values, sweep.simulations)])
            data_path = run_dir / "sweep_results.csv"
            table.to_csv(data_path, index=False)
            figure = comparison_plot(sweep.simulations[0].time, trajectories, xlabel=model.independent_variable, ylabel=output, title=f"{output} across {parameter}")
            st.pyplot(figure)
            figure_paths = _save_figure(figure, run_dir / "figures", "parameter_sweep", {"workflow": "parameter_sweep", "source_config": str(config_path), "parameter": parameter, "parameter_values": raw_values, "output": output, "solver": payload.solver.type, "status": "UNVERIFIED"})
            _record_run(run_id, run_dir, "parameter-sweep", started, status="SUCCEEDED", details={"model": model.name, "solver": payload.solver.type, "configuration": str(configuration_path), "parameter": parameter, "parameter_values": raw_values, "output": output, "data": str(data_path), "figures": [str(path) for path in figure_paths]})
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button("Export sweep data CSV", data_path.read_bytes(), file_name=data_path.name, mime="text/csv", key="sweep_data_export")
            _show_downloads(st, figure_paths, "sweep_figure")
            st.success(f"Saved experiment {run_id} to Experiment History.")
        except Exception as error:
            _record_run(run_id, run_dir, "parameter-sweep", started, status="FAILED", error=str(error))
            st.error(f"Parameter sweep failed.\n\nReason: {error}")


def _render_sensitivity(st: Any) -> None:
    import pandas as pd
    from research_plotter.analysis.sensitivity import logarithmic_sensitivity
    from research_plotter.plotting.line import multi_line_plot

    st.title("Sensitivity Analysis")
    st.caption("The default central relative perturbation is 10%; log sensitivity follows the analysis backend's documented definition.")
    config_path = _load_config_file(st, "sensitivity")
    if config_path is None:
        st.info("Choose a model configuration to configure parameter sensitivity.")
        return
    from research_plotter.config import load_experiment
    try:
        payload = load_experiment(config_path)
        parameter = st.selectbox("Parameter", list(payload.parameters), key="sensitivity_parameter")
        output = st.selectbox("Output state", ["All states", *payload.model.get("states", [])], key="sensitivity_output")
        delta_percent = st.number_input("Perturbation (%)", min_value=0.1, max_value=99.0, value=10.0, step=1.0, key="sensitivity_delta")
        payload = _configured_payload_ui(st, payload, "sensitivity", show_solver=True)
    except Exception as error:
        st.error(f"Sensitivity configuration is invalid.\n\nReason: {error}")
        return
    if st.button("Run sensitivity analysis", type="primary", key="run_sensitivity"):
        from research_plotter.cli import _configured_model, _solver
        from research_plotter.utils.numerics import uniform_time_grid

        run_id, run_dir = _new_run_dir("sensitivity")
        started = time.perf_counter()
        try:
            model = _configured_model(payload)
            grid = uniform_time_grid(payload.time.start, payload.time.end, payload.time.step)
            with st.spinner("Perturbing parameter and solving both cases…"):
                result = logarithmic_sensitivity(model, _solver(payload.solver.type), grid, parameter, delta=delta_percent / 100.0, output=None if output == "All states" else output, initial_conditions=payload.initial_conditions, options=payload.solver.options)
            sensitivity_config = payload.model_dump(mode="json")
            sensitivity_config["sensitivity"] = {"parameter": parameter, "output": output, "delta": result["delta"]}
            configuration_path = _save_configuration(run_dir / "configuration.yaml", sensitivity_config)
            values = result["sensitivity"]
            if getattr(values, "ndim", 1) == 1:
                curves = {result["output"]: values}
            else:
                curves = {name: values[:, index] for index, name in enumerate(model.state_names)}
            table = pd.DataFrame({"time": result["time"], **curves})
            data_path = run_dir / "sensitivity.csv"
            table.to_csv(data_path, index=False)
            figure = multi_line_plot(result["time"], curves, xlabel=model.independent_variable, ylabel=f"log sensitivity ({parameter})", title=f"Sensitivity to {parameter}")
            st.pyplot(figure)
            figure_paths = _save_figure(figure, run_dir / "figures", "sensitivity", {"workflow": "sensitivity", "source_config": str(config_path), "parameter": parameter, "output": output, "delta": result["delta"], "solver": payload.solver.type, "status": "UNVERIFIED", "warnings": result["warnings"]})
            _record_run(run_id, run_dir, "sensitivity", started, status="SUCCEEDED", details={"model": model.name, "solver": payload.solver.type, "configuration": str(configuration_path), "parameter": parameter, "output": output, "delta": result["delta"], "data": str(data_path), "figures": [str(path) for path in figure_paths], "warnings": result["warnings"]})
            st.write({"Parameter": parameter, "Relative perturbation": result["delta"], "Output": output, "Solver": payload.solver.type})
            for warning in result["warnings"]:
                st.warning(warning)
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button("Export sensitivity data CSV", data_path.read_bytes(), file_name=data_path.name, mime="text/csv", key="sensitivity_csv")
            _show_downloads(st, figure_paths, "sensitivity_figure")
        except Exception as error:
            _record_run(run_id, run_dir, "sensitivity", started, status="FAILED", error=str(error))
            st.error(f"Sensitivity analysis failed.\n\nReason: {error}")


def _render_pinn(st: Any) -> None:
    import numpy as np
    import pandas as pd

    st.title("PINN Experiments")
    st.write("Run a lightweight baseline NN, integer-order PINN, or GL-fractional PINN against a configured model trajectory.")
    st.warning("This is a numerical training experiment, not evidence of exact agreement with either reference paper. Paper 2's controlled equations and training details remain incomplete unless supplied by the user.")
    st.caption("For fractional mode, the configured model RHS is paired with the numerical GL residual. Treat this as exploratory unless the configured equations and derivative definition are scientifically justified for that model.")
    config_path = _load_config_file(st, "pinn")
    if config_path is None:
        st.info("Select a model configuration. For paper-specific experiments, configure all unresolved details explicitly first.")
        return
    from research_plotter.config import PINNConfig, load_experiment
    try:
        payload = load_experiment(config_path)
        mode = st.selectbox("Model mode", ["baseline", "integer", "fractional"], key="pinn_mode")
        network_cols = st.columns(4)
        layers = network_cols[0].number_input("Hidden layers", min_value=1, max_value=8, value=2, key="pinn_layers")
        units = network_cols[1].number_input("Hidden units", min_value=2, max_value=256, value=16, step=2, key="pinn_units")
        activation = network_cols[2].selectbox("Activation", ["tanh", "relu", "sigmoid"], key="pinn_activation")
        optimizer = network_cols[3].selectbox("Optimizer", ["Adam", "RMSprop", "SGD"], key="pinn_optimizer")
        train_cols = st.columns(4)
        learning_rate = train_cols[0].number_input("Learning rate", min_value=1e-6, max_value=1.0, value=0.005, format="%.6g", key="pinn_lr")
        epochs = train_cols[1].number_input("Epochs", min_value=1, max_value=5000, value=30, key="pinn_epochs")
        points = train_cols[2].number_input("Training/collocation points", min_value=3, max_value=2000, value=40, key="pinn_points")
        physics_weight = train_cols[3].number_input("Physics loss weight", min_value=0.0, max_value=100.0, value=1.0, key="pinn_physics_weight")
        state_name = st.selectbox("State output", payload.model.get("states", []), key="pinn_state")
        order_default = float(payload.model.get("fractional_order", payload.solver.options.get("fractional_order", .8)))
        order = st.number_input("Fractional order", min_value=.05, max_value=.99, value=min(.99, max(.05, order_default)), step=.05, key="pinn_fractional_order")
        noise_percent = st.selectbox("Observation noise", [0, 5, 10, 15, 20], index=0, key="pinn_noise")
        seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42, key="pinn_seed")
        pinn_config = PINNConfig(enabled=True, hidden_layers=[int(units)] * int(layers), activation=activation, optimizer=optimizer, learning_rate=float(learning_rate), epochs=int(epochs), collocation_points=int(points), data_points=int(points), physics_loss_weight=float(physics_weight), initial_condition_loss_weight=1.0, provenance="USER_PROVIDED")
    except Exception as error:
        st.error(f"PINN configuration is invalid.\n\nReason: {error}")
        return
    if st.button("Run PINN experiment", type="primary", key="run_pinn_experiment"):
        run_id, run_dir = _new_run_dir("pinn")
        started = time.perf_counter()
        try:
            import torch
            from research_plotter.analysis.noise import add_gaussian_noise
            from research_plotter.pinn.experiments import run_pinn_experiment
            from research_plotter.pinn.torch_rhs import torch_rhs_from_model
            from research_plotter.utils.numerics import error_metrics, uniform_time_grid
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            model, solver, reference_result = _run_simulation(payload)
            grid = uniform_time_grid(payload.time.start, payload.time.end, payload.time.step)
            state_index = model.state_names.index(state_name)
            clean_reference = reference_result.states
            # PINN time coordinates and reference observations use the same explicit grid.
            observed = add_gaussian_noise(clean_reference, float(noise_percent) / 100.0, np.random.default_rng(int(seed))) if noise_percent else clean_reference.copy()
            rhs = None if mode == "baseline" else torch_rhs_from_model(model)
            progress = st.progress(0.0, text="Preparing reference trajectory…")
            def update_training_progress(metrics: dict[str, float]) -> None:
                fraction = min(1.0, float(metrics["epoch"]) / int(epochs))
                progress.progress(fraction, text=f"Training · epoch {int(metrics['epoch'])}/{int(epochs)} · loss {metrics['loss']:.3g}")
            with st.spinner("Training network and recording component losses…"):
                result = run_pinn_experiment(grid, observed, pinn_config, mode=mode, rhs=rhs, fractional_order=float(order), progress_callback=update_training_progress)
            clean_metrics = error_metrics(clean_reference[:, state_index], result.prediction[:, state_index])
            prediction_path = run_dir / "predictions.csv"
            pd.DataFrame({"time": grid, f"reference_{state_name}": clean_reference[:, state_index], f"observations_{state_name}": observed[:, state_index], f"prediction_{state_name}": result.prediction[:, state_index]}).to_csv(prediction_path, index=False)
            history_path = run_dir / "training_history.json"
            _write_json(history_path, result.training.history)
            checkpoint = result.training.save(run_dir / "model.pt", configuration={"mode": mode, "pinn": pinn_config.model_dump(mode="json"), "seed": int(seed)})
            experiment_configuration = {
                "model": payload.model,
                "solver": payload.solver.model_dump(mode="json"),
                "time": payload.time.model_dump(mode="json"),
                "parameters": payload.parameters,
                "initial_conditions": payload.initial_conditions,
                "pinn": pinn_config.model_dump(mode="json"),
                "mode": mode,
                "fractional_order": float(order),
                "noise_percent": int(noise_percent),
                "seed": int(seed),
            }
            configuration_path = _save_configuration(run_dir / "configuration.yaml", experiment_configuration)
            from research_plotter.plotting.comparison import comparison_plot
            from research_plotter.plotting.convergence import convergence_plot
            figure = comparison_plot(grid, {"reference": clean_reference[:, state_index], "observations": observed[:, state_index], "prediction": result.prediction[:, state_index]}, xlabel="time", ylabel=state_name, title=f"{mode.title()} PINN · {state_name}")
            st.pyplot(figure)
            figure_paths = _save_figure(figure, run_dir / "figures", "pinn_trajectory", {"workflow": "pinn", "model": model.name, "solver": payload.solver.type, "mode": mode, "state": state_name, "noise_percent": int(noise_percent), "validation_status": "UNVERIFIED"})
            history_frame = pd.DataFrame(result.training.history)
            if not history_frame.empty:
                loss_figure = convergence_plot(history_frame["epoch"], history_frame[["loss", "data_loss", "physics_loss"]])
                st.pyplot(loss_figure)
                figure_paths += _save_figure(loss_figure, run_dir / "figures", "pinn_loss", {"workflow": "pinn", "plot_type": "training_loss", "mode": mode})
            status = "PARTIAL" if mode != "baseline" and result.warnings else "UNVERIFIED"
            details = {"model": model.name, "solver": payload.solver.type, "mode": mode, "state": state_name, "parameters": payload.parameters, "initial_conditions": payload.initial_conditions, "time": payload.time.model_dump(mode="json"), "configuration": str(configuration_path), "noise_percent": int(noise_percent), "initial_loss": result.initial_loss, "final_loss": result.final_loss, "metrics_vs_clean_reference": clean_metrics, "training_runtime_seconds": result.training.runtime_seconds, "status": status, "warnings": result.warnings, "data": str(prediction_path), "history": str(history_path), "checkpoint": str(checkpoint), "figures": [str(path) for path in figure_paths]}
            _record_run(run_id, run_dir, "pinn", started, status="SUCCEEDED", details=details)
            st.subheader("Training and comparison")
            summary_cols = st.columns(4)
            summary_cols[0].metric("Initial loss", f"{result.initial_loss:.4g}" if result.initial_loss is not None else "n/a")
            summary_cols[1].metric("Final loss", f"{result.final_loss:.4g}" if result.final_loss is not None else "n/a")
            summary_cols[2].metric("Training time", f"{result.training.runtime_seconds:.3f} s")
            summary_cols[3].metric("Validation", status)
            st.write("**Metrics vs clean reference**", clean_metrics)
            st.dataframe(history_frame, use_container_width=True, hide_index=True)
            st.download_button("Download predictions CSV", prediction_path.read_bytes(), file_name=prediction_path.name, mime="text/csv", key="pinn_prediction_csv")
            _show_downloads(st, figure_paths, "pinn_figures")
            st.download_button("Download model checkpoint", checkpoint.read_bytes(), file_name=checkpoint.name, mime="application/octet-stream", key="pinn_checkpoint")
            for warning in result.warnings:
                st.warning(warning)
        except Exception as error:
            _record_run(run_id, run_dir, "pinn", started, status="FAILED", error=str(error))
            st.error(f"PINN experiment failed.\n\nReason: {error}")


def _render_gallery(st: Any) -> None:
    st.title("Figure Gallery")
    candidates = sorted((path for path in OUTPUTS.rglob("*.png") if path.is_file()), key=lambda path: path.stat().st_mtime, reverse=True) if OUTPUTS.exists() else []
    if not candidates:
        st.info("No figures have been generated yet.")
        return
    candidates = candidates[:300]
    selected = st.selectbox("Select a figure", candidates, format_func=lambda path: str(path.relative_to(OUTPUTS)), key="gallery_figure")
    columns = st.columns([2, 1])
    columns[0].image(str(selected), caption=selected.stem, use_container_width=True)
    metadata_path = selected.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {"metadata_error": "Could not parse sidecar JSON."}
    columns[1].write({
        "Experiment": metadata.get("experiment", selected.parent.name),
        "Source": metadata.get("source", metadata.get("source_config", "Not recorded")),
        "Figure type": metadata.get("plot_type", selected.stem),
        "Created": datetime.fromtimestamp(selected.stat().st_mtime, timezone.utc).isoformat(),
        "Reproduction status": metadata.get("reproducibility_status", metadata.get("status", metadata.get("validation_status", "Not assessed"))),
    })
    formats = _figure_file_map(selected.with_suffix(""))
    _show_downloads(st, formats, "gallery_download")
    possible_data = [selected.with_suffix(".csv"), selected.parent.parent / "simulation_data" / f"{selected.stem}.csv", selected.parent.parent / "simulation.csv", selected.parent.parent / "sweep_results.csv", selected.parent.parent / "predictions.csv", selected.parent.parent / "sensitivity.csv"]
    for data_path in possible_data:
        if data_path.is_file():
            st.download_button("Download associated data", data_path.read_bytes(), file_name=data_path.name, mime="text/csv", key=f"gallery_data_{data_path}")
            break
    if metadata:
        with st.expander("Figure metadata", expanded=True):
            st.json(metadata)
    elif metadata_path.is_file():
        st.caption(f"Metadata file: {metadata_path}")


def _render_history(st: Any) -> None:
    import pandas as pd

    st.title("Experiment History")
    files = sorted(WORKBENCH.rglob("experiment.json"), key=lambda path: path.stat().st_mtime, reverse=True) if WORKBENCH.exists() else []
    rows, records = [], {}
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records[record.get("experiment_id", path.parent.name)] = (record, path)
        details = record.get("details", {})
        rows.append({"Experiment ID": record.get("experiment_id"), "Created": record.get("created_at"), "Workflow": record.get("workflow"), "Model": details.get("model", "—"), "Solver": details.get("solver", "—"), "Status": record.get("status"), "Runtime (s)": record.get("runtime_seconds")})
    if not rows:
        st.info("Workbench runs will be listed here, including clear failure records.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    selected_id = st.selectbox("Experiment details", list(records), key="history_selected")
    record, path = records[selected_id]
    st.json(record)
    config_path = path.parent / "configuration.yaml"
    if config_path.is_file():
        st.download_button("Download run configuration", config_path.read_bytes(), file_name=config_path.name, mime="text/yaml", key="history_config")
        st.button("Load configuration for another run", on_click=_load_run_config, args=(str(config_path),), key="history_rerun_config")
    for artifact in record.get("artifacts", []):
        artifact_path = ROOT / artifact
        if artifact_path.is_file() and artifact_path.suffix.lower() in {".json", ".csv"}:
            st.download_button(f"Download {artifact_path.name}", artifact_path.read_bytes(), file_name=artifact_path.name, key=f"history_artifact_{artifact_path}")


def _load_run_config(path: str) -> None:
    """Streamlit callback: set config and route to the simulation workflow."""
    import streamlit as st

    st.session_state["last_config_path"] = path
    st.session_state["main_navigation"] = "Model Simulation"


def _render_reports(st: Any) -> None:
    st.title("Reports / Export")
    report_roots = [OUTPUTS / "reports", OUTPUTS / "paper_reproduction"]
    reports = sorted((path for root in report_roots if root.exists() for path in root.rglob("*.json") if path.is_file()), key=lambda path: path.stat().st_mtime, reverse=True)
    if not reports:
        st.info("Analysis, simulation, validation, and paper reports will appear here.")
        return
    chosen = st.selectbox("Report", reports[:500], format_func=lambda path: str(path.relative_to(OUTPUTS)), key="report_selected")
    try:
        payload = json.loads(chosen.read_text(encoding="utf-8"))
        st.json(payload)
        st.download_button("Download report JSON", chosen.read_bytes(), file_name=chosen.name, mime="application/json", key="report_download")
        if chosen.name.endswith("profile.json"):
            csv_candidates = [path for path in (OUTPUTS / "uploaded" / "datasets").glob("*") if path.is_file()] if (OUTPUTS / "uploaded" / "datasets").exists() else []
            if csv_candidates:
                st.caption("Dataset source files remain available under outputs/uploaded/datasets.")
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not open report.\n\nReason: {error}")


def _render_figure_reproduction(st: Any) -> None:
    from research_plotter.ui.paper_reproduction_page import render_paper_reproduction_page

    render_paper_reproduction_page(st, root=ROOT, outputs=OUTPUTS, new_run_dir=_new_run_dir, record_run=_record_run)


def _render_equation_plotter(st: Any) -> None:
    from research_plotter.ui.formula_page import render_formula_page

    render_formula_page(st, output_root=OUTPUTS, new_run_dir=_new_run_dir, save_figure=_save_figure, record_run=_record_run)


def _render_graph_image_analyzer(st: Any) -> None:
    from research_plotter.ui.graph_image_page import render_graph_image_page

    render_graph_image_page(st, outputs=OUTPUTS, new_run_dir=_new_run_dir, record_run=_record_run)


def _render_noise(st: Any) -> None:
    from research_plotter.ui.analysis_pages import render_noise_page

    render_noise_page(st, root=ROOT, outputs=OUTPUTS, new_run_dir=_new_run_dir, save_figure=_save_figure, record_run=_record_run)


def _render_convergence(st: Any) -> None:
    from research_plotter.ui.analysis_pages import render_convergence_page

    render_convergence_page(st, load_config=_load_config_file, new_run_dir=_new_run_dir, save_configuration=_save_configuration, save_figure=_save_figure, record_run=_record_run)


def main() -> None:
    try:
        import streamlit as st
    except ImportError as error:  # pragma: no cover - depends on optional UI install
        raise RuntimeError("Install Streamlit to run the Research Workbench: python -m pip install streamlit") from error

    st.set_page_config(page_title="ResearchPlotter Workbench", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    from research_plotter.ui.workbench_shell import render_sidebar, render_theme_styles, render_top_navigation

    render_theme_styles(st)
    with st.sidebar:
        section = render_sidebar(st, NAVIGATION)

    render_top_navigation(st, root=ROOT, outputs=OUTPUTS)

    renderers = {
        "Dashboard": _render_dashboard,
        "Dataset Analyzer": _render_dataset,
        "Paper Analyzer": _render_paper,
        "Figure Reproduction": _render_figure_reproduction,
        "Equation / Formula Plotter": _render_equation_plotter,
        "Graph Image Analyzer": _render_graph_image_analyzer,
        "Model Simulation": _render_simulation,
        "Parameter Sweep": _render_sweep,
        "Sensitivity Analysis": _render_sensitivity,
        "Noise / Uncertainty": _render_noise,
        "Convergence / Validation": _render_convergence,
        "PINN Experiments": _render_pinn,
        "Figure Gallery": _render_gallery,
        "Experiment History": _render_history,
        "Reports / Export": _render_reports,
    }
    renderer = renderers[section]
    try:
        renderer(st)
    except Exception as error:
        st.error(f"{section} could not be loaded.\n\nReason: {error}")
        if st.checkbox("Show diagnostic details", key=f"debug_{section}"):
            st.exception(error)


if __name__ == "__main__":
    main()
