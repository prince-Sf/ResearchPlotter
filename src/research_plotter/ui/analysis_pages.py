"""Streamlit views for existing convergence and noise-analysis services."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd


def render_convergence_page(st: Any, *, load_config: Callable[..., Path | None], new_run_dir: Callable[..., Any], save_configuration: Callable[..., Any], save_figure: Callable[..., Any], record_run: Callable[..., Any]) -> None:
    st.title("Convergence / Validation")
    st.write("Compare solver outputs under time-grid refinement. The finest requested grid is a numerical reference, not an exact solution.")
    config_path = load_config(st, "convergence")
    if config_path is None:
        st.info("Choose or upload a model configuration first.")
        return
    try:
        from research_plotter.config import load_experiment, SolverConfig, TimeConfig
        from research_plotter.cli import _configured_model, _solver

        payload = load_experiment(config_path)
        cols = st.columns(3)
        start = cols[0].number_input("Start", value=float(payload.time.start), key="convergence_start")
        end = cols[1].number_input("End", value=float(payload.time.end), min_value=float(start) + 1e-9, key="convergence_end")
        step = cols[2].number_input("Base time step", value=float(payload.time.step), min_value=1e-8, format="%.6g", key="convergence_step")
        solver_names = ["ode", "explicit_euler", "grunwald_letnikov"]
        selected_solver = st.selectbox("Solver", solver_names, index=solver_names.index(payload.solver.type) if payload.solver.type in solver_names else 0, key="convergence_solver")
        refinement_text = st.text_input("Refinement factors", value="1, 2, 4", help="Factors n mean steps dt/n.", key="convergence_factors")
        solver_options = dict(payload.solver.options)
        if selected_solver == "grunwald_letnikov":
            fallback = float(solver_options.get("fractional_order", payload.model.get("fractional_order", .9)))
            solver_options["fractional_order"] = st.number_input("GL fractional order", min_value=.01, max_value=1.0, value=min(1.0, max(.01, fallback)), key="convergence_order")
        model_payload = payload.model_copy(update={"time": TimeConfig(start=float(start), end=float(end), step=float(step)), "solver": SolverConfig(type=selected_solver, options=solver_options)})
        model = _configured_model(model_payload)
        factors = tuple(int(item.strip()) for item in refinement_text.split(",") if item.strip())
    except Exception as error:
        st.error(f"Convergence configuration is invalid.\n\nReason: {error}")
        return
    if st.button("Run refinement experiment", type="primary", key="run_convergence"):
        from research_plotter.analysis.convergence import run_refinement_experiment, refinement_plot

        run_id, run_dir = new_run_dir("convergence")
        started = time.perf_counter()
        try:
            with st.spinner("Running solver at each requested grid resolution…"):
                result = run_refinement_experiment(model, _solver(selected_solver), start=float(start), end=float(end), base_step=float(step), refinements=factors, parameters=payload.parameters, initial_conditions=payload.initial_conditions, options=solver_options)
            outputs = result.save(run_dir / "results")
            config_out = save_configuration(run_dir / "configuration.yaml", {"source_config": str(config_path), "model": model.name, "solver": selected_solver, "time": {"start": start, "end": end, "base_step": step}, "refinement_factors": factors, "parameters": payload.parameters, "initial_conditions": payload.initial_conditions, "solver_options": solver_options, "provenance": "USER_PROVIDED_CONFIGURATION"})
            figure = refinement_plot(result)
            figure_files = save_figure(figure, run_dir / "figures", "convergence", {"workflow": "convergence", "solver": selected_solver, "reference_factor": result.reference_factor, "status": result.status, "configuration": str(config_out)})
            table = pd.DataFrame([{"factor": factor, "step": result.steps[index], **result.errors[factor], "observed_order_estimate": result.observed_orders.get(factor)} for index, factor in enumerate(result.factors)])
            table_path = run_dir / "refinement_metrics.csv"
            table.to_csv(table_path, index=False)
            report_path = run_dir / "convergence_report.json"
            report_path.write_text(json.dumps(result.to_dict() | {"outputs": outputs, "configuration": str(config_out)}, indent=2, default=str), encoding="utf-8")
            record_run(run_id, run_dir, "convergence", started, status=result.status, details={"model": model.name, "solver": selected_solver, "configuration": str(config_out), "report": str(report_path), "data": str(table_path), "figures": [str(path) for path in figure_files], "warnings": result.warnings})
            st.session_state["latest_convergence"] = {"run_dir": str(run_dir), "result": result.to_dict(), "metrics": str(table_path), "figures": [str(path) for path in figure_files]}
        except Exception as error:
            record_run(run_id, run_dir, "convergence", started, status="FAILED", error=str(error))
            st.error(f"Refinement experiment failed.\n\nReason: {error}\n\nCheck that the end time is divisible by every requested refinement step.")
    latest = st.session_state.get("latest_convergence")
    if latest:
        result = latest["result"]
        st.subheader(f"Status: {result['status']}")
        st.caption("Observed order estimates are descriptive estimates for this run only; no theoretical convergence order is asserted.")
        st.dataframe(pd.read_csv(latest["metrics"]), use_container_width=True, hide_index=True)
        for figure in latest["figures"]:
            if Path(figure).suffix == ".png" and Path(figure).is_file():
                st.image(figure, caption="RMSE against the finest requested grid", use_container_width=True)
        for warning in result.get("warnings", []):
            st.warning(warning)
        st.download_button("Download convergence metrics CSV", Path(latest["metrics"]).read_bytes(), file_name="refinement_metrics.csv", mime="text/csv", key="convergence_metrics_download")


def render_noise_page(st: Any, *, root: Path, outputs: Path, new_run_dir: Callable[..., Any], save_figure: Callable[..., Any], record_run: Callable[..., Any]) -> None:
    st.title("Noise / Uncertainty")
    st.write("Apply configurable synthetic Gaussian perturbations to a selected observed data column. The clean source data is retained separately and is never overwritten.")
    source_kind = st.radio("Dataset source", ["Bundled example", "Upload dataset"], horizontal=True, key="noise_source_kind")
    source: Path | None = None
    if source_kind == "Bundled example":
        examples = sorted((root / "data" / "examples").glob("*.csv"))
        if examples:
            source = st.selectbox("Example dataset", examples, format_func=lambda path: path.name, key="noise_example")
    else:
        upload = st.file_uploader("CSV, TXT table, Excel, JSON, or Parquet dataset", type=["csv", "txt", "xlsx", "xls", "json", "parquet"], key="noise_dataset_upload")
        if upload is not None:
            from research_plotter.utils.paths import persist_uploaded_file

            source = persist_uploaded_file(upload, "datasets", output_root=outputs)
    if source is None:
        st.info("Select or upload a dataset to begin.")
        return
    try:
        from research_plotter.pipelines.dataset_analysis import load_dataset

        frame = load_dataset(source)
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            st.warning("This dataset has no numeric column for a Gaussian-noise experiment.")
            return
        value_column = st.selectbox("Observed numeric column", numeric, key="noise_value_column")
        candidates = [name for name in frame.columns if name != value_column]
        default_x = next((name for name in candidates if "time" in str(name).lower() or "date" in str(name).lower()), None)
        x_options = ["Row index", *candidates]
        x_choice = st.selectbox("X axis", x_options, index=(x_options.index(default_x) if default_x in x_options else 0), key="noise_x_axis")
        levels_text = st.text_input("Noise levels in percent", value="0, 5, 10, 15, 20", key="noise_levels")
        seed = int(st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, key="noise_seed"))
        levels = [float(value.strip()) / 100.0 for value in levels_text.split(",") if value.strip()]
        if not levels or not np.isfinite(levels).all() or any(level < 0 for level in levels):
            raise ValueError("Provide one or more finite, non-negative noise percentages")
        x_values = np.arange(len(frame)) if x_choice == "Row index" else frame[x_choice]
        keep = frame[value_column].notna() & np.isfinite(frame[value_column].astype(float))
        clean = frame.loc[keep, value_column].to_numpy(dtype=float)
        x_values = np.asarray(x_values)[keep.to_numpy()]
        if clean.size < 2:
            raise ValueError("At least two finite numeric observations are required")
    except Exception as error:
        st.error(f"Noise experiment input is invalid.\n\nReason: {error}")
        return
    st.caption("Noise levels are fractions of the selected clean column's standard deviation; experiments are exploratory data perturbations, not simulated measurement evidence.")
    if st.button("Run noise experiment", type="primary", key="run_noise"):
        from research_plotter.analysis.noise import noise_experiment
        from research_plotter.plotting.line import multi_line_plot

        run_id, run_dir = new_run_dir("noise")
        started = time.perf_counter()
        try:
            with st.spinner("Generating separate noisy observation series and metrics…"):
                rows = noise_experiment(clean, levels=levels, seed=seed)
            curves: dict[str, Any] = {"clean reference": clean}
            long_rows = []
            metrics_rows = []
            for item in rows:
                percent = 100.0 * item["noise_level"]
                curves[f"noise {percent:g}%"] = item["values"]
                metrics_rows.append({key: value for key, value in item.items() if key != "values"})
                long_rows.extend({"x": x, "reference": float(reference), "observation": float(observation), "noise_percent": percent} for x, reference, observation in zip(x_values, clean, item["values"]))
            figure = multi_line_plot(x_values, curves, xlabel=x_choice if x_choice != "Row index" else "row", ylabel=value_column, title=f"Synthetic noise levels · {value_column}")
            st.pyplot(figure)
            figure_paths = save_figure(figure, run_dir / "figures", "noise_comparison", {"workflow": "noise", "source": str(source), "value_column": value_column, "levels": levels, "seed": seed, "status": "EXPLORATORY"})
            data_path = run_dir / "noise_observations.csv"
            pd.DataFrame(long_rows).to_csv(data_path, index=False)
            metrics_path = run_dir / "noise_metrics.json"
            for metric_row in metrics_rows:
                for key, value in list(metric_row.items()):
                    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                        metric_row[key] = None
            metrics_path.write_text(json.dumps(metrics_rows, indent=2, allow_nan=False), encoding="utf-8")
            config_path = run_dir / "configuration.json"
            config_path.write_text(json.dumps({"source": str(source), "value_column": value_column, "x_axis": x_choice, "levels": levels, "seed": seed, "scale": "fraction of selected clean data standard deviation", "clean_data_unchanged": True, "status": "EXPLORATORY"}, indent=2), encoding="utf-8")
            record_run(run_id, run_dir, "noise", started, status="EXPLORATORY", details={"source": str(source), "column": value_column, "configuration": str(config_path), "data": str(data_path), "metrics": str(metrics_path), "figures": [str(path) for path in figure_paths]})
            st.session_state["latest_noise"] = {"metrics": metrics_rows, "data": str(data_path), "figures": [str(path) for path in figure_paths], "clean": clean.tolist()}
        except Exception as error:
            record_run(run_id, run_dir, "noise", started, status="FAILED", error=str(error))
            st.error(f"Noise experiment failed.\n\nReason: {error}")
    latest = st.session_state.get("latest_noise")
    if latest:
        metrics = pd.DataFrame(latest["metrics"])
        metrics["noise_level"] *= 100.0
        st.dataframe(metrics.rename(columns={"noise_level": "noise_percent (%)"}), use_container_width=True, hide_index=True)
        st.download_button("Download noise metrics JSON", json.dumps(latest["metrics"], indent=2), file_name="noise_metrics.json", mime="application/json", key="noise_metrics_download")
        st.download_button("Download noisy observations CSV", Path(latest["data"]).read_bytes(), file_name="noise_observations.csv", mime="text/csv", key="noise_data_download")
        st.caption("The clean reference series was copied before perturbation; only the exported observation column contains synthetic noise.")
