from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

import numpy as np

from research_plotter.analysis.validation import validate_simulation
from research_plotter.analysis.convergence import run_refinement_experiment
from research_plotter.paper_adapters.alzheimer import alzheimer_analysis_notes, build_alzheimer_model
from research_plotter.paper_adapters.cancer import cancer_analysis_notes, build_cancer_model
from research_plotter.paper_adapters.facts import identify_document
from research_plotter.config import Paper2Config, dump_yaml, load_paper2_config
from research_plotter.paper_analysis.paper_parser import parse_paper
from research_plotter.paper_analysis.report_generator import save_analysis_report
from research_plotter.solvers.fractal_fractional import FractalFractionalGMLSolver
from research_plotter.solvers.grunwald_letnikov import GrunwaldLetnikovSolver
from research_plotter.utils.numerics import uniform_time_grid
from research_plotter.plotting.core import save_figure
from research_plotter.plotting.line import multi_line_plot
from research_plotter.pipelines.pinn_experiment import run_pinn_export, run_pinn_noise_export


def _paper_key(path: str | Path) -> str:
    source = Path(path)
    known_key = identify_document(source)
    if known_key:
        return known_key
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source.stem).strip("_") or "document"
    return f"paper_{stem.lower()}"


def analyze_paper(path: str | Path, output_root: str | Path = "outputs/paper_reproduction") -> Path:
    analysis = parse_paper(path)
    destination = Path(output_root) / _paper_key(path) / "paper_analysis.json"
    return save_analysis_report(analysis, destination)


def reproduce_paper(
    path: str | Path,
    output_root: str | Path = "outputs/paper_reproduction",
    initial_conditions: Mapping[str, float] | None = None,
    time_end: float | None = None,
    step: float | None = None,
    config_path: str | Path | None = None,
    paper2_config: Paper2Config | None = None,
    figure_id: str | None = None,
    figure_parameter_values: Mapping[str, list[float]] | None = None,
) -> dict[str, object]:
    source = Path(path); key = _paper_key(source); destination = Path(output_root) / key; destination.mkdir(parents=True, exist_ok=True)
    analysis_path = analyze_paper(source, output_root)
    analysis = parse_paper(source)
    manifest_path = destination / "figure_manifests.json"
    manifest_path.write_text(json.dumps([item.model_dump(mode="json") for item in analysis.figures], indent=2, ensure_ascii=False), encoding="utf-8")
    report: dict[str, object] = {"source_document": str(source), "analysis_report": str(analysis_path), "figure_manifests": str(manifest_path), "status": "BLOCKED", "warnings": [], "experiments": []}
    if key == "paper_1":
        report["paper_notes"] = cancer_analysis_notes()
        report["warnings"] = [
            "BLOCKED: paper does not clearly provide all simulation initial conditions and step-size details.",
            "BLOCKED: displayed recurrence has an index-at-zero ambiguity.",
            "NEEDS_REVIEW: Figure 11 ks and Figure 12 kT sweep values are not tabulated in the source.",
        ]
        report["unresolved_experiments"] = [
            "Figure 11 (ks) and Figure 12 (kT) parameter values are not tabulated in the source."
        ]
        if figure_id is not None and not re.fullmatch(r"figure_\d{2}", figure_id):
            report["warnings"].append(f"BLOCKED: invalid figure identifier {figure_id!r}.")
            report["unresolved_information"] = ["Select a figure identifier present in the paper analysis manifest."]
            (destination / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report
        if initial_conditions is None or time_end is None or step is None:
            report["unresolved_information"] = [
                "Supply initial_conditions, time_end, and step after reviewing paper_analysis.json.",
                "Review the Figure 11 ks and Figure 12 kT sweep values before running those experiments.",
            ]
            (destination / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (destination / "warnings.json").write_text(json.dumps(report["warnings"], indent=2), encoding="utf-8")
            return report
        figure_ids = {item.figure_id for item in analysis.figures}
        if figure_id is not None and figure_id not in figure_ids:
            report["warnings"].append(f"BLOCKED: {figure_id} is not present in the extracted figure manifest.")
            report["unresolved_information"] = ["Choose a figure listed in figure_manifests.json."]
            (destination / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report
        model = build_cancer_model(initial_conditions)
        grid = uniform_time_grid(0, time_end, step)
        solver = FractalFractionalGMLSolver()
        supplied_sweep_parameters = set((figure_parameter_values or {}).keys())
        report["configuration_provenance"] = {
            "initial_conditions": "USER_PROVIDED",
            "time_start": "ASSUMED",
            "time_end": "USER_PROVIDED",
            "time_step": "USER_PROVIDED",
            "published_parameter_table": "SOURCE_PAPER",
            "selected_figure_values": "USER_PROVIDED" if supplied_sweep_parameters else "SOURCE_PAPER",
        }
        result = solver.solve(model, None, initial_conditions, grid, {"fractional_order": .95, "fractal_dimension": .6, "epsilon_origin": True})
        experiment_specs = [("figure_01", {}, {"fractional_order": .95, "fractal_dimension": .6})]
        experiment_specs += [(f"figure_{number:02d}", {"alpha": alpha}, {"fractional_order": alpha, "fractal_dimension": .6}) for number in range(2, 6) for alpha in (.95, .97, 1.0)]
        experiment_specs += [(f"figure_{number:02d}", {"beta": beta}, {"fractional_order": .95, "fractal_dimension": beta}) for number in range(6, 10) for beta in (.6, .7, .8)]
        experiment_specs += [("figure_10", {"delta2": value}, {"fractional_order": .95, "fractal_dimension": .6}) for value in (5.4, 6.4, 7.4)]
        report["unresolved_experiments"] = [
            "Figure 11 (ks) and Figure 12 (kT) parameter values are not tabulated in the source."
        ]
        if figure_id is not None:
            number = int(figure_id.removeprefix("figure_"))
            varying_parameter = (
                "alpha" if 2 <= number <= 5 else
                "beta" if 6 <= number <= 9 else
                "delta2" if number == 10 else
                "ks" if number == 11 else
                "kT" if number == 12 else None
            )
            user_values = list((figure_parameter_values or {}).get(varying_parameter, [])) if varying_parameter else []
            known_values: dict[int, list[float]] = {
                **{item: [.95, .97, 1.0] for item in range(2, 6)},
                **{item: [.6, .7, .8] for item in range(6, 10)},
                10: [5.4, 6.4, 7.4],
            }
            selected_values = user_values or known_values.get(number, [])
            if number in {11, 12} and not selected_values:
                report["unresolved_information"] = [f"Supply explicitly reviewed user values for Figure {number} ({varying_parameter}); the source does not tabulate them."]
                report["warnings"].append(f"NEEDS_REVIEW: Figure {number} sweep values are required and were not supplied.")
                (destination / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                (destination / "warnings.json").write_text(json.dumps(report["warnings"], indent=2), encoding="utf-8")
                return report
            if varying_parameter:
                experiment_specs = [
                    (figure_id, {varying_parameter: value}, {
                        "fractional_order": value if varying_parameter == "alpha" else .95,
                        "fractal_dimension": value if varying_parameter == "beta" else .6,
                        "epsilon_origin": True,
                    })
                    for value in selected_values
                ]
            else:
                experiment_specs = [(figure_id, {}, {"fractional_order": .95, "fractal_dimension": .6, "epsilon_origin": True})]
            report["selected_figure"] = figure_id
            report["selected_figure_parameter"] = varying_parameter
            report["selected_figure_values"] = selected_values
            report["selected_figure_values_provenance"] = "USER_PROVIDED" if user_values else "SOURCE_PAPER"
        for figure_id, overrides, options in experiment_specs:
            experiment_parameters = {name: value for name, value in overrides.items() if name in model.parameters}
            experiment = solver.solve(model, experiment_parameters or None, initial_conditions, grid, {**options, "epsilon_origin": True})
            item: dict[str, object] = {"figure_id": figure_id, "parameter_overrides": overrides, "solver": experiment.solver_name, "warnings": experiment.warnings}
            if np.isfinite(experiment.states).all():
                data_file = destination / "simulation_data" / f"{figure_id}_{len(report['experiments']):03d}.csv"; data_file.parent.mkdir(parents=True, exist_ok=True); experiment.save_csv(str(data_file))
                figure = multi_line_plot(experiment.time, experiment.to_frame().drop(columns="time"), xlabel="time", ylabel="state", title=figure_id)
                figure_files = save_figure(figure, destination / "figures" / f"{figure_id}_{len(report['experiments']):03d}.png", formats=["png", "svg"])
                metadata_file = destination / "figures" / f"{figure_id}_{len(report['experiments']):03d}.json"
                metadata_file.write_text(json.dumps({
                    "figure_id": figure_id,
                    "source_document": str(source),
                    "solver": experiment.solver_name,
                    "parameter_overrides": overrides,
                    "parameter_overrides_provenance": report.get("selected_figure_values_provenance", "SOURCE_PAPER"),
                    "configuration_provenance": report.get("configuration_provenance", {}),
                    "solver_options": options,
                    "time_grid": {"start": float(grid[0]), "end": float(grid[-1]), "step": float(grid[1] - grid[0])},
                    "reproducibility_status": "UNVERIFIED",
                    "warnings": experiment.warnings,
                }, indent=2), encoding="utf-8")
                item.update({"simulation_data": str(data_file), "figures": [str(path) for path in figure_files], "figure_metadata": str(metadata_file), "status": "UNVERIFIED"})
            else:
                item["status"] = "BLOCKED"
            report["experiments"].append(item)
    elif key == "paper_2":
        report["paper_notes"] = alzheimer_analysis_notes()
        config = paper2_config or (load_paper2_config(config_path) if config_path is not None else None)
        report["warnings"] = [
            "PARTIAL: Paper-specified controlled trajectories and exact PINN comparison cannot be reproduced because control equations and training hyperparameters are incomplete; user-configured runs remain exploratory.",
            "PARTIAL: the source does not sufficiently specify NSFD update/denominator details; no NSFD execution is claimed.",
        ]
        if config is not None:
            configuration_path = destination / "configuration.yaml"
            dump_yaml(config, configuration_path)
            report["configuration"] = str(configuration_path)
            report["configuration_provenance"] = config.resolved_provenance()
        selected_initial = (config.initial_conditions if config and config.initial_conditions else initial_conditions)
        selected_parameters = config.parameters if config else None
        config_model = config.model if config else {}
        solver_options = dict(config.solver.options) if config and config.solver else {}
        raw_model_order = config_model.get("fractional_order")
        raw_solver_order = solver_options.get("fractional_order")
        parsed_orders: dict[str, float] = {}
        invalid_order_details = []
        for label, raw_value in (("model.fractional_order", raw_model_order), ("solver.options.fractional_order", raw_solver_order)):
            if raw_value is None:
                continue
            try:
                parsed_value = float(raw_value)
            except (TypeError, ValueError):
                invalid_order_details.append(f"{label} must be a finite number in (0, 1].")
                continue
            if not np.isfinite(parsed_value) or not 0 < parsed_value <= 1:
                invalid_order_details.append(f"{label} must be a finite number in (0, 1].")
                continue
            parsed_orders[label] = parsed_value
        order_value = parsed_orders.get("model.fractional_order", parsed_orders.get("solver.options.fractional_order"))
        selected_end = time_end if time_end is not None else (config.time.end if config and config.time else None)
        selected_step = step if step is not None else (config.time.step if config and config.time else None)
        selected_start = config.time.start if config and config.time else 0.0
        solver_type = config.solver.type.lower() if config and config.solver else None
        model_order_value = parsed_orders.get("model.fractional_order")
        solver_order_value = parsed_orders.get("solver.options.fractional_order")

        provenance = dict(report.get("configuration_provenance", {
            "model": "SOURCE_PAPER",
            "parameters": "SOURCE_PAPER",
            "initial_conditions": "SOURCE_PAPER",
            "time": "UNKNOWN",
            "solver": "UNKNOWN",
            "fractional_order": "UNKNOWN",
            "controls": "UNKNOWN",
            "nsfd": "UNKNOWN",
            "pinn": "UNKNOWN",
        }))
        if time_end is not None or step is not None:
            provenance["time"] = "USER_PROVIDED"
        report["configuration_provenance"] = provenance

        missing_configuration = []
        if selected_end is None:
            missing_configuration.append("time.end")
        if selected_step is None:
            missing_configuration.append("time.step")
        if order_value is None:
            missing_configuration.append("fractional order")
        if solver_type is None:
            missing_configuration.append("solver type")
        order_conflict = (
            model_order_value is not None
            and solver_order_value is not None
            and not np.isclose(float(model_order_value), float(solver_order_value))
        )
        unsupported_solver = solver_type is not None and solver_type not in {
            "grunwald_letnikov", "grunwald-letnikov", "gl"
        }
        if missing_configuration or unsupported_solver or order_conflict or invalid_order_details:
            details = []
            if missing_configuration:
                details.append("Missing explicit configuration: " + ", ".join(missing_configuration) + ".")
            if unsupported_solver:
                details.append(
                    f"Solver {solver_type!r} is not executable by the Paper 2 adapter; "
                    "only the configured Grünwald-Letnikov comparison path is available, and NSFD remains unsupported."
                )
            if order_conflict:
                details.append(
                    "model.fractional_order conflicts with solver.options.fractional_order; "
                    "use one consistent value before running."
                )
            details.extend(invalid_order_details)
            details.append(
                "No numerical simulation was run. The paper does not establish the omitted baseline grid/order settings; "
                "supply reviewed values in a YAML configuration and retain their provenance."
            )
            report.update({
                "status": "PARTIAL",
                "simulation_status": "BLOCKED",
                "unresolved_information": details,
            })
            report["warnings"].append("BLOCKED: Paper 2 numerical execution requires explicit time range, step, fractional order, and supported solver configuration.")
            (destination / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (destination / "warnings.json").write_text(json.dumps(report["warnings"], indent=2), encoding="utf-8")
            return report

        fractional_order = float(order_value)
        model_provenance = config.provenance.get("model") if config else None
        order_provenance = provenance.get("fractional_order", "UNKNOWN")
        model_metadata = {
            "configuration_provenance": model_provenance.value if model_provenance is not None else "UNKNOWN",
            "fractional_order_provenance": order_provenance,
        }
        model = build_alzheimer_model(selected_initial, fractional_order=fractional_order, parameters=selected_parameters, metadata=model_metadata)
        grid = uniform_time_grid(selected_start, selected_end, selected_step)
        solver_options.setdefault("fractional_order", fractional_order)
        result = GrunwaldLetnikovSolver().solve(model, None, selected_initial, grid, solver_options)
        baseline_validation = validate_simulation(result, model, require_nonnegative=True)
        configuration_payload = {
            "source_document": str(source),
            "provenance": report.get("configuration_provenance", {"model": "SOURCE_PAPER", "parameters": "SOURCE_PAPER", "initial_conditions": "SOURCE_PAPER", "fractional_order": "SOURCE_PAPER"}),
            "model": model.metadata | {"fractional_order": fractional_order},
            "parameters": result.parameters,
            "initial_conditions": result.initial_conditions,
            "time": {"start": float(grid[0]), "end": float(grid[-1]), "step": float(grid[1] - grid[0])},
            "solver": {"name": result.solver_name, "options": solver_options},
            "controls": config.controls.model_dump(mode="json") if config else {"status": "BLOCKED", "provenance": "UNKNOWN"},
            "nsfd": config.nsfd.model_dump(mode="json") if config else {"status": "PARTIAL", "provenance": "UNKNOWN"},
            "pinn": config.pinn.model_dump(mode="json") if config else {"enabled": False, "provenance": "UNKNOWN"},
            "run_convergence": config.run_convergence if config else False,
            "refinements": config.refinements if config else [1, 2, 4],
        }
        configuration_json = destination / "configuration.json"
        configuration_json.write_text(json.dumps(configuration_payload, indent=2, default=str), encoding="utf-8")
        report["configuration_json"] = str(configuration_json)
        # Keep the source-paper limitation visible even when optional user
        # experiments are requested.  Additional run-specific warnings are
        # appended below and must not be overwritten.
        report["warnings"] = [
            "PARTIAL: Paper-specified controlled trajectories and exact PINN comparison cannot be reproduced because control equations and training hyperparameters are incomplete; user-configured runs remain exploratory.",
            "PARTIAL: the source does not sufficiently specify NSFD update/denominator details; no NSFD execution is claimed.",
        ]
        if config and config.run_convergence:
            try:
                refinement = run_refinement_experiment(model, GrunwaldLetnikovSolver(), start=selected_start, end=selected_end, base_step=selected_step, refinements=config.refinements, parameters=selected_parameters, initial_conditions=selected_initial, options=solver_options)
                report["convergence"] = refinement.to_dict() | {"outputs": refinement.save(destination / "convergence")}
            except ValueError as error:
                report["warnings"].append(f"BLOCKED: convergence experiment could not run: {error}")
        if config:
            if config.controls.equations:
                if set(config.controls.equations) == set(model.state_names) and config.controls.provenance.value == "USER_PROVIDED":
                    missing_controls = config.controls.missing_values(config.controls.equations)
                    if missing_controls:
                        report["warnings"].append("BLOCKED: controlled equations reference control variables without user-provided numeric values: " + ", ".join(missing_controls))
                    else:
                        controlled_model = build_alzheimer_model(selected_initial, fractional_order=fractional_order, parameters=selected_parameters, additional_parameters=config.controls.values, expressions=config.controls.equations, metadata={"control_equations_provenance": "USER_PROVIDED", "control_values_provenance": "USER_PROVIDED"})
                        controlled = GrunwaldLetnikovSolver().solve(controlled_model, None, selected_initial, grid, solver_options)
                        controlled_path = destination / "simulation_data" / "controlled.csv"
                        controlled.save_csv(str(controlled_path))
                        controlled_validation = validate_simulation(controlled, controlled_model, require_nonnegative=True)
                        controlled_status = "UNVERIFIED" if controlled_validation["status"] == "VALID" else "INVALID"
                        report["experiments"].append({
                            "name": "controlled_gl",
                            "status": controlled_status,
                            "simulation_data": str(controlled_path),
                            "provenance": "USER_PROVIDED",
                            "control_values": config.controls.values,
                            "validation": controlled_validation,
                            "diagnostics": controlled.diagnostics,
                        })
                        if controlled_status == "INVALID":
                            report["warnings"].append("INVALID: the user-configured controlled GL trajectory failed numerical validation; its CSV is retained as a diagnostic, not a validated result.")
                else:
                    report["warnings"].append("BLOCKED: controlled equations must define every state and be marked USER_PROVIDED before controlled simulation.")
            if config.nsfd.enabled:
                report["warnings"].append("BLOCKED: an NSFD solver is not implemented; supplied NSFD settings were recorded but not executed.")
            if config.pinn.enabled:
                if baseline_validation["status"] != "VALID":
                    report["pinn"] = {
                        "status": "BLOCKED",
                        "warnings": ["The baseline trajectory failed finite/positivity validation; it cannot be used as PINN reference data."],
                    }
                    report["warnings"].append("BLOCKED: PINN training was not started because the configured Paper 2 baseline trajectory is numerically invalid.")
                else:
                    pinn_report = run_pinn_export(model, result, config.pinn, destination / "pinn", mode=str(config_model.get("pinn_mode", "fractional")), fractional_order=fractional_order)
                    report["pinn"] = pinn_report
                    if pinn_report.get("status") == "BLOCKED":
                        report["warnings"].extend(pinn_report.get("warnings", []))
                    else:
                        noise_report = run_pinn_noise_export(model, result, config.pinn, config.noise_levels, destination / "pinn_noise", mode=str(config_model.get("pinn_mode", "fractional")), fractional_order=fractional_order)
                        report["pinn_noise"] = noise_report
                        if noise_report.get("status") == "BLOCKED":
                            report["warnings"].extend(noise_report.get("warnings", []))
    else:
        report["warnings"] = ["No paper-specific adapter is registered for this document."]
        (destination / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    data_path = destination / "simulation_data" / "baseline.csv"; data_path.parent.mkdir(parents=True, exist_ok=True); result.save_csv(str(data_path))
    validation = validate_simulation(result, model, require_nonnegative=True)
    report.update({"status": "PARTIAL" if report["warnings"] else "UNVERIFIED", "simulation_status": validation["status"], "simulation_data": str(data_path), "solver": result.solver_name, "validation": validation, "parameters": result.parameters, "initial_conditions": result.initial_conditions, "time_grid": {"start": float(result.time[0]), "end": float(result.time[-1]), "step": float(result.time[1] - result.time[0])}, "diagnostics": result.diagnostics})
    if key == "paper_2":
        if validation["status"] == "VALID":
            baseline_figure = multi_line_plot(result.time, result.to_frame().drop(columns="time"), xlabel="time", ylabel="state", title="Paper 2 configured baseline · partial")
            baseline_figure_paths = save_figure(baseline_figure, destination / "figures" / "partial_baseline.png", formats=["png", "svg", "pdf"])
            baseline_metadata = {"workflow": "paper_reproduction", "source_document": str(source), "plot_type": "multi_line", "status": "PARTIAL", "simulation_status": validation["status"], "solver": result.solver_name, "configuration": str(report.get("configuration_json") or report.get("configuration") or "configured baseline"), "warning": "Partial configured baseline, not a selected published figure reproduction."}
            (destination / "figures" / "partial_baseline.json").write_text(json.dumps(baseline_metadata, indent=2), encoding="utf-8")
            report["baseline_figures"] = [str(path) for path in baseline_figure_paths]
        else:
            report["baseline_figures"] = []
            report["warnings"].append("INVALID: the configured Paper 2 baseline failed numerical validation; no baseline figure was generated. The saved CSV is diagnostic data only.")
            report["numerical_failure_diagnostics"] = {
                "status": validation["status"],
                "completed_steps": result.diagnostics.get("completed_steps"),
                "number_of_steps": result.diagnostics.get("number_of_steps"),
                "failure_index": result.diagnostics.get("failure_index"),
                "time_step": result.diagnostics.get("time_step"),
                "minimum_finite_state": validation.get("min_state"),
                "interpretation": "The explicit GL configuration failed numerical validation. This does not establish failure of the scientific model and does not justify inferring a replacement step size.",
            }
    (destination / "validation_data.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (destination / "warnings.json").write_text(json.dumps(report["warnings"], indent=2), encoding="utf-8")
    (destination / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
