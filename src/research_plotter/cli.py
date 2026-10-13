from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from research_plotter.analysis.parameter_sweep import run_parameter_sweep
from research_plotter.analysis.sensitivity import logarithmic_sensitivity
from research_plotter.analysis.convergence import run_refinement_experiment
from research_plotter.config import load_experiment
from research_plotter.models import ModelSpec, Parameter, StateVariable
from research_plotter.pipelines.dataset_analysis import analyze_dataset, load_dataset
from research_plotter.pipelines.figure_generation import recommend_dataset_figures
from research_plotter.pipelines.paper_reproduction import analyze_paper, reproduce_paper
from research_plotter.plotting.line import multi_line_plot
from research_plotter.solvers import ExplicitEulerSolver, GrunwaldLetnikovSolver, ODESolver
from research_plotter.utils.numerics import uniform_time_grid

app = typer.Typer(add_completion=False, no_args_is_help=True, help="General-purpose scientific modeling and visualization")


def _solver(name: str):
    normalized = name.lower().replace("-", "_")
    if normalized in {"ode", "solve_ivp", "scipy"}: return ODESolver()
    if normalized in {"euler", "explicit_euler"}: return ExplicitEulerSolver()
    if normalized in {"grunwald_letnikov", "gl", "fractional"}: return GrunwaldLetnikovSolver()
    raise typer.BadParameter(f"Unsupported solver: {name}")


def _configured_model(payload):
    model_config = payload.model
    return ModelSpec.from_expressions(
        name=model_config.get("name", "configured_model"),
        state_variables=[StateVariable(name=name) for name in model_config["states"]],
        parameters=[Parameter(name=name, value=value) for name, value in payload.parameters.items()],
        expressions=model_config["equations"],
        initial_conditions=payload.initial_conditions,
        derivative_type=model_config.get("derivative_type", "ordinary"),
        fractional_order=model_config.get("fractional_order"),
    )


@app.command("analyze-paper")
def analyze_paper_command(path: Path, output_root: Path = Path("outputs/paper_reproduction")):
    destination = analyze_paper(path, output_root)
    typer.echo(f"Saved evidence-marked analysis: {destination}")


@app.command("analyze-data")
def analyze_data_command(path: Path, output_dir: Path = Path("outputs/reports")):
    frame, profile, destination = analyze_dataset(path, output_dir)
    typer.echo(json.dumps(profile.model_dump(mode="json"), indent=2))
    typer.echo(f"Saved report: {destination}")


@app.command("plot")
def plot_command(path: Path, output_dir: Path = Path("outputs/figures")):
    frame, profile, _ = analyze_dataset(path)
    created = recommend_dataset_figures(frame, profile, output_dir)
    typer.echo("\n".join(str(item) for item in created) or "No recommended figure could be generated from this dataset.")


@app.command("reproduce")
def reproduce_command(
    path: Path,
    output_root: Path = Path("outputs/paper_reproduction"),
    initial_conditions: str | None = typer.Option(None, help="JSON mapping supplied when a paper omits initial conditions, e.g. '{\"S\":0.9,\"E\":0.5,\"T\":0.9,\"C\":0}'"),
    time_end: float | None = typer.Option(None, help="Simulation end time for an exploratory paper run."),
    step: float | None = typer.Option(None, help="Uniform simulation step for an exploratory paper run."),
    config: Path | None = typer.Option(None, help="Optional Paper 2 YAML override configuration."),
):
    supplied_initial_conditions = None
    if initial_conditions:
        try:
            payload = json.loads(initial_conditions)
        except json.JSONDecodeError as error:
            raise typer.BadParameter("initial-conditions must be valid JSON") from error
        if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
            raise typer.BadParameter("initial-conditions must be a JSON object")
        supplied_initial_conditions = {key: float(value) for key, value in payload.items()}
    report = reproduce_paper(path, output_root, supplied_initial_conditions, time_end, step, config_path=config)
    typer.echo(json.dumps(report, indent=2))


@app.command("sweep")
def sweep_command(config: Path):
    payload = load_experiment(config)
    model = _configured_model(payload)
    solver = _solver(payload.solver.type)
    grid = uniform_time_grid(payload.time.start, payload.time.end, payload.time.step)
    values = payload.model.get("sweep", {})
    result = run_parameter_sweep(model, solver, grid, values, payload.initial_conditions, payload.solver.options, progress=True)
    typer.echo(f"Completed {len(result.simulations)} simulations")


@app.command("sensitivity")
def sensitivity_command(config: Path, parameter: str, output: str | None = None):
    payload = load_experiment(config)
    model = _configured_model(payload)
    result = logarithmic_sensitivity(model, _solver(payload.solver.type), uniform_time_grid(payload.time.start, payload.time.end, payload.time.step), parameter, output=output, options=payload.solver.options)
    typer.echo(json.dumps({"parameter": parameter, "output": output, "delta": result["delta"]}, indent=2))


@app.command("convergence")
def convergence_command(
    config: Path,
    refinements: str = typer.Option("1,2,4", help="Comma-separated refinement factors."),
    output_dir: Path = typer.Option(Path("outputs/reports/convergence"), help="Directory for trajectories, metrics, and figures."),
):
    """Run a solver grid-refinement experiment and save its report."""
    payload = load_experiment(config)
    model = _configured_model(payload)
    try:
        factors = tuple(int(value.strip()) for value in refinements.split(",") if value.strip())
    except ValueError as error:
        raise typer.BadParameter("refinements must be comma-separated positive integers") from error
    result = run_refinement_experiment(
        model,
        _solver(payload.solver.type),
        start=payload.time.start,
        end=payload.time.end,
        base_step=payload.time.step,
        refinements=factors,
        options=payload.solver.options,
    )
    outputs = result.save(output_dir)
    typer.echo(json.dumps({"status": result.status, "observed_orders": result.observed_orders, "outputs": outputs}, indent=2, default=str))
