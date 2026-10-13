"""Grid-refinement experiments for ordinary and fractional solvers.

The experiment deliberately reports observed error ratios instead of claiming a
theoretical order.  The finest requested grid is a numerical reference only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from research_plotter.models.model_spec import ModelSpec
from research_plotter.plotting.core import save_figure
from research_plotter.solvers.base import SimulationResult, Solver
from research_plotter.utils.numerics import error_metrics, uniform_time_grid


def _sample_states(reference: SimulationResult, time: np.ndarray) -> np.ndarray:
    """Interpolate a reference trajectory at another grid's time points."""
    return np.column_stack([
        np.interp(time, reference.time, reference.states[:, column])
        for column in range(reference.states.shape[1])
    ])


@dataclass
class RefinementResult:
    factors: list[int]
    steps: list[float]
    simulations: dict[int, SimulationResult]
    errors: dict[int, dict[str, float]]
    observed_orders: dict[int, float | None]
    reference_factor: int
    status: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factors": self.factors,
            "steps": self.steps,
            "reference_factor": self.reference_factor,
            "errors": self.errors,
            "observed_orders": {str(key): value for key, value in self.observed_orders.items()},
            "status": self.status,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "simulations": {
                str(factor): simulation.to_metadata()
                for factor, simulation in self.simulations.items()
            },
        }

    def save(self, directory: str | Path, *, make_plot: bool = True) -> dict[str, str]:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, str] = {}
        for factor, simulation in self.simulations.items():
            path = destination / f"simulation_factor_{factor}.csv"
            simulation.save_csv(str(path))
            outputs[f"simulation_{factor}"] = str(path)
        report_path = destination / "convergence.json"
        report_path.write_text(json.dumps(self.to_dict(), indent=2, default=str, allow_nan=False), encoding="utf-8")
        outputs["report"] = str(report_path)
        if make_plot:
            figure = refinement_plot(self)
            figure_paths = save_figure(figure, destination / "convergence.png", formats=["png", "svg"])
            outputs["figure"] = str(figure_paths[0])
            outputs["figure_svg"] = str(figure_paths[1])
        return outputs


def run_refinement_experiment(
    model: ModelSpec,
    solver: Solver,
    *,
    start: float,
    end: float,
    base_step: float,
    refinements: Iterable[int] = (1, 2, 4),
    parameters: Mapping[str, float] | None = None,
    initial_conditions: Mapping[str, float] | None = None,
    options: Mapping[str, Any] | None = None,
) -> RefinementResult:
    factors = sorted({int(factor) for factor in refinements})
    if not factors or factors[0] < 1 or any(factor <= 0 for factor in factors):
        raise ValueError("refinements must contain positive integer factors")
    if not np.isfinite([start, end, base_step]).all() or end <= start or base_step <= 0:
        raise ValueError("Require finite end > start and a positive base_step")
    span = end - start
    steps = [base_step / factor for factor in factors]
    grids: dict[int, np.ndarray] = {}
    for factor, step in zip(factors, steps):
        ratio = span / step
        if not np.isclose(ratio, round(ratio), rtol=1e-10, atol=1e-12):
            raise ValueError(f"end-start must be divisible by each refinement step; factor {factor} gives {step}")
        grids[factor] = uniform_time_grid(start, end, step)

    simulations: dict[int, SimulationResult] = {}
    for factor in factors:
        simulations[factor] = solver.solve(model, parameters, initial_conditions, grids[factor], options)
    reference_factor = factors[-1]
    reference = simulations[reference_factor]
    errors: dict[int, dict[str, float]] = {}
    warnings: list[str] = []
    for factor in factors:
        candidate = simulations[factor]
        reference_states = _sample_states(reference, candidate.time)
        errors[factor] = error_metrics(reference_states, candidate.states)
        warnings.extend(candidate.warnings)
    observed_orders: dict[int, float | None] = {factors[0]: None}
    for previous, current in zip(factors, factors[1:]):
        coarse_error = errors[previous]["rmse"]
        fine_error = errors[current]["rmse"]
        if coarse_error > 0 and fine_error > 0 and np.isfinite(coarse_error) and np.isfinite(fine_error):
            observed_orders[current] = float(np.log(coarse_error / fine_error) / np.log(steps[ factors.index(previous) ] / steps[factors.index(current)]))
        else:
            observed_orders[current] = None
    unique_warnings = list(dict.fromkeys(warnings))
    finite = all(np.isfinite(simulation.states).all() for simulation in simulations.values())
    errors_finite = all(np.isfinite(metric["rmse"]) for metric in errors.values())
    status = "NUMERICALLY_VALIDATED" if finite and errors_finite else "PARTIAL"
    return RefinementResult(
        factors=factors,
        steps=steps,
        simulations=simulations,
        errors=errors,
        observed_orders=observed_orders,
        reference_factor=reference_factor,
        status=status,
        warnings=unique_warnings,
        metadata={"solver": solver.name, "model": model.name, "reference_is_finest_requested_grid": True},
    )


def refinement_plot(result: RefinementResult):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    rmse = [result.errors[factor]["rmse"] for factor in result.factors]
    axis.loglog(result.steps, rmse, marker="o", label="RMSE vs finest grid")
    axis.set(xlabel="time step", ylabel="RMSE", title="Numerical refinement")
    axis.grid(True, which="both", alpha=.25)
    axis.legend()
    return figure
