from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np

from research_plotter.models.model_spec import ModelSpec
from research_plotter.solvers.base import SimulationResult
from research_plotter.solvers.fractional_base import FractionalSolver


def grunwald_letnikov_weights(order: float, count: int) -> np.ndarray:
    """Return w_k=(-1)^k binomial(order,k), recursively and stably."""
    if not 0 < order <= 1:
        raise ValueError("Grunwald-Letnikov order must be in (0, 1]")
    if count < 1:
        raise ValueError("count must be positive")
    weights = np.empty(count, dtype=float)
    weights[0] = 1.0
    for index in range(1, count):
        weights[index] = weights[index - 1] * (1 - (order + 1) / index)
    return weights


class GrunwaldLetnikovSolver(FractionalSolver):
    """Caputo-corrected, explicit GL reference solver (O(N^2) history cost)."""

    name = "grunwald_letnikov_caputo_corrected"

    def solve(self, model: ModelSpec, parameters: Mapping[str, float] | None, initial_conditions: Mapping[str, float] | None, time_grid: np.ndarray, options: Mapping[str, Any] | None = None) -> SimulationResult:
        opts, time_grid = dict(options or {}), np.asarray(time_grid, dtype=float)
        order = float(opts.pop("fractional_order", model.fractional_order if model.fractional_order is not None else 1.0))
        if len(time_grid) < 2 or not np.isfinite(time_grid).all() or not np.allclose(np.diff(time_grid), np.diff(time_grid)[0]) or time_grid[1] <= time_grid[0]:
            raise ValueError("GL solver requires a finite, increasing uniform grid with at least two points")
        step = float(time_grid[1] - time_grid[0])
        states = np.empty((len(time_grid), len(model.state_names)), dtype=float)
        states[0] = model.initial_vector(initial_conditions)
        values, weights = model.validate_parameters(parameters), grunwald_letnikov_weights(order, len(time_grid))
        started = perf_counter()
        completed_steps = 0
        failure_index: int | None = None
        for index in range(1, len(time_grid)):
            # Caputo correction subtracts y(0), preserving the derivative of a constant as zero.
            history = weights[1 : index + 1] @ (states[index - 1 :: -1] - states[0])
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                derivative = model.rhs(time_grid[index - 1], states[index - 1], values)
                states[index] = states[0] - history + step**order * derivative
            if not np.isfinite(states[index]).all():
                failure_index = index
                states[index:] = np.nan
                break
            completed_steps = index
        warnings = ["Reference GL implementation uses an explicit update; refine the grid and assess convergence."]
        if failure_index is not None:
            warnings.append(
                f"Explicit GL update produced a non-finite state at index {failure_index} "
                f"(time {time_grid[failure_index]:g}); remaining trajectory values are NaN."
            )
        return SimulationResult(time=time_grid, states=states, state_names=model.state_names, parameters=values,
                                initial_conditions=dict(zip(model.state_names, states[0])), solver_name=self.name,
                                runtime_seconds=perf_counter() - started,
                                diagnostics={"fractional_order": order, "time_step": step, "number_of_steps": len(time_grid) - 1,
                                             "completed_steps": completed_steps, "state_dimension": len(model.state_names),
                                             "failure_index": failure_index,
                                             "finite": bool(np.isfinite(states).all()), "history_complexity": "O(N^2)"},
                                warnings=warnings)
