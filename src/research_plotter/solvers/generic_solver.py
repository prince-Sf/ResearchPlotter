from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np

from research_plotter.models.model_spec import ModelSpec
from research_plotter.solvers.base import SimulationResult, Solver


class ExplicitEulerSolver(Solver):
    """Transparent first-order state update reference implementation."""

    name = "explicit_euler"

    def solve(self, model: ModelSpec, parameters: Mapping[str, float] | None, initial_conditions: Mapping[str, float] | None, time_grid: np.ndarray, options: Mapping[str, Any] | None = None) -> SimulationResult:
        if model.derivative_type not in {"ordinary", "integer"}:
            raise ValueError("ExplicitEulerSolver is only for ordinary/integer order systems")
        values, time_grid = model.validate_parameters(parameters), np.asarray(time_grid, dtype=float)
        if time_grid.ndim != 1 or len(time_grid) < 2 or not np.isfinite(time_grid).all() or np.any(np.diff(time_grid) <= 0):
            raise ValueError("time_grid must be finite, strictly increasing, and contain at least two values")
        states = np.empty((len(time_grid), len(model.state_names)), dtype=float)
        states[0] = model.initial_vector(initial_conditions)
        started = perf_counter()
        for index, step in enumerate(np.diff(time_grid)):
            states[index + 1] = states[index] + step * model.rhs(time_grid[index], states[index], values)
        return SimulationResult(time=time_grid, states=states, state_names=model.state_names, parameters=values,
                                initial_conditions=dict(zip(model.state_names, states[0])), solver_name=self.name,
                                runtime_seconds=perf_counter() - started, diagnostics={"steps": len(time_grid) - 1})
