from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from research_plotter.models.model_spec import ModelSpec
from research_plotter.solvers.base import SimulationResult, Solver


class ODESolver(Solver):
    name = "scipy.solve_ivp"

    def solve(self, model: ModelSpec, parameters: Mapping[str, float] | None, initial_conditions: Mapping[str, float] | None, time_grid: np.ndarray, options: Mapping[str, Any] | None = None) -> SimulationResult:
        if model.derivative_type not in {"ordinary", "integer"}:
            raise ValueError("ODESolver cannot replace a fractional derivative; choose a fractional solver")
        opts = dict(options or {})
        values = model.validate_parameters(parameters)
        y0 = model.initial_vector(initial_conditions)
        time_grid = np.asarray(time_grid, dtype=float)
        if len(time_grid) < 2 or np.any(np.diff(time_grid) <= 0):
            raise ValueError("time_grid must be strictly increasing with at least two values")
        started = perf_counter()
        solution = solve_ivp(
            lambda time, state: model.rhs(time, state, values),
            (time_grid[0], time_grid[-1]),
            y0,
            t_eval=time_grid,
            method=opts.pop("method", "RK45"),
            rtol=opts.pop("rtol", 1e-6),
            atol=opts.pop("atol", 1e-9),
            **opts,
        )
        warnings = [] if solution.success else [solution.message]
        return SimulationResult(time=solution.t, states=solution.y.T, state_names=model.state_names, parameters=values,
                                initial_conditions=dict(zip(model.state_names, y0)), solver_name=self.name,
                                runtime_seconds=perf_counter() - started, diagnostics={"success": solution.success, "message": solution.message, "nfev": solution.nfev}, warnings=warnings)
