from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
from scipy.special import gamma

from research_plotter.models.model_spec import ModelSpec
from research_plotter.solvers.atangana_baleanu import ab_normalization
from research_plotter.solvers.base import SimulationResult
from research_plotter.solvers.fractional_base import FractionalSolver


class FractalFractionalGMLSolver(FractionalSolver):
    """Transparent quadrature implementation of a GML fractal-fractional integral.

    It implements the stated integral relation using left product integration,
    rather than pretending that a classical ODE integrator models memory. When
    beta < 1, the origin-local term is singular in several publications; callers
    must opt in to an epsilon convention and the result records that assumption.
    """

    name = "fractal_fractional_generalized_mittag_leffler_quadrature"

    def solve(self, model: ModelSpec, parameters: Mapping[str, float] | None, initial_conditions: Mapping[str, float] | None, time_grid: np.ndarray, options: Mapping[str, Any] | None = None) -> SimulationResult:
        opts, grid = dict(options or {}), np.asarray(time_grid, dtype=float)
        alpha = float(opts.pop("fractional_order", model.fractional_order if model.fractional_order is not None else 1.0))
        beta = float(opts.pop("fractal_dimension", model.metadata.get("fractal_dimension", 1.0)))
        epsilon_origin = bool(opts.pop("epsilon_origin", False))
        if not (0 < alpha <= 1 and 0 < beta <= 1) or len(grid) < 2 or not np.isfinite(grid).all() or not np.allclose(np.diff(grid), np.diff(grid)[0]) or grid[1] <= grid[0]:
            raise ValueError("Fractal-fractional quadrature requires alpha,beta in (0,1] and an increasing uniform grid")
        if beta < 1 and np.isclose(grid[0], 0.0) and not epsilon_origin:
            raise ValueError("The local t^(beta-1) term is singular at t=0. Set epsilon_origin=True only for an explicitly exploratory convention.")
        step, normalizer = float(grid[1] - grid[0]), ab_normalization(alpha)
        states = np.empty((len(grid), len(model.state_names)))
        states[0] = model.initial_vector(initial_conditions)
        values = model.validate_parameters(parameters)
        rhs_history = np.empty_like(states)
        rhs_history[0] = model.rhs(grid[0], states[0], values)
        warnings: list[str] = []
        started = perf_counter()
        for index in range(1, len(grid)):
            offsets = np.arange(index - 1, -1, -1, dtype=float)
            weights = (offsets + 1.0) ** alpha - offsets**alpha
            temporal = np.maximum(grid[:index], step if epsilon_origin else np.finfo(float).tiny) ** (beta - 1.0)
            integral = weights @ (temporal[:, None] * rhs_history[:index])
            local_time = max(grid[index - 1], step) if epsilon_origin else grid[index - 1]
            local = beta * local_time ** (beta - 1.0) * (1.0 - alpha) / normalizer * rhs_history[index - 1]
            memory = beta * step**alpha / (normalizer * float(gamma(alpha + 1))) * integral
            states[index] = states[0] + local + memory
            rhs_history[index] = model.rhs(grid[index], states[index], values)
        if epsilon_origin:
            warnings.append("Exploratory epsilon-origin convention used for singular t^(beta-1) local term.")
        warnings.append("Reference product-integration solver has O(N^2) history cost; perform grid refinement before drawing conclusions.")
        return SimulationResult(time=grid, states=states, state_names=model.state_names, parameters=values,
                                initial_conditions=dict(zip(model.state_names, states[0])), solver_name=self.name,
                                runtime_seconds=perf_counter() - started,
                                diagnostics={"fractional_order": alpha, "fractal_dimension": beta, "normalization": normalizer, "time_step": step}, warnings=warnings)
