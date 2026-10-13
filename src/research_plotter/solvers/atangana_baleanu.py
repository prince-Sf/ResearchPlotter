from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
from scipy.special import gamma

from research_plotter.models.model_spec import ModelSpec
from research_plotter.solvers.base import SimulationResult
from research_plotter.solvers.fractional_base import FractionalSolver


def ab_normalization(order: float) -> float:
    return 1.0 - order + order / float(gamma(order))


class AtanganaBaleanuStyleSolver(FractionalSolver):
    """Product-integration reference for a standard AB-Caputo integral relation.

    This is intentionally named *style*: it is not interchangeable with every
    published AB variant. A paper adapter must provide a verified formulation.
    """

    name = "atangana_baleanu_product_integration"

    def solve(self, model: ModelSpec, parameters: Mapping[str, float] | None, initial_conditions: Mapping[str, float] | None, time_grid: np.ndarray, options: Mapping[str, Any] | None = None) -> SimulationResult:
        opts, grid = dict(options or {}), np.asarray(time_grid, dtype=float)
        order = float(opts.pop("fractional_order", model.fractional_order if model.fractional_order is not None else 1.0))
        if not 0 < order <= 1 or len(grid) < 2 or not np.isfinite(grid).all() or not np.allclose(np.diff(grid), np.diff(grid)[0]) or grid[1] <= grid[0]:
            raise ValueError("AB product integration requires an order in (0,1] and an increasing uniform grid")
        step, normalizer = float(grid[1] - grid[0]), ab_normalization(order)
        states = np.empty((len(grid), len(model.state_names)))
        states[0] = model.initial_vector(initial_conditions)
        values = model.validate_parameters(parameters)
        rhs_history = np.empty_like(states)
        rhs_history[0] = model.rhs(grid[0], states[0], values)
        started = perf_counter()
        for index in range(1, len(grid)):
            offsets = np.arange(index - 1, -1, -1, dtype=float)
            weights = (offsets + 1.0) ** order - offsets**order
            integral = weights @ rhs_history[:index]
            local = (1.0 - order) / normalizer * rhs_history[index - 1]
            memory = step**order / (normalizer * float(gamma(order + 1))) * integral
            states[index] = states[0] + local + memory
            rhs_history[index] = model.rhs(grid[index], states[index], values)
        return SimulationResult(time=grid, states=states, state_names=model.state_names, parameters=values,
                                initial_conditions=dict(zip(model.state_names, states[0])), solver_name=self.name,
                                runtime_seconds=perf_counter() - started,
                                diagnostics={"fractional_order": order, "normalization": normalizer},
                                warnings=["AB-style reference uses explicit product integration; confirm the operator definition before scientific use."])
