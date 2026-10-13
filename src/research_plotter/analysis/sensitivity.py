from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from research_plotter.models.model_spec import ModelSpec
from research_plotter.solvers.base import SimulationResult, Solver


def perturbation_sensitivity(
    model: ModelSpec,
    solver: Solver,
    time_grid: np.ndarray,
    parameter: str,
    delta: float = 0.10,
    output: str | int | None = None,
    initial_conditions: Mapping[str, float] | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    baseline = model.parameter_values.get(parameter)
    if baseline is None:
        raise ValueError(f"Unknown parameter: {parameter}")
    plus = solver.solve(model, {parameter: baseline * (1 + delta)}, initial_conditions, time_grid, options)
    minus = solver.solve(model, {parameter: baseline * (1 - delta)}, initial_conditions, time_grid, options)
    if output is None:
        index = slice(None)
        output_name = "all_states"
    elif isinstance(output, int):
        index = output
        output_name = plus.state_names[output]
    else:
        if output not in plus.state_names:
            raise ValueError(f"Unknown output state: {output}")
        index = plus.state_names.index(output)
        output_name = output
    y_plus, y_minus = plus.states[:, index], minus.states[:, index]
    denominator = 2 * np.log1p(delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = (np.log(np.abs(y_plus)) - np.log(np.abs(y_minus))) / denominator
    values = np.where(np.isfinite(values), values, np.nan)
    return {"parameter": parameter, "output": output_name, "delta": delta, "time": plus.time, "sensitivity": values,
            "plus": plus, "minus": minus, "warnings": ["Log sensitivity uses absolute output values; zero/negative states produce NaN or sign-insensitive values."]}


def logarithmic_sensitivity(*args, **kwargs) -> dict[str, Any]:
    return perturbation_sensitivity(*args, **kwargs)
