from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from research_plotter.models.model_spec import ModelSpec
from research_plotter.solvers.base import SimulationResult, Solver


@dataclass
class ParameterSweepResult:
    parameter_values: list[dict[str, float]]
    simulations: list[SimulationResult]
    metadata: dict[str, Any] = field(default_factory=dict)


def _combinations(parameters: Mapping[str, Iterable[float]]) -> list[dict[str, float]]:
    names = list(parameters)
    values = [list(parameters[name]) for name in names]
    if any(not items for items in values):
        raise ValueError("Each swept parameter must have at least one value")
    combinations = [{}]
    for name, candidates in zip(names, values):
        combinations = [{**current, name: float(candidate)} for current in combinations for candidate in candidates]
    return combinations


def run_parameter_sweep(
    model: ModelSpec,
    solver: Solver,
    time_grid: np.ndarray,
    parameters: Mapping[str, Iterable[float]],
    initial_conditions: Mapping[str, float] | None = None,
    options: Mapping[str, Any] | None = None,
    progress: bool = False,
) -> ParameterSweepResult:
    combinations = _combinations(parameters)
    iterator = combinations
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(combinations, desc="parameter sweep")
        except ImportError:
            pass
    simulations = [solver.solve(model, values, initial_conditions, time_grid, options) for values in iterator]
    return ParameterSweepResult(combinations, simulations, {"count": len(simulations), "swept_parameters": list(parameters)})


parameter_sweep = run_parameter_sweep
