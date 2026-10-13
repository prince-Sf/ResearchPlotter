from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from research_plotter.models.model_spec import ModelSpec
from research_plotter.solvers.base import SimulationResult
from research_plotter.utils.numerics import error_metrics


def compare_trajectories(reference: SimulationResult | np.ndarray, candidate: SimulationResult | np.ndarray, state: str | int | None = None) -> dict[str, float]:
    ref = reference.states if isinstance(reference, SimulationResult) else np.asarray(reference, dtype=float)
    pred = candidate.states if isinstance(candidate, SimulationResult) else np.asarray(candidate, dtype=float)
    if isinstance(reference, SimulationResult) and isinstance(candidate, SimulationResult):
        if reference.state_names != candidate.state_names:
            raise ValueError("Reference and candidate must have the same state names and order")
        if not np.array_equal(reference.time, candidate.time):
            if candidate.time[0] > reference.time[0] or candidate.time[-1] < reference.time[-1]:
                raise ValueError("Candidate time range must cover the reference time range for interpolation")
            pred = np.column_stack([np.interp(reference.time, candidate.time, candidate.states[:, column]) for column in range(candidate.states.shape[1])])
    if state is not None:
        if isinstance(state, str):
            names = reference.state_names if isinstance(reference, SimulationResult) else []
            if state not in names:
                raise ValueError(f"Unknown state: {state}")
            state = names.index(state)
        ref, pred = ref[:, state], pred[:, state]
    return error_metrics(ref, pred)


def validate_simulation(result: SimulationResult, model: ModelSpec | None = None, require_nonnegative: bool = False) -> dict[str, Any]:
    states = np.asarray(result.states, dtype=float)
    finite = bool(np.isfinite(states).all() and np.isfinite(result.time).all())
    shape_ok = bool(states.ndim == 2 and states.shape[0] == len(result.time) and states.shape[1] == len(result.state_names))
    warnings = list(result.warnings)
    if not finite:
        if "Simulation contains NaN or infinite values" not in warnings:
            warnings.append("Simulation contains NaN or infinite values")
    if require_nonnegative and bool((states < 0).any()):
        if "Simulation violates the requested nonnegativity constraint" not in warnings:
            warnings.append("Simulation violates the requested nonnegativity constraint")
    if model is not None and set(model.state_names) != set(result.state_names):
        warnings.append("Simulation state names do not match the model")
    status = "VALID" if finite and shape_ok and not (require_nonnegative and bool((states < 0).any())) else "INVALID"
    finite_values = states[np.isfinite(states)]
    return {"status": status, "finite": finite, "shape_ok": shape_ok, "warnings": warnings,
            "min_state": float(finite_values.min()) if finite_values.size else None,
            "max_state": float(finite_values.max()) if finite_values.size else None}


def reproduction_status(metrics: Mapping[str, float], *, close_rmse: float = 1e-3, exact_rmse: float = 1e-8) -> str:
    """Map trajectory evidence to a cautious human-readable status."""
    rmse = float(metrics.get("rmse", float("inf")))
    if not np.isfinite(rmse): return "BLOCKED"
    if rmse <= exact_rmse: return "EXACT"
    if rmse <= close_rmse: return "CLOSE"
    return "UNVERIFIED"


trajectory_metrics = compare_trajectories
