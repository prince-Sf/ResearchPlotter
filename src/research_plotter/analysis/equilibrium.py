from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import root

from research_plotter.models.model_spec import ModelSpec


def find_equilibrium(model: ModelSpec, guess: Mapping[str, float] | np.ndarray, parameters: Mapping[str, float] | None = None) -> dict[str, Any]:
    vector = model.initial_vector(guess) if isinstance(guess, Mapping) else np.asarray(guess, dtype=float)
    values = model.validate_parameters(parameters)
    solution = root(lambda state: model.rhs(0.0, state, values), vector)
    state = np.asarray(solution.x, dtype=float)
    return {"success": bool(solution.success), "message": solution.message, "state": dict(zip(model.state_names, state)), "residual_norm": float(np.linalg.norm(model.rhs(0.0, state, values)))}
