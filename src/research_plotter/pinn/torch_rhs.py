"""Build a differentiable Torch RHS from a validated ModelSpec expression set."""

from __future__ import annotations

from typing import Callable

import sympy as sp

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from research_plotter.models.equation_system import safe_parse_expression
from research_plotter.models.model_spec import ModelSpec


def torch_rhs_from_model(model: ModelSpec) -> Callable:
    """Return ``rhs(time, state)`` for a SymPy-validated expression model.

    This never parses or executes arbitrary Python.  The source expressions
    have already passed the restricted parser used by :class:`ModelSpec`.
    """
    if torch is None:
        raise ImportError("Torch RHS generation requires PyTorch")
    raw_expressions = model.metadata.get("expressions")
    if not isinstance(raw_expressions, dict):
        raise ValueError("Model metadata does not contain validated expressions")
    allowed = set(model.state_names) | set(model.parameters) | {model.independent_variable}
    expressions = [safe_parse_expression(str(raw_expressions[name]), allowed) for name in model.state_names]
    symbols = [sp.Symbol(model.independent_variable)] + [sp.Symbol(name) for name in model.state_names + list(model.parameters)]
    function = sp.lambdify(symbols, expressions, modules=[{
        "Abs": torch.abs,
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": torch.tan,
        "exp": torch.exp,
        "log": torch.log,
        "sqrt": torch.sqrt,
    }, "torch"])

    def rhs(time, state):
        time_values = time[:, 0] if time.ndim > 1 else time
        parameter_values = [torch.as_tensor(model.parameters[name].value, dtype=state.dtype, device=state.device) for name in model.parameters]
        values = function(time_values, *[state[:, index] for index in range(state.shape[1])], *parameter_values)
        if not isinstance(values, (list, tuple)):
            values = [values]
        tensors = []
        for value in values:
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=state.dtype, device=state.device)
            tensors.append(tensor if tensor.ndim else tensor.expand_as(time_values))
        return torch.stack(tensors, dim=1)

    return rhs
