from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def gl_weights(order: float, count: int, *, dtype=None, device=None):
    if not 0 < order <= 1 or count < 1:
        raise ValueError("order must be in (0,1] and count positive")
    if torch is None:
        raise ImportError("PINN fractional residuals require PyTorch")
    values = [1.0]
    for index in range(1, count): values.append(values[-1] * (1 - (order + 1) / index))
    return torch.tensor(values, dtype=dtype, device=device)


def gl_fractional_derivative(values, step: float, order: float, *, caputo: bool = True):
    """Differentiate a batch of uniformly sampled predictions using GL history.

    This is numerical convolution, not automatic differentiation. `values` is
    shaped `(time, state)` and the first sample is used for Caputo correction.
    """
    if torch is None:
        raise ImportError("PINN fractional residuals require PyTorch")
    if step <= 0:
        raise ValueError("step must be positive")
    if values.ndim == 1: values = values[:, None]
    weights = gl_weights(order, values.shape[0], dtype=values.dtype, device=values.device)
    output = []
    baseline = values[0] if caputo else torch.zeros_like(values[0])
    for index in range(values.shape[0]):
        history = values[: index + 1].flip(0) - baseline
        output.append(torch.sum(weights[: index + 1, None] * history, dim=0) / (step**order))
    return torch.stack(output)
