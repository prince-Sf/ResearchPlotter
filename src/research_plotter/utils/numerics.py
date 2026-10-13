from __future__ import annotations

import numpy as np


def uniform_time_grid(start: float, end: float, step: float) -> np.ndarray:
    if not all(np.isfinite(value) for value in (start, end, step)) or step <= 0 or end <= start:
        raise ValueError("Require finite end > start and a positive step")
    count = int(round((end - start) / step))
    grid = start + np.arange(count + 1, dtype=float) * step
    if not np.isclose(grid[-1], end):
        grid = np.append(grid, end)
    return grid


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate must share a shape")
    error = candidate - reference
    mse = float(np.mean(error**2))
    denom = np.maximum(np.abs(reference), np.finfo(float).eps)
    flat_reference, flat_candidate = reference.ravel(), candidate.ravel()
    if reference.size <= 1:
        correlation = float("nan")
    elif np.std(flat_reference) == 0 or np.std(flat_candidate) == 0:
        correlation = 1.0 if np.allclose(flat_reference, flat_candidate) else float("nan")
    else:
        correlation = float(np.corrcoef(flat_reference, flat_candidate)[0, 1])
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "max_absolute_error": float(np.max(np.abs(error))),
        "mean_relative_error": float(np.mean(np.abs(error) / denom)),
        "correlation": correlation,
    }
