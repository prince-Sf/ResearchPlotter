from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from research_plotter.utils.numerics import error_metrics


def add_gaussian_noise(values: np.ndarray, level: float, rng: np.random.Generator | None = None) -> np.ndarray:
    if level < 0:
        raise ValueError("Noise level must be non-negative")
    generator = rng or np.random.default_rng()
    values = np.asarray(values, dtype=float)
    scale = float(np.nanstd(values)) * level
    return values + generator.normal(0.0, scale, size=values.shape)


inject_noise = add_gaussian_noise


def noise_experiment(reference: np.ndarray, levels: Iterable[float] = (0, .05, .10, .15, .20), seed: int = 0) -> list[dict[str, Any]]:
    reference = np.asarray(reference, dtype=float)
    rng = np.random.default_rng(seed)
    results = []
    for level in levels:
        noisy = add_gaussian_noise(reference, float(level), rng)
        metrics = error_metrics(reference, noisy)
        results.append({"noise_level": float(level), "values": noisy, **metrics})
    return results
