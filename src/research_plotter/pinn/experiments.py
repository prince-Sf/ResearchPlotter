"""Small, reproducible PINN experiment helpers.

The helpers are model-agnostic.  A caller supplies the right-hand side when a
physics residual is actually specified; no controlled Paper 2 equations are
invented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np

from research_plotter.analysis.noise import add_gaussian_noise
from research_plotter.utils.numerics import error_metrics
from research_plotter.config import PINNConfig
from research_plotter.pinn.network import StateNetwork
from research_plotter.pinn.trainer import PINNTrainer, TrainingResult


@dataclass
class PINNExperimentResult:
    mode: str
    training: TrainingResult
    prediction: np.ndarray
    metrics: dict[str, float]
    configuration: dict[str, Any]
    warnings: list[str]

    @property
    def final_loss(self) -> float | None:
        return self.training.history[-1]["loss"] if self.training.history else None

    @property
    def initial_loss(self) -> float | None:
        """First recorded full training objective, before its optimizer step."""
        return self.training.history[0]["loss"] if self.training.history else None


def _network_from_config(config: PINNConfig, output_dim: int) -> StateNetwork:
    missing = config.missing_fields()
    if missing:
        raise ValueError(f"PINN configuration is missing: {', '.join(missing)}")
    if config.hidden_layers is not None:
        hidden_layers = tuple(int(width) for width in config.hidden_layers)
    else:
        layer_count = int(config.num_hidden_layers or 2)
        hidden_layers = tuple(int(config.hidden_units) for _ in range(layer_count))
    if any(width < 1 for width in hidden_layers):
        raise ValueError("PINN hidden layer widths must be positive")
    return StateNetwork(output_dim=output_dim, hidden_layers=hidden_layers, activation=str(config.activation))


def run_pinn_experiment(
    time,
    reference,
    config: PINNConfig,
    *,
    mode: str = "fractional",
    rhs: Callable | None = None,
    fractional_order: float = 1.0,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
) -> PINNExperimentResult:
    """Train one baseline, integer-order, or fractional PINN experiment."""
    values = np.asarray(reference, dtype=float)
    times = np.asarray(time, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if times.ndim != 1 or values.ndim != 2 or len(times) != len(values):
        raise ValueError("time must be 1-D and reference must have shape (len(time), states)")
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("time and reference values must be finite")
    normalized_mode = mode.lower()
    if normalized_mode not in {"baseline", "integer", "fractional"}:
        raise ValueError("mode must be baseline, integer, or fractional")
    if normalized_mode != "baseline" and rhs is None:
        raise ValueError("A right-hand side callable is required for PINN physics modes")
    if normalized_mode == "fractional" and not 0 < fractional_order < 1:
        raise ValueError("fractional_order must be in (0,1) for fractional PINN mode")
    if normalized_mode == "integer":
        fractional_order = 1.0
    network = _network_from_config(config, values.shape[1])
    trainer = PINNTrainer(network, learning_rate=float(config.learning_rate), physics_weight=float(config.physics_loss_weight), optimizer=str(config.optimizer), device=config.device)
    order = fractional_order if normalized_mode == "fractional" else 1.0
    training_time, training_values = times, values
    experiment_warnings: list[str] = []
    target_points = config.collocation_points or config.data_points
    if target_points is not None and target_points != len(times):
        training_time = np.linspace(times[0], times[-1], int(target_points))
        training_values = np.column_stack([np.interp(training_time, times, values[:, column]) for column in range(values.shape[1])])
        if config.data_points is not None and config.collocation_points is not None and config.data_points != config.collocation_points:
            experiment_warnings.append("data_points and collocation_points differ; observations were interpolated onto the collocation grid for the shared trainer.")
    training = trainer.train(training_time, training_values, rhs=None if normalized_mode == "baseline" else rhs, order=order, epochs=int(config.epochs), initial_weight=float(config.initial_condition_loss_weight), progress_callback=progress_callback)
    prediction = training.predict(times).detach().cpu().numpy()
    metrics = error_metrics(values, prediction)
    warnings = experiment_warnings + list(training.warnings)
    return PINNExperimentResult(
        mode=normalized_mode,
        training=training,
        prediction=prediction,
        metrics=metrics,
        configuration=config.model_dump(mode="json"),
        warnings=warnings,
    )


def run_noise_experiments(
    time,
    reference,
    config: PINNConfig,
    *,
    levels: Iterable[float] = (0.0, .05, .10, .15, .20),
    mode: str = "baseline",
    rhs: Callable | None = None,
    fractional_order: float = 1.0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Train/evaluate against noisy observations while preserving reference data."""
    if mode != "baseline" and rhs is None:
        raise ValueError("A right-hand side callable is required for physics-mode noise experiments")
    rng = np.random.default_rng(seed)
    reference_array = np.asarray(reference, dtype=float)
    results: list[dict[str, Any]] = []
    for level in levels:
        started = perf_counter()
        noisy = add_gaussian_noise(reference_array, float(level), rng)
        experiment = run_pinn_experiment(time, noisy, config, mode=mode, rhs=rhs, fractional_order=fractional_order)
        # Evaluate predictions against the untouched reference dynamics.
        # The noisy observations remain separate and are retained for audit.
        clean_metrics = error_metrics(reference_array, experiment.prediction)
        results.append({"noise_level": float(level), "final_loss": experiment.final_loss, **clean_metrics,
                        "training_metrics": experiment.metrics,
                        "noisy_observations": noisy.tolist(),
                        "runtime_seconds": perf_counter() - started, "training_history": experiment.training.history,
                        "warnings": experiment.warnings, "mode": mode,
                        "configuration": experiment.configuration})
    return results


def compare_pinn_modes(
    time,
    reference,
    config: PINNConfig,
    *,
    rhs: Callable | None,
    fractional_order: float,
    modes: Iterable[str] = ("baseline", "integer", "fractional"),
) -> list[dict[str, Any]]:
    """Run the baseline, integer-order, and fractional modes on shared data."""
    results: list[dict[str, Any]] = []
    for mode in modes:
        experiment = run_pinn_experiment(time, reference, config, mode=mode, rhs=rhs, fractional_order=fractional_order)
        results.append({"mode": mode, "metrics": experiment.metrics, "final_loss": experiment.final_loss,
                        "runtime_seconds": experiment.training.runtime_seconds, "warnings": experiment.warnings,
                        "training_history": experiment.training.history, "prediction": experiment.prediction,
                        "configuration": experiment.configuration})
    return results
