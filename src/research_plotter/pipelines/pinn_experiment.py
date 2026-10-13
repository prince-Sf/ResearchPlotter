"""Exportable PINN experiment workflow shared by papers and generic models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_plotter.config import PINNConfig
from research_plotter.plotting.comparison import comparison_plot
from research_plotter.plotting.convergence import convergence_plot
from research_plotter.plotting.core import save_figure
from research_plotter.pinn.experiments import run_noise_experiments, run_pinn_experiment
from research_plotter.pinn.torch_rhs import torch_rhs_from_model
from research_plotter.solvers.base import SimulationResult


def run_pinn_export(
    model,
    reference: SimulationResult,
    config: PINNConfig,
    output_dir: str | Path,
    *,
    mode: str = "fractional",
    fractional_order: float | None = None,
) -> dict[str, Any]:
    """Run one configured PINN and persist metrics, history, checkpoint, and figures.

    A missing Torch installation or incomplete user configuration is returned as
    a structured BLOCKED result, so paper reproduction remains inspectable.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    missing = config.missing_fields()
    if missing:
        return {"status": "BLOCKED", "warnings": [f"PINN configuration is missing: {', '.join(missing)}"], "output_dir": str(destination)}
    try:
        rhs = torch_rhs_from_model(model) if mode != "baseline" else None
        result = run_pinn_experiment(reference.time, reference.states, config, mode=mode, rhs=rhs, fractional_order=fractional_order or model.fractional_order or 1.0)
    except ImportError as error:
        return {"status": "BLOCKED", "warnings": [str(error)], "output_dir": str(destination)}
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return {"status": "BLOCKED", "warnings": [f"PINN experiment could not run: {error}"], "output_dir": str(destination)}

    prediction = result.prediction
    prediction_path = destination / "prediction.csv"
    pd.DataFrame(prediction, columns=reference.state_names).assign(time=reference.time).to_csv(prediction_path, index=False)
    history_path = destination / "training_history.json"
    history_path.write_text(json.dumps(result.training.history, indent=2), encoding="utf-8")
    metrics_path = destination / "metrics.json"
    metrics_path.write_text(json.dumps({"mode": mode, "model_type": mode, "configuration": result.configuration,
                                        "metrics": result.metrics, "initial_loss": result.initial_loss, "final_loss": result.final_loss,
                                        "runtime_seconds": result.training.runtime_seconds, "warnings": result.warnings}, indent=2), encoding="utf-8")
    checkpoint_path = result.training.save(destination / "model.pt", configuration=config.model_dump(mode="json"))
    combined = pd.DataFrame({f"reference_{name}": reference.states[:, index] for index, name in enumerate(reference.state_names)})
    for index, name in enumerate(reference.state_names):
        combined[f"pinn_{name}"] = prediction[:, index]
    comparison = comparison_plot(reference.time, combined, title=f"{mode} PINN vs reference")
    comparison_paths = save_figure(comparison, destination / "trajectory_comparison.png", formats=["png", "svg"])
    losses = {key: [item[key] for item in result.training.history] for key in ("loss", "data_loss", "physics_loss", "initial_loss") if result.training.history and key in result.training.history[0]}
    convergence = convergence_plot([item["epoch"] for item in result.training.history], losses)
    convergence_paths = save_figure(convergence, destination / "training_convergence.png", formats=["png", "svg"])
    figure_metadata = destination / "figure_metadata.json"
    figure_metadata.write_text(json.dumps({
        "source": reference.metadata if hasattr(reference, "metadata") else {},
        "mode": mode,
        "fractional_order": fractional_order,
        "configuration": config.model_dump(mode="json"),
        "metrics": result.metrics,
        "figures": [str(path) for path in comparison_paths + convergence_paths],
    }, indent=2, default=str), encoding="utf-8")
    return {"status": "COMPLETED", "validation_status": "UNVERIFIED", "model_type": mode,
            "configuration": result.configuration, "metrics": result.metrics,
            "initial_loss": result.initial_loss, "final_loss": result.final_loss, "runtime_seconds": result.training.runtime_seconds,
            "prediction": str(prediction_path), "training_history": str(history_path), "metrics_file": str(metrics_path),
            "checkpoint": str(checkpoint_path), "figures": [str(path) for path in comparison_paths + convergence_paths],
            "figure_metadata": str(figure_metadata), "warnings": result.warnings}


def run_pinn_noise_export(
    model,
    reference: SimulationResult,
    config: PINNConfig,
    levels,
    output_dir: str | Path,
    *,
    mode: str = "baseline",
    fractional_order: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run configured noise levels and save machine-readable metrics/figures."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    levels = list(levels)
    missing = config.missing_fields()
    if missing:
        return {"status": "BLOCKED", "warnings": [f"PINN configuration is missing: {', '.join(missing)}"]}
    try:
        rhs = torch_rhs_from_model(model) if mode != "baseline" else None
        results = run_noise_experiments(reference.time, reference.states, config, levels=levels, mode=mode, rhs=rhs, fractional_order=fractional_order or model.fractional_order or 1.0, seed=seed)
    except ImportError as error:
        return {"status": "BLOCKED", "warnings": [str(error)]}
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return {"status": "BLOCKED", "warnings": [f"PINN noise experiment could not run: {error}"]}
    report_path = destination / "noise_results.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    x = [item["noise_level"] for item in results]
    axis.plot(x, [item["rmse"] for item in results], marker="o", label="RMSE")
    axis.plot(x, [item["mae"] for item in results], marker="s", label="MAE")
    axis.set(xlabel="noise level", ylabel="error", title=f"{mode} PINN noise robustness")
    axis.legend()
    paths = save_figure(figure, destination / "noise_metrics.png", formats=["png", "svg"])
    metadata_path = destination / "figure_metadata.json"
    metadata_path.write_text(json.dumps({"mode": mode, "levels": levels, "configuration": config.model_dump(mode="json"), "figures": [str(path) for path in paths]}, indent=2, default=str), encoding="utf-8")
    return {"status": "COMPLETED", "results": str(report_path), "figures": [str(path) for path in paths], "figure_metadata": str(metadata_path), "levels": levels}
