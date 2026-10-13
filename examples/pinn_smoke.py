"""Run a tiny deterministic PINN and GL comparison smoke experiment.

This validates execution and artifact generation only.  It does not reproduce
either paper or establish prediction accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from research_plotter.config import PINNConfig, Provenance
from research_plotter.models import ModelSpec, Parameter, StateVariable
from research_plotter.pinn import StateNetwork, run_pinn_experiment, torch_rhs_from_model
from research_plotter.pipelines.pinn_experiment import run_pinn_noise_export
from research_plotter.plotting.core import save_figure
from research_plotter.plotting.line import multi_line_plot
from research_plotter.solvers import GrunwaldLetnikovSolver
from research_plotter.utils.numerics import error_metrics, uniform_time_grid


def main() -> None:
    np.random.seed(2026)
    torch.manual_seed(2026)
    output = Path("outputs/reports/pinn_smoke")
    output.mkdir(parents=True, exist_ok=True)

    model = ModelSpec.from_expressions(
        name="smoke_decay",
        state_variables=[StateVariable(name="x", nonnegative=True, initial_value=1.0)],
        parameters=[Parameter(name="rate", value=0.4)],
        expressions={"x": "-rate*x"},
        initial_conditions={"x": 1.0},
        derivative_type="fractional",
        fractional_order=0.8,
    )
    time = uniform_time_grid(0.0, 0.8, 0.1)
    gl_reference = GrunwaldLetnikovSolver().solve(
        model, None, None, time, {"fractional_order": 0.8}
    )
    if not np.isfinite(gl_reference.states).all():
        raise RuntimeError("GL reference produced non-finite states; comparison is blocked")

    config = PINNConfig(
        enabled=True,
        hidden_layers=[8],
        activation="tanh",
        optimizer="Adam",
        learning_rate=0.01,
        epochs=5,
        collocation_points=len(time),
        data_points=len(time),
        physics_loss_weight=0.1,
        initial_condition_loss_weight=1.0,
        provenance=Provenance.USER_PROVIDED,
    )
    rhs = torch_rhs_from_model(model)
    experiments = {}
    for mode in ("baseline", "integer", "fractional"):
        torch.manual_seed(2026)
        experiment = run_pinn_experiment(
            time,
            gl_reference.states,
            config,
            mode=mode,
            rhs=None if mode == "baseline" else rhs,
            fractional_order=0.8,
        )
        experiments[mode] = experiment

    fractional = experiments["fractional"]
    checkpoint = fractional.training.save(
        output / "fractional_model.pt", configuration=config.model_dump(mode="json")
    )
    restored = fractional.training.load(
        StateNetwork(output_dim=1, hidden_layers=(8,), activation="tanh"), checkpoint
    )
    restored_prediction = restored.predict(time).detach().cpu().numpy()
    checkpoint_round_trip = bool(np.allclose(restored_prediction, fractional.prediction))

    noise_reference = gl_reference.states.copy()
    torch.manual_seed(2026)
    noise_report = run_pinn_noise_export(
        model, gl_reference, config, (0.0, 0.1, 0.2), output / "noise",
        mode="baseline", seed=2026,
    )
    noise_results = json.loads(Path(noise_report["results"]).read_text(encoding="utf-8")) if noise_report.get("results") else []
    reference_unchanged = bool(np.array_equal(noise_reference, gl_reference.states))

    comparison_metrics = error_metrics(gl_reference.states, fractional.prediction)
    finite_comparison = bool(
        np.isfinite(gl_reference.states).all()
        and np.isfinite(fractional.prediction).all()
        and all(np.isfinite(value) for value in comparison_metrics.values())
        and np.array_equal(time, gl_reference.time)
    )
    status = "UNVERIFIED" if finite_comparison else "BLOCKED"
    comparison = {
        "status": status,
        "agreement_status": "UNVERIFIED",
        "reference": gl_reference.solver_name,
        "candidate": "fractional PINN",
        "time_grids_match": bool(np.array_equal(time, gl_reference.time)),
        "metrics": comparison_metrics,
        "note": "Finite same-grid comparison completed, but this smoke run has no acceptance threshold and does not establish paper reproduction.",
    }
    comparison_path = output / "gl_pinn_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
    figure = multi_line_plot(
        time,
        {"GL reference": gl_reference.states[:, 0], "fractional PINN": fractional.prediction[:, 0]},
        xlabel="time",
        ylabel="x",
        title="GL reference vs fractional PINN (smoke test)",
    )
    figure_paths = save_figure(figure, output / "gl_pinn_comparison.png", formats=["png", "svg"])

    mode_report = {
        mode: {
            "model_type": mode,
            "configuration": experiment.configuration,
            "initial_loss": experiment.initial_loss,
            "final_loss": experiment.final_loss,
            "runtime_seconds": experiment.training.runtime_seconds,
            "metrics": experiment.metrics,
            "finite_prediction": bool(np.isfinite(experiment.prediction).all()),
            "warnings": experiment.warnings,
        }
        for mode, experiment in experiments.items()
    }
    report = {
        "seed": 2026,
        "model": model.name,
        "configuration": config.model_dump(mode="json"),
        "modes": mode_report,
        "checkpoint": str(checkpoint),
        "checkpoint_round_trip": checkpoint_round_trip,
        "noise_export": noise_report,
        "noise_results": noise_report.get("results"),
        "noise_levels": [item["noise_level"] for item in noise_results],
        "reference_unchanged_by_noise": reference_unchanged,
        "gl_pinn_comparison": str(comparison_path),
        "comparison_figures": [str(path) for path in figure_paths],
    }
    (output / "smoke_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
