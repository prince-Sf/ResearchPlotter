import numpy as np

from research_plotter.analysis.noise import noise_experiment
from research_plotter.analysis.parameter_sweep import run_parameter_sweep
from research_plotter.analysis.sensitivity import logarithmic_sensitivity
from research_plotter.analysis.validation import compare_trajectories
from research_plotter.analysis.convergence import run_refinement_experiment
from research_plotter.models import ModelSpec, Parameter, StateVariable
from research_plotter.solvers import GrunwaldLetnikovSolver, ODESolver, SimulationResult
from research_plotter.utils.numerics import error_metrics, uniform_time_grid


def model():
    return ModelSpec.from_expressions(name="decay", state_variables=[StateVariable(name="x")], parameters=[Parameter(name="rate", value=.2)], expressions={"x": "-rate*x"}, initial_conditions={"x": 1})


def test_sweep_and_sensitivity():
    grid = uniform_time_grid(0, 1, .1)
    sweep = run_parameter_sweep(model(), ODESolver(), grid, {"rate": [.1, .2, .3]})
    assert len(sweep.simulations) == 3
    sensitivity = logarithmic_sensitivity(model(), ODESolver(), grid, "rate", output="x")
    assert sensitivity["sensitivity"].shape == grid.shape


def test_noise_and_metrics():
    values = np.ones((4, 2))
    experiment = noise_experiment(values, [0, .1], seed=3)
    assert experiment[0]["mse"] == 0
    assert error_metrics(values, values)["rmse"] == 0


def test_trajectory_comparison_interpolates_different_uniform_grids():
    reference = ODESolver().solve(model(), None, None, uniform_time_grid(0, 1, .1))
    candidate = ODESolver().solve(model(), None, None, uniform_time_grid(0, 1, .2))
    metrics = compare_trajectories(reference, candidate)
    assert metrics["rmse"] < 1e-3


def test_validation_handles_all_nonfinite_states_without_reduction_warning():
    result = SimulationResult(
        time=np.array([0.0, 1.0]),
        states=np.full((2, 1), np.nan),
        state_names=["x"],
        parameters={},
        initial_conditions={"x": 0.0},
        solver_name="test",
        runtime_seconds=0.0,
    )
    from research_plotter.analysis.validation import validate_simulation
    report = validate_simulation(result)
    assert report["status"] == "INVALID"
    assert report["min_state"] is None


def test_fractional_refinement_reports_finite_errors(tmp_path):
    forced = ModelSpec.from_expressions(
        name="fractional_forcing",
        state_variables=[StateVariable(name="x")],
        parameters=[],
        expressions={"x": "1"},
        initial_conditions={"x": 0},
        derivative_type="fractional",
        fractional_order=.5,
    )
    result = run_refinement_experiment(
        forced,
        GrunwaldLetnikovSolver(),
        start=0,
        end=1,
        base_step=.2,
        refinements=(1, 2, 4),
        options={"fractional_order": .5},
    )
    assert result.status == "NUMERICALLY_VALIDATED"
    assert result.errors[1]["rmse"] > result.errors[2]["rmse"] > result.errors[4]["rmse"]
    outputs = result.save(tmp_path)
    assert (tmp_path / "convergence.json").exists()
    assert "figure" in outputs
