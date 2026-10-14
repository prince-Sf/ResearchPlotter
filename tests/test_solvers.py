import numpy as np
import pytest
from math import gamma

from research_plotter.analysis.validation import validate_simulation
from research_plotter.paper_adapters.alzheimer import build_alzheimer_model
from research_plotter.models import ModelSpec, Parameter, StateVariable
from research_plotter.solvers import ExplicitEulerSolver, GrunwaldLetnikovSolver, ODESolver, grunwald_letnikov_weights
from research_plotter.utils.numerics import uniform_time_grid


def decay_model(derivative_type="ordinary", order=None):
    return ModelSpec.from_expressions(name="decay", state_variables=[StateVariable(name="x")], parameters=[Parameter(name="rate", value=1)], expressions={"x": "-rate*x"}, initial_conditions={"x": 1}, derivative_type=derivative_type, fractional_order=order)


def test_ode_solver_against_exponential():
    result = ODESolver().solve(decay_model(), None, None, uniform_time_grid(0, 1, .1))
    assert abs(result.states[-1, 0] - np.exp(-1)) < 1e-4
    assert result.diagnostics["success"]


def test_euler_solver():
    result = ExplicitEulerSolver().solve(decay_model(), None, None, uniform_time_grid(0, 1, .01))
    assert abs(result.states[-1, 0] - np.exp(-1)) < .01


def test_gl_weights_and_history_solver():
    weights = grunwald_letnikov_weights(.5, 4)
    assert np.allclose(weights, [1, -.5, -.125, -.0625])
    result = GrunwaldLetnikovSolver().solve(decay_model("fractional", .8), None, None, uniform_time_grid(0, .5, .1), {"fractional_order": .8})
    assert result.states.shape == (6, 1)
    assert np.isfinite(result.states).all()
    assert result.diagnostics["number_of_steps"] == 5
    assert result.diagnostics["state_dimension"] == 1


def test_euler_rejects_nonincreasing_grid():
    with pytest.raises(ValueError):
        ExplicitEulerSolver().solve(decay_model(), None, None, np.array([0.0, 0.1, 0.1]))


def test_gl_constant_forcing_refines_toward_caputo_reference():
    model = ModelSpec.from_expressions(
        name="forced",
        state_variables=[StateVariable(name="x")],
        parameters=[],
        expressions={"x": "1"},
        initial_conditions={"x": 0},
        derivative_type="fractional",
        fractional_order=.5,
    )
    errors = []
    for step in (.2, .1, .05):
        grid = uniform_time_grid(0, 1, step)
        result = GrunwaldLetnikovSolver().solve(model, None, None, grid, {"fractional_order": .5})
        exact = grid[-1] ** .5 / gamma(1.5)
        errors.append(abs(result.states[-1, 0] - exact))
    assert errors[2] < errors[1] < errors[0]


def test_paper2_coarse_gl_update_is_diagnosed_as_numerically_invalid():
    model = build_alzheimer_model(fractional_order=.9)
    grid = uniform_time_grid(0, 200, 1.0)
    result = GrunwaldLetnikovSolver().solve(
        model, None, None, grid, {"fractional_order": .9}
    )

    # At the configured initial state, equation (9) gives the fast amyloid
    # decay right-hand side -9.53504e-6. For h=1, the explicit GL step matches
    # the negative value stored in the historical diagnostic CSV.
    assert result.states[1, model.state_names.index("Abeta")] == pytest.approx(-8.53504e-6)
    assert result.diagnostics["completed_steps"] == 17
    assert result.diagnostics["failure_index"] == 18
    assert not result.diagnostics["finite"]
    assert any("non-finite state at index 18" in warning for warning in result.warnings)
    assert validate_simulation(result, model, require_nonnegative=True)["status"] == "INVALID"
