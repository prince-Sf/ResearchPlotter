import numpy as np
import pytest

from research_plotter.models import ModelSpec, Parameter, StateVariable


def test_validated_expression_model():
    model = ModelSpec.from_expressions(name="decay", state_variables=[StateVariable(name="x")], parameters=[Parameter(name="rate", value=.2)], expressions={"x": "-rate*x"}, initial_conditions={"x": 1})
    assert np.allclose(model.rhs(0, np.array([1.0]), {"rate": .2}), [-.2])


def test_expression_allows_safe_scientific_constants():
    model = ModelSpec.from_expressions(name="constant", state_variables=[StateVariable(name="x")], parameters=[], expressions={"x": "pi"}, initial_conditions={"x": 0})
    assert np.allclose(model.rhs(0, np.array([0.0]), {}), [np.pi])


def test_expression_rejects_unknown_or_code():
    with pytest.raises(ValueError):
        ModelSpec.from_expressions(name="unsafe", state_variables=[StateVariable(name="x")], parameters=[], expressions={"x": "__import__(x)"}, initial_conditions={"x": 1})


def test_model_rejects_duplicate_names():
    with pytest.raises(ValueError):
        ModelSpec.from_expressions(name="duplicate", state_variables=[StateVariable(name="x"), StateVariable(name="x")], parameters=[], expressions={"x": "0"}, initial_conditions={"x": 0})


def test_parameter_bounds():
    parameter = Parameter(name="p", value=1, lower_bound=0, upper_bound=2)
    assert parameter.validate_value(2) == 2
    with pytest.raises(ValueError): parameter.validate_value(3)
    with pytest.raises(ValueError): parameter.validate_value(float("nan"))


def test_nonnegative_initial_condition_validation():
    model = ModelSpec.from_expressions(name="positive", state_variables=[StateVariable(name="x", nonnegative=True)], parameters=[], expressions={"x": "0"}, initial_conditions={"x": 1})
    with pytest.raises(ValueError): model.initial_vector({"x": -1})
