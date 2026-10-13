from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from research_plotter.models.equation_system import EquationSystem
from research_plotter.models.parameter import Parameter
from research_plotter.models.state_variable import StateVariable

RhsFunction = Callable[[float, np.ndarray, Mapping[str, float]], np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    """Generic coupled-system specification; plotting knows nothing about it."""

    name: str
    independent_variable: str
    state_variables: tuple[StateVariable, ...]
    parameters: Mapping[str, Parameter]
    rhs: RhsFunction
    initial_conditions: Mapping[str, float]
    derivative_type: str = "ordinary"
    fractional_order: float | None = None
    solver_requirements: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def state_names(self) -> list[str]:
        return [state.name for state in self.state_variables]

    @property
    def parameter_values(self) -> dict[str, float]:
        return {name: parameter.value for name, parameter in self.parameters.items()}

    def initial_vector(self, overrides: Mapping[str, float] | None = None) -> np.ndarray:
        initial = dict(self.initial_conditions)
        initial.update(overrides or {})
        missing = set(self.state_names) - set(initial)
        if missing:
            raise ValueError(f"Missing initial conditions: {sorted(missing)}")
        values = np.asarray([initial[name] for name in self.state_names], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Initial conditions must be finite")
        for state, value in zip(self.state_variables, values):
            if state.nonnegative and value < 0:
                raise ValueError(f"Initial condition for nonnegative state {state.name} cannot be negative")
        return values

    def validate_parameters(self, overrides: Mapping[str, float] | None = None) -> dict[str, float]:
        values = self.parameter_values
        values.update(overrides or {})
        unknown = set(values) - set(self.parameters)
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}")
        return {name: self.parameters[name].validate_value(float(value)) for name, value in values.items()}

    def with_parameter_values(self, overrides: Mapping[str, float]) -> "ModelSpec":
        checked = self.validate_parameters(overrides)
        parameters = {name: parameter.model_copy(update={"value": checked[name]}) for name, parameter in self.parameters.items()}
        return replace(self, parameters=parameters)

    @classmethod
    def from_expressions(
        cls,
        *,
        name: str,
        state_variables: list[StateVariable],
        parameters: list[Parameter],
        expressions: Mapping[str, str],
        initial_conditions: Mapping[str, float],
        independent_variable: str = "time",
        derivative_type: str = "ordinary",
        fractional_order: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ModelSpec":
        state_names = [state.name for state in state_variables]
        parameter_names = [parameter.name for parameter in parameters]
        if len(set(state_names)) != len(state_names):
            raise ValueError("State variable names must be unique")
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("Parameter names must be unique")
        if independent_variable in set(state_names) | set(parameter_names):
            raise ValueError("The independent variable name must not collide with a state or parameter")
        equations = EquationSystem(
            state_names, parameter_names, expressions, independent_variable
        )
        return cls(
            name=name,
            independent_variable=independent_variable,
            state_variables=tuple(state_variables),
            parameters={parameter.name: parameter for parameter in parameters},
            rhs=equations.evaluate,
            initial_conditions=dict(initial_conditions),
            derivative_type=derivative_type,
            fractional_order=fractional_order,
            metadata={**(metadata or {}), "expressions": dict(expressions)},
        )
