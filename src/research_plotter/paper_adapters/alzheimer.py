"""Paper 2 adapter for the published five-state Alzheimer model."""

from __future__ import annotations

from typing import Any, Mapping

from research_plotter.models import ModelSpec, Parameter, StateVariable


PAPER_2_PARAMETERS = {
    "Pi1N": 1.0, "rho": 0.025, "alpha": 0.08, "phi1": 0.02, "beta1": 0.06,
    "gamma": 0.00017, "phi2": 0.00019, "beta2": 0.002, "db": 9.51, "kappa": 0.025,
    "beta3": 0.001, "dmu": 0.277, "phi3": 0.015,
}
PAPER_2_INITIAL = {"FN": 0.14, "IN": 0.0, "Abeta": 1e-6, "Tmu": 1e-6, "Mrho": 0.02}


def build_alzheimer_model(
    initial_conditions: Mapping[str, float] | None = None,
    *,
    fractional_order: float = 0.9,
    parameters: Mapping[str, float] | None = None,
    additional_parameters: Mapping[str, float] | None = None,
    expressions: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ModelSpec:
    initial = dict(PAPER_2_INITIAL)
    if initial_conditions is not None: initial.update(initial_conditions)
    if set(initial) != set(PAPER_2_INITIAL): raise ValueError(f"Paper 2 initial conditions must contain {sorted(PAPER_2_INITIAL)}")
    states = [StateVariable(name=name, nonnegative=True, initial_value=initial[name]) for name in PAPER_2_INITIAL]
    parameter_values = dict(PAPER_2_PARAMETERS)
    if parameters is not None:
        unknown = set(parameters) - set(parameter_values)
        if unknown:
            raise ValueError(f"Unknown Paper 2 parameters: {sorted(unknown)}")
        parameter_values.update({name: float(value) for name, value in parameters.items()})
    if additional_parameters is not None:
        collisions = set(additional_parameters) & set(parameter_values)
        if collisions:
            raise ValueError(f"Control parameters collide with Paper 2 parameters: {sorted(collisions)}")
        parameter_values.update({name: float(value) for name, value in additional_parameters.items()})
    parameter_specs = [Parameter(name=name, value=value) for name, value in parameter_values.items()]
    default_expressions = {
        "FN": "Pi1N+rho*Tmu-alpha*FN*Abeta-phi1*FN",
        "IN": "alpha*FN*Abeta-beta1*IN*Mrho-(gamma+phi2)*IN",
        "Abeta": "gamma*IN-beta2*Abeta*Mrho-(db+kappa)*Abeta",
        "Tmu": "kappa*Abeta-beta3*Tmu*Mrho-(dmu+rho)*Tmu",
        "Mrho": "(beta1*IN+beta2*Abeta+beta3*Tmu)*Mrho-phi3*Mrho",
    }
    selected_expressions = dict(expressions or default_expressions)
    if set(selected_expressions) != set(PAPER_2_INITIAL):
        raise ValueError(f"Paper 2 equations must define exactly {sorted(PAPER_2_INITIAL)}")
    model_metadata = {"source_document": "papers/fninf-20-1748481.pdf", "paper_equation": 9,
                      "controls": ["z1(t)", "z2(t)"], "control_equations_status": "BLOCKED",
                      "nsfd_status": "PARTIAL", "training_hyperparameters_status": "BLOCKED"}
    model_metadata.update(metadata or {})
    return ModelSpec.from_expressions(
        name="paper_2_fractional_alzheimer",
        state_variables=states, parameters=parameter_specs,
        expressions=selected_expressions,
        initial_conditions=initial, derivative_type="fractional", fractional_order=fractional_order,
        metadata=model_metadata,
    )


def alzheimer_analysis_notes() -> dict[str, object]:
    return {"status": "PARTIAL", "confirmed": ["equation 9 state equations", "Table 1 parameters and initial values", "GL weight definition", "logarithmic sensitivity formula", "noise levels 0%-20%", "fractional/integer PINN and baseline comparison"], "unresolved": ["controlled equations", "NSFD details", "network architecture/training hyperparameters", "figure scale/text inconsistencies"], "assumptions": ["Paper equation state symbols are normalized to ASCII names FN, IN, Abeta, Tmu, Mrho."]}
