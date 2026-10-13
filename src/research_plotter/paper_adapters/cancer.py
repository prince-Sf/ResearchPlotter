"""Paper 1 adapter for the equations and published parameter table.

The source does not clearly report simulation initial conditions or a step size.
This adapter therefore refuses to invent them and requires the caller to supply
initial conditions explicitly.
"""

from __future__ import annotations

from typing import Mapping

from research_plotter.models import ModelSpec, Parameter, StateVariable


PAPER_1_PARAMETERS = {
    "delta1": -0.02825,
    "ks": 1.0,
    "gamma": 0.17,
    "mu": 0.03,
    "xi1": 0.1245,
    "delta3": 1.0,
    "g": 0.18,
    "k": 1e-9,
    "delta4": 0.9,
    "kT": 0.9,
    "delta2": 6.4,
    "J": 1.0,
}


def build_cancer_model(initial_conditions: Mapping[str, float] | None = None, *, alpha: float = 0.95, beta: float = 0.6) -> ModelSpec:
    if initial_conditions is None:
        raise ValueError("Paper 1 initial conditions are not clearly reported; supply S, E, T, and C explicitly.")
    required = {"S", "E", "T", "C"}
    if set(initial_conditions) != required:
        raise ValueError(f"Paper 1 initial conditions must contain exactly {sorted(required)}")
    states = [StateVariable(name=name, nonnegative=True, initial_value=float(initial_conditions[name])) for name in ("S", "E", "T", "C")]
    parameters = [Parameter(name=name, value=value) for name, value in PAPER_1_PARAMETERS.items()]
    return ModelSpec.from_expressions(
        name="paper_1_fractal_fractional_cancer",
        state_variables=states,
        parameters=parameters,
        expressions={"S": "delta1*S-ks*C*S", "E": "gamma-mu*E+xi1*E*S/(S+1)-delta3*(T+C)*E", "T": "g*(1-k*T)*T-(delta4*E+kT*C)*T", "C": "-delta2*C+J"},
        initial_conditions=dict(initial_conditions), derivative_type="fractal_fractional", fractional_order=alpha,
        metadata={"source_document": "papers/333.pdf", "fractal_dimension": beta, "paper_equation": 7,
                  "initial_conditions_status": "NEEDS_REVIEW", "step_size_status": "NEEDS_REVIEW", "recurrence_status": "NEEDS_REVIEW"},
    )


def cancer_analysis_notes() -> dict[str, object]:
    return {"status": "BLOCKED", "confirmed": ["state variables S,E,T,C", "parameter table values", "Figures 1-12 parameter sweep descriptions", "generalized Mittag-Leffler/fractal-fractional formulation"], "unresolved": ["simulation initial conditions", "numerical step size", "index-at-zero ambiguity in displayed recurrence"], "assumptions": []}
