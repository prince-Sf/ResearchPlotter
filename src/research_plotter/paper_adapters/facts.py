"""Evidence-backed facts for the bundled paper adapters.

The generic PDF parser asks this registry for known facts, while all source-
specific decisions stay in the adapter layer. Unknown documents simply return
an empty fact set and are handled conservatively by the parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PAPER_FACTS: dict[str, dict[str, Any]] = {
    "paper_1": {
        "title": "Dynamic analysis of fractal–fractional cancer model under chemotherapy drug with generalized Mittag-Leffler kernel",
        "authors": "Hardik Joshi; Mehmet Yavuz; Osman Taylan; Abdulaziz Alkabaa",
        "state_variables": ["S(t)", "E(t)", "T(t)", "C(t)"],
        "parameter_tokens": ["delta1", "ks", "gamma", "mu", "xi1", "delta3", "g", "k", "delta4", "kT", "delta2", "J(t)"],
        "parameter_values": {"delta1": -0.02825, "ks": 1, "gamma": .17, "mu": .03, "xi1": .1245, "delta3": 1, "g": .18, "k": 1e-9, "delta4": .9, "kT": .9, "delta2": 6.4, "J(t)": 1},
        "initial_conditions": [],
        "initial_conditions_status": "NEEDS_REVIEW",
        "initial_conditions_page": None,
        "initial_conditions_notes": ["The source states symbolic S(0)=S0, E(0)=E0, T(0)=T0 and C(0)=C0=0 provided J0=0; numerical values used for the reported simulation are not established."],
        "unresolved": [
            "Paper 1 source does not clearly provide all simulation initial conditions and step-size details.",
            "Paper 1 displayed recurrence has an index-at-zero ambiguity requiring review.",
        ],
    },
    "paper_2": {
        "title": "A Physics Informed Neural Network (PINN) framework for fractional order modeling of Alzheimer’s disease",
        "authors": "Adnan Mehmood; Muhammad Farman; Farkhanda Afzal; Kottakkaran Sooppy Nisar; Mohammed Altaf Ahmed; Mohamed Hafez",
        "state_variables": ["FN(t)", "IN(t)", "Aβ(t)", "Tμ(t)", "Mρ(t)"],
        "parameter_tokens": ["Pi1N", "rho", "alpha", "phi1", "beta1", "gamma", "phi2", "beta2", "db", "kappa", "beta3", "dmu", "phi3"],
        "parameter_values": {"Pi1N": 1, "rho": .025, "alpha": .08, "phi1": .02, "beta1": .06, "gamma": .00017, "phi2": .00019, "beta2": .002, "db": 9.51, "kappa": .025, "beta3": .001, "dmu": .277, "phi3": .015},
        "initial_conditions": {"FN": .14, "IN": 0, "Aβ": 1e-6, "Tμ": 1e-6, "Mρ": .02},
        "initial_conditions_status": "CONFIRMED",
        "initial_conditions_page": 5,
        "initial_conditions_notes": [],
        "unresolved": [
            "Paper 2 omits sufficient controlled-equation details for exact control reproduction.",
            "Paper 2 does not specify enough NSFD/training hyperparameters for exact PINN reproduction.",
            "Some Paper 2 figure descriptions/scales conflict with surrounding text or tables.",
        ],
    },
}


def identify_document(path: str | Path) -> str | None:
    name = Path(path).name.lower()
    if name == "333.pdf" or "108565" in name:
        return "paper_1"
    if name == "fninf-20-1748481.pdf" or "1748481" in name:
        return "paper_2"
    return None


def known_facts(path: str | Path) -> dict[str, Any]:
    key = identify_document(path)
    return dict(PAPER_FACTS.get(key, {}))


def enrich_figure_facts(figures: list[Any], document_key: str | None) -> list[Any]:
    """Add only values explicitly described by a bundled source paper."""
    by_id = {figure.figure_id: figure for figure in figures}
    if document_key == "paper_1":
        model_parameters = dict(PAPER_FACTS["paper_1"]["parameter_values"])
        required_parameters = list(PAPER_FACTS["paper_1"]["parameter_tokens"])
        missing = [
            "Numerical initial conditions for S(0), E(0), and T(0); the source gives C(0)=0 only conditionally when J(0)=0, so verify that condition.",
            "Simulation time-step value used in the reported plots.",
            "Review the displayed recurrence indexing around the first history-sum term.",
        ]
        for figure in by_id.values():
            number = int(figure.figure_id.removeprefix("figure_"))
            output_state = {
                2: "S", 3: "E", 4: "T", 5: "C", 6: "S", 7: "E", 8: "T", 9: "C",
                10: "S", 11: "S", 12: "T",
            }.get(number)
            figure.x_variable = "time"
            figure.y_variable = output_state
            figure.variables = ["S", "E", "T", "C"] if number == 1 else ([output_state] if output_state else [])
            figure.plot_type = "line"
            figure.required_equation_ids = ["equation_07", "equation_58"]
            figure.equation_link_status = type(figure.equation_link_status).PARTIAL
            figure.reproducibility_status = type(figure.reproducibility_status).BLOCKED
            figure.required_parameters = required_parameters
            figure.required_initial_conditions = ["S(0)", "E(0)", "T(0)", "C(0)"]
            figure.missing_information = list(missing)
            figure.solver = "Atangana-Baleanu Caputo fractal-fractional history recurrence (source Eq. 58; review indexing)"
            figure.time_range = (0.0, 90.0)
            figure.fixed_parameters = dict(model_parameters)
            figure.parameter_values = dict(model_parameters)
            figure.provenance.update({
                "equations": "SOURCE_PAPER",
                "equation_link": "SOURCE_PAPER",
                "parameters": "SOURCE_PAPER",
                "initial_conditions": "UNKNOWN",
                "time_range": "SOURCE_PAPER",
                "solver": "SOURCE_PAPER",
            })
            figure.notes.append("Paper-level linkage: the results section uses model Eq. (7) and the numerical scheme ending at Eq. (58); figure-specific equation assignments are not explicitly enumerated.")
            figure.notes.append("The plotted time axis spans 0-90; the simulation step and numerical initial-condition values are not specified here.")
            if number == 5:
                figure.inset_zoom = "The published panel includes a zoom inset over the early 0-10 time interval."
        for number in range(2, 6):
            figure = by_id.get(f"figure_{number:02d}")
            if figure:
                figure.varying_parameters = {"alpha": [.95, .97, 1.0]}
                figure.fixed_parameters = model_parameters | {"beta": .6}
                figure.parameter_values = figure.fixed_parameters | {"alpha": [.95, .97, 1.0]}
                figure.provenance.update({"varying_parameters": "SOURCE_PAPER", "fixed_parameters": "SOURCE_PAPER"})
                figure.notes.append("Values read from the figure description; initial conditions and step size remain unresolved.")
        for number in range(6, 10):
            figure = by_id.get(f"figure_{number:02d}")
            if figure:
                figure.varying_parameters = {"beta": [.6, .7, .8]}
                figure.fixed_parameters = model_parameters | {"alpha": .95}
                figure.parameter_values = figure.fixed_parameters | {"beta": [.6, .7, .8]}
                figure.provenance.update({"varying_parameters": "SOURCE_PAPER", "fixed_parameters": "SOURCE_PAPER"})
        if (figure := by_id.get("figure_10")):
            figure.varying_parameters = {"delta2": [5.4, 6.4, 7.4]}
            figure.fixed_parameters = {name: value for name, value in model_parameters.items() if name != "delta2"} | {"alpha": .95, "beta": .6}
            figure.parameter_values = figure.fixed_parameters | {"delta2": [5.4, 6.4, 7.4]}
            figure.provenance["varying_parameters"] = "SOURCE_PAPER"
            figure.notes.append("The source describes an increase from 5.4 to 7.4; intermediate values are not explicitly tabulated.")
        for number, parameter in ((11, "ks"), (12, "kT")):
            if (figure := by_id.get(f"figure_{number:02d}")):
                figure.varying_parameters = {parameter: []}
                figure.fixed_parameters = {name: value for name, value in model_parameters.items() if name != parameter} | {"alpha": .95, "beta": .6}
                figure.parameter_values = dict(figure.fixed_parameters)
                figure.provenance["varying_parameters"] = "UNKNOWN"
                figure.missing_information.append(f"Published {parameter} values for this sweep are not listed; enter only reviewed USER_PROVIDED values.")
                figure.notes.append("Parameter is varied, but the plotted values are not explicitly tabulated in the source.")
    elif document_key == "paper_2":
        for number in range(9, 14):
            if (figure := by_id.get(f"figure_{number:02d}")):
                figure.varying_parameters = {"alpha": [.7, .8, .9, 1.0]}
                figure.notes.append("The surrounding text reports alpha={0.7,0.8,0.9,1.0}; displayed legends and scales should be reviewed.")
    return list(by_id.values())
