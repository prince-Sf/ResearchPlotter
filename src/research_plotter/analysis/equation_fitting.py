"""Safe empirical curve fitting over digitized graph data.

Only expressions accepted by the existing FormulaSpec AST allowlist are
evaluated. Curve fitting proposes approximations to sampled pixels; it does
not identify or recover the original paper model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import OptimizeWarning, curve_fit
import warnings as python_warnings

from research_plotter.models.formula_spec import FormulaSpec, evaluate_formula, parse_equation_text, parse_safe_expression


@dataclass(frozen=True)
class EquationFitResult:
    equation: str
    latex: str
    parameters: Mapping[str, float]
    uncertainties: Mapping[str, float | None]
    parameter_bounds: Mapping[str, tuple[float, float]]
    metrics: Mapping[str, float | None]
    fitted_y: np.ndarray
    residuals: np.ndarray
    x: np.ndarray
    status: str
    method: str
    warnings: tuple[str, ...] = ()

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, object]:
        def json_bound(value: float) -> float | str:
            if np.isneginf(value):
                return "-inf"
            if np.isposinf(value):
                return "inf"
            return float(value)

        output: dict[str, object] = {
            "equation": self.equation,
            "latex": self.latex,
            "parameters": dict(self.parameters),
            "parameter_uncertainties_local_covariance": dict(self.uncertainties),
            "parameter_bounds": {key: [json_bound(value[0]), json_bound(value[1])] for key, value in self.parameter_bounds.items()},
            "metrics": dict(self.metrics),
            "status": self.status,
            "method": self.method,
            "warnings": list(self.warnings),
            "evidence_status": "EMPIRICAL_FIT",
            "interpretation": "Empirical candidate fitted to digitized graph pixels; not evidence of the original equation.",
        }
        if include_arrays:
            output["x"] = self.x.tolist()
            output["fitted_y"] = self.fitted_y.tolist()
            output["residuals"] = self.residuals.tolist()
        return output


@dataclass(frozen=True)
class CandidateFit:
    family: str
    result: EquationFitResult
    holdout_rmse: float | None
    bic: float | None
    complexity: int
    holdout_metrics: Mapping[str, float | None] | None = None
    evidence_status: str = "EMPIRICAL_FIT"

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "equation": self.result.equation,
            "parameters": dict(self.result.parameters),
            "metrics": dict(self.result.metrics),
            "holdout_rmse": self.holdout_rmse,
            "holdout_metrics": dict(self.holdout_metrics or {}),
            "bic": self.bic,
            "complexity": self.complexity,
            "status": self.result.status,
            "evidence_status": self.evidence_status,
            "interpretation": "Empirical candidate fitted to digitized graph data; not evidence of the original equation or source model.",
            "parameter_uncertainties_local_covariance": dict(self.result.uncertainties),
            "warnings": list(self.result.warnings),
        }


def _metrics(observed: np.ndarray, predicted: np.ndarray, parameter_count: int) -> dict[str, float | None]:
    residuals = observed - predicted
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    max_error = float(np.max(np.abs(residuals)))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = float(1.0 - np.sum(residuals ** 2) / total) if total > 0 else None
    count = max(1, len(observed))
    rss = max(float(np.sum(residuals ** 2)), np.finfo(float).tiny)
    aic = float(count * np.log(rss / count) + 2 * parameter_count)
    bic = float(count * np.log(rss / count) + parameter_count * np.log(count))
    return {"mae": mae, "mse": mse, "rmse": rmse, "max_absolute_error": max_error, "r_squared": r_squared, "aic": aic, "bic": bic}


def _validate_xy(data: pd.DataFrame, x_column: str, y_column: str) -> tuple[np.ndarray, np.ndarray]:
    if x_column not in data or y_column not in data:
        raise ValueError(f"Selected data must contain columns {x_column!r} and {y_column!r}")
    x = np.asarray(data[x_column], dtype=float)
    y = np.asarray(data[y_column], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 4:
        raise ValueError("At least four finite digitized points are required for fitting")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if np.unique(x).size < 3:
        raise ValueError("At least three distinct x values are required for fitting")
    return x, y


def _compile_formula(template: str, independent_variable: str) -> FormulaSpec:
    spec = parse_equation_text(template, independent_variable=independent_variable)
    if spec.kind != "explicit" or len(spec.dependent_variables) != 1:
        raise ValueError("Curve fitting requires one explicit equation such as y = A*exp(-k*x)")
    for expression in spec.expressions.values():
        parse_safe_expression(expression)
    if not spec.parameter_names:
        raise ValueError("The equation template contains no free parameters to estimate")
    return spec


def fit_equation(
    data: pd.DataFrame,
    template: str,
    *,
    x_column: str = "x",
    y_column: str = "y",
    independent_variable: str = "x",
    initial_guesses: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    max_evaluations: int = 20_000,
) -> EquationFitResult:
    """Estimate free template parameters with bounded nonlinear least squares."""
    spec = _compile_formula(template, independent_variable)
    x, observed = _validate_xy(data, x_column, y_column)
    fixed = dict(spec.parameter_values)
    parameter_names = [name for name in spec.parameter_names if name not in fixed]
    if not parameter_names:
        raise ValueError("All template parameters already have fixed numeric values; remove values to estimate them")
    guesses = dict(initial_guesses or {})
    limits = dict(bounds or {})
    allowed = set(parameter_names)
    if set(guesses) - allowed or set(limits) - allowed:
        raise ValueError("Initial guesses and bounds must name free parameters in the equation")
    if any(not np.isfinite(value) for value in guesses.values()):
        raise ValueError("Initial guesses must be finite numbers")
    p0 = np.asarray([float(guesses.get(name, 1.0)) for name in parameter_names], dtype=float)
    lower, upper = [], []
    resolved_bounds: dict[str, tuple[float, float]] = {}
    for name, guess in zip(parameter_names, p0):
        low, high = limits.get(name, (-np.inf, np.inf))
        low, high = float(low), float(high)
        if np.isnan(low) or np.isnan(high) or low >= high:
            raise ValueError(f"Bounds for {name} must be ordered and non-NaN")
        if guess < low or guess > high:
            raise ValueError(f"Initial guess for {name} is outside its bounds")
        lower.append(low)
        upper.append(high)
        resolved_bounds[name] = (low, high)

    dependent = spec.dependent_variables[0]

    def predictor(values: np.ndarray, *parameters: float) -> np.ndarray:
        substitutions = {**fixed, **dict(zip(parameter_names, parameters))}
        return evaluate_formula(spec, np.asarray(values, dtype=float), substitutions)[dependent]

    try:
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always", OptimizeWarning)
            estimates, covariance = curve_fit(
                predictor,
                x,
                observed,
                p0=p0,
                bounds=(np.asarray(lower), np.asarray(upper)),
                maxfev=max_evaluations,
                method="trf",
            )
        warning_messages = [str(item.message) for item in caught]
    except (RuntimeError, ValueError, FloatingPointError, OverflowError, RuntimeWarning) as error:
        raise ValueError(f"Equation fit did not converge: {error}") from error

    fitted = np.asarray(predictor(x, *estimates), dtype=float)
    if not np.isfinite(fitted).all():
        raise ValueError("Fitted expression produced NaN or infinity on the digitized x range")
    residuals = observed - fitted
    uncertainty: dict[str, float | None] = {}
    covariance_reliable = (
        covariance.shape == (len(parameter_names), len(parameter_names))
        and np.isfinite(covariance).all()
        and len(x) > len(parameter_names)
    )
    if covariance_reliable:
        try:
            covariance_reliable = bool(np.linalg.cond(covariance) < 1e12)
        except np.linalg.LinAlgError:
            covariance_reliable = False
    if covariance_reliable:
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        uncertainty.update({name: float(value) for name, value in zip(parameter_names, standard_errors)})
    else:
        uncertainty.update({name: None for name in parameter_names})
        warning_messages.append("Parameter covariance is singular or poorly conditioned; uncertainty estimates are not reliable.")
    parameter_values = {**fixed, **{name: float(value) for name, value in zip(parameter_names, estimates)}}
    expression = parse_safe_expression(spec.expressions[dependent])
    equation = f"{dependent} = {sp.sstr(expression)}"
    latex = rf"{dependent}({independent_variable}) = {sp.latex(expression)}"
    if warning_messages:
        warning_messages.append("Local covariance uncertainty does not include digitization, calibration, or model-selection uncertainty.")
    return EquationFitResult(
        equation=equation,
        latex=latex,
        parameters=parameter_values,
        uncertainties=uncertainty,
        parameter_bounds=resolved_bounds,
        metrics=_metrics(observed, fitted, len(parameter_names)),
        fitted_y=fitted,
        residuals=residuals,
        x=x,
        status="FIT_SUCCESSFUL",
        method="scipy.optimize.curve_fit (bounded trust-region least squares)",
        warnings=tuple(dict.fromkeys(warning_messages)),
    )


def _candidate_templates(
    x: np.ndarray,
    y: np.ndarray,
    *,
    domain_x: np.ndarray | None = None,
    domain_y: np.ndarray | None = None,
) -> list[tuple[str, str, dict[str, float]]]:
    templates: list[tuple[str, str, dict[str, float]]] = []
    coefficients = np.polyfit(x, y, deg=1)
    templates.append(("linear", "y = a*x + b", {"a": float(coefficients[0]), "b": float(coefficients[1])}))
    for degree in (2, 3, 4):
        if len(x) <= degree + 2:
            continue
        values = np.polyfit(x, y, deg=degree)
        letters = [chr(ord("a") + index) for index in range(degree + 1)]
        expression = "+".join(f"{letter}*x**{power}" for letter, power in zip(letters, range(degree, 0, -1))) + f"+{letters[-1]}"
        initial = {letter: float(value) for letter, value in zip(letters, values)}
        templates.append((f"polynomial degree {degree}", f"y = {expression}", initial))
    y_range = max(float(np.ptp(y)), 1e-6)
    x_range = max(float(np.ptp(x)), 1e-6)
    y_min, y_max = float(np.min(y)), float(np.max(y))
    templates.extend([
        ("exponential", "y = a*exp(b*x)+c", {"a": y_range, "b": 0.0, "c": y_min}),
        ("sinusoidal", "y = a*sin(b*x+c)+d", {"a": y_range / 2, "b": 2 * np.pi / x_range, "c": 0.0, "d": float(np.mean(y))}),
        ("damped sinusoidal", "y = a*exp(-k*x)*sin(w*x+p)+c", {"a": y_range / 2, "k": 0.01 / x_range, "w": 2 * np.pi / x_range, "p": 0.0, "c": float(np.mean(y))}),
        ("logistic", "y = a/(1+exp(-k*(x-x0)))+c", {"a": y_range, "k": 1.0 / x_range, "x0": float(np.median(x)), "c": y_min}),
    ])
    valid_x = x if domain_x is None else np.asarray(domain_x, dtype=float)
    valid_y = y if domain_y is None else np.asarray(domain_y, dtype=float)
    if np.all(valid_x > 0):
        templates.append(("logarithmic", "y = a*log(x)+b", {"a": y_range, "b": float(np.mean(y))}))
    if np.all(valid_x > 0) and np.all(valid_y > 0):
        templates.append(("power law", "y = a*x**b", {"a": float(np.median(y)), "b": 1.0}))
    return templates


def discover_equation_candidates(
    data: pd.DataFrame,
    *,
    x_column: str = "x",
    y_column: str = "y",
    independent_variable: str = "x",
    max_candidates: int = 5,
    max_evaluations_per_fit: int = 5000,
) -> list[CandidateFit]:
    """Compare a small allowlisted family set with deterministic holdout points.

    Every fifth point is held out when enough points exist. Family ranking uses
    holdout RMSE first and BIC as a complexity-aware tie-breaker. Results remain
    exploratory candidates and are never described as recovered source models.
    """
    x, y = _validate_xy(data, x_column, y_column)
    if len(x) < 12:
        raise ValueError("At least 12 points are required for candidate discovery with held-out validation")
    if max_evaluations_per_fit < 100:
        raise ValueError("max_evaluations_per_fit must be at least 100")
    validation_mask = np.arange(len(x)) % 5 == 0
    train = pd.DataFrame({x_column: x[~validation_mask], y_column: y[~validation_mask]})
    validation_x, validation_y = x[validation_mask], y[validation_mask]
    successes: list[CandidateFit] = []
    failures: list[str] = []
    # Candidate scale guesses are derived only from training observations so
    # the deterministic holdout set is not used during model initialization.
    for family, template, initial in _candidate_templates(
        x[~validation_mask], y[~validation_mask], domain_x=x, domain_y=y
    ):
        try:
            spec = _compile_formula(template, independent_variable)
            free = set(spec.parameter_names)
            train_initial = {name: value for name, value in initial.items() if name in free}
            train_fit = fit_equation(
                train,
                template,
                x_column=x_column,
                y_column=y_column,
                independent_variable=independent_variable,
                initial_guesses=train_initial,
                max_evaluations=max_evaluations_per_fit,
            )
            validation_formula = _compile_formula(template, independent_variable)
            train_names = list(validation_formula.parameter_names)
            validation_prediction = evaluate_formula(
                validation_formula,
                validation_x,
                train_fit.parameters,
            )[validation_formula.dependent_variables[0]]
            if validation_prediction.shape != validation_y.shape or not np.isfinite(validation_prediction).all():
                raise ValueError("Candidate produced non-finite or mismatched held-out predictions")
            holdout_rmse = float(np.sqrt(np.mean((validation_y - validation_prediction) ** 2)))
            holdout_residuals = validation_y - validation_prediction
            holdout_total = float(np.sum((validation_y - np.mean(validation_y)) ** 2))
            holdout_metrics = {
                "rmse": holdout_rmse,
                "mae": float(np.mean(np.abs(holdout_residuals))),
                "r_squared": float(1.0 - np.sum(holdout_residuals**2) / holdout_total) if holdout_total > 0 else None,
            }
            final_fit = fit_equation(
                data,
                template,
                x_column=x_column,
                y_column=y_column,
                independent_variable=independent_variable,
                initial_guesses=train_initial,
                max_evaluations=max_evaluations_per_fit,
            )
            successes.append(CandidateFit(
                family=family,
                result=final_fit,
                holdout_rmse=holdout_rmse,
                bic=float(final_fit.metrics["bic"]) if final_fit.metrics["bic"] is not None else None,
                complexity=len(train_names),
                holdout_metrics=holdout_metrics,
            ))
        except (ValueError, RuntimeError, OverflowError, FloatingPointError) as error:
            failures.append(f"{family}: {error}")
    if not successes:
        raise ValueError("No candidate equation family converged. " + "; ".join(failures[:3]))
    successes.sort(key=lambda candidate: (candidate.holdout_rmse if candidate.holdout_rmse is not None else np.inf, candidate.bic if candidate.bic is not None else np.inf, candidate.complexity))
    if failures and successes:
        best = successes[0]
        notes = tuple(dict.fromkeys((*best.result.warnings, f"Candidate families not ranked: {'; '.join(failures[:3])}")))
        successes[0] = replace(best, result=replace(best.result, warnings=notes))

    # Collapse only numerically indistinguishable predictions; similarly good
    # but meaningfully different curves remain visible as competing hypotheses.
    distinct: list[CandidateFit] = []
    prediction_tolerance = max(float(np.ptp(y)), 1.0) * 1e-8
    for candidate in successes:
        if any(
            candidate.result.fitted_y.shape == prior.result.fitted_y.shape
            and np.max(np.abs(candidate.result.fitted_y - prior.result.fitted_y)) <= prediction_tolerance
            for prior in distinct
        ):
            continue
        distinct.append(candidate)
    return distinct[:max(1, min(max_candidates, 5, len(distinct)))]

