"""A deliberately small, validated symbolic-expression layer.

Expressions are not Python programs. They are token-validated before SymPy parses
them, and may only use declared symbols plus a small mathematical function set.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import auto_symbol, convert_xor, parse_expr, standard_transformations

_TOKEN = re.compile(r"^[A-Za-z0-9_+\-*/().,\s^]+$")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}
_CONSTANTS = {"pi": sp.pi, "E": sp.E}
_TRANSFORMS = tuple(item for item in standard_transformations if item is not auto_symbol) + (convert_xor,)


def safe_parse_expression(expression: str, allowed_names: set[str]) -> sp.Expr:
    if not expression or not _TOKEN.fullmatch(expression):
        raise ValueError("Expression contains unsupported syntax")
    identifiers = set(_IDENTIFIER.findall(expression))
    unknown = identifiers - allowed_names - set(_FUNCTIONS) - set(_CONSTANTS)
    if unknown:
        raise ValueError(f"Expression uses undeclared symbols: {sorted(unknown)}")
    if "__" in expression:
        raise ValueError("Dunder names are not permitted")
    local_dict = {name: sp.Symbol(name, real=True) for name in allowed_names}
    local_dict.update(_FUNCTIONS)
    local_dict.update(_CONSTANTS)
    # SymPy's parser emits constructor names for numeric literals. Keep the
    # namespace deliberately small instead of exposing Python builtins.
    safe_globals = {
        "__builtins__": {}, "Integer": sp.Integer, "Float": sp.Float,
        "Rational": sp.Rational, "Symbol": sp.Symbol, "Add": sp.Add,
        "Mul": sp.Mul, "Pow": sp.Pow,
    }
    return parse_expr(
        expression,
        local_dict=local_dict,
        global_dict=safe_globals,
        transformations=_TRANSFORMS,
        evaluate=True,
    )


class EquationSystem:
    """Callable RHS created from validated symbolic expressions."""

    def __init__(self, state_names: list[str], parameter_names: list[str], expressions: Mapping[str, str], independent_variable: str = "time"):
        if set(expressions) != set(state_names):
            raise ValueError("Each state variable must have exactly one RHS expression")
        self.state_names = list(state_names)
        self.parameter_names = list(parameter_names)
        self.independent_variable = independent_variable
        allowed = set(self.state_names) | set(self.parameter_names) | {independent_variable}
        self.expressions = {name: safe_parse_expression(value, allowed) for name, value in expressions.items()}
        symbols = [sp.Symbol(independent_variable)] + [sp.Symbol(name) for name in self.state_names + self.parameter_names]
        self._callable = sp.lambdify(symbols, [self.expressions[name] for name in self.state_names], modules="numpy")

    def evaluate(self, time: float, state: np.ndarray, parameters: Mapping[str, float]) -> np.ndarray:
        if len(state) != len(self.state_names):
            raise ValueError("State vector length does not match the equation system")
        try:
            values = self._callable(time, *np.asarray(state, dtype=float), *(parameters[name] for name in self.parameter_names))
        except KeyError as error:
            raise ValueError(f"Missing parameter: {error.args[0]}") from error
        return np.asarray(values, dtype=float)
