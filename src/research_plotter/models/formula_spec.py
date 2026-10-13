"""Safe, small formula input layer for explicit, parametric, and ODE systems.

User and extracted text is parsed as a restricted Python expression AST and
converted node-by-node to SymPy objects. It is never passed to ``eval`` or
``exec``. The existing :class:`ModelSpec` remains the executable ODE model.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Any, Literal, Mapping

import numpy as np
import sympy as sp

from research_plotter.models.model_spec import ModelSpec
from research_plotter.models.parameter import Parameter
from research_plotter.models.state_variable import StateVariable


FormulaKind = Literal["explicit", "parametric", "ode"]
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_INITIAL = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*\(\s*0\s*\)\s*=\s*(.+)$")
_DERIVATIVE = re.compile(r"^d\s*([A-Za-z][A-Za-z0-9_]*)\s*/\s*d\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)$", re.I)
_PARAMETRIC = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\)\s*=\s*(.+)$")
_ASSIGNMENT = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)$")
_NUMBER = re.compile(r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$")
_FUNCTIONS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
}
_CONSTANTS = {"pi": sp.pi, "E": sp.E}


@dataclass(frozen=True)
class FormulaSpec:
    """Structured result of reviewing a formula text block."""

    kind: FormulaKind
    source_text: str
    independent_variable: str
    expressions: Mapping[str, str]
    dependent_variables: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_values: Mapping[str, float]
    state_variables: tuple[str, ...] = ()
    initial_conditions: Mapping[str, float] | None = None

    @property
    def missing_parameter_values(self) -> list[str]:
        return sorted(set(self.parameter_names) - set(self.parameter_values))

    @property
    def missing_initial_conditions(self) -> list[str]:
        if self.kind != "ode":
            return []
        return sorted(set(self.state_variables) - set(self.initial_conditions or {}))


def _convert_ast(node: ast.AST) -> sp.Expr:
    """Convert only arithmetic, declared math functions, and symbols to SymPy."""
    if isinstance(node, ast.Expression):
        return _convert_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise ValueError("Non-finite numeric literals are not allowed")
        return sp.Integer(node.value) if isinstance(node.value, int) else sp.Float(node.value)
    if isinstance(node, ast.Name):
        if node.id in _FUNCTIONS:
            raise ValueError(f"Function {node.id!r} must be called")
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if not _IDENTIFIER.fullmatch(node.id):
            raise ValueError(f"Invalid symbol name: {node.id}")
        return sp.Symbol(node.id, real=True)
    if isinstance(node, ast.BinOp):
        left, right = _convert_ast(node.left), _convert_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            if right.is_number and right.is_real and abs(float(right)) > 100:
                raise ValueError("Constant powers above magnitude 100 are not supported")
            return left ** right
        raise ValueError(f"Operator {type(node.op).__name__} is not supported")
    if isinstance(node, ast.UnaryOp):
        value = _convert_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError(f"Unary operator {type(node.op).__name__} is not supported")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _FUNCTIONS.get(node.func.id)
        if function is None:
            raise ValueError(f"Function {node.func.id!r} is not supported")
        if node.keywords or len(node.args) != 1:
            raise ValueError(f"Function {node.func.id!r} expects one positional argument")
        return function(_convert_ast(node.args[0]))
    raise ValueError(f"Syntax {type(node).__name__} is not allowed in a formula")


def parse_safe_expression(expression: str) -> sp.Expr:
    """Parse a single expression using a strict AST allowlist."""
    source = expression.strip().replace("^", "**")
    if not source or len(source) > 2048:
        raise ValueError("Enter a non-empty formula shorter than 2048 characters")
    try:
        syntax = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Invalid formula syntax: {error.msg}") from error
    if sum(1 for _ in ast.walk(syntax)) > 256:
        raise ValueError("Formula is too complex (maximum 256 syntax nodes)")
    return _convert_ast(syntax)


def _numeric_value(text: str, *, field: str) -> float:
    token = text.strip()
    if not _NUMBER.fullmatch(token):
        raise ValueError(f"{field} must be a finite numeric literal")
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def parse_numeric_assignments(text: str, *, field: str = "values") -> dict[str, float]:
    """Parse `name=value` pairs; values are numeric literals, never code."""
    result: dict[str, float] = {}
    for raw in re.split(r"[,\n;]+", text):
        item = raw.strip()
        if not item:
            continue
        match = _ASSIGNMENT.fullmatch(item)
        if not match:
            raise ValueError(f"Expected name=value in {field}: {item!r}")
        name, value_text = match.groups()
        if name in result:
            raise ValueError(f"Duplicate {field} entry: {name}")
        result[name] = _numeric_value(value_text, field=f"{field}.{name}")
    return result


def _clean_lines(text: str) -> list[str]:
    if len(text) > 20000:
        raise ValueError("Equation text is too long (maximum 20,000 characters)")
    return [line.split("#", 1)[0].strip().rstrip(";") for line in text.splitlines() if line.split("#", 1)[0].strip().rstrip(";")]


def _parameter_symbols(expressions: Mapping[str, str], excluded: set[str]) -> set[str]:
    symbols: set[str] = set()
    for expression in expressions.values():
        symbols.update(str(symbol) for symbol in parse_safe_expression(expression).free_symbols)
    return symbols - excluded


def parse_equation_text(text: str, *, independent_variable: str = "x") -> FormulaSpec:
    """Interpret reviewed text as an explicit formula, parametric curve, or ODE.

    Supported forms include ``y = sin(x)``, ``x(t)=cos(t)`` plus ``y(t)=sin(t)``,
    and ``dx/dt = a*x`` with optional ``x(0)=...`` and numeric parameter lines.
    """
    if not _IDENTIFIER.fullmatch(independent_variable):
        raise ValueError("Independent variable must be a simple identifier")
    lines = _clean_lines(text)
    if not lines:
        raise ValueError("No equations were found in the supplied text")

    derivatives: dict[str, tuple[str, str]] = {}
    initial_conditions: dict[str, float] = {}
    numeric_assignments: dict[str, float] = {}
    simple_assignments: list[tuple[str, str]] = []
    parametric_assignments: list[tuple[str, str, str]] = []
    bare_expressions: list[str] = []

    for line in lines:
        initial = _INITIAL.fullmatch(line)
        if initial:
            name, value = initial.groups()
            initial_conditions[name] = _numeric_value(value, field=f"{name}(0)")
            continue
        derivative = _DERIVATIVE.fullmatch(line)
        if derivative:
            state, variable, expression = derivative.groups()
            derivatives[state] = (variable, expression)
            continue
        parametric = _PARAMETRIC.fullmatch(line)
        if parametric:
            dependent, variable, expression = parametric.groups()
            parametric_assignments.append((dependent, variable, expression))
            continue
        assignment = _ASSIGNMENT.fullmatch(line)
        if assignment:
            name, expression = assignment.groups()
            if _NUMBER.fullmatch(expression.strip()):
                numeric_assignments[name] = _numeric_value(expression, field=name)
            else:
                simple_assignments.append((name, expression))
            continue
        bare_expressions.append(line)

    if derivatives:
        if parametric_assignments or simple_assignments or bare_expressions:
            raise ValueError("ODE input may contain derivative equations, initial conditions, and numeric parameter assignments only")
        variables = {value[0] for value in derivatives.values()}
        if len(variables) != 1:
            raise ValueError("All ODE equations must use the same independent variable")
        variable = next(iter(variables))
        expressions = {name: value[1] for name, value in derivatives.items()}
        states = set(expressions)
        parameters = _parameter_symbols(expressions, states | {variable})
        if set(numeric_assignments) - parameters:
            raise ValueError(f"Numeric values supplied for undeclared ODE parameters: {sorted(set(numeric_assignments) - parameters)}")
        return FormulaSpec("ode", text, variable, expressions, tuple(expressions), tuple(sorted(parameters)), numeric_assignments, tuple(expressions), initial_conditions)

    if parametric_assignments:
        if simple_assignments or bare_expressions:
            raise ValueError("Parametric input should use assignments such as x(t)=... and y(t)=... on separate lines")
        variables = {item[1] for item in parametric_assignments}
        if len(variables) != 1 or len(parametric_assignments) < 2:
            raise ValueError("A parametric curve needs at least two outputs that share one parameter")
        variable = next(iter(variables))
        expressions = {name: expression for name, _, expression in parametric_assignments}
        if len(expressions) != len(parametric_assignments):
            raise ValueError("Parametric output names must be unique")
        parameters = _parameter_symbols(expressions, set(expressions) | {variable})
        if set(numeric_assignments) - parameters:
            raise ValueError(f"Numeric values supplied for undeclared formula parameters: {sorted(set(numeric_assignments) - parameters)}")
        return FormulaSpec("parametric", text, variable, expressions, tuple(expressions), tuple(sorted(parameters)), numeric_assignments)

    if len(simple_assignments) > 1:
        raise ValueError("Multiple formula outputs require parametric notation such as x(t)=... and y(t)=...")
    if simple_assignments and bare_expressions:
        raise ValueError("Use one formula per input, or separate model equations with the ODE notation dX/dt=...")
    if simple_assignments:
        dependent, expression = simple_assignments[0]
    elif len(bare_expressions) == 1:
        dependent, expression = "y", bare_expressions[0]
    else:
        raise ValueError("Enter one explicit formula, for example y = sin(x)")
    parameters = _parameter_symbols({dependent: expression}, {independent_variable, dependent})
    if set(numeric_assignments) - parameters:
        raise ValueError(f"Numeric values supplied for undeclared formula parameters: {sorted(set(numeric_assignments) - parameters)}")
    return FormulaSpec("explicit", text, independent_variable, {dependent: expression}, (dependent,), tuple(sorted(parameters)), numeric_assignments)


@lru_cache(maxsize=256)
def _numeric_evaluator(expression_text: str, independent_variable: str, parameter_names: tuple[str, ...]):
    """Compile a previously AST-validated expression once for repeated evaluation."""
    expression = parse_safe_expression(expression_text)
    variable = sp.Symbol(independent_variable, real=True)
    args = [variable, *[sp.Symbol(name, real=True) for name in parameter_names]]
    allowed = {str(symbol) for symbol in args}
    if {str(symbol) for symbol in expression.free_symbols} - allowed:
        raise ValueError("Formula contains symbols outside the reviewed variable/parameter list")
    numeric_modules = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "Abs": np.abs, "abs": np.abs}
    # lambdify only sees the SymPy tree produced by the strict expression AST converter.
    return sp.lambdify(args, expression, modules=[numeric_modules, "numpy"], cse=True, docstring_limit=0)


def evaluate_formula(spec: FormulaSpec, independent_values: np.ndarray, parameter_values: Mapping[str, float] | None = None) -> dict[str, np.ndarray]:
    """Evaluate validated explicit/parametric equations on a numeric grid."""
    if spec.kind == "ode":
        raise ValueError("ODE FormulaSpec must be converted to ModelSpec and solved, not evaluated as a closed-form curve")
    values = {**spec.parameter_values, **(parameter_values or {})}
    missing = set(spec.parameter_names) - set(values)
    if missing:
        raise ValueError(f"Supply numeric values for parameters: {', '.join(sorted(missing))}")
    extra = set(values) - set(spec.parameter_names)
    if extra:
        raise ValueError(f"Values supplied for unknown formula parameters: {sorted(extra)}")
    grid = np.asarray(independent_values, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("Independent-variable values must be a finite, non-empty 1D array")
    result: dict[str, np.ndarray] = {}
    for name, expression_text in spec.expressions.items():
        # The cache only accepts expression text after parsing through the strict
        # AST allowlist; arbitrary Python or PDF text is never compiled.
        evaluator = _numeric_evaluator(expression_text, spec.independent_variable, tuple(spec.parameter_names))
        raw = evaluator(grid, *[values[name] for name in spec.parameter_names])
        array = np.asarray(raw, dtype=float)
        if array.ndim == 0:
            array = np.full_like(grid, float(array))
        else:
            array = np.broadcast_to(array, grid.shape).astype(float, copy=True)
        if not np.isfinite(array).all():
            raise ValueError(f"Formula for {name} produced NaN or infinity on this domain; check the domain and parameters")
        result[name] = array
    return result


def build_ode_model(spec: FormulaSpec, parameter_values: Mapping[str, float] | None = None, initial_conditions: Mapping[str, float] | None = None, *, fractional_order: float | None = None) -> ModelSpec:
    """Build an existing ModelSpec from a validated ODE formula system."""
    if spec.kind != "ode":
        raise ValueError("Only ODE formula systems can be converted to ModelSpec")
    parameters = {**spec.parameter_values, **(parameter_values or {})}
    missing_parameters = set(spec.parameter_names) - set(parameters)
    if missing_parameters:
        raise ValueError(f"Supply numeric values for parameters: {', '.join(sorted(missing_parameters))}")
    if set(parameters) - set(spec.parameter_names):
        raise ValueError("Parameter values include names not used by this equation system")
    initial = {**(spec.initial_conditions or {}), **(initial_conditions or {})}
    missing_initial = set(spec.state_variables) - set(initial)
    if missing_initial:
        raise ValueError(f"Supply initial conditions for: {', '.join(sorted(missing_initial))}")
    if set(initial) - set(spec.state_variables):
        raise ValueError("Initial conditions include names that are not ODE states")
    allowed = set(spec.state_variables) | set(spec.parameter_names) | {spec.independent_variable}
    for expression in spec.expressions.values():
        unknown = {str(symbol) for symbol in parse_safe_expression(expression).free_symbols} - allowed
        if unknown:
            raise ValueError(f"ODE expression contains undeclared symbols: {sorted(unknown)}")
    return ModelSpec.from_expressions(
        name="formula_ode_system",
        state_variables=[StateVariable(name=name) for name in spec.state_variables],
        parameters=[Parameter(name=name, value=float(parameters[name])) for name in spec.parameter_names],
        expressions=spec.expressions,
        initial_conditions=initial,
        independent_variable=spec.independent_variable,
        derivative_type="fractional" if fractional_order is not None else "ordinary",
        fractional_order=fractional_order,
        metadata={"input_kind": "reviewed_formula_text", "formula_provenance": "USER_PROVIDED"},
    )

