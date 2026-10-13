from research_plotter.analysis.parameter_sweep import run_parameter_sweep
from research_plotter.models import ModelSpec, Parameter, StateVariable
from research_plotter.solvers import ODESolver
from research_plotter.utils.numerics import uniform_time_grid

model = ModelSpec.from_expressions(
    name="decay", state_variables=[StateVariable(name="x")], parameters=[Parameter(name="rate", value=.2)],
    expressions={"x": "-rate*x"}, initial_conditions={"x": 1.0})
result = run_parameter_sweep(model, ODESolver(), uniform_time_grid(0, 5, .1), {"rate": [.1, .2, .3]})
print(len(result.simulations))
