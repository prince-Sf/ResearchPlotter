# Solver notes

`ODESolver` wraps `scipy.solve_ivp` only for ordinary/integer systems. `GrunwaldLetnikovSolver` and the AB/fractal-fractional reference solvers retain explicit history terms and report their O(N²) reference cost. They are transparent numerical baselines, not claims that every paper's variant has been exactly reproduced. The Paper 1 adapter records its generalized Mittag-Leffler recurrence ambiguity and refuses missing initial conditions.

`run_refinement_experiment` compares requested `dt` refinements against the
finest requested grid and reports empirical error ratios. It does not label an
observed ratio as a theoretical convergence order. The GL reference is tested
against the Caputo constant-forcing solution and should still be refined before
long or stiff Paper 2 runs.

The fractional PINN residual uses a Caputo-corrected Grünwald--Letnikov history
sum on an increasing uniform grid. The history tensor operations retain
gradients to the network outputs; ordinary autograd is used only for
integer-order residuals. This reference residual costs O(N²), and it should not
be interpreted as the paper's exact residual unless its discretization and
training details are independently matched. Smoke comparisons report metrics
and remain `UNVERIFIED` when no accuracy threshold is defined.

The Convergence / Validation page runs the configured solver on a selected
sequence of refined grids and compares the results with the finest requested
grid. Reported observed orders are descriptive estimates for those results only;
they are not theoretical convergence guarantees.
