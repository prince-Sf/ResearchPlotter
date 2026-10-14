# Paper workflow

`analyze-paper` extracts conservative text, figures, equations, parameter tokens, methods, and uncertainty markers to `outputs/paper_reproduction/<paper>/paper_analysis.json`. Review this report before simulation. `reproduce` then invokes a paper adapter only where the source supports it. Missing initial conditions, step sizes, controlled equations, NSFD details, and training hyperparameters remain `NEEDS_REVIEW`, `PARTIAL`, or `BLOCKED`; they are never silently guessed.

For an explicitly exploratory Paper 1 run, supply reviewed values on the CLI, for example `reproduce Papers/333.pdf --initial-conditions '{"S":0.9,"E":0.5,"T":0.9,"C":0}' --time-end 90 --step 0.1`. Such a run is still reported as unverified/partial unless independent trajectory evidence is available.

Paper 2 accepts a user override file with `reproduce Papers/fninf-20-1748481.pdf --config configs/paper_2.yaml`. The structured `controls`, `nsfd`, and `pinn` sections carry provenance tags (`SOURCE_PAPER`, `USER_PROVIDED`, `ASSUMED`, or `UNKNOWN`). Missing user values produce explicit `BLOCKED`/`NEEDS_REVIEW` warnings and are copied to the generated configuration report.

If a controlled right-hand side references a control such as `z1` or `z2`,
provide its numeric value under `controls.values` and mark the section
`provenance: USER_PROVIDED`; otherwise the controlled experiment remains
blocked. NSFD settings are recorded for reproducibility, but are not executed
until an NSFD solver is implemented.

Paper 1 remains `BLOCKED` when required initial conditions or time step are not
provided and reviewed. The displayed recurrence also has an indexing ambiguity,
and some sweep values are not listed. A user-supplied exploratory run is
reported as unverified rather than as a paper reproduction.

Paper 2 remains `PARTIAL`: the source does not sufficiently specify controlled
equations, NSFD update details, or PINN training hyperparameters. Provide
overrides in `configs/paper_2.yaml` or another YAML file and tag their origin;
unknown information stays `UNKNOWN` until explicitly resolved.

The tracked Paper 2 baseline under `outputs/paper_reproduction/paper_2/` is a
historical diagnostic, not a valid reference trajectory. Its configured
explicit GL run used `dt=1` with the recorded model and produced a negative
amyloid state on the first update, then non-finite values after 17 completed
steps. At the initial state, the configured Aβ equation's right-hand side is
`-9.53504e-6`; the explicit `h^0.9` update at `h=1` therefore
produces `Aβ₁ = -8.53504e-6`, matching the saved CSV. This identifies an
unstable coarse configured step for this explicit update, not an invalid
scientific model or a validated paper time step. The source does not establish
the step size or fully specify the main NSFD method, so the project does not
substitute a guessed smaller step. The saved report correctly labels the run
`INVALID`; it is retained as evidence of the failure, not as a generated
published result. New runs with no explicit time/order/solver configuration
are blocked rather than assigned hidden defaults.

The Streamlit Figure Reproduction workspace displays extracted figure manifests
before execution. Paper 1 users may enter missing initial conditions, a time
step, and (for Figures 11/12) otherwise-unlisted sweep values; these are
recorded as `USER_PROVIDED`. Runs remain exploratory/unverified and retain the
source warnings. Paper 2's selected-figure workflow is not asserted: the UI
only offers a partial configured baseline because the source omits needed
controlled and method details. A generated curve is not reproduction evidence
by itself.

Install the declared runtime dependencies with `python -m pip install -e .`.
Run `python examples/pinn_smoke.py` for the tiny deterministic baseline,
integer, fractional, noise, checkpoint, and GL comparison smoke experiment.
It records a same-grid comparison as `UNVERIFIED` unless an accuracy criterion
has been supplied; it does not claim agreement with either published figure.

## Equation-region review and figure links

The PDF parser preserves positioned text/image blocks and classifies regions as
prose, heading, equation, figure, figure caption, table, reference, or metadata.
Equation-like blocks become editable `EquationSpec` candidates with source
page/bbox, mathematical type, confidence, provenance, and review status. The UI
renders the matching PDF crop so a person can compare Greek letters, indices,
fractions, sums, and derivative operators against the source.

PDF text is not mathematical OCR. Font encoding, two-column layout, subscripts,
superscripts, and stacked fractions can be lost or reordered. A candidate
marked `FRACTIONAL_DIFFERENTIAL_EQUATION` or `COUPLED_RECURRENCE` is a
classification, not an exact transcription or executable model. A user may
edit a representation and save an explicit `ACCEPTED` review decision; the
Equation Plotter still runs only forms independently accepted by the restricted
parser (explicit/parameterized functions, parametric curves, or ordinary ODEs).
Arbitrary PDF text is never evaluated.

Figure manifests link figures to equation IDs, required parameters, initial
conditions, time range, solver/method, missing information, and provenance.
Generic linkage comes from explicit equation references. A paper-specific
adapter may add a paper-level model/method link, marked `PARTIAL` if the
figure-specific mapping is not explicit. For Paper 1, result figures link to
the source's Eq. (7) model and the numerical scheme through Eq. (58); the
individual caption does not enumerate separate equation IDs. The published
plots span 0-90 on the x-axis, but this does not supply numerical initial
conditions or a simulation step size. The source says `C(0)=0` conditionally
when `J(0)=0`; verify that condition for the experiment. Recurrence indexing
remains ambiguous, so exact reproduction stays `BLOCKED`. Any configured run
with user-provided values remains exploratory/partial.

Provenance values are `SOURCE_PAPER`, `USER_PROVIDED`, `ASSUMED`, and `UNKNOWN`.
Saved user edits are kept distinct from extracted source text. Acceptance does
not upgrade confidence or independently validate the scientific model.
