# ResearchPlotter

ResearchPlotter is a modular Python workbench for scientific datasets, mathematical models, numerical simulations, research-paper analysis, and publication-quality figures. It treats paper reproduction as an evidence-led workflow: extracted claims, model specifications, solvers, validation, plots, and exported provenance are distinct artifacts.

The two PDFs in `Papers/` are validation cases, not special cases baked into the plotting or solver APIs. Their adapters live under `research_plotter.paper_adapters`; other papers, datasets, and models use the same ingestion, model, solver, analysis, plotting, and export layers.

## Quick start

```powershell
# For a fresh checkout only; if .venv already exists, keep it and verify its Python base.
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m research_plotter analyze-data data/examples/measurements.csv
python -m research_plotter plot data/examples/measurements.csv
python -m research_plotter analyze-paper Papers/333.pdf
python -m research_plotter convergence configs/examples/fractional_alzheimer.yaml
python -c "import research_plotter; print(research_plotter.__file__)"
python -m streamlit run src/research_plotter/app.py
```

`python -m pip install -r requirements.txt` is also supported; the requirements
file installs the project in editable mode so the `src/` package is importable.
In PyCharm, select `.venv/Scripts/python.exe` (Windows) or
`.venv/bin/python` (macOS/Linux), then install the project from the repository
root. Avoid launching the app with a different interpreter from the one where
ResearchPlotter was installed. If that venv points to a Python installation
that has since been removed, create a replacement at a new path and select it;
do not remove research outputs or source files to repair an interpreter.

The project configuration declares PyTorch and Streamlit. If installing them
separately, use `python -m pip install "torch>=2.2" "streamlit>=1.31"`.
Run a small deterministic PINN, noise, checkpoint, and GL comparison smoke
experiment with `python examples/pinn_smoke.py`; it writes reviewable artifacts
under `outputs/reports/pinn_smoke/` and does not reproduce a paper figure.

## Research Workbench

Launch `python -m streamlit run src/research_plotter/app.py` and use the sidebar
to open Dashboard, Dataset Analyzer, Paper Analyzer, Model Simulation,
Figure Reproduction, Equation / Formula Plotter, Graph Image Analyzer,
Parameter Sweep, Sensitivity Analysis, Noise / Uncertainty, Convergence /
Validation, PINN Experiments, Figure Gallery, Experiment History, or Reports /
Export.

- **Dataset Analyzer** profiles uploaded CSV, clearly delimited TXT tables, XLSX, XLS, JSON, and Parquet files
  or the bundled `data/examples/measurements.csv`, then lets you choose which
  suggested Matplotlib plots to generate and export.
- **Paper Analyzer** uses PDF text blocks, font/layout cues, vector-chart bounds,
  equation-region geometry, and page previews to present editable equation
  candidates with confidence and provenance. The parsed symbol sequence is not
  treated as exact LaTeX. Paper reproduction can remain `BLOCKED` or `PARTIAL`;
  the UI preserves that status and keeps unresolved details visible.
- **Figure Reproduction** presents extracted figure manifests and lets users
  provide missing Paper 1 initial conditions, time step, or otherwise-unlisted
  sweep values. Supplied values are recorded as `USER_PROVIDED`; Paper 2 figure-
  specific reproduction remains partial where the source omits equations or
  method details.
- **Equation / Formula Plotter** accepts manual formulas, TXT, or selected PDF
  equation regions. It supports explicit and parametric curves plus ODE systems
  routed through `ModelSpec` and existing solvers. Fractional equations and
  recurrences are classified and shown for review instead of being forced into
  `y=f(x)`. Parsing uses an AST allowlist and never evaluates arbitrary
  uploaded code. See [docs/paper_workflow.md](docs/paper_workflow.md) for
  equation review, provenance, and figure linkage.
- **Graph Image Analyzer** calibrates axes from user-provided pixel/data
  anchors, segments one or more selected curve colors, exports digitized points,
  and fits known safe equation templates or ranks candidate families using
  held-out points. Fits are empirical approximations; they do not recover the
  original paper model. Paper figures can be handed off from Figure
  Reproduction for a digitized-reference comparison. See
  [docs/graph_image_analysis.md](docs/graph_image_analysis.md).
- **Model Simulation**, **Parameter Sweep**, and **Sensitivity Analysis** use
  the existing YAML model configuration, solver, and analysis APIs. Start with
  `configs/examples/basic_ode.yaml` or upload a compatible configuration.
- **PINN Experiments** runs a small configurable baseline, integer, or
  GL-fractional network against a configured model trajectory. Training and
  prediction metrics are exploratory numerical results, not a claim that a
  published figure was reproduced.
- **Noise / Uncertainty** creates separate, seeded noisy observations from a
  clean numeric dataset column. **Convergence / Validation** compares existing
  solvers over user-selected grid refinements without asserting a theoretical
  order.
- **Figure Gallery**, **Experiment History**, and **Reports / Export** read the
  artifacts and JSON records saved under `outputs/`. Workbench runs retain the
  configuration, result data, figure sidecars, warnings, runtime, and status.

The paper adapters intentionally retain missing/ambiguous scientific details.
Paper 1 is blocked for exact reproduction where initial conditions, step size,
recurrence indexing, or sweep values are not specified. Paper 2 remains partial
where controlled equations, NSFD details, or PINN training settings are
incomplete. Supply explicit user configuration for exploratory runs; it is
recorded as user-provided rather than attributed to the source paper.

See `docs/architecture.md` for the component boundaries, `docs/dataset_workflow.md` for data analysis, `docs/paper_workflow.md` for the evidence/reproduction process, and `docs/solver_notes.md` for numerical-method limitations.

Paper commands create a machine-readable analysis report before attempting simulations. They intentionally report a blocked or exploratory reproduction when the source lacks sufficient numerical detail; generating a graph is never treated as an exact reproduction claim.

## Equation-candidate validation

Graph Image Analyzer separates empirical curve fitting from paper-model
reproduction. Candidate discovery fits a small supported family set, derives
initial guesses from the training subset, and holds out every fifth
x-sorted observation. Candidates are ranked by held-out RMSE, with BIC and
parameter count as tie-breakers. The result is an `EMPIRICAL_FIT` candidate,
not the recovered equation from a paper. See
[`docs/graph_image_analysis.md`](docs/graph_image_analysis.md) for supported
families, calibration caveats, and fit diagnostics.

## Repository and contribution notes

Generated artifacts and local uploads under `outputs/` are ignored by default;
the placeholder directories remain available in a checkout. Local raw and
processed datasets are likewise excluded. Research PDFs under `Papers/` and
`papers/` are ignored because redistribution rights vary; see
[`Papers/README.md`](Papers/README.md). Never add credentials, private data,
or licensed source documents unless authorized. The repository currently has
no selected open-source license; no reuse or redistribution grant is implied.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for local development checks and
[`SECURITY.md`](SECURITY.md) for reporting security concerns and handling
untrusted scientific inputs.
