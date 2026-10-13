# Architecture

ResearchPlotter keeps ingestion, evidence extraction, structured models, numerical solvers, analysis, plotting, validation, and export separate. A dataset enters through `ingestion`, a paper through `paper_analysis`, and both can feed generic model/analysis/plotting APIs. Paper-specific equations are isolated in `paper_adapters`.

`ModelSpec` stores state variables, bounded parameters, initial conditions, a validated SymPy expression system, derivative metadata, and provenance. Solvers return `SimulationResult`, which contains trajectories plus parameters, diagnostics, warnings, runtime, and provenance. Plotters consume arrays/dataframes/results and do not inspect paper names.

Fractional refinement experiments live in `analysis.convergence` and use the
finest requested grid only as a numerical reference. PINN training, residuals,
noise robustness, checkpoints, and export are shared under `pinn` and
`pipelines.pinn_experiment`; paper adapters provide equations and provenance,
not duplicate neural-network implementations.

Install the `src/` package in editable mode from the repository root with
`python -m pip install -e ".[dev]"` (or `python -m pip install -r requirements.txt`).
This ensures that the CLI and `python -m streamlit run
src/research_plotter/app.py` resolve `research_plotter` through normal package
discovery rather than a `sys.path` workaround.

The Equation / Formula Plotter parses input through a restricted Python AST
allowlist into SymPy expressions. ODE text becomes the existing `ModelSpec` and
uses the normal solver and plotting APIs. Extracted PDF text must be reviewed
by the user. Uploads use content-addressed names to avoid overwriting an
earlier file with the same original filename.

PDF equations follow a separate review-first path. `ingestion.pdf_loader`
retains positioned text/image blocks, font cues, and page coordinates; large
vector chart bounds are associated with nearby figure captions. The paper
analysis layer groups equation-like blocks into `EquationSpec` candidates with
source page and bounding box, mathematical type, confidence, provenance, and
review status. These candidates are editable and may be rendered as cropped
source previews. They are not executable merely because they were extracted.
Only an accepted, supported formula/ODE that independently passes the restricted
`FormulaSpec` parser may reach the existing solver. Fractional equations and
discrete recurrences stay structured as review candidates unless a validated
paper adapter supplies an appropriate method.
