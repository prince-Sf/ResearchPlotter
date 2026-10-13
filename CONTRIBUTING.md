# Contributing to ResearchPlotter

Thank you for helping improve ResearchPlotter. Keep changes focused and
preserve scientific provenance and uncertainty in existing workflows.

## Local setup and checks

Use the project's configured Python environment and install the declared
dependencies from the repository root:

```powershell
python -m pip install -e ".[dev]"
pytest -q
python -m compileall src
python -c "import research_plotter; print(research_plotter.__file__)"
```

For Streamlit changes, start the app with
`python -m streamlit run src/research_plotter/app.py` and smoke-test the
affected page. Numerical changes should include a small reference or synthetic
test with a stated tolerance. Do not assert a theoretical convergence order
without evidence.

## Scientific and data-handling expectations

- Keep paper-specific logic in adapters, not generic solvers or plotting code.
- Record whether values came from a source paper, a user, an assumption, or
  remain unknown.
- Do not silently fill omissions or describe exploratory output as exact
  reproduction.
- Treat uploaded equations and documents as untrusted input; use the safe
  parser and allowlisted operations.
- Do not commit local uploads, generated results, credentials, or research PDFs
  unless both authorization and repository policy are clear.

There is no project license selected yet. Until one is chosen, do not imply a
license grant when redistributing this project.
