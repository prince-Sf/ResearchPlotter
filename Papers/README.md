# Paper sources

This folder is the optional local location for the source PDFs used as first validation cases. The files remain unmodified.

PDFs are ignored by `.gitignore` by default. Research papers may have separate
redistribution terms, so users should obtain them from an authorized source and
keep them locally unless they have confirmed they may redistribute the files.
The application and tests continue to work without the PDFs; paper-specific
tests skip when a local source is unavailable.

`paper_1` is *Dynamic analysis of fractal-fractional cancer model under chemotherapy drug with generalized Mittag-Leffler kernel*.

`paper_2` is *A Physics Informed Neural Network (PINN) framework for fractional order modeling of Alzheimer's disease*.

The pipeline stores extracted evidence, uncertainty statuses, manifests, and reproduction diagnostics under `outputs/paper_reproduction/`. A paper adapter may only make source-specific decisions; shared solvers and plotting code remain generic.
