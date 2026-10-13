# Scientific validation and evidence

ResearchPlotter distinguishes software execution success from scientific
reproducibility. A run can finish successfully while remaining `PARTIAL`,
`UNVERIFIED`, or `BLOCKED` as a reproduction.

## Graph-derived fits

Digitized coordinates are estimates from raster pixels, affected by image
resolution, line thickness, anti-aliasing, crop choice, calibration, and curve
segmentation. Candidate fitting evaluates supported empirical forms against
the extracted points; it does not identify the original equation uniquely.
Candidate ranking uses a deterministic held-out subset and reports fit and
holdout metrics plus model-complexity information where available. Local
covariance standard errors are conditional on the selected form and data and
do not account for digitization or model-selection uncertainty.

## Paper-model comparisons

Paper reproduction compares a model and method supported by source evidence
against a digitized reference curve. It records calibration and interpolation
caveats and is not equivalent to access to the original numerical data.
Missing initial conditions, solver details, equations, or sweep values remain
visible as unresolved rather than being silently inferred.

## Evidence provenance

When available, configuration and report records distinguish
`SOURCE_PAPER`, `USER_PROVIDED`, `ASSUMED`, and `UNKNOWN`. A user-supplied value
may enable an exploratory run, but it does not become evidence that the paper
used that value. See the paper and graph workflow guides for operational
details.
