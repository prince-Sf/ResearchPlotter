# Graph Image Analyzer

Graph Image Analyzer turns plotted pixels into an estimated numerical dataset.
It does not recover the underlying raw observations, and a fitted equation is
not proof of the equation or model used by the paper.

## Workflow

1. Upload PNG, JPG/JPEG, WEBP, TIFF/TIF, or open a selected page from a PDF.
   PDF page rendering uses the existing PyMuPDF dependency. The source remains
   unchanged; rendered previews and selected crops are separate artifacts.
2. Crop to one main plotting region. Inset plots should be digitized as a
   separate region with their own calibration.
3. Enter two pixel/data anchors on each axis. Pixel coordinates are in the
   rendered image coordinate system, and image y increases downward. Linear
   axes use affine mapping. Log axes interpolate in log10 space and require
   positive calibration values. Tick detection is not treated as authoritative.
4. Review detected RGB candidates, choose and name curves, and adjust color
   tolerance. The reference implementation uses RGB distance segmentation
   and the median detected y pixel in each x column to reduce line-thickness
   effects. It can be confused by overlapping curves, labels, markers, dense
   grids, and same-color annotations; use a tight crop and inspect the pixel
   overlay.
5. Export digitized points in long form (`curve_name`, `pixel_x`, `pixel_y`,
   `x`, `y`) or wide form (`x`, one column per curve). A JSON sidecar records
   the source hash, image/page dimensions, crop, calibration, settings,
   provenance, point counts, and warnings.
6. Fit a user-supplied explicit equation template or compare a bounded set of
   candidate families. The known-form fitter uses the existing safe
   `FormulaSpec` parser/evaluator and SciPy bounded least squares. Initial
   guesses and bounds are editable. Candidate discovery holds out every fifth
   point and ranks by holdout RMSE with BIC/complexity as tie-breakers.
7. Review the data/fit overlay, residuals, and optional overlay mapped onto
   the original crop. Local covariance errors are only estimates conditional
   on the selected formula and digitized points; they omit calibration,
   segmentation, and model-selection uncertainty.

Candidate discovery caps each fit at 5,000 optimizer evaluations to keep the
exploratory search bounded. Logarithmic and power-law families are considered
only when their domain is valid over the complete data range, including held-out
points. Families that fail convergence or domain checks are omitted and noted
in the candidate warnings; use a known-form fit with explicit guesses/bounds
when a specific family matters. Discovery can still take several seconds for
large digitized curves, so the UI shows progress while the existing fitter
works through the allowed families.

## Paper-figure comparisons

From Figure Reproduction, select a paper figure and choose **Digitize this
published figure**. The Graph Image Analyzer opens the source PDF page (or an
uploaded reference image) with the figure identifier and paper provenance
retained. After digitization, a completed adapter simulation can be aligned to
the digitized curve by linear interpolation over the common x range. The report
labels this a `DIGITIZED_REFERENCE_COMPARISON` with status `UNVERIFIED` and
records image/calibration/interpolation caveats. Digitized pixels are not
ground-truth data and do not change Paper 1's `BLOCKED` or Paper 2's `PARTIAL`
source-information status.

## Equation and provenance semantics

Known-form fitting estimates free parameters in a reviewed formula such as
`y = A*exp(-k*x)*sin(w*x)`. Candidate discovery may suggest linear, low-degree
polynomial, exponential, logarithmic, power-law, sinusoidal, damped-sinusoidal,
or logistic forms when the data domain permits. Only the existing arithmetic
operators and allowlisted functions are accepted. No extracted image text or
user formula is passed to Python `eval`/`exec`.

Metadata distinguishes source paper/image context, user-provided axis
calibration, and fitted parameter estimates. An equation candidate should be
described as `FIT_SUCCESSFUL` or `CANDIDATE`, never as the original equation
without independent evidence. A good curve fit does not prove that the
candidate was the original equation or mathematical model.

To correct an extraction, adjust the plot crop, axis anchors, color choice, or
tolerance and digitize again. For non-linear/log axes, verify both calibration
anchors against visible ticks before interpreting results.

