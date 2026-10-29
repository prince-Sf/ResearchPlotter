# Graph Image Analyzer

Graph Image Analyzer turns plotted pixels into an estimated numerical dataset.
It does not recover the underlying raw observations, and a fitted equation is
not proof of the equation or model used by the paper.

## Workflow

1. Upload PNG, JPG/JPEG, WEBP, TIFF/TIF, or open a selected page from a PDF.
   PDF page rendering uses the existing PyMuPDF dependency. The source remains
   unchanged; rendered previews and selected crops are separate artifacts.
2. Select one main plotting region. This rectangle controls which pixels are
   extracted. Inset plots should be selected as a separate region with their
   own calibration; inset pixels are never interpreted with the main axes.
   The analysis crop may be broader than the extraction region in backend
   workflows, and the selected bounds are saved in digitization metadata.
3. Graph Image Analyzer attempts to detect the plot axes, tick marks, and
   nearby tick labels. It uses local Tesseract OCR when available; no image is
   sent to a network service. Tick-label alignment helps distinguish plot axes
   from a decorative image border or gridline. If visible axis strokes do not
   form an intersecting pair, aligned OCR label rows/columns can still provide
   a candidate mapping, which is marked low confidence. The detected values
   prefill the same editable eight calibration fields used by manual
   calibration. Review the highlighted axis/tick preview and table before
   digitizing. If OCR is unavailable or a label is ambiguous, affected fields
   remain blank for manual entry. Pixel coordinates are in original rendered-
   image coordinates even when a crop is selected, and image y increases
   downward. Linear axes use affine mapping; log axes interpolate in log10
   space and require positive calibration values. With only two ticks, linear
   versus logarithmic scaling cannot be inferred from spacing alone and is
   flagged for review. A logarithmic candidate must itself explain at least
   three distinct tick positions; a two-point zero-residual log fit is not
   sufficient evidence to override a multi-tick linear mapping.
   Geometry detection combines long-line evidence, local contrast, and tick
   text alignment; arrowhead strokes without nearby labels are suppressed near
   axis endpoints. Small-figure OCR can confuse π with
   similar glyphs or mistake a decimal point; context-supported π corrections
   are warned and must be checked against the source. Manual cropping and
   calibration remain available; curve shape is never used to invent axis
   values.
4. Review detected RGB candidates, choose and name curves, and adjust color
   tolerance. Calibration anchors define a coordinate mapping only; interior
   tick anchors do not clip curve observations outside their pixel interval.
   The mapping may be extrapolated within the selected plot region (including
   in log coordinates), and edge extrapolation is called out for review. The
   reference implementation uses RGB distance segmentation and retains
   separate resolvable vertical pixel clusters in each x column. It can still
   be confused by overlapping curves, labels, markers, dense grids, and
   same-color annotations; use exclusions and inspect the pixel overlay.
5. Each selected curve remains an independent named series. The combined
   extraction action runs candidate discovery separately for each selected
   series and records per-series errors without discarding successful fits.
   Export digitized points in long form (`curve_name`, `pixel_x`, `pixel_y`,
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

## Optional local OCR setup

Axis geometry detection works without OCR, but geometry alone does not provide
the numerical tick values. For automatic label recognition, install the
optional Python adapter and a local Tesseract executable:

```powershell
python -m pip install -e ".[ocr]"
```

Install Tesseract for the operating system using its normal package/installer
distribution, then restart ResearchPlotter. On Windows, the adapter uses the
executable on `PATH` or checks the standard Program Files installation
location. It tries the nested `script/Greek` model when present, then falls
back to English OCR; the nested script model may not appear in Python's
language-list API even when Tesseract itself can use it. The application
reports when the optional executable is unavailable and keeps all eight
manual fields usable. No OCR service, uploaded figure, or paid API is used.
OCR output is an estimate; fractional/Greek labels, scientific notation,
superscripts, and crowded legends can still require manual correction. The
detector reports low confidence when it relies on label alignment instead of
visible axis strokes, and it does not infer a formula from curve shape.

