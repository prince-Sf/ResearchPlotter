from research_plotter.analysis.noise import add_gaussian_noise, noise_experiment
from research_plotter.analysis.parameter_sweep import ParameterSweepResult, run_parameter_sweep
from research_plotter.analysis.sensitivity import logarithmic_sensitivity, perturbation_sensitivity
from research_plotter.analysis.statistics import summarize_numeric
from research_plotter.analysis.validation import validate_simulation, compare_trajectories
from research_plotter.analysis.convergence import RefinementResult, run_refinement_experiment
from research_plotter.analysis.equation_fitting import CandidateFit, EquationFitResult, discover_equation_candidates, fit_equation
from research_plotter.analysis.graph_digitization import AxisCalibration, DigitizationResult, GraphImage, detect_curve_colors, digitize_color_curves, load_graph_image, render_pdf_page, validate_exclusion_regions
from research_plotter.analysis.axis_calibration import AutomaticCalibrationResult, OCRWord, TickLabel, auto_calibrate_axes, local_ocr_available, parse_tick_label, validate_auto_calibration

__all__ = [
    "ParameterSweepResult", "RefinementResult", "add_gaussian_noise", "compare_trajectories", "logarithmic_sensitivity",
    "noise_experiment", "perturbation_sensitivity", "run_parameter_sweep", "run_refinement_experiment", "summarize_numeric", "validate_simulation",
    "AxisCalibration", "AutomaticCalibrationResult", "CandidateFit", "DigitizationResult", "EquationFitResult", "GraphImage", "OCRWord", "TickLabel", "auto_calibrate_axes", "detect_curve_colors", "digitize_color_curves", "discover_equation_candidates", "fit_equation", "load_graph_image", "local_ocr_available", "parse_tick_label", "render_pdf_page", "validate_auto_calibration", "validate_exclusion_regions",
]
