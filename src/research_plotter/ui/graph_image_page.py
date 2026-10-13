"""Streamlit presentation for the graph digitization backend services."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import ImageDraw

from research_plotter.analysis.equation_fitting import CandidateFit, EquationFitResult, discover_equation_candidates, fit_equation
from research_plotter.analysis.graph_digitization import AxisCalibration, CurveColor, DigitizationResult, GraphImage, detect_curve_colors, digitize_color_curves, load_graph_image, render_pdf_page, validate_crop
from research_plotter.ingestion.dataset_profiler import profile_dataset
from research_plotter.models.formula_spec import parse_numeric_assignments
from research_plotter.pipelines.graph_image_analysis import compare_digitized_reference, save_digitization_artifacts, save_digitized_comparison, save_fit_artifacts


def equation_for_formula_plotter(result: EquationFitResult) -> str:
    """Return reviewed safe formula text with estimated values filled in."""
    lines = [result.equation]
    lines.extend(f"{name} = {value:.12g}" for name, value in result.parameters.items())
    return "\n".join(lines)


def handoff_to_formula_plotter(st: Any, equation_text: str, x_domain: tuple[float, float] | None = None) -> None:
    """Streamlit callback: make a fitted candidate editable in Formula Plotter."""
    st.session_state["formula_manual_text"] = equation_text
    st.session_state["formula_input_mode"] = "Manual formula"
    if x_domain is not None:
        low, high = sorted((float(x_domain[0]), float(x_domain[1])))
        st.session_state["formula_domain_start"] = low
        st.session_state["formula_domain_end"] = high
    st.session_state["main_navigation"] = "Equation / Formula Plotter"


def handoff_to_dataset_analyzer(st: Any, data: pd.DataFrame, new_run_dir: Callable[[str], tuple[str, Path]], record_run: Callable[..., Path]) -> None:
    """Streamlit callback: expose digitized data through the existing profiler UI."""
    run_id, run_dir = new_run_dir("digitized-dataset")
    st.session_state["dataset_frame"] = data.copy()
    st.session_state["dataset_profile"] = profile_dataset(data).model_dump(mode="json")
    st.session_state["dataset_source_path"] = "Digitized graph points · long-form CSV"
    st.session_state["dataset_run_dir"] = str(run_dir)
    st.session_state["dataset_run_id"] = run_id
    record_run(run_id, run_dir, "dataset", time.perf_counter(), status="SUCCEEDED", details={"source": "digitized graph points", "rows": len(data), "columns": len(data.columns), "source_provenance": "DIGITIZED_IMAGE"})
    st.session_state["main_navigation"] = "Dataset Analyzer"


def _set_default(st: Any, key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _open_paper_analyzer() -> None:
    """Navigate to source-evidence analysis instead of fitting paper pixels."""
    import streamlit as st

    st.session_state["main_navigation"] = "Paper Analyzer"


def _parse_bounds(text: str) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        if "=" not in token or ":" not in token:
            raise ValueError("Bounds use name=lower:upper entries, e.g. k=0:2,w=0:20")
        name, pair = token.split("=", 1)
        low_text, high_text = pair.split(":", 1)
        try:
            low = float(low_text.strip())
            high = float(high_text.strip())
        except ValueError as error:
            raise ValueError("Bounds must be numeric; use -inf or inf for an open bound") from error
        if np.isnan(low) or np.isnan(high) or low >= high:
            raise ValueError(f"Bounds for {name.strip()} must be ordered numeric values")
        if name.strip() in result:
            raise ValueError(f"Duplicate bound entry for parameter {name.strip()}")
        result[name.strip()] = (low, high)
    return result


def _parameter_guesses(text: str) -> dict[str, float]:
    if not text.strip():
        return {}
    return parse_numeric_assignments(text, field="initial guesses")


def _load_source(st: Any, outputs: Path) -> GraphImage | None:
    st.subheader("1 · Upload image or open a paper figure")
    source_options = ["Upload image or PDF"]
    handoff_path = st.session_state.get("graph_image_source_path")
    if handoff_path and Path(handoff_path).is_file():
        source_options.insert(0, "Paper figure handoff")
    source_kind = st.radio("Input source", source_options, horizontal=True, key="graph_image_source_kind")
    uploaded = None
    source_path: Path | None = None
    page_number: int | None = None
    if source_kind == "Upload image or PDF":
        st.session_state.pop("graph_image_paper_context", None)
        uploaded = st.file_uploader("Scientific graph image or PDF", type=["png", "jpg", "jpeg", "webp", "tif", "tiff", "pdf"], key="graph_image_upload")
        if uploaded is None:
            st.info("Upload a graph image, or open a paper figure from Figure Reproduction.")
            return None
        from research_plotter.utils.paths import persist_uploaded_file

        source_path = persist_uploaded_file(uploaded, "graph_images", output_root=outputs)
    else:
        source_path = Path(handoff_path)
        context_page = st.session_state.get("graph_image_page_number")
        if source_path.suffix.lower() == ".pdf":
            _set_default(st, "graph_image_pdf_page", int(context_page or 1))
            page_number = int(st.number_input("Source PDF page", min_value=1, max_value=10000, step=1, key="graph_image_pdf_page"))
        else:
            page_number = None
        context = st.session_state.get("graph_image_paper_context")
        if context:
            st.caption(f"Paper context: {context.get('source_document')} · {context.get('figure_id')} · page {page_number or 'image'}")
    if source_path is None:
        return None
    try:
        if source_path.suffix.lower() == ".pdf":
            if page_number is None:
                page_number = int(st.number_input("PDF page to render", min_value=1, max_value=10000, value=1, step=1, key="graph_image_upload_pdf_page"))
            graph_source = render_pdf_page(source_path, page_number)
        else:
            graph_source = load_graph_image(source_path)
        if uploaded is not None:
            graph_source = GraphImage(
                image=graph_source.image,
                source_name=uploaded.name,
                source_format=graph_source.source_format,
                page_number=graph_source.page_number,
                source_sha256=graph_source.source_sha256,
            )
        return graph_source
    except Exception as error:
        st.error(f"Image/PDF could not be loaded.\n\nReason: {error}\n\nUse a valid PNG, JPG, WEBP, TIFF, or readable PDF page.")
        return None


def _axis_calibration_controls(st: Any, image: GraphImage, crop: tuple[int, int, int, int], prefix: str) -> AxisCalibration | None:
    st.subheader("3 · Calibrate axes")
    st.caption(
        f"Image dimensions: {image.image.width} × {image.image.height} pixels. "
        "Manual calibration is required: enter coordinates on the actual plotted axis lines, not the image boundaries. "
        "The app does not infer tick values. For an inset, select it as its own crop and calibrate it separately."
    )
    x_cols = st.columns(4)
    x_left = x_cols[0].number_input("X left pixel", min_value=0.0, max_value=float(image.image.width), value=None, step=1.0, help="Pixel coordinate of the left axis anchor, in the original image.", key=f"{prefix}_cal_x_left")
    x_right = x_cols[1].number_input("X right pixel", min_value=0.0, max_value=float(image.image.width), value=None, step=1.0, help="Pixel coordinate of the right axis anchor, in the original image.", key=f"{prefix}_cal_x_right")
    x_min = x_cols[2].number_input("X value at left", value=None, help="Data value printed at the selected left pixel anchor.", key=f"{prefix}_cal_x_min")
    x_max = x_cols[3].number_input("X value at right", value=None, help="Data value printed at the selected right pixel anchor.", key=f"{prefix}_cal_x_max")
    y_cols = st.columns(4)
    y_top = y_cols[0].number_input("Y top pixel", min_value=0.0, max_value=float(image.image.height), value=None, step=1.0, help="Pixel coordinate of the top axis anchor, in the original image.", key=f"{prefix}_cal_y_top")
    y_bottom = y_cols[1].number_input("Y bottom pixel", min_value=0.0, max_value=float(image.image.height), value=None, step=1.0, help="Pixel coordinate of the bottom axis anchor, in the original image.", key=f"{prefix}_cal_y_bottom")
    y_max = y_cols[2].number_input("Y value at top", value=None, help="Data value printed at the selected top pixel anchor.", key=f"{prefix}_cal_y_max")
    y_min = y_cols[3].number_input("Y value at bottom", value=None, help="Data value printed at the selected bottom pixel anchor.", key=f"{prefix}_cal_y_min")
    scales = st.columns(2)
    x_scale = scales[0].selectbox("X scale", ["linear", "log"], key=f"{prefix}_cal_x_scale")
    y_scale = scales[1].selectbox("Y scale", ["linear", "log"], key=f"{prefix}_cal_y_scale")
    anchors = (x_left, x_right, x_min, x_max, y_top, y_bottom, y_min, y_max)
    if any(value is None for value in anchors):
        st.info("Enter all eight pixel/data anchor values to configure the two-point axis mapping.")
        return None
    try:
        calibration = AxisCalibration(
            x_left_px=float(x_left), x_right_px=float(x_right), x_min=float(x_min), x_max=float(x_max),
            y_top_px=float(y_top), y_bottom_px=float(y_bottom), y_min=float(y_min), y_max=float(y_max),
            x_scale=x_scale, y_scale=y_scale,
        )
        calibration.validate_image_bounds(image.image.size, crop)
        st.caption("Configured mapping: left/right pixel anchors map to the supplied x values; top/bottom pixel anchors map to the supplied y values. Image y pixels increase downward.")
        st.json(calibration.to_dict())
        return calibration
    except ValueError as error:
        st.warning(f"Calibration needs review: {error}")
        return None


def _exclusion_region_controls(st: Any, image: GraphImage, crop: tuple[int, int, int, int], prefix: str) -> tuple[tuple[int, int, int, int], ...] | None:
    """Collect absolute-pixel exclusion rectangles for inset/legend masking."""
    left, top, right, bottom = crop
    st.subheader("2b · Exclude inset or legend regions (optional)")
    st.caption("Rectangles are in original-image pixels and must sit inside the selected crop. If an exclusion crosses a curve, its points become a gap; the app does not interpolate across it. Re-run on a separate crop to analyze an inset independently.")
    count = int(st.number_input("Number of exclusion rectangles", min_value=0, max_value=8, value=0, step=1, key=f"{prefix}_exclusion_count"))
    regions: list[tuple[int, int, int, int]] = []
    for index in range(count):
        st.caption(f"Exclusion {index + 1} · bounds are left/top inclusive, right/bottom exclusive")
        cols = st.columns(4)
        values = (
            cols[0].number_input(f"Exclusion {index + 1} left pixel", min_value=0, max_value=image.image.width, value=None, step=1, key=f"{prefix}_exclude_{index}_left"),
            cols[1].number_input(f"Exclusion {index + 1} top pixel", min_value=0, max_value=image.image.height, value=None, step=1, key=f"{prefix}_exclude_{index}_top"),
            cols[2].number_input(f"Exclusion {index + 1} right pixel", min_value=0, max_value=image.image.width, value=None, step=1, key=f"{prefix}_exclude_{index}_right"),
            cols[3].number_input(f"Exclusion {index + 1} bottom pixel", min_value=0, max_value=image.image.height, value=None, step=1, key=f"{prefix}_exclude_{index}_bottom"),
        )
        if any(value is None for value in values):
            st.info(f"Enter all four coordinates for exclusion {index + 1}.")
            return None
        regions.append(tuple(int(value) for value in values))
    try:
        from research_plotter.analysis.graph_digitization import validate_exclusion_regions

        return validate_exclusion_regions(image.image.size, crop, regions)
    except ValueError as error:
        st.warning(f"Exclusion region needs review: {error}")
        return None


def _display_fit(st: Any, fit: EquationFitResult, fit_paths: dict[str, Path] | None, *, selected_curve: str, equation_source: str) -> None:
    st.subheader("Fit completed · empirical equation candidate")
    st.success(f"Selected curve: {selected_curve} · {equation_source} · status: {fit.status}")
    st.latex(fit.latex)
    st.code(fit.equation, language="text")
    st.dataframe(pd.DataFrame([{"Parameter": key, "Estimate": value, "Local covariance SE": fit.uncertainties.get(key), "Bounds": fit.parameter_bounds.get(key, "fixed/user-provided")} for key, value in fit.parameters.items()]), use_container_width=True, hide_index=True)
    metrics = st.columns(4)
    for column, key in zip(metrics, ("rmse", "mae", "r_squared", "bic")):
        value = fit.metrics.get(key)
        column.metric(key.upper(), "not defined" if value is None else f"{value:.6g}")
    st.caption(f"Status: {fit.status} · method: {fit.method}. A good fit does not establish the source equation.")
    for warning in fit.warnings:
        st.warning(warning)
    if fit_paths:
        if fit_paths.get("overlay_png", Path()).is_file():
            st.image(str(fit_paths["overlay_png"]), caption="Digitized data and empirical fitted candidate", use_container_width=True)
        if fit_paths.get("residual_png", Path()).is_file():
            st.image(str(fit_paths["residual_png"]), caption="Residual analysis", use_container_width=True)


def _render_paper_comparison(st: Any, latest: dict[str, Any], new_run_dir, record_run) -> None:
    context = st.session_state.get("graph_image_paper_context")
    reproduction = st.session_state.get("latest_selected_reproduction")
    if not context or not reproduction or reproduction.get("selected_figure") != context.get("figure_id"):
        st.info("For a paper-model comparison, open this graph from Figure Reproduction and run a supported model first. A digitized curve is an approximate reference, not ground truth.")
        return
    experiments = reproduction.get("report", {}).get("experiments", [])
    simulation_options = [(item.get("name", item.get("figure_id", f"experiment {index + 1}")), Path(item["simulation_data"])) for index, item in enumerate(experiments) if item.get("simulation_data") and Path(item["simulation_data"]).is_file()]
    if not simulation_options:
        st.warning("The selected figure has no completed simulation data available for comparison; source limitations may leave the reproduction BLOCKED.")
        return
    simulation_label = st.selectbox("Paper-adapter trajectory", [label for label, _ in simulation_options], key="graph_paper_simulation")
    simulation_path = dict(simulation_options)[simulation_label]
    try:
        simulation = pd.read_csv(simulation_path)
        state_columns = [name for name in simulation.columns if name != "time"]
        reference_data = latest.get("reviewed_data", latest["result"].data)
        curve_names = list(reference_data["curve_name"].unique())
        cols = st.columns(2)
        curve_name = cols[0].selectbox("Digitized curve", curve_names, key="graph_paper_curve")
        state_name = cols[1].selectbox("Simulation state", state_columns, key="graph_paper_state")
        if st.button("Compare simulation to digitized reference", type="primary", key="graph_compare_paper"):
            comparison = compare_digitized_reference(reference_data, simulation, curve_name=curve_name, state_name=state_name)
            image_source = latest["source"]
            comparison["digitized_reference_metadata"] = {
                "source_document_or_image": image_source.source_name,
                "source_format": image_source.source_format,
                "source_page": image_source.page_number,
                "image_dimensions_pixels": list(image_source.image.size),
                "crop_coordinates_pixels": list(latest["crop"]),
                "exclusion_regions_pixels": [list(region) for region in latest.get("exclusion_regions", ())],
                "axis_calibration": latest["calibration"].to_dict(),
                "axis_calibration_provenance": "USER_PROVIDED",
                "curve_extraction_warnings": list(latest["result"].warnings),
                "paper_figure_context": context,
                "simulation_data": str(simulation_path),
            }
            run_id, run_dir = new_run_dir("graph-reference-comparison")
            started = time.perf_counter()
            paths = save_digitized_comparison(comparison, reference_data, output_dir=run_dir)
            record_run(run_id, run_dir, "graph-reference-comparison", started, status="UNVERIFIED", details={"source_document": context.get("source_document"), "figure_id": context.get("figure_id"), "comparison_report": str(paths["report"]), "simulation_data": str(simulation_path), "reference_provenance": "DIGITIZED_IMAGE"})
            st.session_state["latest_graph_reference_comparison"] = {"comparison": comparison, "paths": {key: str(path) for key, path in paths.items()}, "run_dir": str(run_dir)}
    except Exception as error:
        st.error(f"Reference comparison could not be prepared.\n\nReason: {error}")
    comparison = st.session_state.get("latest_graph_reference_comparison")
    if comparison:
        st.warning("Comparison status: UNVERIFIED · digitized-reference comparison, not exact-ground-truth validation.")
        st.json({"metrics": comparison["comparison"].get("metrics"), "warnings": comparison["comparison"].get("warnings")})
        png = Path(comparison["paths"].get("png", ""))
        if png.is_file():
            st.image(str(png), caption="Paper simulation versus digitized graph", use_container_width=True)


def render_graph_image_page(
    st: Any,
    *,
    outputs: Path,
    new_run_dir: Callable[[str], tuple[str, Path]],
    record_run: Callable[..., Path],
) -> None:
    """Render image calibration, digitization, empirical fitting, and handoffs."""
    st.title("Graph Image Analyzer")
    st.write("Choose source-evidence analysis for a research paper, or digitize a graph and explore empirical equation candidates. A graph fit is not proof of the paper's original model.")
    choice_columns = st.columns(2)
    with choice_columns[0].container(border=True):
        st.markdown("**Find the source equation**")
        st.caption("Analyze equations, methods, parameters, and figure links from a research-paper PDF.")
        st.button("Open Paper Analyzer", on_click=_open_paper_analyzer, key="graph_open_paper_analyzer")
    with choice_columns[1].container(border=True):
        st.markdown("**Discover candidates from a graph**")
        st.caption("Upload a graph, calibrate its axes, digitize a curve, and compare safe equation families.")
    source = _load_source(st, outputs)
    if source is None:
        return
    image = source.image
    source_context = st.session_state.get("graph_image_paper_context") or {}
    st.image(image, caption=f"Source · {source.source_name} · {source.source_format}{f' · page {source.page_number}' if source.page_number else ''} · {image.width}×{image.height}px", use_container_width=True)
    if image.width * image.height > 24_000_000:
        st.warning("Large image: crop to the plot area before curve extraction to keep the workbench responsive.")
    digest = source.source_sha256 or f"{image.width}x{image.height}"
    prefix = f"graph_{digest[:12]}_{source.page_number or 0}"

    with st.expander("Advanced cropping and exclusion regions", expanded=False):
        st.subheader("2 · Select plot region")
        st.caption("Crop to the actual plotting axes. Exclude legends or inset areas here; an inset should be analyzed separately with its own crop and calibration.")
        crop_cols = st.columns(4)
        crop_left = crop_cols[0].number_input("Crop left pixel", min_value=0, max_value=image.width - 1, value=0, key=f"{prefix}_crop_left")
        crop_right = crop_cols[1].number_input("Crop right pixel", min_value=1, max_value=image.width, value=image.width, key=f"{prefix}_crop_right")
        crop_top = crop_cols[2].number_input("Crop top pixel", min_value=0, max_value=image.height - 1, value=0, key=f"{prefix}_crop_top")
        crop_bottom = crop_cols[3].number_input("Crop bottom pixel", min_value=1, max_value=image.height, value=image.height, key=f"{prefix}_crop_bottom")
        try:
            crop = validate_crop(image.size, (int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)))
            crop_image = image.crop(crop)
        except ValueError as error:
            st.error(f"Plot region is invalid: {error}")
            return
        exclusion_regions = _exclusion_region_controls(st, source, crop, prefix)
        if exclusion_regions is not None:
            crop_preview = crop_image.copy()
            draw = ImageDraw.Draw(crop_preview)
            for index, (ex_left, ex_top, ex_right, ex_bottom) in enumerate(exclusion_regions, start=1):
                local_box = (ex_left - crop[0], ex_top - crop[1], ex_right - crop[0] - 1, ex_bottom - crop[1] - 1)
                draw.rectangle(local_box, outline="#ff8c00", width=3)
                draw.text((local_box[0] + 3, local_box[1] + 3), f"Exclude {index}", fill="#ff8c00", stroke_width=2, stroke_fill="white")
            st.image(crop_preview, caption="Selected plot region · orange boxes are excluded from extraction", use_container_width=True)
        else:
            st.image(crop_image, caption="Selected plot region", use_container_width=True)

    if "crop" not in locals():
        crop = (0, 0, image.width, image.height)
        crop_image = image.copy()
        exclusion_regions = ()
        st.caption("Current plot region uses the full image. Use Advanced cropping if labels, legend, or insets are present.")
    calibration = _axis_calibration_controls(st, source, crop, prefix)
    st.subheader("4 · Choose curves and analyze")
    colors = detect_curve_colors(image, crop, exclusion_regions=exclusion_regions or ())
    if not colors:
        st.warning("No likely line colors were detected. Tighten the crop or choose a more colorful/high-contrast source image.")
        return
    color_by_label = {color.label: color for color in colors}
    with st.expander("Curve selection and extraction settings", expanded=False):
        with st.expander("Detected curve colors (confirm before using)", expanded=False):
            for color in colors:
                st.write(color.label)
                st.color_picker(f"Preview {color.label}", value="#%02x%02x%02x" % color.rgb, disabled=True, key=f"{prefix}_preview_{color.rgb}")
        selected_labels = st.multiselect("Curves to extract", list(color_by_label), default=list(color_by_label)[:min(3, len(color_by_label))], key=f"{prefix}_selected_colors")
        named_colors: list[tuple[str, tuple[int, int, int]]] = []
        for index, label in enumerate(selected_labels):
            candidate: CurveColor = color_by_label[label]
            name = st.text_input(f"Curve label for {label}", value=f"Curve {index + 1}", key=f"{prefix}_curve_name_{candidate.rgb}")
            named_colors.append((name, candidate.rgb))
        add_manual_color = st.checkbox("Add/override a curve color manually", key=f"{prefix}_manual_color_toggle")
        if add_manual_color:
            manual_cols = st.columns(2)
            manual_hex = manual_cols[0].color_picker("Curve RGB color", value="#d62728", key=f"{prefix}_manual_rgb")
            manual_name = manual_cols[1].text_input("Manual curve name", value=f"Curve {len(named_colors) + 1}", key=f"{prefix}_manual_name")
            manual_rgb = tuple(int(manual_hex[index:index + 2], 16) for index in (1, 3, 5))
            named_colors = [item for item in named_colors if item[0] != manual_name]
            named_colors.append((manual_name, manual_rgb))
        tolerance = st.slider("Color distance tolerance", min_value=15, max_value=140, value=62, help="Higher values include anti-aliased edge pixels but can merge nearby colors.", key=f"{prefix}_tolerance")
        max_points = st.slider("Maximum points per curve", min_value=100, max_value=5000, value=2000, step=100, help="Long curves are evenly downsampled to keep fitting and editing responsive.", key=f"{prefix}_max_points")
    if "named_colors" not in locals():
        named_colors = [(f"Curve {index + 1}", color.rgb) for index, color in enumerate(colors[:min(3, len(colors))])]
        tolerance = float(st.session_state.get(f"{prefix}_tolerance", 62))
        max_points = int(st.session_state.get(f"{prefix}_max_points", 2000))
    st.caption("Calibration is manual unless you supply two pixel/value anchors per axis. Review the curve overlay and warnings before interpreting a fit.")
    analyze_clicked = st.button("Analyze Graph and Find Equations", type="primary", disabled=calibration is None or not named_colors, key=f"{prefix}_analyze_graph")
    digitize_clicked = st.button("Extract digitized data", disabled=calibration is None or not named_colors, key=f"{prefix}_run_digitization")
    if (digitize_clicked or analyze_clicked) and calibration is not None and exclusion_regions is not None:
        run_id, run_dir = new_run_dir("graph-digitization")
        started = time.perf_counter()
        try:
            with st.spinner("Segmenting curve pixels and applying axis calibration…"):
                result = digitize_color_curves(image, crop=crop, calibration=calibration, colors=named_colors, color_tolerance=float(tolerance), max_points=int(max_points), exclusion_regions=exclusion_regions)
                artifacts = save_digitization_artifacts(
                    source, result, calibration=calibration, crop=crop, output_dir=run_dir, experiment_id=run_id,
                    extraction_settings={"colors": [{"name": name, "rgb": rgb} for name, rgb in named_colors], "color_tolerance": tolerance, "maximum_points_per_curve": max_points},
                    source_context=source_context,
                    exclusion_regions=exclusion_regions,
                )
            record_run(run_id, run_dir, "graph-digitization", started, status="DIGITIZED", details={"source": source.source_name, "source_page": source.page_number, "point_count": len(result.data), "curve_count": len(result.curve_names), "metadata": str(artifacts["metadata"]), "warnings": result.warnings})
            candidates: list[CandidateFit] = []
            discovery_report: str | None = None
            if analyze_clicked:
                preferred_curve = named_colors[0][0]
                discovery_id, discovery_dir = new_run_dir("graph-equation-discovery")
                discovery_started = time.perf_counter()
                try:
                    with st.spinner(f"Ranking equation candidates for {preferred_curve}…"):
                        selected_points = result.data[result.data["curve_name"] == preferred_curve]
                        candidates = discover_equation_candidates(selected_points, max_candidates=5)
                        discovery_payload = {
                            "status": "CANDIDATE",
                            "evidence_status": "EMPIRICAL_FIT",
                            "digitization_run": run_id,
                            "curve": preferred_curve,
                            "validation_policy": "Every fifth x-sorted training observation was held out; training-only initial guesses; ranked by held-out RMSE, then full-fit BIC and parameter count.",
                            "interpretation": "Candidate equations approximate digitized points; they are not evidence of the original source equation.",
                            "candidates": [candidate.to_dict() for candidate in candidates],
                        }
                        report_path = discovery_dir / "equation_candidates.json"
                        report_path.write_text(json.dumps(discovery_payload, indent=2, allow_nan=False), encoding="utf-8")
                        discovery_report = str(report_path)
                    record_run(discovery_id, discovery_dir, "graph-equation-discovery", discovery_started, status="CANDIDATE", details={"digitization_run": run_id, "curve": preferred_curve, "candidate_count": len(candidates), "candidate_report": discovery_report, "evidence_status": "EMPIRICAL_FIT"})
                except Exception as error:
                    record_run(discovery_id, discovery_dir, "graph-equation-discovery", discovery_started, status="FAILED", error=str(error), details={"digitization_run": run_id, "curve": preferred_curve, "evidence_status": "EMPIRICAL_FIT"})
                    st.warning(f"Digitization completed, but equation discovery needs review: {error}")
            st.session_state["latest_graph_analysis"] = {
                "run_id": run_id, "run_dir": str(run_dir), "source": source, "source_context": source_context,
                "result": result, "calibration": calibration, "crop": crop,
                "extraction_settings": {"colors": [{"name": name, "rgb": rgb} for name, rgb in named_colors], "color_tolerance": tolerance, "maximum_points_per_curve": max_points},
                "exclusion_regions": exclusion_regions,
                "artifacts": {key: str(path) for key, path in artifacts.items()}, "fit": None,
                "candidates": candidates, "candidate_report": discovery_report,
                "candidate_curve": named_colors[0][0] if candidates and analyze_clicked else None,
                "settings_signature": (crop, calibration, tuple(named_colors), float(tolerance), int(max_points), exclusion_regions),
            }
        except Exception as error:
            record_run(run_id, run_dir, "graph-digitization", started, status="FAILED", error=str(error), details={"source": source.source_name, "source_page": source.page_number})
            st.error(f"Curve digitization failed.\n\nReason: {error}\n\nReview the crop, calibration, selected curve colors, and tolerance.")

    latest = st.session_state.get("latest_graph_analysis")
    current_signature = (crop, calibration, tuple(named_colors), float(tolerance), int(max_points), exclusion_regions)
    if not latest or latest.get("source", source).source_sha256 != source.source_sha256 or latest.get("source", source).page_number != source.page_number or latest.get("settings_signature") != current_signature:
        st.info("Set a crop, confirm calibration and curve colors, then extract points. Changed settings require a fresh digitization run.")
        return
    result: DigitizationResult = latest["result"]
    data = latest.get("reviewed_data", result.data)
    st.subheader("5 · Inspect digitized data")
    summary = st.columns(4)
    summary[0].metric("Points", len(data))
    summary[1].metric("Curves", data["curve_name"].nunique())
    summary[2].metric("X range", f"{data['x'].min():.5g} – {data['x'].max():.5g}")
    summary[3].metric("Y range", f"{data['y'].min():.5g} – {data['y'].max():.5g}")
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)
        st.warning("Review extraction warnings and the pixel overlay before fitting. Partial coverage, overlapping same-color marks, or exclusion gaps can materially affect an equation estimate.")
    if latest.get("candidates"):
        st.info("Curve extraction and candidate search completed. Continue to Fit / Compare to review ranked equations for the selected curve.")
    else:
        st.info("Curve extraction completed. Continue to Fit / Compare to estimate an equation from the selected curve.")
    st.caption("CSV uses long form: one row per extracted point with curve_name, source pixel coordinates, and calibrated x/y.")
    data_tab, fit_tab, export_tab = st.tabs(["Digitized points", "Fit / Compare · next step", "Export / hand off"])
    with data_tab:
        st.caption("Correct a value or delete a point below. Manual edits are USER_PROVIDED corrections and remain separate from the original pixel extraction.")
        edited_data = st.data_editor(data, num_rows="dynamic", use_container_width=True, hide_index=True, key=f"{prefix}_digitized_editor")
        if not edited_data.equals(data):
            data = edited_data
            latest["reviewed_data"] = edited_data
            # Never keep presenting a fit against the pre-edit observations.
            latest["fit"] = None
            latest.pop("fit_paths", None)
            latest.pop("candidates", None)
            st.session_state["latest_graph_analysis"] = latest
        if "reviewed_data" in latest and st.button("Save reviewed point corrections", key=f"{prefix}_save_reviewed_points"):
            reviewed_path = Path(latest["run_dir"]) / "reviewed_digitized_data.csv"
            reviewed_path.parent.mkdir(parents=True, exist_ok=True)
            edited_data.to_csv(reviewed_path, index=False)
            review_metadata = {
                "status": "NEEDS_REVIEW",
                "review_status": "USER_REVIEWED",
                "provenance": "USER_PROVIDED",
                "original_digitization_run": latest["run_id"],
                "original_point_count": int(len(result.data)),
                "reviewed_point_count": int(len(edited_data)),
                "note": "Manual corrections are separate from the original pixel extraction and require scientific interpretation.",
            }
            (Path(latest["run_dir"]) / "reviewed_digitized_data.json").write_text(json.dumps(review_metadata, indent=2), encoding="utf-8")
            record_run(latest["run_id"], Path(latest["run_dir"]), "graph-digitization", time.perf_counter(), status="DIGITIZED", details={"reviewed_data": str(reviewed_path), "review_status": "USER_REVIEWED", "reviewed_point_count": len(edited_data)})
            st.success(f"Saved {len(edited_data)} reviewed point(s) to {reviewed_path.name}.")
        data_png = Path(latest["artifacts"]["data_figure"])
        pixel_overlay = Path(latest["artifacts"]["pixel_overlay"])
        if data_png.is_file():
            st.image(str(data_png), caption="Digitized data coordinates", use_container_width=True)
        if pixel_overlay.is_file():
            st.image(str(pixel_overlay), caption="Median pixel picks over source crop", use_container_width=True)
    with fit_tab:
        st.subheader("6 · Fit or compare equations")
        curve_names = list(data["curve_name"].drop_duplicates())
        curve_name = st.selectbox("Curve to fit", curve_names, key=f"{prefix}_fit_curve")
        fitting_mode = st.radio("Choose fitting mode", ["Fit a known equation form", "Discover candidate equations"], horizontal=True, key=f"{prefix}_fit_mode")
        selected_curve = data[data["curve_name"] == curve_name]
        if fitting_mode == "Fit a known equation form":
            st.caption("Enter the equation family you want fitted; for example, y = a*x+b or y = A*exp(-k*x)*sin(w*x). These are empirical candidates, not claims about the source model.")
            template = st.text_input("Equation template", value="", placeholder="y = A*exp(-k*x)*sin(w*x)", key=f"{prefix}_fit_template")
            guesses_text = st.text_input("Initial guesses (optional)", value="", placeholder="A=1,k=0.1,w=3", key=f"{prefix}_fit_guesses")
            bounds_text = st.text_input("Parameter bounds (optional)", value="", placeholder="k=0:inf,w=0:inf", key=f"{prefix}_fit_bounds")
            current_fit_signature = ("known", curve_name, template, guesses_text, bounds_text)
            if st.button("Fit known equation form", type="primary", key=f"{prefix}_fit_known"):
                run_id, run_dir = new_run_dir("graph-equation-fit")
                started = time.perf_counter()
                try:
                    fit = fit_equation(selected_curve, template, initial_guesses=_parameter_guesses(guesses_text), bounds=_parse_bounds(bounds_text))
                    paths = save_fit_artifacts(data, fit, curve_name=curve_name, output_dir=run_dir, crop_image=source.image.crop(crop), crop_origin=(crop[0], crop[1]), calibration=latest["calibration"], fit_configuration={"experiment_id": run_id, "mode": "known_equation_form", "template": template, "initial_guesses_text": guesses_text, "bounds_text": bounds_text, "curve": curve_name, "digitization_run": latest["run_id"], "source": {"name": source.source_name, "format": source.source_format, "page_number": source.page_number, "sha256": source.source_sha256, "dimensions_pixels": list(source.image.size), "provenance": "USER_PROVIDED"}, "crop_coordinates_pixels": dict(zip(("left", "top", "right", "bottom"), latest["crop"])), "exclusion_regions_pixels": [list(region) for region in latest.get("exclusion_regions", ())], "extraction_settings": latest["extraction_settings"], "axis_calibration": latest["calibration"].to_dict(), "source_context": latest["source_context"]})
                    record_run(run_id, run_dir, "graph-equation-fit", started, status=fit.status, details={"digitization_run": latest["run_id"], "curve": curve_name, "equation": fit.equation, "fit_result": str(paths["result"])})
                    latest["fit"] = fit
                    latest["fit_paths"] = {key: str(path) for key, path in paths.items()}
                    latest["fit_run_dir"] = str(run_dir)
                    latest["fit_signature"] = current_fit_signature
                    latest["fit_description"] = f"user-supplied template: {template}"
                    st.session_state["latest_graph_analysis"] = latest
                except Exception as error:
                    record_run(run_id, run_dir, "graph-equation-fit", started, status="FAILED", error=str(error), details={"digitization_run": latest["run_id"], "curve": curve_name, "equation_template": template})
                    st.error(f"Equation fitting failed.\n\nReason: {error}\n\nCheck the safe formula syntax, initial guesses, parameter bounds, and data domain.")
        else:
            st.caption("A deterministic subset of points is held out for ranking. Fitting only identifies plausible approximations; graph data rarely uniquely determines an equation.")
            current_fit_signature = ("discovery", curve_name, None)
            if st.button("Discover equation candidates", type="primary", key=f"{prefix}_discover"):
                run_id, run_dir = new_run_dir("graph-equation-discovery")
                started = time.perf_counter()
                try:
                    with st.spinner("Fitting supported equation families and validating on held-out points…"):
                        candidates = discover_equation_candidates(selected_curve)
                    payload = {
                        "status": "CANDIDATE",
                        "digitization_run": latest["run_id"],
                        "curve": curve_name,
                        "validation_policy": "Every fifth x-sorted observation was held out; initial guesses were derived from the training subset only; candidates are ranked by held-out RMSE, then BIC and parameter count.",
                        "interpretation": "Candidate equations approximate digitized points; they are not evidence of the original source equation.",
                        "candidates": [candidate.to_dict() for candidate in candidates],
                    }
                    (run_dir / "equation_candidates.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
                    record_run(run_id, run_dir, "graph-equation-discovery", started, status="CANDIDATE", details={"digitization_run": latest["run_id"], "curve": curve_name, "candidate_count": len(candidates), "candidate_report": str(run_dir / "equation_candidates.json"), "validation_policy": payload["validation_policy"]})
                    latest["candidates"] = candidates
                    latest["candidate_curve"] = curve_name
                    latest["candidate_run_dir"] = str(run_dir)
                    st.session_state["latest_graph_analysis"] = latest
                except Exception as error:
                    record_run(run_id, run_dir, "graph-equation-discovery", started, status="FAILED", error=str(error), details={"digitization_run": latest["run_id"], "curve": curve_name})
                    st.error(f"Candidate discovery failed.\n\nReason: {error}")
            candidates: list[CandidateFit] = latest.get("candidates", [])
            if candidates and latest.get("candidate_curve") not in {None, curve_name}:
                st.info(f"The visible candidates belong to {latest['candidate_curve']}. Run discovery for {curve_name} to replace them.")
                candidates = []
            if candidates:
                st.dataframe(pd.DataFrame([{
                    "Rank": index + 1,
                    "Family": candidate.family,
                    "Status": candidate.result.status,
                    "Evidence": candidate.evidence_status,
                    "Holdout RMSE": candidate.holdout_metrics.get("rmse") if candidate.holdout_metrics else candidate.holdout_rmse,
                    "Holdout MAE": candidate.holdout_metrics.get("mae") if candidate.holdout_metrics else None,
                    "Fit RMSE": candidate.result.metrics.get("rmse"),
                    "Fit MAE": candidate.result.metrics.get("mae"),
                    "R²": candidate.result.metrics.get("r_squared"),
                    "BIC": candidate.bic,
                } for index, candidate in enumerate(candidates)]), use_container_width=True, hide_index=True)
                labels = [f"{candidate.family} · holdout RMSE {candidate.holdout_rmse:.4g}" for candidate in candidates]
                candidate_index = st.selectbox("Candidate to inspect", range(len(candidates)), format_func=lambda index: labels[index], key=f"{prefix}_candidate_selected")
                selected_candidate = candidates[candidate_index]
                current_fit_signature = ("discovery", curve_name, selected_candidate.family)
                st.markdown("#### Selected ranked candidate")
                st.latex(selected_candidate.result.latex)
                st.code(selected_candidate.result.equation, language="text")
                parameter_table = pd.DataFrame([{
                    "Parameter": key,
                    "Estimate": value,
                    "Local covariance SE": selected_candidate.result.uncertainties.get(key),
                    "Evidence": "ESTIMATED_FROM_DIGITIZED_PIXELS",
                } for key, value in selected_candidate.result.parameters.items()])
                st.dataframe(parameter_table, use_container_width=True, hide_index=True)
                candidate_metrics = st.columns(5)
                candidate_metrics[0].metric("Holdout RMSE", f"{selected_candidate.holdout_rmse:.6g}")
                holdout_mae = (selected_candidate.holdout_metrics or {}).get("mae")
                candidate_metrics[1].metric("Holdout MAE", "n/a" if holdout_mae is None else f"{holdout_mae:.6g}")
                candidate_metrics[2].metric("Fit RMSE", f"{selected_candidate.result.metrics['rmse']:.6g}")
                candidate_metrics[3].metric("Fit MAE", f"{selected_candidate.result.metrics['mae']:.6g}")
                r_squared = selected_candidate.result.metrics.get("r_squared")
                candidate_metrics[4].metric("R²", "not defined" if r_squared is None else f"{r_squared:.6g}")
                st.caption(f"Evidence classification: **{selected_candidate.evidence_status}** · ranking: held-out RMSE, with BIC and parameter count as tie-breakers. Estimates are conditional on the selected curve, calibration, and candidate family.")
                for warning in selected_candidate.result.warnings:
                    st.warning(warning)
                from research_plotter.plotting.graph_image import plot_fit_overlay, plot_fit_residuals

                candidate_figure = plot_fit_overlay(data, selected_candidate.result, curve_name=curve_name)
                st.pyplot(candidate_figure, clear_figure=True)
                residual_figure = plot_fit_residuals(selected_candidate.result)
                st.pyplot(residual_figure, clear_figure=True)
                st.warning("Candidate equations approximate digitized graph pixels. A close fit does not verify the original paper equation or model.")
                st.button(
                    "Open selected candidate in Formula Plotter",
                    on_click=handoff_to_formula_plotter,
                    args=(st, equation_for_formula_plotter(selected_candidate.result), (float(np.min(selected_candidate.result.x)), float(np.max(selected_candidate.result.x)))),
                    key=f"{prefix}_candidate_formula_handoff_{selected_candidate.family}",
                )
                if st.button("Use and save selected candidate", key=f"{prefix}_candidate_save"):
                    run_id, run_dir = new_run_dir("graph-equation-fit")
                    started = time.perf_counter()
                    paths = save_fit_artifacts(data, selected_candidate.result, curve_name=curve_name, output_dir=run_dir, crop_image=source.image.crop(crop), crop_origin=(crop[0], crop[1]), calibration=latest["calibration"], fit_configuration={"experiment_id": run_id, "mode": "candidate_discovery", "selected_family": selected_candidate.family, "holdout_rmse": selected_candidate.holdout_rmse, "curve": curve_name, "digitization_run": latest["run_id"], "source": {"name": source.source_name, "format": source.source_format, "page_number": source.page_number, "sha256": source.source_sha256, "dimensions_pixels": list(source.image.size), "provenance": "USER_PROVIDED"}, "crop_coordinates_pixels": dict(zip(("left", "top", "right", "bottom"), latest["crop"])), "exclusion_regions_pixels": [list(region) for region in latest.get("exclusion_regions", ())], "extraction_settings": latest["extraction_settings"], "axis_calibration": latest["calibration"].to_dict(), "source_context": latest["source_context"]})
                    record_run(run_id, run_dir, "graph-equation-fit", started, status="CANDIDATE", details={"digitization_run": latest["run_id"], "candidate_family": selected_candidate.family, "fit_result": str(paths["result"]), "holdout_rmse": selected_candidate.holdout_rmse})
                    latest["fit"] = selected_candidate.result
                    latest["fit_paths"] = {key: str(path) for key, path in paths.items()}
                    latest["fit_run_dir"] = str(run_dir)
                    latest["selected_candidate_family"] = selected_candidate.family
                    latest["fit_signature"] = current_fit_signature
                    latest["fit_description"] = f"discovered candidate family: {selected_candidate.family}"
                    st.session_state["latest_graph_analysis"] = latest
        fit = latest.get("fit")
        fit_paths = {key: Path(path) for key, path in latest.get("fit_paths", {}).items()}
        if fit and latest.get("fit_signature") == current_fit_signature:
            _display_fit(st, fit, fit_paths, selected_curve=curve_name, equation_source=latest.get("fit_description", "selected candidate"))
            st.warning("Equation candidates approximate the extracted graph data. A good curve fit does not prove that the candidate was the original equation or mathematical model.")
            st.button("Open selected equation in Formula Plotter", on_click=handoff_to_formula_plotter, args=(st, equation_for_formula_plotter(fit), (float(np.min(fit.x)), float(np.max(fit.x)))), key=f"{prefix}_open_formula")
        elif fit:
            st.info("A completed fit exists for a different curve or fitting configuration. Run the current selection to display a matching result.")
        else:
            st.caption("No fitted equation is shown yet. Choose a known template or discover candidates, then run the selected fitting action.")
        st.divider()
        st.subheader("Paper model vs digitized reference")
        _render_paper_comparison(st, latest, new_run_dir, record_run)
    with export_tab:
        st.write("**Digitization status:** DIGITIZED · axis calibration provenance USER_PROVIDED")
        st.write("**Source context:**", source_context or "generic image upload")
        for name in ("long_csv", "wide_csv", "metadata", "data_figure", "data_svg", "data_pdf", "pixel_overlay", "pixel_overlay_svg", "pixel_overlay_pdf"):
            artifact = Path(latest["artifacts"][name])
            if artifact.is_file():
                st.download_button(f"Download {name.replace('_', ' ').title()}", artifact.read_bytes(), file_name=artifact.name, key=f"{prefix}_download_{name}")
        reviewed_data = latest.get("reviewed_data")
        if reviewed_data is not None:
            st.download_button("Download edited points CSV", reviewed_data.to_csv(index=False), file_name="reviewed_digitized_data.csv", mime="text/csv", key=f"{prefix}_download_reviewed_points")
        fit = latest.get("fit")
        fit_paths = {key: Path(path) for key, path in latest.get("fit_paths", {}).items()}
        if fit:
            for key in ("data", "result", "overlay_png", "overlay_svg", "overlay_pdf", "residual_png", "residual_svg", "residual_pdf"):
                artifact = fit_paths.get(key)
                if artifact and artifact.is_file():
                    st.download_button(f"Download {key.replace('_', ' ').title()}", artifact.read_bytes(), file_name=artifact.name, key=f"{prefix}_fit_download_{key}")
        st.button("Send digitized data to Dataset Analyzer", on_click=handoff_to_dataset_analyzer, args=(st, data, new_run_dir, record_run), key=f"{prefix}_open_dataset")
        st.caption("The long-form digitized points can open in the existing Dataset Analyzer without requiring an equation fit.")

