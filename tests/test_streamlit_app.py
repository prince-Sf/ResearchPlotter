from pathlib import Path

import pytest
from PIL import Image, ImageDraw

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from research_plotter.app import NAVIGATION


APP_PATH = Path(__file__).parents[1] / "src" / "research_plotter" / "app.py"
PAPER_1_PDF = APP_PATH.parents[2] / "Papers" / "333.pdf"


def _enter_graph_calibration(app, *, x_right=230, y_top=15, y_bottom=145, x_min=0.0, x_max=10.0, y_top_value=10.0, y_bottom_value=0.0):
    values = {
        "X left pixel": 10,
        "X right pixel": x_right,
        "X value at left": x_min,
        "X value at right": x_max,
        "Y top pixel": y_top,
        "Y bottom pixel": y_bottom,
        "Y value at top": y_top_value,
        "Y value at bottom": y_bottom_value,
    }
    for label, value in values.items():
        next(item for item in app.number_input if item.label == label).set_value(value)
    app.run()


def test_workbench_navigation_and_bundled_dataset_flow():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    navigation = next(item for item in app.radio if item.key == "main_navigation")
    rendered_labels = list(navigation.options)
    normalized_pages = [next((page for page in NAVIGATION if str(label).endswith(page)), None) for label in rendered_labels]
    assert normalized_pages == NAVIGATION

    # Every page must render without an uncaught startup/import exception.
    for page in NAVIGATION:
        next(item for item in app.radio if item.key == "main_navigation").set_value(page).run()
        assert not app.exception, f"Workbench page {page!r} raised: {[item.value for item in app.exception]}"

    for page, action in (
        ("Model Simulation", "Run simulation"),
        ("Parameter Sweep", "Run parameter sweep"),
        ("Sensitivity Analysis", "Run sensitivity analysis"),
    ):
        next(item for item in app.radio if item.key == "main_navigation").set_value(page).run()
        next(button for button in app.button if button.label == action).click().run()
        assert not app.exception, f"{page} raised: {[item.value for item in app.exception]}"
        assert not app.error, f"{page} displayed an error: {[item.value for item in app.error]}"

    if PAPER_1_PDF.is_file():
        next(item for item in app.radio if item.key == "main_navigation").set_value("Paper Analyzer").run()
        next(button for button in app.button if button.label == "Analyze paper").click().run()
        assert not app.exception
        assert not app.error, [item.value for item in app.error]

    # Use the bundled example to exercise the page -> profiler -> plotting API
    # path without a browser upload or a network-dependent fixture.
    next(item for item in app.radio if item.key == "main_navigation").set_value("Dataset Analyzer").run()
    next(button for button in app.button if button.label == "Analyze dataset").click().run()
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric}["Rows"] == "10"
    next(button for button in app.button if button.label == "Generate selected figures").click().run()
    assert not app.exception
    assert any("Generated 1 selected figure" in message.value for message in app.success)

    next(item for item in app.radio if item.key == "main_navigation").set_value("Equation / Formula Plotter").run()
    next(button for button in app.button if button.label == "Analyze Equation").click().run()
    assert not app.exception
    assert any(metric.label == "Type" and metric.value == "EXPLICIT" for metric in app.metric)
    next(button for button in app.button if button.label == "Plot / Run Equation").click().run()
    assert not app.exception
    assert not app.error

    next(item for item in app.radio if item.key == "main_navigation").set_value("Convergence / Validation").run()
    next(button for button in app.button if button.label == "Run refinement experiment").click().run()
    assert not app.exception
    assert not app.error

    next(item for item in app.radio if item.key == "main_navigation").set_value("Noise / Uncertainty").run()
    next(button for button in app.button if button.label == "Run noise experiment").click().run()
    assert not app.exception
    assert not app.error, [item.value for item in app.error]


def test_graph_image_digitized_data_handoff_does_not_require_a_fit(tmp_path):
    image = Image.new("RGB", (240, 160), "white")
    ImageDraw.Draw(image).line((10, 145, 230, 15), fill=(214, 39, 40), width=3)
    source = tmp_path / "red-line.png"
    image.save(source)

    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["main_navigation"] = "Graph Image Analyzer"
    app.session_state["graph_image_source_path"] = str(source)
    app.session_state["graph_image_source_kind"] = "Paper figure handoff"
    app.run()
    assert not app.exception
    _enter_graph_calibration(app)
    assert any("actual plotted axis lines" in item.value for item in app.caption)
    next(button for button in app.button if button.label == "Extract digitized data").click().run()
    assert not app.exception, [item.value for item in app.exception]
    assert any(button.label == "Send digitized data to Dataset Analyzer" for button in app.button)
    assert "latest_graph_analysis" in app.session_state
    assert app.session_state["latest_graph_analysis"].get("fit") is None
    assert any("Curve extraction completed. Continue to Fit / Compare" in item.value for item in app.info)
    assert any("Fit / Compare · next step" in item.label for item in app.tabs)

    next(button for button in app.button if button.label == "Send digitized data to Dataset Analyzer").click().run()
    assert not app.exception, [item.value for item in app.exception]
    assert app.session_state["main_navigation"] == "Dataset Analyzer"
    assert app.session_state["dataset_frame"].shape[0] > 100
    assert app.session_state["dataset_profile"]["rows"] == app.session_state["dataset_frame"].shape[0]


def test_graph_image_fit_compare_ui_completes_and_transfers_equation(tmp_path):
    image = Image.new("RGB", (240, 160), "white")
    ImageDraw.Draw(image).line((10, 145, 230, 15), fill=(214, 39, 40), width=3)
    source = tmp_path / "red-line-fit.png"
    image.save(source)

    app = AppTest.from_file(str(APP_PATH), default_timeout=90)
    app.session_state["main_navigation"] = "Graph Image Analyzer"
    app.session_state["graph_image_source_path"] = str(source)
    app.session_state["graph_image_source_kind"] = "Paper figure handoff"
    app.run()
    _enter_graph_calibration(app)
    next(button for button in app.button if button.label == "Extract digitized data").click().run()
    assert not app.exception
    assert app.session_state["latest_graph_analysis"].get("fit") is None
    mode_control = next(item for item in app.radio if item.label == "Choose fitting mode")
    mode_control.set_value("Discover candidate equations").run()
    next(button for button in app.button if button.label == "Discover equation candidates").click().run()
    assert not app.exception
    assert app.session_state["latest_graph_analysis"]["candidates"], [item.value for item in app.error]
    assert any(item.value == "#### Selected ranked candidate" for item in app.markdown)
    mode_control = next(item for item in app.radio if item.label == "Choose fitting mode")
    mode_control.set_value("Fit a known equation form").run()
    next(item for item in app.text_input if item.label == "Equation template").set_value("y=a*x+b")
    next(item for item in app.text_input if item.label == "Initial guesses (optional)").set_value("a=1,b=0")
    next(item for item in app.text_input if item.label == "Parameter bounds (optional)").set_value("")
    app.run()
    assert not app.exception
    assert "Fit a known equation form" in next(item for item in app.radio if item.label == "Choose fitting mode").options
    next(button for button in app.button if button.label == "Fit known equation form").click().run()
    assert not app.exception, [item.value for item in app.exception]
    assert not app.error, [item.value for item in app.error]
    assert app.session_state["latest_graph_analysis"]["fit"].status == "FIT_SUCCESSFUL"
    assert any("Fit completed · empirical equation candidate" in item.value for item in app.subheader)
    assert any(item.label == "Open selected equation in Formula Plotter" for item in app.button)
    next(button for button in app.button if button.label == "Open selected equation in Formula Plotter").click().run()
    assert app.session_state["main_navigation"] == "Equation / Formula Plotter"
    assert "a =" in app.session_state["formula_manual_text"]


@pytest.mark.skipif(not PAPER_1_PDF.is_file(), reason="Local research PDF is optional and unavailable in this checkout.")
def test_paper_figure_button_opens_graph_analyzer_with_source_context():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["main_navigation"] = "Figure Reproduction"
    app.run()
    assert not app.exception
    assert any(button.label == "Digitize this published figure" for button in app.button)
    next(button for button in app.button if button.label == "Digitize this published figure").click().run()
    assert not app.exception, [item.value for item in app.exception]
    assert app.session_state["main_navigation"] == "Graph Image Analyzer"
    context = app.session_state["graph_image_paper_context"]
    assert context["source_document"].lower().endswith(".pdf")
    assert context["figure_id"].startswith("figure_")
    assert context["source_page"] is not None
    assert app.session_state["graph_image_page_number"] == context["source_page"]
