from pathlib import Path

from research_plotter.config import load_yaml, load_experiment, load_paper2_config
from research_plotter.paper_analysis.paper_parser import parse_paper
from research_plotter.pipelines.paper_reproduction import reproduce_paper


def test_paper_1_analysis_preserves_uncertainty(tmp_path: Path, paper_1_pdf: Path):
    analysis = parse_paper(paper_1_pdf)
    assert analysis.page_count == 12
    assert any("initial conditions" in item.lower() for item in analysis.unresolved_information)
    report = reproduce_paper(paper_1_pdf, tmp_path)
    assert report["status"] == "BLOCKED"
    assert any("ks" in item and "kT" in item for item in report["warnings"])
    assert report["unresolved_experiments"]


def test_paper_1_selected_figure_runs_user_configured_exploratory_experiment(tmp_path: Path, paper_1_pdf: Path):
    report = reproduce_paper(
        paper_1_pdf,
        tmp_path,
        initial_conditions={"S": .9, "E": .5, "T": .9, "C": 0.0},
        time_end=.2,
        step=.1,
        figure_id="figure_01",
    )
    assert report["status"] == "PARTIAL"
    assert report["selected_figure"] == "figure_01"
    assert report["experiments"]
    assert report["experiments"][0]["status"] == "UNVERIFIED"
    assert Path(report["experiments"][0]["simulation_data"]).is_file()
    assert any("index-at-zero" in warning for warning in report["warnings"])


def test_paper_1_unlisted_selected_sweep_values_remain_required(tmp_path: Path, paper_1_pdf: Path):
    report = reproduce_paper(
        paper_1_pdf, tmp_path,
        initial_conditions={"S": .9, "E": .5, "T": .9, "C": 0.0},
        time_end=.2, step=.1, figure_id="figure_11",
    )
    assert report["status"] == "BLOCKED"
    assert any("NEEDS_REVIEW" in warning for warning in report["warnings"])
    assert not report["experiments"]


def test_paper_1_user_supplied_sweep_values_are_tagged_and_run(tmp_path: Path, paper_1_pdf: Path):
    report = reproduce_paper(
        paper_1_pdf, tmp_path,
        initial_conditions={"S": .9, "E": .5, "T": .9, "C": 0.0},
        time_end=.1, step=.1, figure_id="figure_11",
        figure_parameter_values={"ks": [.8, 1.0]},
    )
    assert report["status"] == "PARTIAL"
    assert report["selected_figure_values_provenance"] == "USER_PROVIDED"
    assert report["configuration_provenance"]["initial_conditions"] == "USER_PROVIDED"
    assert [item["parameter_overrides"]["ks"] for item in report["experiments"]] == [.8, 1.0]
    assert all(item["status"] == "UNVERIFIED" for item in report["experiments"])


def test_paper_2_analysis_detects_fractional_and_pinn(paper_2_pdf: Path):
    analysis = parse_paper(paper_2_pdf)
    assert analysis.page_count == 23
    assert analysis.neural_network_methodology.status.value in {"PARTIAL", "UNCERTAIN"}
    assert any("controlled" in item.lower() for item in analysis.unresolved_information)


def test_paper2_without_explicit_numerical_configuration_is_blocked(tmp_path: Path, paper_2_pdf: Path):
    report = reproduce_paper(paper_2_pdf, tmp_path)

    assert report["status"] == "PARTIAL"
    assert report["simulation_status"] == "BLOCKED"
    assert any("time.end" in item and "time.step" in item and "solver type" in item for item in report["unresolved_information"])
    assert not (tmp_path / "paper_2" / "simulation_data" / "baseline.csv").exists()


def test_paper2_invalid_coarse_config_is_diagnostic_not_a_figure(tmp_path: Path, paper_2_pdf: Path):
    config = Path(__file__).parents[1] / "configs" / "paper_2.yaml"
    report = reproduce_paper(paper_2_pdf, tmp_path, config_path=config)

    assert report["status"] == "PARTIAL"
    assert report["simulation_status"] == "INVALID"
    assert report["validation"]["status"] == "INVALID"
    assert report["diagnostics"]["completed_steps"] == 17
    assert report["numerical_failure_diagnostics"]["failure_index"] == 18
    assert report["baseline_figures"] == []
    assert not (tmp_path / "paper_2" / "figures" / "partial_baseline.png").exists()


def test_paper2_conflicting_fractional_orders_are_blocked(tmp_path: Path, paper_2_pdf: Path):
    config = tmp_path / "paper2_conflicting_order.yaml"
    config.write_text(
        """
model:
  fractional_order: 0.8
solver:
  type: grunwald_letnikov
  options:
    fractional_order: 0.9
time:
  start: 0
  end: 0.2
  step: 0.1
""",
        encoding="utf-8",
    )

    report = reproduce_paper(paper_2_pdf, tmp_path / "conflict", config_path=config)

    assert report["simulation_status"] == "BLOCKED"
    assert any("conflicts with solver.options.fractional_order" in item for item in report["unresolved_information"])


def test_paper2_invalid_baseline_blocks_pinn_training(tmp_path: Path, paper_2_pdf: Path, monkeypatch):
    config = tmp_path / "paper2_invalid_pinn.yaml"
    config.write_text(
        """
model:
  fractional_order: 0.9
solver:
  type: grunwald_letnikov
  options:
    fractional_order: 0.9
time:
  start: 0
  end: 200
  step: 1
pinn:
  enabled: true
  hidden_layers: [2]
  activation: tanh
  optimizer: Adam
  learning_rate: 0.001
  epochs: 1
  collocation_points: 3
  data_points: 3
  physics_loss_weight: 1.0
  initial_condition_loss_weight: 1.0
""",
        encoding="utf-8",
    )

    def training_must_not_start(*_args, **_kwargs):
        raise AssertionError("PINN training must not consume an invalid reference trajectory")

    monkeypatch.setattr("research_plotter.pipelines.paper_reproduction.run_pinn_export", training_must_not_start)
    report = reproduce_paper(paper_2_pdf, tmp_path / "invalid-pinn", config_path=config)

    assert report["simulation_status"] == "INVALID"
    assert report["pinn"]["status"] == "BLOCKED"
    assert "pinn_noise" not in report


def test_paper_configs_are_valid_and_keep_review_status():
    paper_1 = load_yaml("configs/paper_1.yaml")
    paper_2 = load_experiment("configs/paper_2.yaml")
    assert paper_1["status"] == "BLOCKED"
    assert paper_2.model["controls"]["equations_status"] == "BLOCKED"


def test_user_paper2_configuration_runs_controlled_model_when_complete(tmp_path: Path, paper_2_pdf: Path):
    config = tmp_path / "paper2_user.yaml"
    config.write_text(
        """
model:
  fractional_order: 0.8
solver:
  type: grunwald_letnikov
  options:
    fractional_order: 0.8
time:
  start: 0
  end: 1
  step: 0.1
controls:
  variables: [z1, z2]
  equations:
    FN: Pi1N + rho*Tmu - alpha*FN*Abeta - phi1*FN
    IN: alpha*FN*Abeta - beta1*IN*Mrho - (gamma+phi2)*IN
    Abeta: gamma*IN - beta2*Abeta*Mrho - (db+kappa)*Abeta
    Tmu: kappa*Abeta - beta3*Tmu*Mrho - (dmu+rho)*Tmu
    Mrho: (beta1*IN+beta2*Abeta+beta3*Tmu)*Mrho - phi3*Mrho
  status: USER_PROVIDED
  provenance: USER_PROVIDED
provenance:
  controls: USER_PROVIDED
""",
        encoding="utf-8",
    )
    loaded = load_paper2_config(config)
    assert loaded.controls.is_complete
    report = reproduce_paper(paper_2_pdf, tmp_path / "output", config_path=config)
    assert any(item["name"] == "controlled_gl" for item in report["experiments"])
    assert any("NSFD" in item for item in report["warnings"])
    assert (tmp_path / "output" / "paper_2" / "configuration.yaml").exists()


def test_user_control_values_are_required_and_recorded(tmp_path: Path, paper_2_pdf: Path):
    config = tmp_path / "paper2_controls.yaml"
    config.write_text(
        """
controls:
  variables: [z1, z2]
  equations:
    FN: Pi1N + rho*Tmu - alpha*FN*Abeta - phi1*FN + z1
    IN: alpha*FN*Abeta - beta1*IN*Mrho - (gamma+phi2)*IN
    Abeta: gamma*IN - beta2*Abeta*Mrho - (db+kappa)*Abeta
    Tmu: kappa*Abeta - beta3*Tmu*Mrho - (dmu+rho)*Tmu
    Mrho: (beta1*IN+beta2*Abeta+beta3*Tmu)*Mrho - phi3*Mrho
  status: USER_PROVIDED
  provenance: USER_PROVIDED
provenance:
  controls: USER_PROVIDED
time:
  start: 0
  end: 0.2
  step: 0.1
solver:
  type: grunwald_letnikov
  options:
    fractional_order: 0.8
""",
        encoding="utf-8",
    )
    blocked = reproduce_paper(paper_2_pdf, tmp_path / "blocked", config_path=config)
    assert any("control variables" in warning for warning in blocked["warnings"])

    config.write_text(config.read_text(encoding="utf-8").replace("  status: USER_PROVIDED\n  provenance: USER_PROVIDED", "  values: {z1: 0.0, z2: 0.0}\n  status: USER_PROVIDED\n  provenance: USER_PROVIDED"), encoding="utf-8")
    completed = reproduce_paper(paper_2_pdf, tmp_path / "completed", config_path=config)
    assert any(item["name"] == "controlled_gl" for item in completed["experiments"])
    assert completed["experiments"][0]["control_values"] == {"z1": 0.0, "z2": 0.0}
    assert completed["configuration_provenance"]["controls"] == "USER_PROVIDED"
