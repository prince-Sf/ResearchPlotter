from pathlib import Path

import numpy as np
import pytest

from research_plotter.config import PINNConfig, Provenance


def _tiny_config() -> PINNConfig:
    return PINNConfig(
        enabled=True,
        hidden_layers=[8],
        activation="tanh",
        optimizer="Adam",
        learning_rate=.01,
        epochs=3,
        collocation_points=5,
        data_points=5,
        physics_loss_weight=1.0,
        initial_condition_loss_weight=1.0,
        provenance=Provenance.USER_PROVIDED,
    )


def test_pinn_config_reports_missing_required_values():
    config = PINNConfig(enabled=True, provenance=Provenance.USER_PROVIDED)
    assert "activation" in config.missing_fields()
    assert "hidden_layers or hidden_units" in config.missing_fields()
    nested = PINNConfig(network={"hidden_layers": [4, 4], "activation": "relu"})
    assert nested.hidden_layers == [4, 4]
    assert nested.activation == "relu"
    compact = PINNConfig(network={"hidden_layers": 3, "hidden_units": 4, "activation": "tanh"})
    assert compact.hidden_layers is None
    assert compact.num_hidden_layers == 3
    assert compact.hidden_units == 4


def test_pinn_forward_and_fractional_residual_shape():
    torch = pytest.importorskip("torch", reason="PyTorch is not installed in this runtime; PINN runtime tests remain available after installing torch.")
    from research_plotter.pinn import StateNetwork, gl_fractional_derivative

    network = StateNetwork(output_dim=5, hidden_layers=(8,), activation="tanh")
    time = torch.linspace(0, 1, 5)
    prediction = network(time)
    assert tuple(prediction.shape) == (5, 5)
    derivative = gl_fractional_derivative(prediction, .25, .8)
    assert tuple(derivative.shape) == (5, 5)
    assert torch.isfinite(derivative).all()


def test_gl_fractional_residual_uses_history_and_keeps_parameter_gradients():
    torch = pytest.importorskip("torch", reason="PyTorch is not installed in this runtime; fractional residual test requires torch.")
    from research_plotter.pinn import gl_fractional_derivative

    time = torch.linspace(0, 1, 9)
    values = (time.square() + 0.2 * time).reshape(-1, 1).requires_grad_()
    fractional = gl_fractional_derivative(values, 1 / 8, .7)
    integer_like = 2 * time.reshape(-1, 1) + 0.2
    assert fractional.shape == values.shape
    assert torch.isfinite(fractional).all()
    assert not torch.allclose(fractional[1:], integer_like[1:])
    fractional.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    constant = gl_fractional_derivative(torch.ones((9, 1)), 1 / 8, .7)
    assert torch.allclose(constant, torch.zeros_like(constant))


def test_torch_rhs_from_validated_model():
    torch = pytest.importorskip("torch", reason="PyTorch is not installed in this runtime; Torch RHS tests remain available after installing torch.")
    from research_plotter.models import ModelSpec, Parameter, StateVariable
    from research_plotter.pinn import torch_rhs_from_model

    model = ModelSpec.from_expressions(
        name="decay",
        state_variables=[StateVariable(name="x")],
        parameters=[Parameter(name="rate", value=.2)],
        expressions={"x": "-rate*x"},
        initial_conditions={"x": 1},
    )
    rhs = torch_rhs_from_model(model)
    state = torch.ones((3, 1))
    values = rhs(torch.linspace(0, 1, 3), state)
    assert tuple(values.shape) == (3, 1)
    assert torch.allclose(values, torch.full((3, 1), -.2))


def test_tiny_baseline_integer_fractional_and_checkpoint(tmp_path: Path):
    torch = pytest.importorskip("torch", reason="PyTorch is not installed in this runtime; PINN runtime tests remain available after installing torch.")
    from research_plotter.pinn import StateNetwork, run_pinn_experiment

    time = np.linspace(0, 1, 5)
    observations = np.column_stack([np.exp(-time), np.exp(-2 * time)])

    torch.manual_seed(17)
    baseline = run_pinn_experiment(time, observations, _tiny_config(), mode="baseline")
    assert baseline.prediction.shape == observations.shape
    assert baseline.final_loss is not None
    assert baseline.mode == "baseline"
    assert baseline.configuration["epochs"] == 3
    assert all(np.isfinite(row["loss"]) for row in baseline.training.history)
    checkpoint = baseline.training.save(tmp_path / "baseline.pt")
    restored = baseline.training.load(StateNetwork(output_dim=2, hidden_layers=(8,), activation="tanh"), checkpoint)
    assert restored.predict(time).shape == (5, 2)

    def rhs(t, y):
        return torch.cat((-y[:, 0:1], -2 * y[:, 1:2]), dim=1)

    torch.manual_seed(17)
    integer = run_pinn_experiment(time, observations, _tiny_config(), mode="integer", rhs=rhs)
    assert integer.metrics["rmse"] >= 0
    assert integer.training.history[0]["physics_loss"] > 0
    torch.manual_seed(17)
    fractional = run_pinn_experiment(time, observations, _tiny_config(), mode="fractional", rhs=rhs, fractional_order=.8)
    assert fractional.prediction.shape == observations.shape
    assert fractional.mode == "fractional"
    assert all(np.isfinite(row["loss"]) for row in fractional.training.history)
    assert fractional.training.history[0]["physics_loss"] > 0
    assert not np.allclose(integer.prediction, fractional.prediction)


def test_tiny_noise_experiment_records_metrics():
    torch = pytest.importorskip("torch", reason="PyTorch is not installed in this runtime; PINN noise tests remain available after installing torch.")
    from research_plotter.pinn import run_noise_experiments

    time = np.linspace(0, 1, 5)
    observations = np.column_stack([np.exp(-time)])
    original = observations.copy()
    torch.manual_seed(4)
    results = run_noise_experiments(time, observations, _tiny_config(), levels=(0.0, .1), seed=4)
    assert [item["noise_level"] for item in results] == [0.0, .1]
    assert all("final_loss" in item and "runtime_seconds" in item and "rmse" in item for item in results)
    assert np.array_equal(observations, original)
    assert np.allclose(results[0]["noisy_observations"], original)
    assert not np.allclose(results[1]["noisy_observations"], original)
    assert "training_metrics" in results[1]
