from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from research_plotter.pinn.fractional_residual import gl_fractional_derivative
from research_plotter.pinn.losses import combined_loss, data_loss, physics_loss


@dataclass
class TrainingResult:
    history: list[dict[str, float]]
    runtime_seconds: float
    model: object
    warnings: list[str] = field(default_factory=list)

    def predict(self, time):
        if torch is None:
            raise ImportError("PINN prediction requires PyTorch")
        network = self.model
        device = next(network.parameters()).device
        values = time.to(device) if hasattr(time, "to") else torch.as_tensor(time, dtype=torch.float32, device=device)
        if values.ndim == 1:
            values = values[:, None]
        was_training = network.training
        network.eval()
        with torch.no_grad():
            prediction = network(values)
        if was_training:
            network.train()
        return prediction

    def save(self, path: str | Path, *, configuration: dict[str, Any] | None = None) -> Path:
        if torch is None:
            raise ImportError("Saving PINN checkpoints requires PyTorch")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "history": self.history,
                    "runtime_seconds": self.runtime_seconds, "warnings": self.warnings,
                    "configuration": configuration or {}}, destination)
        return destination

    @classmethod
    def load(cls, network, path: str | Path, *, map_location: str = "cpu") -> "TrainingResult":
        if torch is None:
            raise ImportError("Loading PINN checkpoints requires PyTorch")
        payload = torch.load(path, map_location=map_location, weights_only=False)
        network.load_state_dict(payload["state_dict"])
        return cls(payload.get("history", []), float(payload.get("runtime_seconds", 0.0)), network, payload.get("warnings", []))


class PINNTrainer:
    """Small explicit trainer for supervised, integer, or GL-fractional PINNs."""

    def __init__(self, network, *, learning_rate: float = 1e-3, physics_weight: float = 1.0, optimizer: str = "Adam", device: str = "cpu"):
        if torch is None: raise ImportError("PINN training requires PyTorch")
        if learning_rate <= 0 or physics_weight < 0:
            raise ValueError("learning_rate must be positive and physics_weight non-negative")
        self.network = network.to(device)
        self.physics_weight = physics_weight
        self.device = device
        optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
        optimizer_type = optimizers.get(str(optimizer).lower())
        if optimizer_type is None:
            raise ValueError(f"Unsupported PINN optimizer: {optimizer}")
        self.optimizer = optimizer_type(self.network.parameters(), lr=learning_rate)

    def train(self, time, observations, *, rhs: Callable | None = None, order: float = 1.0, epochs: int = 1000, initial_weight: float = 1.0, progress_callback: Callable[[dict[str, float]], None] | None = None) -> TrainingResult:
        if torch is None: raise ImportError("PINN training requires PyTorch")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if not 0 < order <= 1:
            raise ValueError("order must be in (0, 1]")
        time = time.to(self.device) if hasattr(time, "to") else torch.as_tensor(time, dtype=torch.float32, device=self.device)
        observations = observations.to(self.device) if hasattr(observations, "to") else torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if time.ndim not in {1, 2} or time.shape[0] < 2:
            raise ValueError("time must contain at least two samples")
        if not torch.isfinite(time).all() or not torch.isfinite(observations).all():
            raise ValueError("time and observations must be finite")
        if time.ndim == 1: time_input = time[:, None]
        else: time_input = time
        if order < 1:
            differences = time_input[1:, 0] - time_input[:-1, 0]
            if not torch.all(differences > 0) or not torch.allclose(differences, differences[0].expand_as(differences)):
                raise ValueError("fractional PINN training requires an increasing uniform time grid")
        if order >= 1:
            time_input = time_input.detach().clone().requires_grad_(True)
        if observations.ndim == 1: observations = observations[:, None]
        if observations.shape[0] != time_input.shape[0]:
            raise ValueError("time and observations must have the same number of samples")
        history: list[dict[str, float]] = []
        started = perf_counter()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            prediction = self.network(time_input)
            dloss = data_loss(prediction, observations)
            if rhs is None:
                residual = torch.zeros_like(prediction)
            elif order < 1:
                derivative = gl_fractional_derivative(prediction, float(time_input[1, 0] - time_input[0, 0]), order)
                residual = derivative - rhs(time_input, prediction)
            else:
                # Differentiate each state output separately.  A single
                # autograd call with ``ones_like(prediction)`` would return
                # the derivative of the sum of all state outputs.
                derivative_columns = []
                for column in range(prediction.shape[1]):
                    gradient = torch.autograd.grad(
                        prediction[:, column].sum(),
                        time_input,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    derivative_columns.append(gradient if gradient is not None else torch.zeros_like(time_input))
                derivative = torch.cat(derivative_columns, dim=1)
                residual = derivative
                residual = residual - rhs(time_input, prediction)
            ploss = physics_loss(residual)
            initial = data_loss(prediction[:1], observations[:1])
            loss = combined_loss(dloss, ploss, self.physics_weight, initial, initial_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"PINN loss became non-finite at epoch {epoch + 1}")
            loss.backward(); self.optimizer.step()
            if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 10) == 0:
                metrics = {"epoch": float(epoch + 1), "loss": float(loss.detach().cpu()), "data_loss": float(dloss.detach().cpu()), "physics_loss": float(ploss.detach().cpu()), "initial_loss": float(initial.detach().cpu())}
                history.append(metrics)
                if progress_callback is not None:
                    progress_callback(dict(metrics))
        return TrainingResult(history, perf_counter() - started, self.network, ["Fractional derivatives were evaluated by GL convolution; automatic differentiation was used only for ordinary-order residuals."] if order < 1 else [])

    def predict(self, time):
        result = TrainingResult([], 0.0, self.network)
        return result.predict(time)

    def save_checkpoint(self, path: str | Path, *, configuration: dict[str, Any] | None = None) -> Path:
        return TrainingResult([], 0.0, self.network).save(path, configuration=configuration)
