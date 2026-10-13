from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in minimal installs
    torch = None
    nn = object


if torch is not None:
    class StateNetwork(nn.Module):
        """Feed-forward time-to-state network used by baseline and PINN runs."""

        def __init__(self, input_dim: int = 1, output_dim: int = 1, hidden_layers: tuple[int, ...] = (64, 64), activation: str = "tanh"):
            super().__init__()
            activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}
            if activation not in activations:
                raise ValueError(f"Unsupported activation: {activation}")
            layers: list[nn.Module] = []
            width = input_dim
            for hidden in hidden_layers:
                layers.extend([nn.Linear(width, hidden), activations[activation]()])
                width = hidden
            layers.append(nn.Linear(width, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, time):
            if time.ndim == 1: time = time[:, None]
            return self.network(time)
else:
    class StateNetwork:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PINN support requires PyTorch. Install the project's torch dependency.")
