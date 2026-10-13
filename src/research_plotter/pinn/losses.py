from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def data_loss(prediction, observation):
    if torch is None: raise ImportError("PINN losses require PyTorch")
    return torch.mean((prediction - observation) ** 2)


def physics_loss(residual):
    if torch is None: raise ImportError("PINN losses require PyTorch")
    return torch.mean(residual ** 2)


def combined_loss(data_term, physics_term, physics_weight: float = 1.0, initial_term=None, initial_weight: float = 1.0):
    if torch is None: raise ImportError("PINN losses require PyTorch")
    loss = data_term + physics_weight * physics_term
    if initial_term is not None: loss = loss + initial_weight * initial_term
    return loss
