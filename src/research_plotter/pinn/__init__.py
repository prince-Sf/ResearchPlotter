"""Optional PyTorch PINN components.

The package imports without PyTorch so dataset, paper-analysis, and classical
solver workflows remain usable in lightweight environments.
"""

from research_plotter.pinn.fractional_residual import gl_fractional_derivative
from research_plotter.pinn.losses import data_loss, physics_loss
from research_plotter.pinn.network import StateNetwork
from research_plotter.pinn.trainer import PINNTrainer, TrainingResult
from research_plotter.pinn.experiments import PINNExperimentResult, compare_pinn_modes, run_noise_experiments, run_pinn_experiment
from research_plotter.pinn.torch_rhs import torch_rhs_from_model

__all__ = ["PINNExperimentResult", "PINNTrainer", "StateNetwork", "TrainingResult", "compare_pinn_modes", "data_loss", "gl_fractional_derivative", "physics_loss", "run_noise_experiments", "run_pinn_experiment", "torch_rhs_from_model"]
