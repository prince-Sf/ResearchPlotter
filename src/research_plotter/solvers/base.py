from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from research_plotter.models.model_spec import ModelSpec


@dataclass
class SimulationResult:
    time: np.ndarray
    states: np.ndarray
    state_names: list[str]
    parameters: dict[str, float]
    initial_conditions: dict[str, float]
    solver_name: str
    runtime_seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.states = np.asarray(self.states, dtype=float)
        if self.time.ndim != 1 or len(self.time) == 0:
            raise ValueError("time must be a non-empty one-dimensional array")
        if self.states.ndim != 2 or self.states.shape[0] != len(self.time) or self.states.shape[1] != len(self.state_names):
            raise ValueError("states must have shape (len(time), len(state_names))")
        if len(self.time) > 1 and np.any(np.diff(self.time) <= 0):
            raise ValueError("time must be strictly increasing")
        if not np.isfinite(self.time).all() or not np.isfinite(self.states).all():
            if "Simulation contains NaN or infinite values" not in self.warnings:
                self.warnings.append("Simulation contains NaN or infinite values")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.states, columns=self.state_names).assign(time=self.time)[["time", *self.state_names]]

    def save_csv(self, path: str) -> None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)

    def to_metadata(self) -> dict[str, Any]:
        return {"solver_name": self.solver_name, "runtime_seconds": self.runtime_seconds, "parameters": self.parameters,
                "initial_conditions": self.initial_conditions, "state_names": self.state_names,
                "time_grid": {"start": float(self.time[0]), "end": float(self.time[-1]), "count": len(self.time), "step": float(self.time[1] - self.time[0]) if len(self.time) > 1 else None},
                "diagnostics": self.diagnostics, "warnings": self.warnings, "metadata": self.metadata}


class Solver(ABC):
    name = "base"

    @abstractmethod
    def solve(
        self,
        model: ModelSpec,
        parameters: Mapping[str, float] | None,
        initial_conditions: Mapping[str, float] | None,
        time_grid: np.ndarray,
        options: Mapping[str, Any] | None = None,
    ) -> SimulationResult:
        raise NotImplementedError

    @staticmethod
    def start_timer() -> float:
        return perf_counter()
