"""Validated, serializable experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from enum import Enum
import re

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TimeConfig(BaseModel):
    start: float = 0.0
    end: float = 10.0
    step: float = 0.1

    @field_validator("end")
    @classmethod
    def end_after_start(cls, value: float, info):
        if value <= info.data.get("start", 0.0):
            raise ValueError("time.end must be greater than time.start")
        return value

    @field_validator("step")
    @classmethod
    def positive_step(cls, value: float):
        if value <= 0:
            raise ValueError("time.step must be positive")
        return value


class SolverConfig(BaseModel):
    type: str = "ode"
    options: dict[str, Any] = Field(default_factory=dict)


class PlotConfig(BaseModel):
    type: str = "line"
    x: str = "time"
    y: str | list[str] | None = None
    title: str | None = None
    formats: list[str] = Field(default_factory=lambda: ["png"])
    dpi: int = 180


class ExperimentConfig(BaseModel):
    """Configuration for a concrete, reproducible simulation or analysis."""

    model_config = ConfigDict(extra="allow")
    model: dict[str, Any] = Field(default_factory=dict)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    time: TimeConfig = Field(default_factory=TimeConfig)
    parameters: dict[str, float] = Field(default_factory=dict)
    initial_conditions: dict[str, float] = Field(default_factory=dict)
    plots: list[PlotConfig] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Provenance(str, Enum):
    """Origin of a value used in a reproducibility configuration."""

    SOURCE_PAPER = "SOURCE_PAPER"
    USER_PROVIDED = "USER_PROVIDED"
    ASSUMED = "ASSUMED"
    UNKNOWN = "UNKNOWN"


class ControlConfig(BaseModel):
    variables: list[str] = Field(default_factory=lambda: ["z1", "z2"])
    equations: dict[str, str] = Field(default_factory=dict)
    # Numeric control values are deliberately explicit.  The source paper does
    # not provide enough information to infer control trajectories, so a
    # configured controlled run may only use values supplied by the user.
    values: dict[str, float] = Field(default_factory=dict)
    status: str = "BLOCKED"
    provenance: Provenance = Provenance.UNKNOWN
    notes: list[str] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return bool(self.equations) and self.provenance == Provenance.USER_PROVIDED

    def missing_values(self, expressions: dict[str, str] | None = None) -> list[str]:
        """Return control symbols referenced by equations without a value."""
        if expressions is None:
            expressions = self.equations
        referenced: set[str] = set()
        for expression in expressions.values():
            # This is only a preflight check; the restricted symbolic parser
            # remains the authoritative validator for actual execution.
            referenced.update(token for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(expression)) if token in self.variables)
        return sorted(referenced - set(self.values))


class NSFDConfig(BaseModel):
    enabled: bool = False
    scheme: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    status: str = "UNKNOWN"
    provenance: Provenance = Provenance.UNKNOWN
    notes: list[str] = Field(default_factory=list)


class PINNConfig(BaseModel):
    enabled: bool = False
    hidden_layers: list[int] | None = None
    num_hidden_layers: int | None = None
    hidden_units: int | None = None
    activation: str | None = None
    optimizer: str | None = None
    learning_rate: float | None = None
    epochs: int | None = None
    collocation_points: int | None = None
    data_points: int | None = None
    physics_loss_weight: float | None = None
    initial_condition_loss_weight: float | None = None
    device: str = "cpu"
    network: dict[str, Any] | None = None
    provenance: Provenance = Provenance.UNKNOWN
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def flatten_network_block(cls, data):
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        network = payload.get("network")
        if isinstance(network, dict):
            for key in ("hidden_layers", "num_hidden_layers", "hidden_units", "activation"):
                if key not in payload and key in network:
                    payload[key] = network[key]
        # Accept the compact ``hidden_layers: 3`` form as a layer count when
        # paired with ``hidden_units``; a list remains the explicit per-layer
        # architecture.
        if isinstance(payload.get("hidden_layers"), int):
            payload.setdefault("num_hidden_layers", payload["hidden_layers"])
            payload["hidden_layers"] = None
        return payload

    @field_validator("learning_rate")
    @classmethod
    def positive_learning_rate(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("PINN learning_rate must be positive")
        return value

    @field_validator("epochs", "collocation_points", "data_points", "num_hidden_layers", "hidden_units")
    @classmethod
    def positive_counts(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("PINN counts must be positive")
        return value

    def missing_fields(self) -> list[str]:
        if not self.enabled:
            return []
        required = {
            "activation": self.activation,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "collocation_points": self.collocation_points,
            "data_points": self.data_points,
            "physics_loss_weight": self.physics_loss_weight,
            "initial_condition_loss_weight": self.initial_condition_loss_weight,
        }
        missing = [name for name, value in required.items() if value is None]
        if self.hidden_layers is None and self.hidden_units is None:
            missing.append("hidden_layers or hidden_units")
        return missing


class Paper2Config(BaseModel):
    """User-overrides for Paper 2 without fabricating omitted paper details."""

    model: dict[str, Any] = Field(default_factory=dict)
    solver: SolverConfig | None = None
    time: TimeConfig | None = None
    parameters: dict[str, float] = Field(default_factory=dict)
    initial_conditions: dict[str, float] = Field(default_factory=dict)
    controls: ControlConfig = Field(default_factory=ControlConfig)
    nsfd: NSFDConfig = Field(default_factory=NSFDConfig)
    pinn: PINNConfig = Field(default_factory=PINNConfig)
    noise_levels: list[float] = Field(default_factory=lambda: [0.0, .05, .10, .15, .20])
    run_convergence: bool = False
    refinements: list[int] = Field(default_factory=lambda: [1, 2, 4])
    provenance: dict[str, Provenance] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")

    @field_validator("noise_levels")
    @classmethod
    def valid_noise_levels(cls, values: list[float]) -> list[float]:
        if any(value < 0 for value in values):
            raise ValueError("noise_levels must be non-negative")
        return values

    @field_validator("refinements")
    @classmethod
    def valid_refinements(cls, values: list[int]) -> list[int]:
        if not values or any(value < 1 for value in values):
            raise ValueError("refinements must contain positive integers")
        return values

    def missing_for_pinn(self) -> list[str]:
        return self.pinn.missing_fields()

    def resolved_provenance(self) -> dict[str, str]:
        """Return explicit provenance without guessing the source of values.

        A YAML file cannot tell us whether an unannotated value was copied from
        a paper, chosen by a user, or assumed during an exploratory run.  Such
        entries therefore remain ``UNKNOWN`` until the author marks them.
        """
        explicit = {key: value.value if isinstance(value, Provenance) else str(value) for key, value in self.provenance.items()}
        fields = ("model", "parameters", "initial_conditions", "time", "solver",
                  "fractional_order", "controls", "nsfd", "pinn")
        return {field: explicit.get(field, Provenance.UNKNOWN.value) for field in fields}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML only as data; no custom tags or executable expressions are accepted."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("YAML configuration must contain a mapping at its root")
    return payload


def load_experiment(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))


def load_paper2_config(path: str | Path) -> Paper2Config:
    """Load a Paper 2 override config and normalize legacy nested fields.

    Older project configs stored ``controls`` and status fields under
    ``model``.  They remain accepted, but the structured top-level fields are
    preferred for new user-provided information.
    """
    payload = load_yaml(path)
    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    if "controls" not in payload and isinstance(model.get("controls"), dict):
        payload["controls"] = model["controls"]
    if "nsfd" not in payload and "nsfd_status" in model:
        payload["nsfd"] = {"status": model["nsfd_status"]}
    if "pinn" not in payload and "training_hyperparameters_status" in model:
        payload["pinn"] = {"notes": [f"Source status: {model['training_hyperparameters_status']}"]}
    return Paper2Config.model_validate(payload)


def dump_yaml(payload: BaseModel | dict[str, Any], path: str | Path) -> Path:
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return destination
