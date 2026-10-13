from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    PARTIAL = "PARTIAL"
    UNCERTAIN = "UNCERTAIN"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED = "BLOCKED"


ReproducibilityStatus = EvidenceStatus


class FigureManifest(BaseModel):
    figure_id: str
    source_document: str
    source_page: int | None = None
    caption: str = ""
    plot_type: str | None = None
    x_variable: str | None = None
    y_variable: str | None = None
    variables: list[str] = Field(default_factory=list)
    fixed_parameters: dict[str, Any] = Field(default_factory=dict)
    varying_parameters: dict[str, list[Any]] = Field(default_factory=dict)
    parameter_values: dict[str, Any] = Field(default_factory=dict)
    time_range: tuple[float, float] | None = None
    solver: str | None = None
    inset_zoom: str | None = None
    required_equation_ids: list[str] = Field(default_factory=list)
    equation_link_status: EvidenceStatus = EvidenceStatus.NEEDS_REVIEW
    required_parameters: list[str] = Field(default_factory=list)
    required_initial_conditions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    confidence: EvidenceStatus = EvidenceStatus.PARTIAL
    reproducibility_status: EvidenceStatus = EvidenceStatus.NEEDS_REVIEW
