from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator


class Parameter(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    value: float
    description: str = ""
    unit: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Parameter value must be finite")
        return value

    def validate_value(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{self.name} must be finite")
        if self.lower_bound is not None and value < self.lower_bound:
            raise ValueError(f"{self.name} must be >= {self.lower_bound}")
        if self.upper_bound is not None and value > self.upper_bound:
            raise ValueError(f"{self.name} must be <= {self.upper_bound}")
        return value
