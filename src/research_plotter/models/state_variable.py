from __future__ import annotations

from pydantic import BaseModel, Field


class StateVariable(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    description: str = ""
    unit: str | None = None
    nonnegative: bool = False
    initial_value: float | None = None
