from __future__ import annotations

from typing import Any


def json_safe(value: Any) -> Any:
    """Convert common numerical values recursively to JSON-friendly values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
