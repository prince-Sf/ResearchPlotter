from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_json(path: str | Path, **options) -> pd.DataFrame:
    source = Path(path)
    try:
        frame = pd.read_json(source, **options)
        # pandas can accept an object containing a list of records but leave
        # that list in a single cell. Normalize that common document shape.
        for column in frame.columns:
            values = frame[column].tolist()
            if values and all(isinstance(item, dict) for item in values):
                rows = values
                normalized = pd.json_normalize(rows)
                normalized.columns = [f"{column}.{name}" for name in normalized.columns]
                return normalized
        return frame
    except ValueError as first_error:
        # Handle ordinary nested JSON documents as records without executing
        # any content.  Callers still receive a useful error for scalar JSON.
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not parse JSON dataset {source}: {error}") from first_error
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                    normalized = pd.json_normalize(value)
                    normalized.columns = [f"{key}.{column}" for column in normalized.columns]
                    return normalized
            return pd.json_normalize(payload)
        raise ValueError("JSON dataset must contain an object or an array of records") from first_error
