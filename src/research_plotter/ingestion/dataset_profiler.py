"""Dataset profiling stays data-centric and makes no unsupported time-series claim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    kind: str
    missing_count: int
    unique_count: int
    constant: bool
    low_cardinality: bool
    numeric_range: tuple[float, float] | None = None
    statistics: dict[str, float] = Field(default_factory=dict)
    outlier_count_iqr: int | None = None


class DatasetProfile(BaseModel):
    rows: int
    columns: int
    duplicate_rows: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    time_like_columns: list[str]
    constant_columns: list[str]
    columns_detail: list[ColumnProfile]
    correlations: dict[str, dict[str, float]] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


def _datetime_candidate(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series):
        sample = series.dropna().astype(str).head(50)
        if sample.empty:
            return False
        converted = pd.to_datetime(sample, errors="coerce")
        return bool(converted.notna().mean() >= 0.8)
    return False


def _time_like(name: str, series: pd.Series, is_datetime: bool) -> bool:
    token = name.lower().replace("_", " ")
    semantic = token.strip() in {"t", "time", "date", "datetime", "timestamp", "year", "day", "month"} or any(word in token for word in ("time", "date", "timestamp", "year", "day", "month"))
    if is_datetime:
        return True
    if semantic and pd.api.types.is_numeric_dtype(series):
        # Experimental datasets often repeat a time grid for treatment groups;
        # semantic names plus a nontrivial numeric domain are enough to flag a
        # candidate without claiming it is a valid time series.
        return bool(series.dropna().nunique() >= 3)
    return False


def _profile_column(name: str, series: pd.Series, row_count: int) -> ColumnProfile:
    is_datetime = _datetime_candidate(series)
    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    unique = int(series.nunique(dropna=True))
    constant = unique <= 1
    low_cardinality = unique > 0 and unique <= min(20, max(2, row_count // 10))
    kind = "datetime" if is_datetime else "numeric" if numeric else "categorical"
    profile = ColumnProfile(name=name, dtype=str(series.dtype), kind=kind, missing_count=int(series.isna().sum()), unique_count=unique,
                            constant=constant, low_cardinality=low_cardinality)
    if numeric:
        valid = series.dropna().astype(float)
        if not valid.empty:
            first, third = valid.quantile([0.25, 0.75])
            iqr = third - first
            profile.numeric_range = (float(valid.min()), float(valid.max()))
            profile.statistics = {"mean": float(valid.mean()), "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
                                  "median": float(valid.median()), "q1": float(first), "q3": float(third)}
            profile.outlier_count_iqr = int(((valid < first - 1.5 * iqr) | (valid > third + 1.5 * iqr)).sum()) if iqr > 0 else 0
    return profile


def _recommendations(frame: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    suggestions: list[str] = []
    if profile.time_like_columns and profile.numeric_columns:
        suggestions.append(f"Use line plots against '{profile.time_like_columns[0]}' for numeric trajectories.")
    elif len(profile.numeric_columns) >= 2:
        suggestions.append("Use scatter plots and a correlation heatmap for the numeric columns; no time variable was detected.")
    elif profile.numeric_columns:
        suggestions.append("Use a histogram or box plot for the numeric measurement column.")
    if profile.categorical_columns and profile.numeric_columns:
        suggestions.append("Use grouped box plots to compare numeric values across categorical groups.")
    if profile.duplicate_rows:
        suggestions.append(f"Review {profile.duplicate_rows} duplicated rows before inferential analysis.")
    if not suggestions:
        suggestions.append("Inspect column semantics before selecting a plot; this dataset has no obvious quantitative relationship.")
    return suggestions


def profile_dataset(frame: pd.DataFrame) -> DatasetProfile:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("profile_dataset expects a pandas DataFrame")
    detail = [_profile_column(str(name), frame[name], len(frame)) for name in frame.columns]
    numeric = [item.name for item in detail if item.kind == "numeric"]
    categorical = [item.name for item in detail if item.kind == "categorical"]
    datetime_columns = [item.name for item in detail if item.kind == "datetime"]
    time_like = [name for name in frame.columns if _time_like(str(name), frame[name], str(name) in datetime_columns)]
    correlation = frame[numeric].corr(numeric_only=True) if len(numeric) >= 2 else pd.DataFrame()
    model = DatasetProfile(rows=len(frame), columns=len(frame.columns), duplicate_rows=int(frame.duplicated().sum()),
                           numeric_columns=numeric, categorical_columns=categorical, datetime_columns=datetime_columns,
                           time_like_columns=[str(item) for item in time_like], constant_columns=[item.name for item in detail if item.constant],
                           columns_detail=detail, correlations={str(row): {str(column): float(value) for column, value in values.dropna().items()} for row, values in correlation.iterrows()})
    return model.model_copy(update={"recommendations": _recommendations(frame, model)})
