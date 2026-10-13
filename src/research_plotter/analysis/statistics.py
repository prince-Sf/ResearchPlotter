from __future__ import annotations

import pandas as pd


def summarize_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T
