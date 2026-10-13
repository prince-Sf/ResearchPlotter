"""Conservative loader for clearly delimited plain-text tables."""

from __future__ import annotations

import csv
from pathlib import Path
import re

import pandas as pd


def load_delimited_text(path: str | Path) -> pd.DataFrame:
    """Load a TXT file only when a consistent multi-column table is evident.

    Common comma/tab/semicolon/pipe delimiters are sniffed first. Whitespace
    tables are accepted only when multiple non-empty lines have consistent
    widths greater than one. Plain prose is rejected instead of guessed into
    a dataframe.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    sample = text[:8192]
    delimiter: str | None = None
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        rows = [line.split() for line in sample.splitlines() if line.strip()]
        widths = {len(row) for row in rows}
        numeric_cell = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
        has_numeric_data = any(numeric_cell.fullmatch(value) for row in rows[1:] for value in row)
        if len(rows) >= 2 and len(widths) == 1 and next(iter(widths), 0) > 1 and has_numeric_data:
            delimiter = r"\s+"
    if delimiter is None:
        raise ValueError("TXT input does not contain a recognizable, consistent table delimiter; use a structured dataset format instead.")
    frame = pd.read_csv(source, sep=delimiter, engine="python")
    if frame.shape[1] < 2 or frame.empty:
        raise ValueError("TXT input must contain a non-empty table with at least two columns")
    return frame
