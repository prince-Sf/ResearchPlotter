from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research_plotter.ingestion.csv_loader import load_csv
from research_plotter.ingestion.dataset_profiler import DatasetProfile, profile_dataset
from research_plotter.ingestion.excel_loader import load_excel
from research_plotter.ingestion.json_loader import load_json
from research_plotter.ingestion.parquet_loader import load_parquet
from research_plotter.ingestion.text_loader import load_delimited_text


def load_dataset(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv": return load_csv(source)
    if suffix in {".xlsx", ".xls"}: return load_excel(source)
    if suffix == ".json": return load_json(source)
    if suffix == ".parquet": return load_parquet(source)
    if suffix == ".txt": return load_delimited_text(source)
    raise ValueError(f"Unsupported dataset format: {suffix}. Use CSV, TXT tables, XLSX/XLS, JSON, or Parquet.")


def analyze_dataset(path: str | Path, output_dir: str | Path = "outputs/reports") -> tuple[pd.DataFrame, DatasetProfile, Path]:
    frame = load_dataset(path)
    profile = profile_dataset(frame)
    destination = Path(output_dir) / f"{Path(path).stem}_profile.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(profile.model_dump(mode="json"), indent=2), encoding="utf-8")
    return frame, profile, destination
