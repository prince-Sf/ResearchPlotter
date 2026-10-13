"""Bundle simulation provenance without coupling it to a paper or plot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_plotter.solvers.base import SimulationResult


def save_simulation_bundle(result: SimulationResult, directory: str | Path, *, configuration: dict[str, Any] | None = None, source_document: str | None = None, plotting_configuration: Any = None) -> dict[str, str]:
    destination = Path(directory); destination.mkdir(parents=True, exist_ok=True)
    data_path = destination / "simulation.csv"; result.save_csv(str(data_path))
    metadata = result.to_metadata() | {"configuration": configuration or {}, "source_document": source_document, "plotting_configuration": plotting_configuration}
    metadata_path = destination / "experiment.json"; metadata_path.write_text(json.dumps(metadata, indent=2, default=str, allow_nan=False), encoding="utf-8")
    return {"data": str(data_path), "metadata": str(metadata_path)}
