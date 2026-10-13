from __future__ import annotations

import json
from pathlib import Path

from research_plotter.paper_analysis.paper_parser import PaperAnalysis


def save_analysis_report(analysis: PaperAnalysis, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(analysis.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    return destination
