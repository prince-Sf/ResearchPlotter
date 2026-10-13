from __future__ import annotations

from research_plotter.paper_analysis.figure_manifest import FigureManifest
from research_plotter.paper_analysis.paper_parser import PaperAnalysis


def detect_figures(analysis: PaperAnalysis) -> list[FigureManifest]:
    return analysis.figures
