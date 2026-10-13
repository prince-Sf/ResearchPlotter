from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research_plotter.plotting.core import save_figure
from research_plotter.plotting.heatmap import heatmap
from research_plotter.plotting.line import multi_line_plot
from research_plotter.plotting.scatter import scatter_plot


def recommend_dataset_figures(frame: pd.DataFrame, profile, output_dir: str | Path = "outputs/figures") -> list[Path]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    numeric = profile.numeric_columns
    if profile.time_like_columns and numeric:
        time_name = profile.time_like_columns[0]
        figure = multi_line_plot(frame[time_name], frame[numeric], xlabel=time_name, ylabel="value")
        created.extend(save_figure(figure, output / "dataset_timeseries.png", formats=["png", "svg"]))
        (output / "dataset_timeseries.json").write_text(json.dumps({
            "plot_type": "multi_line",
            "source_columns": {"x": time_name, "y": numeric},
            "formats": ["png", "svg"],
            "profile_rows": profile.rows,
        }, indent=2), encoding="utf-8")
    elif len(numeric) >= 2:
        created.append(save_figure(scatter_plot(frame[numeric[0]], frame[numeric[1]], xlabel=numeric[0], ylabel=numeric[1]), output / "dataset_scatter.png")[0])
        created.extend(save_figure(heatmap(frame[numeric].corr(), xlabels=numeric, ylabels=numeric, title="Numeric correlation"), output / "dataset_correlation.png", formats=["png", "svg"]))
        (output / "dataset_scatter.json").write_text(json.dumps({
            "plot_type": "scatter",
            "source_columns": {"x": numeric[0], "y": numeric[1]},
            "formats": ["png"],
            "profile_rows": profile.rows,
        }, indent=2), encoding="utf-8")
        (output / "dataset_correlation.json").write_text(json.dumps({
            "plot_type": "heatmap",
            "source_columns": numeric,
            "formats": ["png", "svg"],
            "profile_rows": profile.rows,
        }, indent=2), encoding="utf-8")
    return [Path(item) for item in created]
