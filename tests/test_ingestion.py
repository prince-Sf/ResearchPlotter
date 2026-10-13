from pathlib import Path

import pandas as pd
import pytest

from research_plotter.ingestion.dataset_profiler import profile_dataset
from research_plotter.pipelines.dataset_analysis import load_dataset
from research_plotter.ingestion.json_loader import load_json


def test_csv_loader_and_profile():
    frame = load_dataset(Path("data/examples/measurements.csv"))
    profile = profile_dataset(frame)
    assert profile.rows == 10
    assert "signal" in profile.numeric_columns
    assert "time" in profile.time_like_columns
    assert profile.categorical_columns == ["group"]


def test_profiler_handles_no_time():
    profile = profile_dataset(pd.DataFrame({"x": [1, 2, 3], "y": [3, 2, 1], "label": ["a", "b", "a"]}))
    assert profile.time_like_columns == []
    assert any("scatter" in message for message in profile.recommendations)


def test_profiler_recognizes_common_symbolic_time_column():
    profile = profile_dataset(pd.DataFrame({"t": [0.0, 0.5, 1.0], "value": [1.0, 1.5, 1.2]}))
    assert profile.time_like_columns == ["t"]


def test_json_loader_normalizes_nested_records(tmp_path: Path):
    source = tmp_path / "records.json"
    source.write_text('{"records": [{"x": 1, "meta": {"label": "a"}}, {"x": 2, "meta": {"label": "b"}}]}', encoding="utf-8")
    frame = load_json(source)
    assert list(frame["records.x"]) == [1, 2]


@pytest.mark.parametrize("content", ["time\tsignal\n0\t1.2\n1\t1.4\n", "time,signal\n0,1.2\n1,1.4\n", "time signal\n0 1.2\n1 1.4\n"])
def test_txt_loader_accepts_clearly_delimited_tables(tmp_path: Path, content: str):
    source = tmp_path / "table.txt"
    source.write_text(content, encoding="utf-8")
    frame = load_dataset(source)
    assert frame.shape == (2, 2)
    assert list(frame.columns) == ["time", "signal"]


def test_txt_loader_rejects_plain_prose_instead_of_guessing_columns(tmp_path: Path):
    source = tmp_path / "notes.txt"
    source.write_text("These are ordinary research notes.\nNo table is present here.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="recognizable, consistent table"):
        load_dataset(source)
