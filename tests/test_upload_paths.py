from dataclasses import dataclass

import pytest

from research_plotter.utils.paths import output_path, persist_uploaded_file


@dataclass
class Upload:
    name: str
    content: bytes

    def getbuffer(self):
        return self.content


def test_upload_storage_is_content_addressed_and_does_not_overwrite(tmp_path):
    first = persist_uploaded_file(Upload("results.csv", b"x,y\n1,2\n"), "datasets", output_root=tmp_path)
    second = persist_uploaded_file(Upload("results.csv", b"x,y\n3,4\n"), "datasets", output_root=tmp_path)
    assert first != second
    assert first.read_bytes() == b"x,y\n1,2\n"
    assert second.read_bytes() == b"x,y\n3,4\n"
    assert persist_uploaded_file(Upload("results.csv", b"x,y\n1,2\n"), "datasets", output_root=tmp_path) == first


def test_upload_storage_rejects_directory_traversal_category(tmp_path):
    with pytest.raises(ValueError, match="safe directory"):
        persist_uploaded_file(Upload("data.csv", b"a"), "../datasets", output_root=tmp_path)


def test_output_path_stays_within_outputs_root(tmp_path):
    safe = output_path("reports", "summary.json", root=tmp_path)
    assert safe == (tmp_path / "outputs" / "reports" / "summary.json").resolve()

    with pytest.raises(ValueError, match="inside the outputs directory"):
        output_path("..", "outside.txt", root=tmp_path)

    with pytest.raises(ValueError, match="inside the outputs directory"):
        output_path(str(tmp_path / "outside.txt"), root=tmp_path)
