from dataclasses import dataclass
import hashlib

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


def _directory_symlink_or_skip(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Directory symlinks are unavailable in this environment: {error}")


def test_output_path_rejects_existing_symlink_escape(tmp_path):
    outputs = tmp_path / "outputs"
    reports = outputs / "reports"
    outside = tmp_path / "outside"
    reports.mkdir(parents=True)
    outside.mkdir()
    _directory_symlink_or_skip(reports / "external", outside)

    with pytest.raises(ValueError, match="inside the outputs directory"):
        output_path("reports", "external", "summary.json", root=tmp_path)

    assert not (outside / "summary.json").exists()


def test_output_path_rejects_symlinked_outputs_root(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "external-output"
    outside.mkdir()
    _directory_symlink_or_skip(project / "outputs", outside)

    with pytest.raises(ValueError, match="outputs directory must remain"):
        output_path("reports", "summary.json", root=project)

    assert list(outside.iterdir()) == []


def test_workbench_run_directory_rejects_unsafe_workflow_component(tmp_path, monkeypatch):
    from research_plotter import app

    monkeypatch.setattr(app, "ROOT", tmp_path)
    run_id, run_dir = app._new_run_dir("simulation")
    assert run_id.startswith("simulation-")
    assert run_dir.is_dir()
    assert run_dir.is_relative_to((tmp_path / "outputs").resolve())

    with pytest.raises(ValueError, match="inside the outputs directory"):
        app._new_run_dir("../../../../escape")


def test_upload_rejects_category_symlink_escape(tmp_path):
    root = tmp_path / "store"
    (root / "uploaded").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    _directory_symlink_or_skip(root / "uploaded" / "datasets", outside)

    with pytest.raises(ValueError, match="inside the selected output directory"):
        persist_uploaded_file(Upload("data.csv", b"value\n1\n"), "datasets", output_root=root)

    assert list(outside.iterdir()) == []


def test_upload_rejects_symlink_destination_without_overwriting_target(tmp_path):
    root = tmp_path / "store"
    category = root / "uploaded" / "datasets"
    category.mkdir(parents=True)
    content = b"value\n1\n"
    filename = "data.csv"
    digest = hashlib.sha256(content).hexdigest()[:16]
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"preserve me")
    target = category / f"{digest}_{filename}"
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"File symlinks are unavailable in this environment: {error}")

    with pytest.raises(ValueError, match="symlink upload destination"):
        persist_uploaded_file(Upload(filename, content), "datasets", output_root=root)

    assert outside.read_bytes() == b"preserve me"


def test_upload_filename_traversal_is_stored_inside_upload_category(tmp_path):
    saved = persist_uploaded_file(
        Upload("../../outside.csv", b"x\n1\n"),
        "datasets",
        output_root=tmp_path,
    )

    expected = (tmp_path / "uploaded" / "datasets").resolve()
    assert saved.is_file()
    assert saved.resolve().is_relative_to(expected)
    assert saved.name.endswith("_outside.csv")
