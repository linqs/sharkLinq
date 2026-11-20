"""Contracts that keep path handling independent of the checkout location and working directory."""

import os
import shutil
import sqlite3
import subprocess
import sys

import pytest

from core import app_constants
from core import app_utils
from db import database
from db.scripts import create_sample_dataset
from scripts import shark_identification


def test_portable_csv_dir_prefers_database_format_then_data_root(monkeypatch, tmp_path):
    """Sidecars name CSV directories relative to the data roots so annotations move between machines."""
    data_root = tmp_path / "data"
    monkeypatch.setattr(app_constants, "BASE_DATA_DIR", str(data_root))
    monkeypatch.setattr(app_constants, "DB_DATA_DIR", str(data_root / "database-format"))

    assert app_utils.portable_csv_dir(data_root / "database-format" / "thesis") == "thesis"
    assert app_utils.portable_csv_dir(data_root / "custom" / "csv") == "custom/csv"
    outside = tmp_path / "outside" / "csv"
    assert app_utils.portable_csv_dir(outside) == str(outside)
    assert app_utils.portable_csv_dir(data_root / "..hidden") == "..hidden"


def test_sample_annotations_sidecar_is_relative_inside_the_data_root(monkeypatch, tmp_path):
    """The sample-dataset builder records a data-root-relative source CSV directory."""
    data_root = tmp_path / "data"
    annotations = data_root / "annotations" / "input"
    annotations.mkdir(parents=True)
    first, second = "1\tAN00XXXX01.jpeg\t0\n", "2\tAN01XXXX01.jpeg\t1\n"
    for name, rows in (("full", first + second), ("train", first), ("valid", second)):
        (annotations / f"{name}-annotations.tsv").write_text(rows, encoding="utf-8")
    monkeypatch.setattr(app_constants, "BASE_DATA_DIR", str(data_root))
    monkeypatch.setattr(app_constants, "DB_DATA_DIR", str(data_root / "database-format"))
    output = data_root / "annotations" / "output"

    create_sample_dataset.filter_annotations_split(
        str(annotations), str(output), cutoff_year=2099,
        source_csv_dir=str(data_root / "database-format" / "sample"))

    assert (output / ".source_csv_dir").read_text(encoding="utf-8") == "sample\n"


@pytest.mark.skipif(shutil.which("sqlite3") is None, reason="requires the sqlite3 command-line tool")
def test_sqlite_dot_commands_accept_paths_with_spaces_and_quotes(tmp_path):
    """Schema and CSV paths containing spaces, quotes, or backslashes reach sqlite3 intact."""
    folder = tmp_path / "it's a \\folder"
    folder.mkdir()
    (folder / "schema.sql").write_text("CREATE TABLE sample (id INTEGER, name TEXT);\n", encoding="utf-8")
    (folder / "rows.csv").write_text("1,alpha\n", encoding="utf-8")
    database_path = tmp_path / "sample.db"

    database.run_sqlite_cmd(str(database_path), f".read {database._sqlite_shell_arg(folder / 'schema.sql')}\n")
    database.run_sqlite_cmd(
        str(database_path),
        f".import --csv {database._sqlite_shell_arg(folder / 'rows.csv')} sample\n")

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT id, name FROM sample").fetchall() == [(1, "alpha")]


def test_orchestrator_steps_run_from_the_checkout_not_the_callers_directory(monkeypatch, tmp_path):
    """Module-style steps resolve from the checkout even when started elsewhere."""
    captured = {}

    class _Process:
        returncode = 0

        def wait(self):
            """Report an immediately finished step."""
            return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shark_identification.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        shark_identification.subprocess, "Popen",
        lambda *_args, **kwargs: captured.update(kwargs) or _Process())

    shark_identification.run_step(["child"], "Step")

    assert captured["cwd"] == app_constants.THIS_DIR


def test_data_integrity_step_runs_from_the_checkout(monkeypatch, tmp_path):
    """The populate integrity check imports its module from the checkout, whatever the caller's directory."""
    database_path = tmp_path / "app.db"
    database_path.write_bytes(b"")
    captured = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(database, "_database_is_initialized", lambda _path: True)
    monkeypatch.setattr(database, "_database_has_data", lambda _path: False)
    monkeypatch.setattr(database, "default_repopulate", lambda *_args: None)
    monkeypatch.setattr(database, "_configured_database_path", lambda _args: str(database_path))
    monkeypatch.setattr(
        database.subprocess, "run",
        lambda command, **kwargs: captured.update(kwargs, command=command) or subprocess.CompletedProcess(command, 0))
    args = database.argparse.Namespace(data_path=str(tmp_path), all=False, tables=None, force=False,
                                       overwrite=False)

    database.handle_populate(args)

    assert captured["cwd"] == app_constants.THIS_DIR
    assert captured["command"][:3] == [sys.executable, "-m", "tests.manual.test_data_integrity"]
    assert os.path.exists(database_path)
