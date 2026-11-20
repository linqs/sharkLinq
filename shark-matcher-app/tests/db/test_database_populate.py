"""Tests for db.database population safety on failure paths."""

import argparse
import sqlite3
from pathlib import Path

import pytest

import db.database as database

SCHEMA_PATH = Path(database.__file__).parent / "sql" / "create_database.sql"


def _populate_args(data_path, **overrides):
    """Return populate arguments for the default (non-snapshot) workflow."""
    values = {
        "data_path": str(data_path),
        "all": False,
        "tables": None,
        "users_file": "users.csv",
        "force": False,
        "overwrite": False,
        "name": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _staging_path(db_path):
    """Return the temporary database path that populate stages into."""
    return Path(str(db_path) + ".tmp")


@pytest.mark.parametrize("create_empty_file", [False, True])
def test_populate_refuses_a_database_without_schema(
        tmp_path, monkeypatch, capsys, create_empty_file):
    """Populate stops with an init hint and neither creates nor stages a database."""
    db_path = tmp_path / "app.db"
    if create_empty_file:
        db_path.touch()
    monkeypatch.setattr(database, "_configured_database_path", lambda _args: str(db_path))

    with pytest.raises(SystemExit) as raised:
        database.handle_populate(_populate_args(tmp_path))

    assert raised.value.code == 1
    assert "db.database init" in capsys.readouterr().out
    assert db_path.exists() is create_empty_file
    if create_empty_file:
        assert db_path.stat().st_size == 0
    assert not _staging_path(db_path).exists()


def test_failed_population_discards_staging_copy_and_keeps_database(tmp_path, monkeypatch):
    """A failing import leaves the configured database byte-identical and removes the staging copy."""
    db_path = tmp_path / "app.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    original = db_path.read_bytes()

    def failing_sqlite_command(_db_path, _commands):
        raise SystemExit(5)

    monkeypatch.setattr(database, "_configured_database_path", lambda _args: str(db_path))
    monkeypatch.setattr(database, "run_sqlite_cmd", failing_sqlite_command)

    with pytest.raises(SystemExit) as raised:
        database.handle_populate(_populate_args(tmp_path))

    assert raised.value.code == 5
    assert db_path.read_bytes() == original
    assert not _staging_path(db_path).exists()


def test_missing_sqlite_command_line_tool_is_reported(monkeypatch, capsys):
    """A missing sqlite3 binary produces an actionable error instead of a traceback."""
    def missing_binary(*_args, **_kwargs):
        raise FileNotFoundError("sqlite3")

    monkeypatch.setattr(database.subprocess, "run", missing_binary)

    with pytest.raises(SystemExit) as raised:
        database.run_sqlite_cmd("unused.db", ".tables\n")

    assert raised.value.code == 1
    assert "sqlite3 command-line tool is required" in capsys.readouterr().out


def test_populate_requires_a_data_path(monkeypatch, capsys):
    """Populate without --data-path stops with the usual missing-argument error."""
    monkeypatch.setattr("sys.argv", ["database.py", "populate", "--all"])
    with pytest.raises(SystemExit) as raised:
        database.main()

    assert raised.value.code == 2
    assert "--data-path" in capsys.readouterr().err


def _record_and_stop(reached):
    """Return a stand-in for the repopulate step that records it ran and stops populate there."""
    def stop(*_args):
        reached.append(True)
        raise SystemExit(7)
    return stop


def test_populate_refuses_a_database_with_data_unless_overwrite_is_given(tmp_path, monkeypatch, capsys):
    """Populate leaves a database with data untouched and names --overwrite; with the flag it proceeds."""
    db_path = tmp_path / "app.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        connection.execute("INSERT INTO Media (id, hash, name, filename) VALUES (1, 'h', 'AB12345678', 'a.jpeg')")
    original = db_path.read_bytes()
    monkeypatch.setattr(database, "_configured_database_path", lambda _args: str(db_path))

    with pytest.raises(SystemExit) as raised:
        database.handle_populate(_populate_args(tmp_path, all=True))
    assert raised.value.code == 1
    assert "--overwrite" in capsys.readouterr().out
    assert db_path.read_bytes() == original

    reached = []
    monkeypatch.setattr(database, "full_repopulate", _record_and_stop(reached))
    with pytest.raises(SystemExit) as raised:
        database.handle_populate(_populate_args(tmp_path, all=True, overwrite=True))
    assert reached == [True]


def test_populate_runs_on_a_freshly_initialised_database(tmp_path, monkeypatch):
    """The documented first run, init then populate, is not blocked by the guard."""
    db_path = tmp_path / "app.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(database, "_configured_database_path", lambda _args: str(db_path))
    reached = []
    monkeypatch.setattr(database, "full_repopulate", _record_and_stop(reached))
    with pytest.raises(SystemExit):
        database.handle_populate(_populate_args(tmp_path, all=True))
    assert reached == [True]
