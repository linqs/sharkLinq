"""Tests that db.database init never wipes a database holding data without --force."""

import argparse
import sqlite3
from pathlib import Path

import pytest

import db.database as database

SCHEMA_PATH = Path(database.__file__).parent / "sql" / "create_database.sql"


def _init_args(**overrides):
    values = {"name": None, "no_indexes": True, "force": False}
    values.update(overrides)
    return argparse.Namespace(**values)


def _create_schema(db_path):
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def _add_row(db_path, table):
    with sqlite3.connect(db_path) as connection:
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
        values = {"status": "'open'"}
        connection.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join(values.get(column, '1') for column in columns)})")


@pytest.fixture(name="db_path")
def db_path_fixture(tmp_path, monkeypatch):
    path = tmp_path / "app.db"
    monkeypatch.setattr(database, "_configured_database_path", lambda _args: str(path))
    return path


def test_init_creates_a_missing_database(db_path):
    database.handle_init(_init_args())
    assert database._database_is_initialized(str(db_path))


def test_init_recreates_an_empty_database_without_the_flag(db_path):
    _create_schema(db_path)
    database.handle_init(_init_args())
    assert database._database_is_initialized(str(db_path))


@pytest.mark.parametrize("table", ["Media", "Users", "Actions", "Proposals", "Tokens"])
def test_init_refuses_a_database_with_data_and_leaves_it_untouched(db_path, capsys, table):
    _create_schema(db_path)
    _add_row(db_path, table)
    original = db_path.read_bytes()

    with pytest.raises(SystemExit) as raised:
        database.handle_init(_init_args())

    assert raised.value.code == 1
    assert "--force" in capsys.readouterr().out
    assert db_path.read_bytes() == original


def test_force_wipes_a_database_with_data(db_path):
    _create_schema(db_path)
    _add_row(db_path, "Media")
    database.handle_init(_init_args(force=True))
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM Media").fetchone()[0] == 0
