"""Configuration and fixtures for opt-in parity tests comparing CSV store contents against a SQLite database."""

import os
import sqlite3

import pytest

from src.core import constants
import src.csv_store._loader as _loader


def _resolve_db_path() -> str:
    return os.getenv("PARITY_DB_PATH", "")


def _resolve_csv_dir() -> str:
    return os.getenv("PARITY_CSV_DIR", "")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "parity: one-time local validation tests comparing CSV store against the live DB"
    )


@pytest.fixture(scope="session", autouse=True)
def parity_paths(tmp_path_factory):
    """Validate paths and point the csv_store at the right data directory."""
    db_path = _resolve_db_path()
    csv_dir = _resolve_csv_dir()

    missing = []
    if not os.path.isfile(db_path):
        missing.append(f"DB not found: {db_path!r}  (set PARITY_DB_PATH)")
    if not csv_dir or not os.path.isdir(csv_dir):
        missing.append(f"CSV dir not found: {csv_dir!r}  (set PARITY_CSV_DIR)")

    if missing:
        pytest.skip("\n".join(["Parity tests require live data:"] + missing))

    # Point the csv_store loader at the user-supplied CSV directory.
    constants.DB_DATA_DIR = csv_dir
    _loader.clear_cache()

    return {"db_path": db_path, "csv_dir": csv_dir}


@pytest.fixture(scope="session")
def db(parity_paths):
    """Return an open sqlite3 connection to the live database."""
    conn = sqlite3.connect(parity_paths["db_path"])
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
