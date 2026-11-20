"""Shared pytest configuration for the Shark Matcher application."""

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import pytest

# Keep the suite's database and outputs in a disposable data root, never a real sm-data workspace.
_TEST_DATA_ROOT = tempfile.mkdtemp(prefix="shark-matcher-app-tests-")
os.environ["SHARK_DATA_ROOT"] = _TEST_DATA_ROOT
# Set test mode before importing constants to ensure DB_PATH is correct.
os.environ["SHARK_TEST_MODE"] = "true"

from core import app_constants  # pylint: disable=wrong-import-position


def pytest_unconfigure(config):  # pylint: disable=unused-argument
    """Remove the disposable data root when the test session ends."""
    shutil.rmtree(_TEST_DATA_ROOT, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def setup_test_database(tmp_path_factory):
    """Populate the disposable test database at the start of a test session."""
    _populate_ci_test_database()
    snapshot = tmp_path_factory.mktemp('ci-snapshot') / 'fixture.db'
    with sqlite3.connect(app_constants.DB_PATH) as original:
        with sqlite3.connect(snapshot) as copy:
            original.backup(copy)
    yield snapshot


@pytest.fixture(scope="module")
def reset_ci_test_database(request):
    """Reset a read-only contract module to the immutable CI fixture state."""
    request.getfixturevalue("setup_test_database")
    _populate_ci_test_database()


def _populate_ci_test_database():
    """Rebuild the disposable test database from the committed CI fixture."""
    data_path = app_constants.CI_TEST_DATABASE_FORMAT_DIR
    if not os.path.isdir(data_path):
        pytest.fail(f"Required immutable CI test data not found at {data_path}")

    base_cmd = [sys.executable, "-m", "db.database", "-t"]

    # Initialize a fresh schema (equivalent to the old -d/drop flag).
    print(f"Initializing test database schema: {app_constants.DB_PATH}")
    init_result = subprocess.run(
        base_cmd + ["init", "--force"],
        cwd=app_constants.SHARK_APP_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if init_result.returncode != 0:
        pytest.fail(f"Failed to initialize test database: {init_result.stderr}")

    print(f"Using immutable CI test data from: {data_path}")
    populate_cmd = base_cmd + ["populate", "--all", "--data-path", data_path]
    print(f"Populating test database: {app_constants.DB_PATH}")
    populate_result = subprocess.run(
        populate_cmd,
        cwd=app_constants.SHARK_APP_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if populate_result.returncode != 0:
        pytest.fail(
            f"Failed to populate test database: {populate_result.stderr}"
        )
