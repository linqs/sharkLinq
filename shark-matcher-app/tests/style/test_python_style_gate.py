"""Tests for the repository Python lint gate's file discovery."""

import os

from core import app_constants
from tests.tools import python_style_tests


def test_default_lint_targets_reach_modules_outside_packages(monkeypatch):
    """The gate lints files in directories without __init__.py, which Pylint skips when given a directory."""
    monkeypatch.chdir(app_constants.THIS_DIR)
    files = {
        os.path.normpath(path)
        for path in python_style_tests.python_files(python_style_tests.DEFAULT_PATHS)
    }

    for expected in (
        "db/database.py",
        "tests/backend/db/test_user.py",
        "tests/browser/conftest.py",
        "tests/manual/test_data_integrity.py",
    ):
        assert os.path.normpath(expected) in files
    assert not any("__pycache__" in path for path in files)
