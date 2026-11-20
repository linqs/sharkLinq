"""Pytest fixtures pointing csv_store loader at fixture CSV files for src unit tests."""

import os

import pytest

from src.core import constants
import src.csv_store._loader as _loader

_FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")


def _use_csv_fixture(name):
    """Context-manager fixture generator redirecting csv_store to a named fixture directory."""
    @pytest.fixture
    def _fixture():
        path = os.path.join(_FIXTURES_DIR, name)
        original = constants.DB_DATA_DIR
        original_custom = _loader._custom_db_dir
        constants.DB_DATA_DIR = path
        _loader._custom_db_dir = None
        _loader.clear_cache()
        yield path
        constants.DB_DATA_DIR = original
        _loader._custom_db_dir = original_custom
        _loader.clear_cache()
    return _fixture


csv_fine_evaluation_full = _use_csv_fixture("fine_evaluation_full")
csv_fine_evaluation_partial = _use_csv_fixture("fine_evaluation_partial")
csv_convert_data = _use_csv_fixture("convert_data")
