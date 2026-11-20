"""Tests for the standalone biology-to-database-format converter."""

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch


MATCHER_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = MATCHER_ROOT / "src" / "data_scripts" / "biology_to_db_format.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "biology_to_db_format",
    MODULE_PATH,
)
biology_to_db_format = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = biology_to_db_format
MODULE_SPEC.loader.exec_module(biology_to_db_format)


def test_main_defaults_output_root_to_matcher_data_authority(tmp_path):
    """The default output is a named child of constants.DB_DATA_DIR."""
    input_path = Path("biology-input.csv")
    with (
        patch.object(biology_to_db_format.constants, "DB_DATA_DIR", str(tmp_path)),
        patch.object(biology_to_db_format.os.path, "isfile", return_value=True),
        patch.object(biology_to_db_format, "convert") as mock_convert,
        patch.object(
            sys,
            "argv",
            ["biology_to_db_format.py", "--input", str(input_path)],
        ),
    ):
        biology_to_db_format.main()

    mock_convert.assert_called_once_with(
        str(input_path),
        str(tmp_path / input_path.stem),
    )


def test_main_accepts_external_output_root(tmp_path):
    """The CLI output-root option overrides the matcher data authority."""
    output_root = tmp_path / "external-datasets"
    with (
        patch.object(biology_to_db_format.os.path, "isfile", return_value=True),
        patch.object(biology_to_db_format, "convert") as mock_convert,
        patch.object(
            sys,
            "argv",
            [
                "biology_to_db_format.py",
                "--input",
                "biology-input.csv",
                "--output-root",
                str(output_root),
                "--output-name",
                "reviewed-dataset",
            ],
        ),
    ):
        biology_to_db_format.main()

    mock_convert.assert_called_once_with(
        "biology-input.csv",
        str(output_root / "reviewed-dataset"),
    )
