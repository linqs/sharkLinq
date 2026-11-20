"""Tests for the src.cli.run_pipeline orchestrator's child-command construction."""

import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

from src.core import constants
import src.cli.run_pipeline as run_pipeline


def _run_orchestrator(argv):
    """Run the orchestrator with argv and return every child subprocess call."""
    with patch.object(sys, "argv", ["src.cli.run_pipeline", *argv]), patch(
            "src.cli.run_pipeline.subprocess.run",
            return_value=MagicMock(returncode=0),
    ) as subprocess_run, patch(
            "src.cli.run_pipeline.utils.resolve_directory_path",
            side_effect=lambda path, _fallback: path,
    ), patch("src.cli.run_pipeline.utils.print_script_banner"), patch(
            "src.cli.run_pipeline.utils.notify_completion"):
        run_pipeline.main()
    return subprocess_run.call_args_list


def test_pipeline_children_run_as_modules_from_the_repository_root():
    """Importable steps use -m and every step runs from the repository root so imports resolve."""
    calls = _run_orchestrator([
        "--input-images", "images",
        "--input-csv", "csvs",
        "--model-name", "model",
        "--configs", "config_key",
        "--no-email",
    ])
    commands = [call.args[0] for call in calls]

    assert [call.kwargs.get("cwd") for call in calls] == [constants.THIS_DIR] * len(calls)
    modules = [command[2] for command in commands if command[1] == "-m"]
    assert modules == [
        "src.cli.convert_images",
        "src.cli.convert_data",
        "src.learning.trainer.new_trainer",
        "src.retrieval.embeddings",
        "src.retrieval.evaluation",
    ]
    for module in modules:
        assert importlib.util.find_spec(module) is not None, module
    script_steps = [command[1] for command in commands if command[1] != "-m"]
    assert len(script_steps) == 1
    assert script_steps[0].endswith(os.path.join("data_scripts", "analyze_data.py"))
    assert os.path.isfile(script_steps[0])


def _embedding_command(extra_args):
    """Return the Step 5 embeddings command built for the given extra flags."""
    calls = _run_orchestrator([
        "--input-images", "images",
        "--input-csv", "csvs",
        "--model-name", "model",
        "--configs", "config_key",
        "--no-email",
        *extra_args,
    ])
    return next(call.args[0] for call in calls
                if "src.retrieval.embeddings" in call.args[0])


def test_embeddings_step_is_plain_unless_stacked_is_requested():
    """A default run avoids the PSL and Java requirements of --stacked."""
    assert "--stacked" not in _embedding_command([])


def test_stacked_flag_is_forwarded_to_the_embeddings_step():
    """--stacked on the orchestrator reaches the embeddings command."""
    assert "--stacked" in _embedding_command(["--stacked"])


def test_missing_configs_fails_before_any_step_runs():
    """Training without --configs stops before steps 1-3 write any data."""
    with patch.object(sys, "argv", ["src.cli.run_pipeline", "--input-images", "images", "--input-csv", "csvs",
                                  "--model-name", "model"]), patch(
            "src.cli.run_pipeline.subprocess.run") as subprocess_run:
        try:
            run_pipeline.main()
        except SystemExit as exit_info:
            assert exit_info.code == 1
        else:
            raise AssertionError("expected SystemExit")
    subprocess_run.assert_not_called()


def _training_command(extra_args):
    """Return the Step 4 trainer command built for the given extra flags."""
    calls = _run_orchestrator([
        "--input-images", "images",
        "--input-csv", "csvs",
        "--model-name", "model",
        "--configs", "config_key",
        "--no-email",
        *extra_args,
    ])
    return next(call.args[0] for call in calls
                if "src.learning.trainer.new_trainer" in call.args[0])


def test_training_step_keeps_the_overwrite_guard_unless_asked():
    """A default rerun lets the trainer refuse to replace existing weights."""
    assert "--overwrite" not in _training_command([])


def test_overwrite_flag_is_forwarded_to_the_training_step():
    """--overwrite on the orchestrator reaches the trainer command."""
    assert "--overwrite" in _training_command(["--overwrite"])
