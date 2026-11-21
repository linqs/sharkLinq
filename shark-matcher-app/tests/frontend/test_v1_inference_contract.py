"""Tests for inference client requests, embedding updates, and startup CLI."""

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from core import app_constants
from backend import inference_snapshot
import frontend.inference_client as client
from frontend import project
from scripts import shark_identification


APP_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ID = "11111111-1111-4111-8111-111111111111"


def _response(status=200, body=None):
    """Build a mock JSON HTTP response."""
    response = MagicMock()
    response.status = status
    response.read.return_value = json.dumps(body or {}).encode("utf-8")
    return response


def _connection(response):
    """Build a mock HTTP connection returning the supplied response."""
    connection = MagicMock()
    connection.getresponse.return_value = response
    return connection


def test_app_production_code_does_not_import_matcher_modules():
    """Check that app runtime code does not import matcher modules."""
    forbidden = []
    production_roots = (APP_ROOT / "backend", APP_ROOT / "db", APP_ROOT / "frontend")
    for production_root in production_roots:
        for source_path in production_root.rglob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if name == "src" or name.startswith("src.") or name == "constants":
                        forbidden.append(f"{source_path.relative_to(APP_ROOT)}:{name}")

    assert not forbidden


def test_health_request_remains_public():
    """Check that the health request sends no Authorization header."""
    connection = _connection(_response(200, {"status": "ok"}))
    with patch("http.client.HTTPConnection", return_value=connection):
        with patch.object(
            app_constants,
            "INFERENCE_API_KEY",
            "contract-test-key-that-is-long-enough",
            create=True,
        ):
            client.health()

    request_kwargs = connection.request.call_args.kwargs
    assert "Authorization" not in request_kwargs.get("headers", {})


def test_embedding_generation_sends_snapshot_identifier_only():
    """Check that generate_embeddings sends only model_name and snapshot_id."""
    connection = _connection(_response(200, {"status": "ok"}))
    with patch("http.client.HTTPConnection", return_value=connection):
        with patch.object(
            app_constants,
            "INFERENCE_API_KEY",
            "contract-test-key-that-is-long-enough",
            create=True,
        ):
            client.generate_embeddings(
                "MiewID Multimodal", snapshot_id=SNAPSHOT_ID
            )

    payload = json.loads(connection.request.call_args.kwargs["body"])
    assert payload == {
        "model_name": "MiewID Multimodal",
        "snapshot_id": SNAPSHOT_ID,
    }


def test_update_handler_reuses_one_snapshot_for_every_loaded_model():
    """Check that update_embeddings creates one snapshot and reuses it for all loaded models."""
    project._reset_embeddings_update_job()
    models = ["MiewID Multimodal", "MiewID Image"]
    with patch.object(
            inference_snapshot,
            "create_inference_snapshot",
            return_value=SNAPSHOT_ID,
    ) as create_snapshot, patch.object(
            client, "get_stats", return_value={"loaded_models": models}
    ), patch.object(
            client, "generate_embeddings", return_value=True
    ) as generate_embeddings, patch.object(
            project,
            "get_dataset_stats",
            return_value=({"images": 2}, 200, None),
    ):
        result, status, headers = project.update_embeddings(
            MagicMock(), "/api/embeddings/update", sync=True
        )

    create_snapshot.assert_called_once_with()
    assert generate_embeddings.call_args_list == [
        call(model_name=model_name, snapshot_id=SNAPSHOT_ID)
        for model_name in models
    ]
    assert result == {"images": 2}
    assert status == 200
    assert headers is None


def test_update_handler_stops_when_snapshot_publication_fails():
    """Check that update_embeddings returns 500 and skips inference if snapshot creation fails."""
    project._reset_embeddings_update_job()
    with patch.object(
            inference_snapshot,
            "create_inference_snapshot",
            side_effect=OSError("injected snapshot failure"),
    ), patch.object(
            client,
            "get_stats",
            return_value={"loaded_models": ["MiewID Multimodal"]},
    ), patch.object(client, "generate_embeddings") as generate_embeddings:
        result, status, headers = project.update_embeddings(
            MagicMock(), "/api/embeddings/update", sync=True
        )

    generate_embeddings.assert_not_called()
    assert result == "Could not create inference snapshot."
    assert status == 500
    assert headers is None


def test_update_handler_starts_asynchronous_generation():
    """Check that async update_embeddings starts a background thread and returns in_progress."""
    project._reset_embeddings_update_job()
    with patch("threading.Thread") as mock_thread, patch.object(
            client, "get_stats", return_value={}):
        result, status, headers = project.update_embeddings(
            MagicMock(), "/api/embeddings/update"
        )
        assert status == 200
        assert result["status"] == "in_progress"
        assert "job_id" in result
        assert headers is None
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()


def test_update_handler_rejects_concurrent_runs_with_conflict():
    """A second update request while one is in progress returns HTTP 409 Conflict."""
    project._reset_embeddings_update_job()
    with patch.dict(project._embeddings_update_job, {"status": "in_progress"}):
        result, status, _ = project.update_embeddings(
            MagicMock(), "/api/embeddings/update"
        )
        assert status == 409
        assert "already in progress" in result["error"]


def test_get_embeddings_status_endpoint_reports_state():
    """The status endpoint reflects the current background job state."""
    project._reset_embeddings_update_job()
    result, status, _ = project.get_embeddings_update_status()
    assert status == 200
    assert result["status"] == "idle"


def test_dataset_stats_includes_update_in_progress_flag():
    """Check that dataset stats include update_in_progress and embeddings_update_status."""
    project._reset_embeddings_update_job()
    with patch.object(client, "get_stats", return_value={}):
        stats, status, _ = project.get_dataset_stats(MagicMock(), "/api/data/stats")
        assert status == 200
        assert stats["update_in_progress"] is False
        assert stats["embeddings_update_status"] == "idle"



def test_stats_request_sends_bearer_credential():
    """Check that get_stats sends the Bearer API key."""
    connection = _connection(_response(200, {}))
    api_key = "contract-test-key-that-is-long-enough"
    with patch("http.client.HTTPConnection", return_value=connection):
        with patch.object(
            app_constants,
            "INFERENCE_API_KEY",
            api_key,
            create=True,
        ):
            client.get_stats()

    headers = connection.request.call_args.kwargs["headers"]
    assert headers["Authorization"] == f"Bearer {api_key}"


def test_unconfigured_protected_request_does_not_open_a_connection():
    """Check that get_stats skips the HTTP request when INFERENCE_API_KEY is empty."""
    connection_class = MagicMock()
    with patch("http.client.HTTPConnection", connection_class):
        with patch.object(
            app_constants,
            "INFERENCE_API_KEY",
            "",
            create=True,
        ):
            result = client.get_stats()

    assert result is None
    connection_class.assert_not_called()


@pytest.mark.parametrize(
    ("arguments", "step_name", "test_mode"),
    [
        (["--run"], "Run Server", False),
        (["--test"], "Run Server (Test Mode)", True),
    ],
)
def test_orchestrator_starts_app_server_without_model_arguments(
        arguments, step_name, test_mode):
    """Check that shark_identification starts the app server without model arguments."""
    argv = ["shark_identification", *arguments]
    with patch.object(shark_identification.sys, "argv", argv), patch.object(
            shark_identification.app_utils, "print_script_banner"
    ), patch.object(shark_identification.os, "makedirs"), patch(
            "builtins.open", mock_open()
    ), patch.object(shark_identification, "run_step") as run_step:
        shark_identification.main()

    run_step.assert_called_once()
    command, actual_step_name = run_step.call_args.args
    assert command == [
        shark_identification.sys.executable,
        "-m",
        "frontend.cli.serve",
        ".",
        "--no-browser",
    ]
    assert actual_step_name == step_name
    if test_mode:
        assert run_step.call_args.kwargs["env"]["SHARK_TEST_MODE"] == "true"
    else:
        assert "env" not in run_step.call_args.kwargs


def test_orchestrator_repopulates_test_database_from_committed_fixture():
    """Check that --test --repopulate loads the committed CI test data."""
    argv = ["shark_identification", "--test", "--repopulate"]
    with patch.object(shark_identification.sys, "argv", argv), patch.object(
            shark_identification.app_utils, "print_script_banner"
    ), patch.object(shark_identification.os, "makedirs"), patch(
            "builtins.open", mock_open()
    ), patch.object(shark_identification, "run_step") as run_step:
        shark_identification.main()

    step_names = [step.args[1] for step in run_step.call_args_list]
    assert step_names == [
        "Init Test Database", "Populate Test Database", "Run Server (Test Mode)"]
    populate_command = run_step.call_args_list[1].args[0]
    data_path = populate_command[populate_command.index("--data-path") + 1]
    assert data_path == app_constants.CI_TEST_DATABASE_FORMAT_DIR
    assert (Path(data_path) / "tokens.csv").is_file()
    assert all(
        step.kwargs["env"]["SHARK_TEST_MODE"] == "true"
        for step in run_step.call_args_list
    )


def test_orchestrator_rejects_app_side_model_selection():
    """Check that shark_identification rejects the --model flag."""
    argv = ["shark_identification", "--model", "test-model"]
    with patch.object(shark_identification.sys, "argv", argv), patch.object(
            shark_identification.app_utils, "print_script_banner"
    ), patch.object(shark_identification.os, "makedirs"), patch(
            "builtins.open", mock_open()
    ), pytest.raises(SystemExit) as raised:
        shark_identification.main()

    assert raised.value.code != 0
