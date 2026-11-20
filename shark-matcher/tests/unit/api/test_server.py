"""Unit tests for the FastAPI server verifying endpoints, locking invariants, and error responses."""

import time
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient

import src.api.server as server
from src.api.server import app
from src.contracts import ModelDiagnostic, ModelState, ModelStatus


SNAPSHOT_ID = "11111111-1111-4111-8111-111111111111"
SERVICE_API_KEY = "contract-test-service-key-with-32-characters"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_generation_lock():
    """Ensure _generation_lock is free before and after each test."""
    _release_lock()
    yield
    _release_lock()


def _release_lock():
    """Drop _generation_lock if held. Releasing an unlocked Lock raises."""
    if server._generation_lock.locked():
        server._generation_lock.release()


def _generation_payload(model_name="miew"):
    """Return the snapshot-only V1 embedding generation request."""
    return {"model_name": model_name, "snapshot_id": SNAPSHOT_ID}


def _model_status(display_name, state, *, required=True, error=None):
    """Return one immutable model-status fixture."""
    diagnostic = ModelDiagnostic(**error) if error is not None else None
    return ModelStatus(
        display_name=display_name,
        embedding_set=display_name.lower(),
        required=required,
        state=ModelState(state),
        error=diagnostic,
    )


def _status_payloads(statuses):
    """Serialize model-status fixtures as FastAPI response JSON."""
    return [status.model_dump(mode="json") for status in statuses]


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r._load_model_data.return_value = _model_status("miew", "ready")
    r.get_neighbors.return_value = {
        "ids": [101, 201],
        "distances": [0.1, 0.5],
        "labels": [10, 20],
    }
    # The generate endpoint resolves model_name -> config -> (dir, model) before
    # calling the pipeline. With no configs registered, both fall back to the
    # resolved name, which keeps the forwarding assertions readable.
    r._resolve_model.side_effect = lambda name: name or 'miew'
    r.retrieval_methods = {0: None, 1: None, 2: None}
    r.allowed_models = {'miew'}
    r._model_configs = {}
    return r


@pytest.fixture
def client(mock_retriever):
    """TestClient with Retriever creation patched out of the lifespan."""
    with patch.object(
            server.constants,
            "INFERENCE_API_KEY",
            SERVICE_API_KEY,
            create=True,
    ), patch("src.api.server.Retriever", return_value=mock_retriever):
        with TestClient(
                app,
                headers={"Authorization": f"Bearer {SERVICE_API_KEY}"},
        ) as c:
            yield c, mock_retriever


@pytest.fixture
def unauthenticated_client(mock_retriever):
    """TestClient with a configured server key but no default credential."""
    with patch.object(
            server.constants,
            "INFERENCE_API_KEY",
            SERVICE_API_KEY,
            create=True,
    ), patch("src.api.server.Retriever", return_value=mock_retriever):
        with TestClient(app) as c:
            yield c, mock_retriever


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_remains_public_without_credential(self, unauthenticated_client):
        """The liveness report does not require a credential."""
        c, _ = unauthenticated_client
        assert c.get("/health").status_code == 200

    def test_returns_200(self, client):
        c, _ = client
        assert c.get("/health").status_code == 200

    def test_returns_ok_body(self, client):
        c, _ = client
        assert c.get("/health").json() == {"status": "ok"}

    def test_does_not_include_model_state(self, client):
        """Model readiness cannot change or expand the liveness response."""
        c, mock_r = client
        mock_r.get_model_statuses.return_value = [
            _model_status("broken", "failed", error={
                "code": "artifact_missing",
                "message": "Required model data is missing.",
            })
        ]
        assert c.get("/health").json() == {"status": "ok"}
        mock_r.get_model_statuses.assert_not_called()


class TestReadiness:
    """Pin readiness decisions and reuse of Retriever model statuses."""

    def test_all_required_models_ready_returns_200(self, client):
        """Readiness succeeds only when every required model is ready."""
        c, mock_r = client
        statuses = [_model_status("miew", "ready")]
        mock_r.get_model_statuses.return_value = statuses

        response = c.get("/ready")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "model_statuses": _status_payloads(statuses),
        }

    def test_partial_failure_is_live_but_not_ready(self, client):
        """One failed required model returns 503 while health stays live."""
        c, mock_r = client
        statuses = [
            _model_status("miew", "ready"),
            _model_status(
                "vit",
                "failed",
                error={
                    "code": "artifact_missing",
                    "message": "Required model data is missing.",
                },
            ),
        ]
        mock_r.get_model_statuses.return_value = statuses

        readiness = c.get("/ready")
        liveness = c.get("/health")

        assert readiness.status_code == 503
        assert readiness.json() == {
            "status": "not_ready",
            "model_statuses": _status_payloads(statuses),
        }
        assert liveness.status_code == 200
        assert liveness.json() == {"status": "ok"}

    def test_loading_model_is_not_ready(self, client):
        """A delayed required load remains machine-detectably not ready."""
        c, mock_r = client
        statuses = [_model_status("miew", "loading")]
        mock_r.get_model_statuses.return_value = statuses

        response = c.get("/ready")

        assert response.status_code == 503
        assert response.json()["model_statuses"] == _status_payloads(statuses)

    def test_disabled_optional_model_does_not_block_readiness(self, client):
        """Explicitly disabled optional models remain visible but non-blocking."""
        c, mock_r = client
        statuses = [
            _model_status("miew", "ready"),
            _model_status("vit", "disabled", required=False),
        ]
        mock_r.get_model_statuses.return_value = statuses

        response = c.get("/ready")

        assert response.status_code == 200
        assert response.json()["model_statuses"] == _status_payloads(statuses)

    def test_readiness_requires_service_credential(self, unauthenticated_client):
        """Readiness follows the protected non-liveness route convention."""
        c, _ = unauthenticated_client

        response = c.get("/ready")

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"


class TestRealStartupMonitoring:
    """Exercise monitoring through the real lifespan and Retriever loader."""

    def test_missing_required_model_is_live_not_ready_and_reported(
            self, tmp_path):
        """A real failed preload separates liveness, readiness, and stats."""
        model_name = "startup-test"
        models = [{
            "display_name": model_name,
            "dir": model_name,
            "model": model_name,
            "retrieval_mode": "baseline",
            "rerank": "none",
            "required": True,
        }]
        expected_statuses = [
            _model_status(
                model_name,
                "failed",
                error={
                    "code": "artifact_missing",
                    "message": "Required model data is missing.",
                },
            )
        ]
        authorization = {
            "Authorization": f"Bearer {SERVICE_API_KEY}"
        }

        with patch.object(
                server.constants,
                "INFERENCE_API_KEY",
                SERVICE_API_KEY,
                create=True,
        ), patch.object(
                server.constants,
                "EMBEDDINGS_DIR",
                str(tmp_path),
        ), patch.object(
                server,
                "DEFAULT_MODELS",
                models,
        ), patch(
                "src.retrieval.retrieval.utils.get_torch_device",
                return_value="cpu",
        ):
            with TestClient(app) as real_client:
                health_response = real_client.get("/health")
                ready_response = real_client.get(
                    "/ready", headers=authorization
                )
                stats_response = real_client.get(
                    "/stats", headers=authorization
                )

        assert type(app.state.retriever) is server.Retriever
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "ok"}
        assert ready_response.status_code == 503
        assert ready_response.json() == {
            "status": "not_ready",
            "model_statuses": _status_payloads(expected_statuses),
        }
        assert stats_response.status_code == 200
        assert stats_response.json() == {
            "loaded_models": [],
            "embeddings_metadata": {
                "start_time": None,
                "update_time": None,
            },
            "filenames": [],
            "model_statuses": _status_payloads(expected_statuses),
            "generation_id": None,
            "snapshot_id": None,
        }
        assert str(tmp_path) not in ready_response.text
        assert str(tmp_path) not in stats_response.text


# ---------------------------------------------------------------------------
# Service authentication contract
# ---------------------------------------------------------------------------

class TestServiceAuthenticationContract:
    """Pin credential parsing, route coverage, and startup failure behavior."""

    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("POST", "/match", {"image_id": 101}),
            ("GET", "/ready", None),
            ("GET", "/stats", None),
            ("POST", "/embeddings/generate", _generation_payload()),
            ("POST", "/suggestions", {}),
        ],
    )
    def test_each_protected_route_rejects_a_missing_credential(
            self, unauthenticated_client, method, path, body):
        """Every non-liveness route rejects a request without Authorization."""
        c, _ = unauthenticated_client
        with patch("src.api.server._run_embedding_pipeline"):
            response = c.request(method, path, json=body)

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_malformed_authorization_scheme_is_rejected(
            self, unauthenticated_client):
        """A non-Bearer Authorization scheme receives the standard challenge."""
        c, _ = unauthenticated_client
        response = c.post(
            "/match",
            json={"image_id": 101},
            headers={"Authorization": f"Basic {SERVICE_API_KEY}"},
        )

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_incorrect_credential_is_rejected_without_disclosure(
            self, unauthenticated_client, capsys):
        """An incorrect key receives no configured or presented key material."""
        c, _ = unauthenticated_client
        incorrect_key = "incorrect-contract-key-with-32-characters"
        response = c.post(
            "/match",
            json={"image_id": 101},
            headers={"Authorization": f"Bearer {incorrect_key}"},
        )

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert SERVICE_API_KEY not in response.text
        assert incorrect_key not in response.text
        captured = capsys.readouterr()
        assert SERVICE_API_KEY not in captured.out + captured.err
        assert incorrect_key not in captured.out + captured.err

    @pytest.mark.parametrize(
        "configured_key",
        [
            "",
            "too-short",
            " contract-test-service-key-with-32-characters ",
        ],
    )
    def test_invalid_server_key_stops_startup_before_model_loading(
            self, configured_key):
        """A missing, short, or padded key prevents model initialization."""
        with patch.object(
                server.constants,
                "INFERENCE_API_KEY",
                configured_key,
                create=True,
        ), patch(
                "src.api.server._log_model_config"
        ) as log_model_config, patch(
                "src.api.server.Retriever"
        ) as retriever_class:
            with pytest.raises(RuntimeError, match="service API key") as raised:
                with TestClient(app):
                    pass

        message = str(raised.value)
        assert "SHARK_INFERENCE_API_KEY" in message
        assert "32 characters" in message
        if configured_key.strip():
            assert configured_key.strip() not in message
        log_model_config.assert_not_called()
        retriever_class.assert_not_called()

    def test_main_binds_to_the_configured_address(self, monkeypatch):
        """The module entry point uses the resolved host and port."""
        monkeypatch.setenv("SHARK_INFERENCE_HOST", "127.0.0.1")
        monkeypatch.setenv("SHARK_INFERENCE_PORT", "18000")
        with patch("uvicorn.run") as uvicorn_run:
            server.main()
        uvicorn_run.assert_called_once_with(
            server.app, host="127.0.0.1", port=18000, reload=False)

    def test_framework_documentation_routes_are_not_public(self):
        """Only declared inference routes exist; framework docs are disabled."""
        paths = {
            route.path for route in app.routes if hasattr(route, "path")
        }
        assert "/docs" not in paths
        assert "/redoc" not in paths
        assert "/openapi.json" not in paths


def test_correct_service_credential_preserves_protected_route_behavior(client):
    """The configured Bearer credential allows the underlying route to run."""
    c, mock_r = client
    response = c.post("/match", json={"image_id": 101})

    assert response.status_code == 200
    mock_r.get_neighbors.assert_called_once_with(0, 101, model=None)


# ---------------------------------------------------------------------------
# POST /match
# ---------------------------------------------------------------------------

class TestMatch:
    def test_happy_path_returns_parallel_lists(self, client):
        # The response is the retriever's own {ids, distances, labels} shape
        # rather than the old list of {id, distance} objects; labels are now
        # carried through instead of being dropped.
        c, _ = client
        resp = c.post("/match", json={"image_id": 101})
        assert resp.status_code == 200
        assert resp.json() == {
            "ids": [101, 201],
            "distances": [0.1, 0.5],
            "labels": [10, 20],
        }

    def test_empty_result_returns_200_not_404(self, client):
        """A valid empty ranking stays successful, distinct from service failure."""
        c, mock_r = client
        mock_r.get_neighbors.return_value = {"ids": [], "distances": [], "labels": []}
        resp = c.post("/match", json={"image_id": 999})
        assert resp.status_code == 200
        assert resp.json() == {"ids": [], "distances": [], "labels": []}

    def test_default_retrieval_method_is_0(self, client):
        c, mock_r = client
        c.post("/match", json={"image_id": 101})
        mock_r.get_neighbors.assert_called_once_with(0, 101, model=None)

    def test_retrieval_method_forwarded(self, client):
        c, mock_r = client
        c.post("/match", json={"image_id": 101, "retrieval_method": 2})
        mock_r.get_neighbors.assert_called_once_with(2, 101, model=None)

    def test_model_name_forwarded(self, client):
        c, mock_r = client
        c.post("/match", json={"image_id": 101, "model": "miew"})
        mock_r.get_neighbors.assert_called_once_with(0, 101, model="miew")

    def test_missing_image_id_returns_422(self, client):
        c, _ = client
        resp = c.post("/match", json={"retrieval_method": 0})
        assert resp.status_code == 422

    @pytest.mark.parametrize('parameters', [
        {'retrieval_method': 99}, {'model': 'unknown-model'},
    ])
    def test_unknown_match_options_are_errors(self, client, parameters):
        """Bad options cannot masquerade as successful empty rankings."""
        c, mock_r = client
        response = c.post('/match', json={'image_id': 101, **parameters})
        assert response.status_code == 400
        mock_r.get_neighbors.assert_not_called()

    def test_unavailable_model_is_service_unavailable(self, client):
        """An unavailable configured model is distinguishable from no matches."""
        c, mock_r = client
        mock_r._get_model_data.return_value = None
        response = c.post('/match', json={'image_id': 101})
        assert response.status_code == 503
        mock_r.get_neighbors.assert_not_called()


class TestStats:
    def test_reports_shared_status_and_default_source_identity(self, client):
        """Stats serializes Retriever status and active generation identity."""
        c, mock_r = client
        statuses = [_model_status("miew", "ready")]
        mock_r.get_model_statuses.return_value = statuses
        mock_r.get_loaded_display_names.return_value = ["miew"]
        mock_r.default_model_name = "miew"
        mock_r.loaded_models = {
            "miew": {
                "embeddings_metadata": {
                    "start_time": 1000,
                    "update_time": 20,
                },
                "image_neighbors": {"shark.jpg": []},
                "generation_id": "22222222-2222-4222-8222-222222222222",
                "snapshot_id": SNAPSHOT_ID,
            }
        }

        response = c.get("/stats")

        assert response.status_code == 200
        assert response.json() == {
            "loaded_models": ["miew"],
            "embeddings_metadata": {
                "start_time": 1000,
                "update_time": 20,
            },
            "filenames": ["shark.jpg"],
            "model_statuses": _status_payloads(statuses),
            "generation_id": "22222222-2222-4222-8222-222222222222",
            "snapshot_id": SNAPSHOT_ID,
        }

    def test_failed_default_has_status_without_source_identity(self, client):
        """A failed default remains visible while generation IDs stay absent."""
        c, mock_r = client
        statuses = [
            _model_status(
                "miew",
                "failed",
                error={
                    "code": "artifact_incompatible",
                    "message": "Required model data is incompatible.",
                },
            )
        ]
        mock_r.get_model_statuses.return_value = statuses
        mock_r.get_loaded_display_names.return_value = []
        mock_r.default_model_name = "miew"
        mock_r.loaded_models = {}

        response = c.get("/stats")

        assert response.status_code == 200
        assert response.json() == {
            "loaded_models": [],
            "embeddings_metadata": {
                "start_time": None,
                "update_time": None,
            },
            "filenames": [],
            "model_statuses": _status_payloads(statuses),
            "generation_id": None,
            "snapshot_id": None,
        }


# ---------------------------------------------------------------------------
# POST /embeddings/generate
# ---------------------------------------------------------------------------

class TestEmbeddingsGenerate:
    def test_returns_ok(self, client):
        # The endpoint is synchronous now, so a 200 means the pipeline finished
        # rather than that it was scheduled.
        c, _ = client
        with patch("src.api.server._run_embedding_pipeline"):
            resp = c.post("/embeddings/generate", json=_generation_payload())
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_returns_503_when_generation_in_progress(self, client):
        c, _ = client
        server._generation_lock.acquire()
        resp = c.post("/embeddings/generate", json=_generation_payload())
        assert resp.status_code == 503
        assert "already in progress" in resp.json()["detail"]

    def test_busy_request_does_not_run_pipeline(self, client):
        """A rejected request must not touch the embeddings on disk."""
        c, _ = client
        server._generation_lock.acquire()
        with patch("src.api.server._run_embedding_pipeline") as mock_pipeline:
            c.post("/embeddings/generate", json=_generation_payload())
        mock_pipeline.assert_not_called()

    def test_different_model_is_also_blocked(self, client):
        """Verify the global generation lock blocks concurrent generation across different models."""
        c, _ = client
        server._generation_lock.acquire()
        with patch("src.api.server._run_embedding_pipeline") as mock_pipeline:
            resp = c.post(
                "/embeddings/generate",
                json=_generation_payload("vit_triplet"),
            )
        assert resp.status_code == 503
        mock_pipeline.assert_not_called()

    def test_missing_model_name_returns_422(self, client):
        c, _ = client
        resp = c.post("/embeddings/generate", json={})
        assert resp.status_code == 422

    @pytest.mark.parametrize("snapshot_id", [
        "../../etc/passwd", "not-a-uuid", "", 7])
    def test_invalid_snapshot_id_returns_422_before_any_work(self, client, snapshot_id):
        c, _ = client
        with patch("src.api.server._run_embedding_pipeline") as mock_pipeline:
            resp = c.post(
                "/embeddings/generate",
                json={"model_name": "miew", "snapshot_id": snapshot_id})
        assert resp.status_code == 422
        mock_pipeline.assert_not_called()

    def test_snapshot_id_forwarded_to_pipeline(self, client):
        c, _ = client
        with patch("src.api.server._run_embedding_pipeline") as mock_pipeline:
            c.post("/embeddings/generate", json=_generation_payload())
        mock_pipeline.assert_called_once_with(
            "miew", "miew", SNAPSHOT_ID, "multimodal"
        )

    def test_legacy_annotations_field_cannot_replace_snapshot_id(self, client):
        c, _ = client
        with patch("src.api.server._run_embedding_pipeline") as mock_pipeline:
            response = c.post(
                "/embeddings/generate",
                json={"model_name": "miew", "annotations_version": "v2"},
            )
        assert response.status_code == 422
        mock_pipeline.assert_not_called()

    def test_dir_and_model_resolved_from_config(self, client):
        """When a config maps the model, dir and model names are forwarded separately."""
        c, mock_r = client
        mock_r._model_configs = {
            "miew": {"dir": "miew_arcface", "model": "initial_miew_arcface"},
        }
        with patch("src.api.server._run_embedding_pipeline") as mock_pipeline:
            c.post("/embeddings/generate", json=_generation_payload())
        mock_pipeline.assert_called_once_with(
            "miew_arcface", "initial_miew_arcface", SNAPSHOT_ID, "multimodal"
        )

    def test_baseline_directory_uses_baseline_generation(self, client):
        """A directory with only baseline views skips multimodal generation."""
        c, mock_r = client
        mock_r._model_configs = {
            "miew": {
                "dir": "miew",
                "model": "miew",
                "retrieval_mode": "baseline",
            },
        }
        with patch("src.api.server._run_embedding_pipeline") as mock_pipeline:
            response = c.post(
                "/embeddings/generate", json=_generation_payload()
            )

        assert response.status_code == 200
        mock_pipeline.assert_called_once_with(
            "miew", "miew", SNAPSHOT_ID, "baseline"
        )

    def test_shared_multimodal_view_selects_multimodal_generation(self, client):
        """One shared multimodal view makes the directory build multimodal files."""
        c, mock_r = client
        mock_r._model_configs = {
            "image": {
                "dir": "shared",
                "model": "trained",
                "retrieval_mode": "baseline",
            },
            "fused": {
                "dir": "shared",
                "model": "trained",
                "retrieval_mode": "multimodal",
            },
        }
        with patch("src.api.server._run_embedding_pipeline") as mock_pipeline:
            response = c.post(
                "/embeddings/generate", json=_generation_payload("image")
            )

        assert response.status_code == 200
        mock_pipeline.assert_called_once_with(
            "shared", "trained", SNAPSHOT_ID, "multimodal"
        )

    def test_live_generation_rejects_psl_views(self, client):
        """The server cannot publish stale PSL files after rebuilding base data."""
        c, mock_r = client
        mock_r._model_configs = {
            "psl": {
                "dir": "shared",
                "model": "trained",
                "retrieval_mode": "psl",
            },
        }
        with patch("src.api.server._run_embedding_pipeline") as mock_pipeline:
            response = c.post(
                "/embeddings/generate", json=_generation_payload("psl")
            )

        assert response.status_code == 500
        assert "cannot safely rebuild" in response.json()["detail"]
        mock_pipeline.assert_not_called()

    def test_success_reloads_every_view_sharing_the_published_directory(
            self, client):
        """Hot reload occurs only after publication and covers shared views."""
        c, mock_r = client
        mock_r._model_configs = {
            "image": {
                "dir": "shared",
                "model": "trained",
                "retrieval_mode": "baseline",
            },
            "fused": {
                "dir": "shared",
                "model": "trained",
                "retrieval_mode": "multimodal",
            },
            "other": {
                "dir": "other",
                "model": "other-trained",
                "retrieval_mode": "baseline",
            },
        }
        with patch("src.api.server._run_embedding_pipeline"):
            response = c.post(
                "/embeddings/generate", json=_generation_payload("image")
            )

        assert response.status_code == 200
        assert mock_r._load_model_data.call_args_list == [
            call("image"),
            call("fused"),
        ]

    def test_failed_publication_does_not_reload_retriever(self, client):
        """A publication error leaves the long-lived in-memory model untouched."""
        c, mock_r = client
        mock_r._model_configs = {
            "miew": {
                "dir": "miew",
                "model": "trained",
                "retrieval_mode": "baseline",
            },
        }
        with patch(
                "src.api.server._run_embedding_pipeline",
                side_effect=RuntimeError("publish failed"),
        ):
            response = c.post(
                "/embeddings/generate", json=_generation_payload()
            )

        assert response.status_code == 500
        mock_r._load_model_data.assert_not_called()

    def test_required_reload_failure_is_not_reported_as_success(self, client):
        """A published generation is not healthy until required reload passes."""
        c, mock_r = client
        mock_r._model_configs = {
            "miew": {
                "dir": "miew",
                "model": "trained",
                "retrieval_mode": "baseline",
                "required": True,
            },
        }
        mock_r._load_model_data.return_value = _model_status(
            "miew",
            "failed",
            error={
                "code": "artifact_incompatible",
                "message": "Required model data is incompatible.",
            },
        )

        with patch("src.api.server._run_embedding_pipeline"):
            response = c.post(
                "/embeddings/generate", json=_generation_payload()
            )

        assert response.status_code == 500
        assert response.json()["detail"] == (
            "Embedding generation failed: "
            "A required model failed to reload after generation."
        )


class TestServerPipelineAdapter:
    """Verify that server generation delegates to the shared CLI coordinator."""

    @patch("src.api.server.load_inference_snapshot")
    @patch("src.api.server.time.time", return_value=1_000.0)
    @patch("src.api.server._embeddings.run_pipeline")
    def test_snapshot_generation_validates_and_delegates_to_shared_pipeline(
            self, mock_run_pipeline, _mock_time, mock_load_snapshot):
        """Server generation passes one validated snapshot to the coordinator."""
        mock_run_pipeline.return_value = MagicMock(generation_id="generation-id")
        source_snapshot = mock_load_snapshot.return_value

        result = server._run_embedding_pipeline(
            "embedding-set", "trained-model", SNAPSHOT_ID, "multimodal"
        )

        assert result is mock_run_pipeline.return_value
        mock_load_snapshot.assert_called_once_with(SNAPSHOT_ID)
        positional = mock_run_pipeline.call_args.args
        assert positional[:2] == ("trained-model", "embedding-set")
        pipeline_args = positional[2]
        assert positional[3] == 1_000.0
        assert pipeline_args.update_embeddings is True
        assert pipeline_args.annotations_version is None
        assert pipeline_args.csv_dir is None
        assert pipeline_args.use_multimodal is True
        assert pipeline_args.use_psl is False
        assert pipeline_args.stacked is False
        assert mock_run_pipeline.call_args.kwargs == {
            "source_snapshot": source_snapshot
        }

    @patch("src.api.server.load_inference_snapshot")
    @patch("src.api.server.time.time", return_value=2_000.0)
    @patch("src.api.server._embeddings.run_pipeline")
    def test_baseline_snapshot_generation_reuses_pipeline_configuration(
            self, mock_run_pipeline, _mock_time, mock_load_snapshot):
        """Baseline snapshot generation retains the shared mode adapter."""
        mock_run_pipeline.return_value = MagicMock(generation_id="generation-id")

        server._run_embedding_pipeline(
            "embedding-set", "trained-model", SNAPSHOT_ID, "baseline"
        )

        mock_load_snapshot.assert_called_once_with(SNAPSHOT_ID)
        pipeline_args = mock_run_pipeline.call_args.args[2]
        assert pipeline_args.update_embeddings is True
        assert pipeline_args.annotations_version is None
        assert pipeline_args.use_multimodal is False
        assert mock_run_pipeline.call_args.kwargs == {
            "source_snapshot": mock_load_snapshot.return_value
        }

    @patch(
        "src.api.server.load_inference_snapshot",
        side_effect=ValueError("invalid snapshot"),
    )
    @patch("src.api.server._embeddings.run_pipeline")
    def test_invalid_snapshot_stops_before_generation_preparation(
            self, mock_run_pipeline, _mock_load_snapshot):
        """Unvalidated request data cannot reach the artifact builders."""
        with pytest.raises(ValueError, match="invalid snapshot"):
            server._run_embedding_pipeline(
                "embedding-set", "trained-model", SNAPSHOT_ID, "baseline"
            )

        mock_run_pipeline.assert_not_called()


# Verify generation lock release ensures the endpoint remains usable after runs.

class TestGenerationLockRelease:
    def test_lock_released_after_successful_run(self, client):
        c, _ = client
        with patch("src.api.server._run_embedding_pipeline"):
            resp = c.post("/embeddings/generate", json=_generation_payload())
        assert resp.status_code == 200
        assert not server._generation_lock.locked()

    def test_lock_released_after_pipeline_error(self, client):
        """A failing pipeline must not wedge the endpoint.

        Without the `finally`, one raised exception would leave the lock held
        and every later request would 503 forever.
        """
        c, _ = client
        with patch("src.api.server._run_embedding_pipeline",
                   side_effect=RuntimeError("disk full")):
            resp = c.post("/embeddings/generate", json=_generation_payload())
        assert resp.status_code == 500
        assert not server._generation_lock.locked()

    def test_second_request_accepted_after_first_fails(self, client):
        """The endpoint recovers: a failed run does not poison the next one."""
        c, _ = client
        with patch("src.api.server._run_embedding_pipeline",
                   side_effect=RuntimeError("disk full")):
            c.post("/embeddings/generate", json=_generation_payload())

        with patch("src.api.server._run_embedding_pipeline"):
            resp = c.post("/embeddings/generate", json=_generation_payload())
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestSubprocessIsolationAndBackgroundJobs:
    """Verify subprocess fault isolation and non-blocking job status reporting."""

    @patch("src.api.server.load_inference_snapshot")
    @patch("src.api.server.subprocess.run")
    def test_subprocess_invocation_passes_snapshot_and_mode(
            self, mock_subproc, _mock_load_snapshot):
        """Production pipeline execution runs in an isolated child process."""
        mock_subproc.return_value = MagicMock(
            returncode=0, stdout="", stderr=""
        )
        server._run_embedding_pipeline(
            "embedding-set", "trained-model", SNAPSHOT_ID, "multimodal"
        )
        mock_subproc.assert_called_once()
        cmd = mock_subproc.call_args.args[0]
        assert "--model-name" in cmd
        assert "trained-model" in cmd
        assert "--embeddings-name" in cmd
        assert "embedding-set" in cmd
        assert "--snapshot-id" in cmd
        assert SNAPSHOT_ID in cmd
        assert "--use-multimodal" in cmd

    @patch("src.api.server.load_inference_snapshot")
    @patch("src.api.server.subprocess.run")
    def test_subprocess_sigill_raises_diagnostic_runtime_error(
            self, mock_subproc, _mock_load_snapshot):
        """A child killed by SIGILL (-4) raises RuntimeError without crashing server."""
        mock_subproc.return_value = MagicMock(
            returncode=-4, stdout="", stderr=""
        )
        with pytest.raises(RuntimeError, match="SIGILL"):
            server._run_embedding_pipeline(
                "embedding-set", "trained-model", SNAPSHOT_ID, "baseline"
            )

    def test_background_generation_and_status_polling(self, client):
        """Background generation returns immediately and updates /embeddings/status."""
        c, _ = client
        payload = {**_generation_payload(), "background": True}
        with patch("src.api.server._run_embedding_pipeline"):
            resp = c.post("/embeddings/generate", json=payload)
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "started"
            assert body.get("job_id")

            # Wait briefly for daemon thread completion.
            for _ in range(50):
                status_resp = c.get("/embeddings/status")
                assert status_resp.status_code == 200
                status_body = status_resp.json()
                if status_body["status"] in ("succeeded", "failed"):
                    break
                time.sleep(0.01)

            assert status_body["status"] == "succeeded"
            assert status_body["job_id"] == body["job_id"]
            assert not server._generation_lock.locked()


class TestModelsSetting:
    """SHARK_MODELS entries with unsupported values stop startup instead of silently falling back."""

    def _parse(self, monkeypatch, models, **extra):
        monkeypatch.setenv("SHARK_MODELS", models)
        for name in ("SHARK_RETRIEVAL_MODE", "SHARK_RERANK"):
            monkeypatch.delenv(name, raising=False)
        for name, value in extra.items():
            monkeypatch.setenv(name, value)
        with patch.object(server, "_resolve_manifest_model", return_value=None):
            return server._parse_models_env()

    def test_full_entry_is_parsed(self, monkeypatch):
        (spec,) = self._parse(monkeypatch, "Demo:baseline:none:demo-set:demo-ckpt")
        assert (spec.display_name, spec.retrieval_mode, spec.rerank,
                spec.embedding_set, spec.checkpoint) == (
                    "Demo", "baseline", "none", "demo-set", "demo-ckpt")

    @pytest.mark.parametrize("models, named", [
        ("Demo:fusion:none:set:ckpt", "MODE"),
        ("Demo:baseline:sometimes:set:ckpt", "RERANK"),
    ])
    def test_unknown_mode_or_rerank_stops_startup(self, monkeypatch, models, named):
        with pytest.raises(ValueError, match=f"Unknown {named}.*use one of"):
            self._parse(monkeypatch, models)

    def test_mode_without_rerank_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="MODE without RERANK"):
            self._parse(monkeypatch, "Demo:baseline")

    @pytest.mark.parametrize("setting, value", [
        ("SHARK_RETRIEVAL_MODE", "fusion"), ("SHARK_RERANK", "sometimes")])
    def test_unknown_global_setting_stops_startup(self, monkeypatch, setting, value):
        with pytest.raises(ValueError, match=f"Unknown {setting}"):
            self._parse(monkeypatch, "Demo", **{setting: value})

    def test_valid_values_come_from_the_shared_contract(self):
        assert server._VALID_MODES == {"baseline", "psl", "multimodal", "stacked"}
        assert server._VALID_RERANKS == {"none", "kreciprocal"}

    def test_server_runs_the_app_object_so_the_module_is_imported_once(self):
        with patch("uvicorn.run") as run:
            server.main()
        assert run.call_args.args[0] is server.app
