"""Tests for frontend.inference_client using mocked HTTP connections."""

import json
import unittest
from unittest.mock import MagicMock, patch

import pytest

from core import app_constants
import frontend.common
import frontend.inference_client as client

SNAPSHOT_ID = "11111111-1111-4111-8111-111111111111"
SERVICE_API_KEY = "contract-test-service-key-with-32-characters"

class ConfiguredInferenceClientTest(unittest.TestCase):
    """Provide the required service credential to transport unit tests."""

    def setUp(self):
        """Configure one valid credential for each isolated test case."""
        super().setUp()
        key_patch = patch.object(
            app_constants, "INFERENCE_API_KEY", SERVICE_API_KEY, create=True
        )
        key_patch.start()
        self.addCleanup(key_patch.stop)


class TestHealth(ConfiguredInferenceClientTest):

    def test_returns_true_on_ok_status(self):
        conn = _make_conn(_make_response(200, {"status": "ok"}))
        with patch("http.client.HTTPConnection", return_value=conn):
            self.assertTrue(client.health())

    def test_returns_false_on_non_200(self):
        conn = _make_conn(_make_response(503, {}))
        with patch("http.client.HTTPConnection", return_value=conn):
            self.assertFalse(client.health())

    def test_returns_false_when_status_field_not_ok(self):
        conn = _make_conn(_make_response(200, {"status": "degraded"}))
        with patch("http.client.HTTPConnection", return_value=conn):
            self.assertFalse(client.health())

    def test_returns_false_on_connection_error(self):
        conn = MagicMock()
        conn.request.side_effect = ConnectionRefusedError("refused")
        with patch("http.client.HTTPConnection", return_value=conn):
            self.assertFalse(client.health())

    def test_uses_health_timeout(self):
        conn = _make_conn(_make_response(200, {"status": "ok"}))
        with patch("http.client.HTTPConnection", return_value=conn) as MockConn:
            client.health()
            _, kwargs = MockConn.call_args
            self.assertEqual(kwargs["timeout"], client._HEALTH_TIMEOUT)

    def test_calls_get_health_endpoint(self):
        conn = _make_conn(_make_response(200, {"status": "ok"}))
        with patch("http.client.HTTPConnection", return_value=conn):
            client.health()
            conn.request.assert_called_once_with("GET", "/health")


class TestGetMatches(ConfiguredInferenceClientTest):

    _SUCCESS_BODY = {"ids": [1, 2, 3], "distances": [0.1, 0.5, 0.9], "labels": [10, 20, 30]}

    def test_returns_result_on_success(self):
        conn = _make_conn(_make_response(200, self._SUCCESS_BODY))
        with patch("http.client.HTTPConnection", return_value=conn):
            result = client.get_matches(image_id=7, retrieval_method=0, model="miew")
        self.assertEqual(result["ids"], [1, 2, 3])
        self.assertEqual(result["distances"], [0.1, 0.5, 0.9])
        self.assertEqual(result["labels"], [10, 20, 30])

    def test_reports_failure_instead_of_empty_results(self):
        conn = _make_conn(_make_response(500, {}))
        with patch("http.client.HTTPConnection", return_value=conn):
            with self.assertRaises(frontend.common.APIError) as error:
                client.get_matches(image_id=7, retrieval_method=0, model="miew")
        self.assertEqual(error.exception.status, 502)

    def test_reports_connection_error_instead_of_empty_results(self):
        conn = MagicMock()
        conn.request.side_effect = OSError("unreachable")
        with patch("http.client.HTTPConnection", return_value=conn):
            with self.assertRaises(frontend.common.APIError) as error:
                client.get_matches(image_id=7, retrieval_method=0, model="miew")
        self.assertEqual(error.exception.status, 502)

    def test_posts_to_match_endpoint(self):
        conn = _make_conn(_make_response(200, self._SUCCESS_BODY))
        with patch("http.client.HTTPConnection", return_value=conn):
            client.get_matches(image_id=7, retrieval_method=2, model="vit_triplet")
        self.assertEqual(_request_method(conn), "POST")
        self.assertEqual(_request_path(conn), "/match")

    def test_sends_correct_body_fields(self):
        conn = _make_conn(_make_response(200, self._SUCCESS_BODY))
        with patch("http.client.HTTPConnection", return_value=conn):
            client.get_matches(image_id=42, retrieval_method=2, model="vit_triplet")
        body = _request_body(conn)
        self.assertEqual(body["image_id"], 42)
        self.assertEqual(body["retrieval_method"], 2)
        self.assertEqual(body["model"], "vit_triplet")

    def test_omits_model_when_none(self):
        conn = _make_conn(_make_response(200, self._SUCCESS_BODY))
        with patch("http.client.HTTPConnection", return_value=conn):
            client.get_matches(image_id=1, retrieval_method=0)
        body = _request_body(conn)
        self.assertNotIn("model", body)

    def test_default_retrieval_method_is_zero(self):
        conn = _make_conn(_make_response(200, self._SUCCESS_BODY))
        with patch("http.client.HTTPConnection", return_value=conn):
            client.get_matches(image_id=1)
        body = _request_body(conn)
        self.assertEqual(body["retrieval_method"], 0)


class TestGetStats(ConfiguredInferenceClientTest):

    _SUCCESS_BODY = {
        "loaded_models": ["miew"],
        "embeddings_metadata": {"start_time": 1000, "update_time": 200},
        "filenames": ["shark_001.jpeg", "shark_002.jpeg"],
        "model_statuses": [
            {
                "display_name": "miew",
                "embedding_set": "miew",
                "required": True,
                "state": "ready",
                "error": None,
            }
        ],
        "generation_id": "22222222-2222-4222-8222-222222222222",
        "snapshot_id": SNAPSHOT_ID,
    }

    def test_returns_response_dict_on_success(self):
        conn = _make_conn(_make_response(200, self._SUCCESS_BODY))
        with patch("http.client.HTTPConnection", return_value=conn):
            result = client.get_stats()
        self.assertEqual(result["loaded_models"], ["miew"])
        self.assertIn("shark_001.jpeg", result["filenames"])
        self.assertEqual(result["embeddings_metadata"]["start_time"], 1000)
        self.assertEqual(
            result["model_statuses"], self._SUCCESS_BODY["model_statuses"]
        )
        self.assertEqual(
            result["generation_id"], self._SUCCESS_BODY["generation_id"]
        )
        self.assertEqual(result["snapshot_id"], SNAPSHOT_ID)

    def test_returns_none_on_non_200(self):
        conn = _make_conn(_make_response(503, {}))
        with patch("http.client.HTTPConnection", return_value=conn):
            self.assertIsNone(client.get_stats())

    def test_returns_none_on_connection_error(self):
        conn = MagicMock()
        conn.request.side_effect = OSError("timeout")
        with patch("http.client.HTTPConnection", return_value=conn):
            self.assertIsNone(client.get_stats())

    def test_calls_get_stats_endpoint(self):
        conn = _make_conn(_make_response(200, self._SUCCESS_BODY))
        with patch("http.client.HTTPConnection", return_value=conn):
            client.get_stats()
        self.assertEqual(_request_method(conn), "GET")
        self.assertEqual(_request_path(conn), "/stats")

    def test_sends_no_body(self):
        conn = _make_conn(_make_response(200, self._SUCCESS_BODY))
        with patch("http.client.HTTPConnection", return_value=conn):
            client.get_stats()
        # The GET to /stats should not include a request body.
        self.assertIsNone(_request_body(conn))


class TestGenerateEmbeddings(ConfiguredInferenceClientTest):

    def test_returns_true_on_success(self):
        conn = _make_conn(_make_response(200, {"status": "ok"}))
        with patch("http.client.HTTPConnection", return_value=conn):
            self.assertTrue(client.generate_embeddings("miew", SNAPSHOT_ID))

    def test_returns_false_on_non_200(self):
        conn = _make_conn(_make_response(500, {}))
        with patch("http.client.HTTPConnection", return_value=conn):
            self.assertFalse(client.generate_embeddings("miew", SNAPSHOT_ID))

    def test_returns_false_on_connection_error(self):
        conn = MagicMock()
        conn.request.side_effect = OSError("timeout")
        with patch("http.client.HTTPConnection", return_value=conn):
            self.assertFalse(client.generate_embeddings("miew", SNAPSHOT_ID))

    def test_posts_to_embeddings_generate_endpoint(self):
        conn = _make_conn(_make_response(200, {"status": "ok"}))
        with patch("http.client.HTTPConnection", return_value=conn):
            client.generate_embeddings("miew", SNAPSHOT_ID)
        self.assertEqual(_request_method(conn), "POST")
        self.assertEqual(_request_path(conn), "/embeddings/generate")

    def test_sends_model_name(self):
        conn = _make_conn(_make_response(200, {"status": "ok"}))
        with patch("http.client.HTTPConnection", return_value=conn):
            client.generate_embeddings("miew", SNAPSHOT_ID)
        body = _request_body(conn)
        self.assertEqual(body["model_name"], "miew")
        self.assertEqual(body["snapshot_id"], SNAPSHOT_ID)
        self.assertNotIn("annotations_version", body)

    def test_sends_only_snapshot_generation_fields(self):
        conn = _make_conn(_make_response(200, {"status": "ok"}))
        with patch("http.client.HTTPConnection", return_value=conn):
            client.generate_embeddings("miew", SNAPSHOT_ID)
        body = _request_body(conn)
        self.assertEqual(
            body,
            {"model_name": "miew", "snapshot_id": SNAPSHOT_ID},
        )

    def test_uses_embeddings_timeout(self):
        conn = _make_conn(_make_response(200, {"status": "ok"}))
        with patch("http.client.HTTPConnection", return_value=conn) as MockConn:
            client.generate_embeddings("miew", SNAPSHOT_ID)
            _, kwargs = MockConn.call_args
            self.assertEqual(kwargs["timeout"], client._EMBEDDINGS_TIMEOUT)

    def test_start_embeddings_generation_sends_background_true(self):
        conn = _make_conn(_make_response(200, {"status": "started", "job_id": "abc"}))
        with patch("http.client.HTTPConnection", return_value=conn) as MockConn:
            result = client.start_embeddings_generation("miew", SNAPSHOT_ID)
            _, kwargs = MockConn.call_args
            self.assertEqual(kwargs["timeout"], client._STATUS_TIMEOUT)
        body = _request_body(conn)
        self.assertEqual(
            body,
            {"model_name": "miew", "snapshot_id": SNAPSHOT_ID, "background": True},
        )
        self.assertEqual(result["status"], "started")

    def test_get_embeddings_status_calls_status_endpoint(self):
        conn = _make_conn(_make_response(200, {"status": "running", "job_id": "abc"}))
        with patch("http.client.HTTPConnection", return_value=conn) as MockConn:
            result = client.get_embeddings_status()
            _, kwargs = MockConn.call_args
            self.assertEqual(kwargs["timeout"], client._STATUS_TIMEOUT)
        self.assertEqual(_request_method(conn), "GET")
        self.assertEqual(_request_path(conn), "/embeddings/status")
        self.assertEqual(result["status"], "running")


class TestConnectionBehaviour(ConfiguredInferenceClientTest):
    """Verify host/port parsing from INFERENCE_SERVICE_URL and connection closing."""

    def test_connection_closed_on_success(self):
        conn = _make_conn(_make_response(200, {"status": "ok"}))
        with patch("http.client.HTTPConnection", return_value=conn):
            client.health()
        conn.close.assert_called_once()

    def test_connection_closed_on_error(self):
        conn = MagicMock()
        conn.request.side_effect = OSError("refused")
        with patch("http.client.HTTPConnection", return_value=conn):
            client.health()
        conn.close.assert_called_once()

    def test_json_content_type_header_sent(self):
        conn = _make_conn(_make_response(200, {"ids": [], "distances": [], "labels": []}))
        with patch("http.client.HTTPConnection", return_value=conn):
            client.get_matches(image_id=1, retrieval_method=0, model="miew")
        _, kwargs = conn.request.call_args
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")

    def test_custom_inference_url_parsed(self):
        conn = _make_conn(_make_response(200, {"status": "ok"}))
        with patch("http.client.HTTPConnection", return_value=conn) as MockConn:
            with patch.object(
                __import__("core.app_constants", fromlist=["app_constants"]),
                "INFERENCE_SERVICE_URL",
                "http://inference-host:9090",
            ):
                client.health()
            args, kwargs = MockConn.call_args
            self.assertEqual(args[0], "inference-host")
            self.assertEqual(args[1], 9090)

@pytest.mark.parametrize('status, body, expected', [
    (400, {}, 400), (503, {}, 503), (500, {}, 502),
    (200, {'ids': [1], 'distances': []}, 502), (200, {'ids': 'bad'}, 502),
])
def test_match_failure_status_and_schema(status, body, expected, monkeypatch):
    """Bad upstream status or schema cannot masquerade as empty success."""
    monkeypatch.setattr(app_constants, 'INFERENCE_API_KEY', SERVICE_API_KEY)
    connection = _make_conn(_make_response(status, body))
    with patch('http.client.HTTPConnection', return_value=connection):
        with pytest.raises(frontend.common.APIError) as error:
            client.get_matches(1)
    assert error.value.status == expected


def test_match_timeout_is_gateway_timeout(monkeypatch):
    """Timeouts are distinct from other upstream failures."""
    monkeypatch.setattr(app_constants, 'INFERENCE_API_KEY', SERVICE_API_KEY)
    connection = _make_conn(_make_response())
    connection.request.side_effect = TimeoutError
    with patch('http.client.HTTPConnection', return_value=connection):
        with pytest.raises(frontend.common.APIError) as error:
            client.get_matches(1)
    assert error.value.status == 504


def test_missing_credential_names_the_environment_variable(monkeypatch, capsys):
    """An unconfigured credential tells operators which variable to set."""
    monkeypatch.setattr(app_constants, 'INFERENCE_API_KEY', '')
    with patch('http.client.HTTPConnection') as connection_class:
        assert client.get_stats() is None
    connection_class.assert_not_called()
    assert 'SHARK_INFERENCE_API_KEY' in capsys.readouterr().out

def _make_response(status=200, body=None):
    """Build a mock HTTPResponse with the given status and JSON body."""
    mock_resp = MagicMock()
    mock_resp.status = status
    encoded = json.dumps(body or {}).encode("utf-8")
    mock_resp.read.return_value = encoded
    return mock_resp


def _make_conn(response):
    """Build a mock HTTPConnection that returns *response* from getresponse()."""
    mock_conn = MagicMock()
    mock_conn.getresponse.return_value = response
    return mock_conn


def _request_method(mock_conn):
    """Extract the HTTP method from the most recent conn.request(...) call."""
    return mock_conn.request.call_args[0][0]


def _request_path(mock_conn):
    """Extract the path from the most recent conn.request(...) call."""
    return mock_conn.request.call_args[0][1]


def _request_body(mock_conn):
    """Extract and decode the JSON body sent via conn.request(...)."""
    call_args = mock_conn.request.call_args
    body = call_args[1].get("body")
    return json.loads(body) if body is not None else None

if __name__ == "__main__":
    unittest.main()
