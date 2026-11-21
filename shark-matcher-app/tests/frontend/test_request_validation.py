"""Real-HTTP tests that malformed API requests get a 400, never a server error."""

import http
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import frontend.project
import frontend.server
import frontend.user
from tests.frontend import test_server as server_tests

# Every authenticated POST route; an empty body must never produce a 5xx.
POST_ENDPOINTS = (
    "data/export", "data/stats", "embeddings/status",
    "user/get", "user/filter", "user/post", "user/delete",
    "individual/get", "individual/filter", "individual/post", "individual/delete",
    "individual/merge", "individual/unmatch", "individual/reference",
    "media/get", "media/filter", "media/post", "media/delete", "media/save",
    "media/upload_batch", "object/get", "object/filter", "object/post", "object/delete",
    "action/filter", "proposal/get", "proposal/filter", "proposal/post", "proposal/delete",
    "match/get", "match/filter",
)

# Endpoint, body, and the field the 400 message must name.
MISSING_FIELD_CASES = (
    ("match/filter", {"image_name": "x"}, "method"),
    ("data/export", {}, "content"),
    ("data/export", {"content": "annotations"}, "format"),
    ("object/get", {}, "table"),
    ("object/filter", {}, "table"),
    ("object/filter", {"table": "label"}, "individual"),
    ("object/post", {"table": "label", "name": "x"}, "individual"),
    ("object/delete", {"table": "label"}, "id"),
    ("individual/merge", {}, "major_id"),
    ("individual/unmatch", {}, "id"),
    ("individual/reference", {}, "id"),
    ("media/post", {}, "name"),
    ("user/delete", {}, "id"),
    ("data/export", {"content": "nope", "format": "snapshot"}, "content"),
    ("data/export", {"content": "annotations", "format": "nope"}, "format"),
    ("media/upload_batch", {"items": [{"name": "a"}]}, "hash"),
    ("media/upload_batch", {"items": [{"name": "a", "hash": "h", "labels": ["x"]}]}, "labels"),
)

# Endpoint, body, and the field the 400 message must name for a wrongly shaped value.
INVALID_VALUE_CASES = (
    ("individual/filter", {"sex": "abc"}, "sex"),
    ("individual/filter", {"qualities": [1, 2]}, "qualities"),
    ("individual/get", {"ids": ["abc"]}, "ids"),
    ("individual/unmatch", {"id": "abc"}, "id"),
    ("individual/merge", {"major_id": {"x": 1}, "minor_id": 2}, "major_id"),
    ("object/get", {"table": 5}, "table"),
    ("media/filter", {"locations": {"x": 1}}, "locations"),
    ("action/filter", {"keywords": [1, 2]}, "keywords"),
    ("action/filter", {"original": {"x": 1}}, "original"),
    ("action/filter", {"final": "abc"}, "final"),
    ("proposal/filter", {"page": "many"}, "page"),
    ("proposal/filter", {"original": {"x": 1}}, "original"),
    ("proposal/filter", {"final": "abc"}, "final"),
    ("proposal/post", {"action": "create", "original": {"x": 1}, "final": {"id": 1}}, "original"),
    ("proposal/post", {"action": "create", "original": "", "final": {"id": 1}}, "original"),
    ("proposal/post", {"action": "create", "original": None, "final": 123}, "final"),
    ("media/post", {"name": "abc"}, "hash"),
)

# Endpoint and body naming an identifier that no record has.
NOT_FOUND_CASES = (
    ("user/delete", {"id": 999999}),
    ("individual/post", {"id": 999999, "name": "x"}),
    ("individual/unmatch", {"id": 999999}),
    ("individual/reference", {"id": 999999}),
    ("proposal/post", {"id": 999999, "status": "approved"}),
)

# Body keys the handlers read; each is sent alone with a wrongly shaped value.
FUZZ_KEYS = (
    "id", "ids", "name", "names", "table", "individual", "source", "media", "length", "sex",
    "hash", "labels", "tags", "type", "time", "method", "content", "format", "major_id",
    "minor_id", "image_name", "preview", "items", "keywords", "locations", "qualities",
)
FUZZ_VALUES = ({"x": 1}, [1, 2], "abc", -1, 999999, True, 7.5, "")


class TestRequestValidation(unittest.TestCase):
    """Send malformed requests to a production-handler server and check status codes."""

    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        database_path = Path(cls._directory.name) / "validation.db"
        cls.user, derivative = server_tests.TestServerFrontend._initialize_auth_database(
            database_path, "validation_owner", "validation-password")
        cls._patch = patch.object(frontend.server.app_constants, "DB_PATH", database_path)
        cls._patch.start()
        frontend.user._AUTHENTICATION_LIMITER.clear()
        cls.server, cls.thread = server_tests.TestServerFrontend._start_test_server()
        status, _, response = server_tests.TestServerFrontend._post_json(
            cls.server, "user/login",
            {"username": cls.user.username, "password": derivative})
        assert status == http.HTTPStatus.OK
        cls.token = response["content"]["token"]

    @classmethod
    def tearDownClass(cls):
        server_tests.TestServerFrontend._stop_test_server(cls.server, cls.thread)
        frontend.user._AUTHENTICATION_LIMITER.clear()
        cls._patch.stop()
        cls._directory.cleanup()

    def _post(self, endpoint, payload):
        body = {"user_id": self.user.id, "user_username": self.user.username,
                "user_token": self.token, **payload}
        status, _, response = server_tests.TestServerFrontend._post_json(self.server, endpoint, body)
        return status, response

    def test_empty_bodies_never_cause_server_errors(self):
        """No route turns a missing field into a 5xx."""
        failures = []
        for endpoint in POST_ENDPOINTS:
            status, _ = self._post(endpoint, {})
            if status >= 500:
                failures.append((endpoint, status))
        self.assertEqual(failures, [])

    def test_no_single_wrong_value_causes_a_server_error(self):
        """Any one key holding the wrong shape, or an unknown id, is never a 5xx."""
        failures = []
        for endpoint in POST_ENDPOINTS:
            for key in FUZZ_KEYS:
                for value in FUZZ_VALUES:
                    status, _ = self._post(endpoint, {key: value})
                    if status >= 500:
                        failures.append((endpoint, key, value, status))
        self.assertEqual(failures, [])

    def test_wrongly_shaped_value_is_a_400_that_names_the_field(self):
        """The 400 message says which field holds the impossible value."""
        for endpoint, payload, field in INVALID_VALUE_CASES:
            with self.subTest(endpoint=endpoint, field=field):
                status, response = self._post(endpoint, payload)
                self.assertEqual(status, http.HTTPStatus.BAD_REQUEST)
                self.assertIn(field, response["message"])

    def test_unknown_identifier_is_a_404(self):
        """An identifier that no record has is reported as not found."""
        for endpoint, payload in NOT_FOUND_CASES:
            with self.subTest(endpoint=endpoint):
                status, _ = self._post(endpoint, payload)
                self.assertEqual(status, http.HTTPStatus.NOT_FOUND)

    def test_recent_embeddings_update_is_429_with_a_retry_after_header(self):
        """The cooldown reaches the HTTP client as 429 plus Retry-After; an owner may force it."""
        stats = {"snapshot_id": "33333333-3333-4333-8333-333333333333",
                 "embeddings_metadata": {"start_time": time.time() - 600}}
        body = {"user_id": self.user.id, "user_username": self.user.username,
                "user_token": self.token}
        harness = server_tests.TestServerFrontend
        frontend.project._reset_embeddings_update_job()
        try:
            with patch("frontend.inference_client.get_stats", return_value=stats), \
                    patch("backend.inference_snapshot.snapshot_created_at",
                          return_value=time.time() - 600), \
                    patch.object(frontend.project, "_run_update_embeddings_worker"):
                status, headers, response = harness._post_json(
                    self.server, "embeddings/update", body)
                self.assertEqual(status, http.HTTPStatus.TOO_MANY_REQUESTS)
                self.assertAlmostEqual(int(headers["Retry-After"]), 3000, delta=10)
                self.assertIn("last hour", response["message"])
                status, _, _ = harness._post_json(
                    self.server, "embeddings/update", {**body, "force": True})
                self.assertEqual(status, http.HTTPStatus.OK)
        finally:
            frontend.project._reset_embeddings_update_job()

    def test_missing_required_field_is_a_400_that_names_it(self):
        """The 400 message says which field is missing."""
        for endpoint, payload, field in MISSING_FIELD_CASES:
            with self.subTest(endpoint=endpoint, field=field):
                status, response = self._post(endpoint, payload)
                self.assertEqual(status, http.HTTPStatus.BAD_REQUEST)
                self.assertIn(field, response["message"])

    def test_proposal_and_action_filters_accept_empty_and_id_object_filters(self):
        """History and proposal search accept empty strings and integer IDs for original and final."""
        base_payload = {
            "min_time": "", "max_time": "", "object": "", "action": "",
            "original": "", "final": "", "author": "", "ids": "", "keywords": "",
            "paginate": True, "page": 1, "page_size": 50,
            "attribute": "time", "order": "descending",
        }
        endpoints = (
            ("action/filter", base_payload),
            ("proposal/filter", {**base_payload, "status": ""}),
        )
        for endpoint, payload in endpoints:
            for original, final in (("", ""), (None, None), (42, "42"), (-1, "-1")):
                with self.subTest(endpoint=endpoint, original=original, final=final):
                    status, _ = self._post(endpoint, {**payload, "original": original, "final": final})
                    self.assertEqual(status, http.HTTPStatus.OK)


if __name__ == "__main__":
    unittest.main()
