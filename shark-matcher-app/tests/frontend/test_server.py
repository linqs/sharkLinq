"""Tests for application server routing, request parsing, and startup checks."""

import base64
import contextlib
import hashlib
import http
import http.client
import http.server
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import pytest

import backend.db.user
import db.database
import frontend.handlers
import frontend.server
import frontend.user


class TestServerFrontend(unittest.TestCase):
    """Exercise application HTTP server behavior."""
    def test_check_database_rejects_legacy_token_schema(self):
        """Startup rejects expiry-less sessions and directs the operator to migrate."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "legacy.db"
            with sqlite3.connect(database_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE Individuals (id INTEGER PRIMARY KEY);
                    CREATE TABLE Tokens (
                        id INTEGER PRIMARY KEY,
                        token TEXT NOT NULL,
                        user INTEGER NOT NULL
                    );
                    """
                )

            with patch.object(frontend.server.app_constants, "DB_PATH", database_path):
                with self.assertRaises(SystemExit):
                    frontend.server._check_database()

    def test_check_database_accepts_current_token_schema(self):
        """Startup accepts an initialized database with persisted session expiry."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "current.db"
            with sqlite3.connect(database_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE Individuals (id INTEGER PRIMARY KEY);
                    CREATE TABLE Tokens (
                        id INTEGER PRIMARY KEY,
                        token TEXT NOT NULL,
                        user INTEGER NOT NULL,
                        expires INTEGER NOT NULL
                    );
                    """
                )

            with patch.object(frontend.server.app_constants, "DB_PATH", database_path):
                frontend.server._check_database()

    def test_http_authentication_survives_restart_and_revokes_sessions(self):
        """Real HTTP login persists across restart, logout, and expiry cleanup."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "lifecycle.db"
            user, password_derivative = self._initialize_auth_database(
                database_path, "lifecycle_owner", "lifecycle-password"
            )
            credentials = {
                "user_id": user.id,
                "user_username": user.username,
            }

            with patch.object(frontend.server.app_constants, "DB_PATH", database_path):
                frontend.server._check_database()
                frontend.user._AUTHENTICATION_LIMITER.clear()
                first_server, first_thread = self._start_test_server()
                try:
                    status, _, response = self._post_json(
                        first_server,
                        "user/login",
                        {
                            "username": user.username,
                            "password": password_derivative,
                        },
                    )
                    self.assertEqual(status, http.HTTPStatus.OK)
                    first_token = response["content"]["token"]
                finally:
                    self._stop_test_server(first_server, first_thread)

                restarted_server, restarted_thread = self._start_test_server()
                try:
                    active_credentials = {
                        **credentials,
                        "user_token": first_token,
                        "id": user.id,
                        "username": user.username,
                    }
                    status, _, _ = self._post_json(
                        restarted_server, "user/get", active_credentials
                    )
                    self.assertEqual(status, http.HTTPStatus.OK)

                    status, _, _ = self._post_json(
                        restarted_server,
                        "user/logout",
                        {
                            "user_username": user.username,
                            "user_token": first_token,
                        },
                    )
                    self.assertEqual(status, http.HTTPStatus.OK)
                    status, _, _ = self._post_json(
                        restarted_server, "user/get", active_credentials
                    )
                    self.assertEqual(status, http.HTTPStatus.UNAUTHORIZED)

                    status, _, response = self._post_json(
                        restarted_server,
                        "user/login",
                        {
                            "username": user.username,
                            "password": password_derivative,
                        },
                    )
                    self.assertEqual(status, http.HTTPStatus.OK)
                    expiring_token = response["content"]["token"]
                    expiring_digest = backend.db.user._session_token_digest(
                        expiring_token
                    )
                    with sqlite3.connect(database_path) as connection:
                        connection.execute(
                            "UPDATE Tokens SET expires = ? WHERE token = ?",
                            (int(time.time()) - 1, expiring_digest),
                        )

                    expired_credentials = {
                        **credentials,
                        "user_token": expiring_token,
                        "id": user.id,
                        "username": user.username,
                    }
                    status, _, _ = self._post_json(
                        restarted_server, "user/get", expired_credentials
                    )
                    self.assertEqual(status, http.HTTPStatus.UNAUTHORIZED)
                    with sqlite3.connect(database_path) as connection:
                        stored_session = connection.execute(
                            "SELECT id FROM Tokens WHERE token = ?",
                            (expiring_digest,),
                        ).fetchone()
                    self.assertIsNone(stored_session)
                finally:
                    self._stop_test_server(restarted_server, restarted_thread)
                    frontend.user._AUTHENTICATION_LIMITER.clear()

    def test_migrated_auth_database_starts_and_preserves_login(self):
        """A legacy database can migrate, start, and log in its existing user."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "legacy.db"
            user, password_derivative = self._initialize_auth_database(
                database_path, "migrated_owner", "migration-password"
            )
            with sqlite3.connect(database_path) as connection:
                connection.executescript(
                    """
                    DROP TABLE Tokens;
                    CREATE TABLE Tokens (
                        id INTEGER PRIMARY KEY,
                        token TEXT NOT NULL,
                        user INTEGER REFERENCES Users(id) ON DELETE CASCADE NOT NULL
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO Tokens (id, token, user) VALUES (?, ?, ?)",
                    (1, "legacy-session", user.id),
                )

            self.assertTrue(
                db.database.migrate_authentication_schema(database_path)
            )
            self.assertFalse(
                db.database.migrate_authentication_schema(database_path)
            )

            with patch.object(frontend.server.app_constants, "DB_PATH", database_path):
                frontend.server._check_database()
                frontend.user._AUTHENTICATION_LIMITER.clear()
                server, thread = self._start_test_server()
                try:
                    status, _, response = self._post_json(
                        server,
                        "user/login",
                        {
                            "username": user.username,
                            "password": password_derivative,
                        },
                    )
                finally:
                    self._stop_test_server(server, thread)
                    frontend.user._AUTHENTICATION_LIMITER.clear()

            self.assertEqual(status, http.HTTPStatus.OK)
            self.assertEqual(response["content"]["username"], user.username)
            with sqlite3.connect(database_path) as connection:
                tokens = connection.execute(
                    "SELECT token, expires FROM Tokens"
                ).fetchall()
            self.assertEqual(len(tokens), 1)
            self.assertNotEqual(tokens[0][0], "legacy-session")

    def test_http_password_and_token_attempts_are_throttled(self):
        """Real HTTP password and token failures reach their shared limiter."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "throttling.db"
            user, password_derivative = self._initialize_auth_database(
                database_path, "throttled_owner", "throttling-password"
            )
            wrong_password = hashlib.sha256(b"wrong-password").hexdigest()

            with patch.object(frontend.server.app_constants, "DB_PATH", database_path):
                frontend.user._AUTHENTICATION_LIMITER.clear()
                server, thread = self._start_test_server()
                try:
                    for _ in range(frontend.user.AUTHENTICATION_ATTEMPT_LIMIT):
                        status, _, _ = self._post_json(
                            server,
                            "user/login",
                            {
                                "username": user.username,
                                "password": wrong_password,
                            },
                        )
                        self.assertEqual(status, http.HTTPStatus.OK)

                    status, headers, blocked_response = self._post_json(
                        server,
                        "user/login",
                        {
                            "username": user.username,
                            "password": password_derivative,
                        },
                    )
                    self.assertEqual(status, http.HTTPStatus.TOO_MANY_REQUESTS)
                    self.assertGreater(int(headers["Retry-After"]), 0)
                    self.assertNotIn(
                        password_derivative, json.dumps(blocked_response)
                    )
                    self.assertNotIn(wrong_password, json.dumps(blocked_response))

                    frontend.user._AUTHENTICATION_LIMITER.clear()
                    status, _, response = self._post_json(
                        server,
                        "user/login",
                        {
                            "username": user.username,
                            "password": password_derivative,
                        },
                    )
                    self.assertEqual(status, http.HTTPStatus.OK)
                    token = response["content"]["token"]
                    request = {
                        "id": user.id,
                        "username": user.username,
                        "user_id": user.id,
                        "user_username": user.username,
                        "user_token": "invalid-token",
                    }

                    for _ in range(frontend.user.AUTHENTICATION_ATTEMPT_LIMIT):
                        status, _, _ = self._post_json(
                            server, "user/get", request
                        )
                        self.assertEqual(status, http.HTTPStatus.UNAUTHORIZED)

                    status, headers, blocked_response = self._post_json(
                        server, "user/get", request
                    )
                    self.assertEqual(status, http.HTTPStatus.TOO_MANY_REQUESTS)
                    self.assertGreater(int(headers["Retry-After"]), 0)
                    self.assertNotIn("invalid-token", json.dumps(blocked_response))

                    # A valid session is still served while the bucket is full.
                    request["user_token"] = token
                    status, _, _ = self._post_json(server, "user/get", request)
                    self.assertEqual(status, http.HTTPStatus.OK)
                finally:
                    self._stop_test_server(server, thread)
                    frontend.user._AUTHENTICATION_LIMITER.clear()

    def test_fresh_login_reopens_image_requests_after_stale_cookie_failures(self):
        """Cookieless thumbnail requests lock the image bucket for bad cookies, never for a valid session."""
        def image_status(cookie_value=None):
            headers = {}
            if cookie_value:
                headers["Cookie"] = f"{frontend.user.IMAGE_COOKIE}={cookie_value}"
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                connection.request("GET", "/static/frontend-images/absent.jpg", headers=headers)
                response = connection.getresponse()
                response.read()
                return response.status
            finally:
                connection.close()

        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "stale-cookie.db"
            user, derivative = self._initialize_auth_database(
                database_path, "stale_owner", "stale-password")
            login = {"username": user.username, "password": derivative}

            with patch.object(frontend.server.app_constants, "DB_PATH", database_path):
                frontend.user._AUTHENTICATION_LIMITER.clear()
                server, thread = self._start_test_server()
                try:
                    _, _, response = self._post_json(server, "user/login", login)
                    for _ in range(frontend.user.AUTHENTICATION_ATTEMPT_LIMIT):
                        self.assertEqual(image_status(), http.HTTPStatus.UNAUTHORIZED)
                    self.assertEqual(image_status(), http.HTTPStatus.TOO_MANY_REQUESTS)
                    self.assertEqual(
                        image_status(response["content"]["token"]), http.HTTPStatus.NOT_FOUND)

                    _, _, response = self._post_json(server, "user/login", login)
                    self.assertEqual(image_status(), http.HTTPStatus.UNAUTHORIZED)
                finally:
                    self._stop_test_server(server, thread)
                    frontend.user._AUTHENTICATION_LIMITER.clear()

    def test_http_export_cannot_request_trusted_authentication_records(self):
        """A real browser export cannot promote itself to trusted dump mode."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            database_path = temporary_path / "export.db"
            export_path = temporary_path / "exports"
            user, password_derivative = self._initialize_auth_database(
                database_path, "export_owner", "export-password"
            )

            with patch.object(
                frontend.server.app_constants, "DB_PATH", database_path
            ), patch.object(
                frontend.server.app_constants, "OUT_DATA_DIR", export_path
            ):
                frontend.user._AUTHENTICATION_LIMITER.clear()
                server, thread = self._start_test_server()
                try:
                    captured_output = io.StringIO()
                    with contextlib.redirect_stdout(captured_output):
                        status, _, login_response = self._post_json(
                            server,
                            "user/login",
                            {
                                "username": user.username,
                                "password": password_derivative,
                            },
                        )
                        self.assertEqual(status, http.HTTPStatus.OK)
                        token = login_response["content"]["token"]
                        request = {
                            "content": "annotations",
                            "format": "snapshot",
                            "include_images": False,
                            "user_id": user.id,
                            "user_username": user.username,
                            "user_token": token,
                        }
                        status, _, refused = self._post_json(
                            server, "data/export",
                            {**request, "export_mode": "trusted", "public": False})
                        self.assertEqual(status, http.HTTPStatus.BAD_REQUEST)
                        self.assertIn("Unsupported field", refused["message"])
                        status, _, export_response = self._post_json(
                            server, "data/export", request)
                finally:
                    self._stop_test_server(server, thread)
                    frontend.user._AUTHENTICATION_LIMITER.clear()

            self.assertEqual(status, http.HTTPStatus.OK)
            self.assertTrue(export_response["success"])
            archive_bytes = base64.b64decode(export_response["content"])
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                filenames = {Path(name).name for name in archive.namelist()}
                exported_bytes = b"\n".join(
                    archive.read(name) for name in archive.namelist()
                )

            self.assertIn("individuals.csv", filenames)
            self.assertNotIn("users.csv", filenames)
            self.assertNotIn("tokens.csv", filenames)
            self.assertNotIn(user.email.encode("utf-8"), exported_bytes)
            self.assertNotIn(user.password.encode("utf-8"), exported_bytes)
            self.assertNotIn(token.encode("utf-8"), exported_bytes)
            output = captured_output.getvalue()
            self.assertNotIn(password_derivative, output)
            self.assertNotIn(user.password, output)
            self.assertNotIn(token, output)

    def test_frontend_handlers_do_not_print_request_payloads(self):
        """Frontend handlers never print complete authenticated request kwargs."""
        frontend_root = Path(__file__).resolve().parents[2] / "frontend"
        offenders = []
        for source_path in frontend_root.rglob("*.py"):
            source = source_path.read_text(encoding="utf-8")
            if "print(kwargs)" in source:
                offenders.append(str(source_path.relative_to(frontend_root)))

        self.assertEqual(offenders, [])

    def test_route_matching(self):
        """Test regex routing logic."""
        # _handler is the class.
        handler = frontend.server._handler
        
        # Test valid route.
        # Changed shark -> individual
        target, auth = handler._route(handler, "/api/individual/get")
        self.assertNotEqual(target, frontend.handlers.not_found)
        # Auth might be present.
        
        # Test invalid route.
        target_bad, _ = handler._route(handler, "/api/invalid/endpoint")
        self.assertEqual(target_bad, frontend.handlers.not_found)

    def test_get_get_data(self):
        """Test parsing GET parameters."""
        path = "/api/test?id=1&name=sharky"
        
        # Create a dummy instance.
        h = MagicMock()
        h.path = path
        
        # Bind the function to the mock or just call it providing self.
        result = frontend.server._handler._get_get_data(h)
        self.assertEqual(result["id"], "1")
        self.assertEqual(result["name"], "sharky")

    def test_get_post_data(self):
        """Test parsing POST JSON body."""
        payload = {"key": "value"}
        body = json.dumps(payload).encode('utf-8')
        
        h = MagicMock()
        h.headers = {'Content-Length': str(len(body)), 'Content-Type': 'application/json'}
        h.rfile = io.BytesIO(body)
        h.path = '/api/test'
        
        result = frontend.server._handler._get_post_data(h)
        self.assertEqual(result, payload)

    @patch("frontend.server._handler._route")
    def test_handle_request_success(self, mock_route):
        """Test handling a successful request."""
        # Setup mocks.
        mock_target = MagicMock(return_value=({"res": "ok"}, http.HTTPStatus.OK, None))
        mock_target.allowed_methods = ('POST',)
        mock_auth = None
        mock_route.return_value = (mock_target, mock_auth)
        
        mock_data_handler = MagicMock(return_value={})
        
        # Create handler instance bypassing __init__.
        h = frontend.server._handler.__new__(frontend.server._handler)
        h.path = "/api/test"
        h.command = 'POST'
        h.send_response = MagicMock()
        h.send_header = MagicMock()
        h.end_headers = MagicMock()
        h.wfile = MagicMock()
        
        # Call.
        h.handle_request(mock_data_handler)
        
        # Verify.
        mock_target.assert_called()
        h.send_response.assert_called_with(http.HTTPStatus.OK)
    
    @patch("frontend.server._handler._route")
    def test_handle_request_auth_fail(self, mock_route):
        """Test handling a request that fails auth."""
        # Setup mocks.
        mock_target = MagicMock()
        mock_target.allowed_methods = ('POST',)
        mock_auth = MagicMock(return_value=("Auth Failed", http.HTTPStatus.UNAUTHORIZED, None))
        mock_route.return_value = (mock_target, mock_auth)
        
        mock_data_handler = MagicMock(return_value={})
        
        # Bypass BaseHTTPRequestHandler.__init__ to test handle_request in isolation.
        h = frontend.server._handler.__new__(frontend.server._handler)
        h._route = mock_route
        h.path = "/api/test"
        h.command = 'POST'
        h.send_response = MagicMock()
        h.send_header = MagicMock()
        h.end_headers = MagicMock()
        h.wfile = MagicMock()
        
        # Call.
        h.handle_request(mock_data_handler)
        
        # Verify.
        mock_auth.assert_called()
        mock_target.assert_not_called()
        h.send_response.assert_called_with(http.HTTPStatus.UNAUTHORIZED)
    def test_simultaneous_connections_are_all_served(self):
        """A burst of parallel connections, like a page loading its thumbnails, is never reset."""
        burst = 64
        server, thread = self._start_test_server()
        try:
            barrier = threading.Barrier(burst)
            outcomes = []

            def fetch():
                barrier.wait()
                try:
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", server.server_port, timeout=30)
                    connection.request("GET", "/static/index.html")
                    outcomes.append(connection.getresponse().status)
                    connection.close()
                except OSError as error:
                    outcomes.append(type(error).__name__)

            workers = [threading.Thread(target=fetch) for _ in range(burst)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=60)
        finally:
            self._stop_test_server(server, thread)

        self.assertEqual(outcomes, [http.HTTPStatus.OK] * burst)

    def test_loading_image_is_served(self):
        """The page's loading indicator is a real PNG, not a 404."""
        server, thread = self._start_test_server()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=30)
            connection.request("GET", "/static/images/shark-loading.png")
            response = connection.getresponse()
            body = response.read()
            connection.close()
        finally:
            self._stop_test_server(server, thread)

        self.assertEqual(response.status, http.HTTPStatus.OK)
        self.assertTrue(body.startswith(b"\x89PNG"))

    @staticmethod
    def _initialize_auth_database(database_path, username, cleartext_password):
        """Create a complete temporary schema containing one login account."""
        app_root = Path(__file__).resolve().parents[2]
        schema = (app_root / "db/sql/create_database.sql").read_text(encoding="utf-8")
        password_derivative = hashlib.sha256(
            cleartext_password.encode("utf-8")
        ).hexdigest()

        with sqlite3.connect(database_path) as connection:
            connection.executescript(schema)
            user = backend.db.user.create_user(
                name="Lifecycle Owner",
                email="lifecycle@example.test",
                username=username,
                role="owner",
                password=password_derivative,
                connection=connection,
            )

        return user, password_derivative

    @staticmethod
    def _start_test_server(demo_enabled=False):
        """Start the production request handler on an ephemeral local port."""
        app_root = Path(__file__).resolve().parents[2]
        frontend.server._handler.init(app_root, demo_enabled=demo_enabled)
        server = frontend.server._create_server(
            ("127.0.0.1", 0), frontend.server._handler
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    @staticmethod
    def _stop_test_server(server, thread):
        """Stop a temporary production-handler server and join its thread."""
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("Temporary application server did not stop.")

    @staticmethod
    def _post_json(server, endpoint, payload, headers=None):
        """POST JSON to a temporary application server and decode its response."""
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        try:
            connection.request(
                "POST",
                f"/api/{endpoint}",
                body=json.dumps(payload),
                headers={"Content-Type": "application/json", **(headers or {})},
            )
            response = connection.getresponse()
            body = response.read()
            decoded = json.loads(body) if body else None
            return response.status, dict(response.getheaders()), decoded
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
