"""Pytest fixtures for browser tests."""

import http.server
import json
import re
import sqlite3
import threading
from types import SimpleNamespace

import PIL.Image
import pytest

from core import app_constants
import backend.db.user
import frontend.project
import frontend.user
from tests.frontend.test_server import TestServerFrontend

WEB_FONT_HOSTS = re.compile(r"^https://fonts\.(googleapis|gstatic)\.com/")


@pytest.fixture(autouse=True)
def block_web_fonts(page):
    """Block web-font requests so page loads do not wait on the network."""
    page.route(WEB_FONT_HOSTS, lambda route: route.abort())


@pytest.fixture
def browser_app(tmp_path, monkeypatch, setup_test_database):
    """Start the app server and a mock inference server with a temporary database."""
    database = tmp_path / "browser.db"
    with sqlite3.connect(setup_test_database) as original:
        with sqlite3.connect(database) as copy:
            original.backup(copy)
    images = tmp_path / "images"
    images.mkdir()
    with sqlite3.connect(database) as connection:
        media = connection.execute("SELECT id, name, filename FROM Media ORDER BY id").fetchall()
    for _, _, filename in media:
        destination = images / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        PIL.Image.new("RGB", (16, 16), "blue").save(destination)

    for key, value in {
        "DB_PATH": database, "IMAGES_DIR": str(images),
        "OUT_DATA_DIR": str(tmp_path / "exports"),
        "INFERENCE_SNAPSHOT_ROOT": str(tmp_path / "snapshots"),
        "INFERENCE_API_KEY": "browser-test-service-key-with-32-characters",
    }.items():
        monkeypatch.setattr(app_constants, key, value)

    users = {}
    for role in frontend.user.role_dict:
        users[role] = backend.db.user.create_user(
            role, f"{role}@example.test", f"browser_{role}", role,
            backend.db.user.derive_client_password("browser-password"))

    class InferenceHandler(http.server.BaseHTTPRequestHandler):
        """Mock inference HTTP handler returning fixed JSON responses."""

        def log_message(self, *_args):
            pass

        def do_GET(self):  # pylint: disable=invalid-name
            self.respond({"status": "ok", "loaded_models": ["ViT"],
                          "filenames": [row[2] for row in media],
                          "embeddings_metadata": {"start_time": 1, "update_time": 1}})

        def do_POST(self):  # pylint: disable=invalid-name
            self.rfile.read(int(self.headers["Content-Length"]))
            self.respond({"ids": [row[0] for row in media],
                          "distances": [index / 100 for index in range(len(media))],
                          "labels": [], "status": "ok"})

        def respond(self, value):
            """Return one fixed JSON response."""
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    inference = http.server.ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    worker = threading.Thread(target=inference.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setattr(app_constants, "INFERENCE_SERVICE_URL",
                        f"http://127.0.0.1:{inference.server_port}")
    frontend.user._AUTHENTICATION_LIMITER.clear()  # pylint: disable=protected-access
    frontend.project.invalidate_dataset_stats()
    server, thread = TestServerFrontend._start_test_server(demo_enabled=True)
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}",
                              database=database, images=images, media=media, users=users)
    finally:
        TestServerFrontend._stop_test_server(server, thread)
        TestServerFrontend._stop_test_server(inference, worker)
        frontend.project.invalidate_dataset_stats()


@pytest.fixture
def login(page, browser_app):
    """Log in through the browser form and fail on unexpected JS errors."""
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def perform(role="owner", password="browser-password"):
        page.goto(browser_app.url + "/#login")
        page.locator(".login.username-input").fill(f"browser_{role}")
        page.locator(".login.password-input").fill(password)
        page.locator(".login.submit").click()
        page.wait_for_function("location.hash !== '#login'")
        page.wait_for_load_state("networkidle")

    yield perform
    assert not errors, errors
