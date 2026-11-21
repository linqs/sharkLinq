"""Tests for demo mode rounds, grading, and server permissions."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import http.client
import json
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from core import app_constants
from core import app_utils
import demo.runtime
import frontend.cli.serve
import frontend.individual
import frontend.match
import frontend.media
import frontend.project
import frontend.server
import frontend.user
from scripts import shark_identification
from tests.frontend import test_server as server_tests


@pytest.fixture(name="demo_database")
def fixture_demo_database(tmp_path, monkeypatch):
    """Create a small temporary database with a test user."""
    path = tmp_path / "live.db"
    user, password = server_tests.TestServerFrontend._initialize_auth_database(
        path, "demo_tester", "demo-password")
    with sqlite3.connect(path) as connection:
        for image in range(1, 8):
            source = 100 if image <= 3 else 200 if image <= 6 else 300
            connection.execute(
                "INSERT INTO Media (id, hash, name, filename, source, time, type, quality) "
                "VALUES (?, ?, ?, ?, ?, ?, 'photo', 'unmatchable')",
                (image, str(image), f"image-{image}", f"image-{image}.jpg", image, image * 100))
            connection.execute(
                "INSERT INTO Individuals (id, name, media, source) VALUES (?, ?, ?, ?)",
                (image, str(image), image, source))
        for source, reference in ((100, 1), (200, 4), (300, 7)):
            connection.execute(
                "INSERT INTO Individuals (id, source, reference) VALUES (?, ?, ?)",
                (source, source, reference))
    monkeypatch.setattr(app_constants, "DB_PATH", path)
    frontend.user._AUTHENTICATION_LIMITER.clear()
    return SimpleNamespace(path=path, user=user, password=password)


@pytest.fixture(name="session")
def fixture_session(demo_database):
    """Create and clean up a DemoSession for the test database."""
    value = demo.runtime.DemoSession(demo_database.path)
    yield value
    value.close()


def dataset_rows(path):
    """Read all rows from demo data tables to compare database state."""
    with sqlite3.connect(path) as connection:
        return {table: connection.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
                for table in demo.runtime.DATA_TABLES}


def selections(session):
    """Return selected individuals with their original and current source IDs."""
    with sqlite3.connect(session.database_path) as connection:
        return connection.execute("""
            SELECT truth.individual, truth.original_source, image.source
            FROM DemoSelections selected JOIN DemoTruth truth USING (individual)
            JOIN Individuals image ON image.id = selected.individual ORDER BY truth.individual
        """).fetchall()


def merge(session, major, minor):
    """Merge two individuals inside the active demo session."""
    with session.scope(), closing(app_utils.db_connect()) as connection:
        reference = connection.execute(
            "SELECT reference FROM Individuals WHERE id = ?", (major,)).fetchone()[0]
        preview = {"reference": reference, "sex": None, "length": None,
                   "alternate_names": [], "tags": [], "labels": []}
        return frontend.individual.merge_individuals(
            None, "", major_id=major, minor_id=minor, preview=preview, user_id=1)


def test_sampling_reset_and_restart(session, demo_database):
    """Check that reset undoes edits, skips singletons, and repeats the sample."""
    live = dataset_rows(demo_database.path)
    assert session.status()["maximum"] == 4
    assert session.status()["selected"] == 0
    session.randomize(4)
    first = selections(session)
    assert len(first) == 4
    assert 7 not in {item[0] for item in first}
    with sqlite3.connect(session.database_path) as connection:
        for individual, original, source in first:
            assert source != original
            assert connection.execute(
                "SELECT COUNT(*) FROM Individuals WHERE source = ? AND media >= 0", (source,)
            ).fetchone()[0] == 1
            assert {row[0] for row in connection.execute(
                "SELECT name FROM Labels WHERE individual = ?", (source,))} == {"unmatched", "randomized"}
        for original in (100, 200, 300):
            assert connection.execute(
                "SELECT COUNT(*) FROM Individuals WHERE source = ? AND media >= 0", (original,)
            ).fetchone()[0] >= 1
        connection.execute("UPDATE Media SET quality = 'good'")
        connection.execute("UPDATE Individuals SET sex = 'F'")
        connection.execute("INSERT INTO Labels (individual, name) VALUES (300, 'edited')")
    # An incorrect merge and profile edits are all undone on the next round.
    _, original, source = first[0]
    assert merge(session, 200 if original == 100 else 100, source)[1] == 200
    old_round = session.round_id
    session.randomize(4)
    assert selections(session) == first
    assert session.round_id != old_round
    with sqlite3.connect(session.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM Individuals WHERE sex IS NOT NULL").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM Labels WHERE name = 'edited'").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM Actions").fetchone()[0] == 0
    assert dataset_rows(demo_database.path) == live
    with closing(demo.runtime.DemoSession(demo_database.path)) as restarted:
        assert restarted.status()["selected"] == 0
        assert dataset_rows(restarted.database_path) == live


@pytest.mark.parametrize("count", [0, -1, 5, 1.5, True, "2", None])
def test_invalid_count_preserves_round(session, count):
    session.randomize(2)
    before = dataset_rows(session.database_path)
    round_id = session.round_id
    with pytest.raises(ValueError):
        session.randomize(count)
    assert dataset_rows(session.database_path) == before
    assert session.round_id == round_id


def test_clear_restores_baseline_and_can_randomize_again(session, demo_database):
    baseline = dataset_rows(demo_database.path)
    session.randomize(2)
    sample = selections(session)
    _, original, source = sample[0]
    assert merge(session, 200 if original == 100 else 100, source)[1] == 200
    with sqlite3.connect(session.database_path) as connection:
        connection.execute("UPDATE Individuals SET sex = 'F'")
        connection.execute("UPDATE Media SET quality = 'good'")
    round_id = session.round_id
    result = session.clear()
    assert result["selected"] == 0
    assert result["round_id"] != round_id
    assert selections(session) == []
    assert dataset_rows(session.database_path) == baseline
    assert dataset_rows(demo_database.path) == baseline
    with session.scope():
        assert session.keys(original, reveal=True) == {"keys": []}
    with sqlite3.connect(session.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM DemoTargets").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM Actions").fetchone()[0] == 0
    # Clearing an empty round is harmless, and the next sample is unchanged.
    assert session.clear()["selected"] == 0
    session.randomize(2)
    assert selections(session) == sample


def test_failed_clear_preserves_round(session):
    session.randomize(2)
    before = dataset_rows(session.database_path)
    selected = selections(session)
    round_id = session.round_id
    with sqlite3.connect(session.database_path) as connection:
        connection.execute("""
            CREATE TRIGGER fail_clear BEFORE DELETE ON Media
            BEGIN SELECT RAISE(ABORT, 'injected clear failure'); END
        """)
    with pytest.raises(sqlite3.IntegrityError, match="injected clear failure"):
        session.clear()
    assert dataset_rows(session.database_path) == before
    assert selections(session) == selected
    assert session.round_id == round_id


def test_failed_randomization_rolls_back(session, monkeypatch):
    session.randomize(1)
    before = dataset_rows(session.database_path)
    selected = selections(session)
    round_id = session.round_id
    original = frontend.individual.detach_individual

    def fail_after_detach(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(frontend.individual, "detach_individual", fail_after_detach)
    with pytest.raises(RuntimeError, match="injected failure"):
        session.randomize(3)
    assert dataset_rows(session.database_path) == before
    assert selections(session) == selected
    assert session.round_id == round_id


@pytest.mark.parametrize("reverse", [False, True])
def test_grading_keys_and_deleted_target_identity(session, reverse):
    session.randomize(4)
    selected = selections(session)
    individual, original, source = selected[0]
    with session.scope():
        key = session.keys(source, reveal=True)["keys"][0]
        assert session.keys(source) == {"count": 1}
    major, minor = (source, original) if reverse else (original, source)
    result, status, _ = merge(session, major, minor)
    assert status == 200
    assert result["demo_results"] == [{"individual": individual, "correct": True}]
    assert "original_source" not in json.dumps(result["demo_results"])
    with session.scope():
        assert key in session.keys(major, reveal=True)["keys"]
    # Rejoin the other randomized image from that original shark after its ID changed.
    second = next(row for row in selected if row[1] == original and row[0] != individual)
    result, status, _ = merge(session, major, second[2])
    assert status == 200
    assert len(result["demo_results"]) == 2
    assert all(row["correct"] for row in result["demo_results"])
    # A later wrong merge is graded again and still commits.
    other = 200 if original == 100 else 100
    result, status, _ = merge(session, major, other)
    assert status == 200
    assert not any(row["correct"] for row in result["demo_results"])
    with sqlite3.connect(session.database_path) as connection:
        assert connection.execute("SELECT 1 FROM Individuals WHERE id = ?", (other,)).fetchone() is None
        assert connection.execute("SELECT COUNT(*) FROM Actions").fetchone()[0] == 0
        assert connection.execute(
            "SELECT name FROM Labels WHERE individual = ?", (major,)).fetchall() == [("randomized",)]


def test_two_detached_images_are_not_the_retained_target(session):
    session.randomize(4)
    selected = selections(session)
    first = selected[0]
    second = next(row for row in selected if row[1] == first[1] and row[0] != first[0])
    result, status, _ = merge(session, first[2], second[2])
    assert status == 200
    assert len(result["demo_results"]) == 2
    assert not any(row["correct"] for row in result["demo_results"])


def test_scopes_are_thread_local(session, demo_database):
    session.randomize(2)

    def count_sources(in_demo):
        def read():
            with closing(app_utils.db_connect()) as connection:
                return connection.execute("SELECT COUNT(*) FROM Individuals WHERE media IS NULL").fetchone()[0]
        if in_demo:
            with session.scope():
                return read()
        return read()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(count_sources, [True, False])) == [5, 3]
    assert app_constants.DB_PATH == demo_database.path


def test_stats_cache_does_not_cross_datasets(session, monkeypatch):
    """Check that the live stats cache is not reused inside a demo session."""
    monkeypatch.setattr(frontend.project.frontend.inference_client, "get_stats", lambda: {})
    frontend.project.invalidate_dataset_stats()
    try:
        assert frontend.project.get_dataset_stats(None, "")[0]["source_individuals"] == 3
        session.randomize(2)
        with session.scope():
            assert frontend.project.get_dataset_stats(None, "")[0]["source_individuals"] == 5
        assert frontend.project.get_dataset_stats(None, "")[0]["source_individuals"] == 3
    finally:
        frontend.project.invalidate_dataset_stats()


def test_grade_failure_rolls_back_merge(session, monkeypatch):
    session.randomize(1)
    _, original, source = selections(session)[0]
    before = dataset_rows(session.database_path)

    def fail_grade(*_args):
        raise RuntimeError("injected grading failure")

    monkeypatch.setattr(session, "grade_merge", fail_grade)
    assert merge(session, original, source)[1] == 500
    assert dataset_rows(session.database_path) == before


@pytest.mark.parametrize("method", ["nearest neighbor", "nearest centroid"])
def test_original_shark_remains_a_match_candidate(session, monkeypatch, method):
    session.randomize(1)
    individual, original, source = selections(session)[0]
    monkeypatch.setattr(frontend.match.frontend.inference_client, "get_matches", lambda **_kwargs: {
        "ids": list(range(1, 8)), "distances": [0.1] * 7, "labels": [100] * 3 + [200] * 3 + [300],
    })
    with session.scope():
        result, status, _ = frontend.match.get_matches(None, "", name=str(individual), method=method)
    assert status == 200
    assert result["query_individual"]["source"] == source
    assert original in {row["source"] for row in result["match_suggestions"]}


def test_http_mode_permissions_restrictions_and_stale_rounds(demo_database):
    server, thread = server_tests.TestServerFrontend._start_test_server(demo_enabled=True)
    session = frontend.server._handler.demo_session
    post = server_tests.TestServerFrontend._post_json
    try:
        status, _, login = post(server, "user/login", {
            "username": demo_database.user.username, "password": demo_database.password})
        assert status == 200
        credentials = {"user_username": demo_database.user.username,
                       "user_token": login["content"]["token"], "user_id": 99999}
        headers = {"X-Shark-Demo": "true", "X-Shark-Demo-Round": session.round_id}
        with sqlite3.connect(demo_database.path) as connection:
            connection.execute("UPDATE Users SET role = 'collaborator'")
        assert post(server, "demo/randomize", {**credentials, "count": 0}, headers)[0] == 400
        assert post(server, "demo/randomize", {**credentials, "count": 1}, headers)[0] == 200
        # A second participant submitting the old round cannot overwrite the new one.
        assert post(server, "demo/randomize", {**credentials, "count": 2}, headers)[0] == 409
        headers["X-Shark-Demo-Round"] = session.round_id
        status, _, response = post(server, "demo/status", credentials, headers)
        assert status == 200 and response["content"]["selected"] == 1
        individual, original, source = selections(session)[0]
        preview = {"reference": 1 if original == 100 else 4, "sex": None, "length": None,
                   "alternate_names": [], "tags": [], "labels": []}
        # Read the current reference; randomization can detach the original reference image.
        with sqlite3.connect(session.database_path) as connection:
            preview["reference"] = connection.execute(
                "SELECT reference FROM Individuals WHERE id = ?", (original,)).fetchone()[0]
        payload = {**credentials, "major_id": original, "minor_id": source, "preview": preview}
        assert post(server, "individual/merge", payload)[0] == 403
        status, _, result = post(server, "individual/merge", payload, headers)
        assert status == 200
        assert result["content"]["demo_results"] == [{"individual": individual, "correct": True}]
        for endpoint in ("media/delete", "individual/delete", "media/save", "media/upload_batch",
                         "embeddings/update", "proposal/post", "user/post",
                         "media/post", "individual/post", "individual/reference",
                         "object/post", "object/delete"):
            assert post(server, endpoint, credentials, headers)[0] == 403
        with sqlite3.connect(session.database_path) as connection:
            multi_source = connection.execute(
                "SELECT source FROM Individuals WHERE media >= 0 GROUP BY source HAVING COUNT(*) > 1"
            ).fetchone()[0]
            unmatch_target = connection.execute(
                "SELECT id FROM Individuals WHERE source = ? AND media >= 0", (multi_source,)
            ).fetchone()[0]
        assert post(server, "individual/unmatch", {**credentials, "id": unmatch_target}, headers)[0] == 200
        status, _, cleared = post(server, "demo/clear", credentials, headers)
        assert status == 200 and cleared["content"]["selected"] == 0
        assert post(server, "demo/clear", credentials, headers)[0] == 409
        headers["X-Shark-Demo-Round"] = session.round_id
        with sqlite3.connect(demo_database.path) as connection:
            connection.execute("UPDATE Users SET role = 'viewer'")
        assert post(server, "demo/randomize", {**credentials, "count": 1}, headers)[0] == 403
        assert post(server, "demo/clear", credentials, headers)[0] == 403
        assert post(server, "individual/merge", payload, headers)[0] == 403
        assert post(server, "individual/unmatch", {**credentials, "id": unmatch_target}, headers)[0] == 403
        assert post(server, "demo/status", credentials, headers)[0] == 200
        assert post(server, "demo/status", credentials)[0] == 404
        assert post(server, "demo/status", {}, headers)[0] == 401
        # Logging out in demo revokes the real session; reset never touches authentication.
        assert post(server, "user/logout", credentials, headers)[0] == 200
        assert post(server, "demo/status", credentials, headers)[0] == 401
    finally:
        server_tests.TestServerFrontend._stop_test_server(server, thread)
        session.close()
        frontend.server._handler.demo_session = None


def test_disabled_demo_cannot_be_enabled_by_client(demo_database):
    server, thread = server_tests.TestServerFrontend._start_test_server()
    post = server_tests.TestServerFrontend._post_json
    try:
        status, _, result = post(server, "runtime/config", {})
        assert status == 200 and result["content"]["demo_enabled"] is False
        assert post(server, "demo/randomize", {"count": 1}, {"X-Shark-Demo": "true"})[0] == 404
        assert post(server, "demo/clear", {}, {"X-Shark-Demo": "true"})[0] == 404
        assert post(server, "individual/merge", {}, {"X-Shark-Demo": "true"})[0] == 404
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        try:
            connection.request("GET", "/demo/demo.js")
            assert connection.getresponse().status == 404
        finally:
            connection.close()
    finally:
        server_tests.TestServerFrontend._stop_test_server(server, thread)


@pytest.mark.parametrize("enabled", [False, True])
def test_server_cli_demo_is_explicit(monkeypatch, enabled):
    calls = []
    monkeypatch.setattr(sys, "argv", ["server", "."] + (["--demo"] if enabled else []))
    monkeypatch.setattr(frontend.cli.serve, "run_cli", lambda **kwargs: calls.append(kwargs))
    frontend.cli.serve.main()
    assert calls[0]["demo"] is enabled


def test_orchestrator_forwards_demo_flag(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sys, "argv", ["shark_identification", "--run", "--demo"])
    monkeypatch.setattr(app_constants, "OUT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(shark_identification, "run_step", lambda command, *_args, **_kwargs: calls.append(command))
    shark_identification.main()
    assert len(calls) == 1
    assert calls[0][1:3] == ["-m", "frontend.cli.serve"]
    assert "--demo" in calls[0]


@pytest.mark.parametrize("reverse", [False, True])
def test_ordinary_merge_keeps_chosen_survivor_and_logs_actions(demo_database, reverse):
    """Check that normal merges keep the chosen survivor and log actions."""
    major, minor = (200, 100) if reverse else (100, 200)
    preview = {"reference": 4 if reverse else 1, "sex": None, "length": None,
               "alternate_names": [], "tags": [], "labels": []}
    result, status, _ = frontend.individual.merge_individuals(
        None, "", major_id=major, minor_id=minor, preview=preview, user_id=demo_database.user.id)
    assert status == 200
    assert "demo_results" not in result
    with sqlite3.connect(demo_database.path) as connection:
        assert connection.execute("SELECT id FROM Individuals WHERE id = ?", (minor,)).fetchone() is None
        assert connection.execute("SELECT COUNT(*) FROM Individuals WHERE source = ? AND media >= 0", (major,)).fetchone()[0] == 6
        assert connection.execute("SELECT COUNT(*) FROM Actions WHERE action = 'merge'").fetchone()[0] == 2


def test_demo_post_media_never_renames_disk_files(session, demo_database, monkeypatch):
    """Check that posting media in demo mode does not rename image files on disk."""
    renamed = []
    monkeypatch.setattr(frontend.media.os, "rename", lambda src, dst: renamed.append((src, dst)))
    with session.scope():
        frontend.media.post_media(
            None, "", id=1, hash="1", name="renamed-image", source=1, individual=1,
            metadata=None, location=None, time=100, type="photo", quality="good",
            length=None, sex=None, user_id=1)
    assert renamed == []

