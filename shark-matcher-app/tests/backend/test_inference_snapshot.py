"""Tests for creating and cleaning up inference snapshots."""

import csv
import hashlib
import json
import os
import sqlite3
import time
from unittest.mock import patch

import pytest

from db.scripts import db_dump


def _snapshot_api():
    """Import the snapshot creation and cleanup functions."""
    from backend.inference_snapshot import (  # pylint: disable=import-outside-toplevel
        cleanup_abandoned_snapshot_staging,
        create_inference_snapshot,
    )
    return create_inference_snapshot, cleanup_abandoned_snapshot_staging


def _create_database(path):
    """Create a minimal database with public and sensitive tables."""
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE Individuals (
                id INTEGER, name TEXT, sex TEXT, length REAL,
                media INTEGER, source INTEGER, reference INTEGER
            );
            CREATE TABLE Media (
                id INTEGER, hash TEXT, name TEXT, filename TEXT,
                source INTEGER, metadata TEXT, location TEXT, time INTEGER,
                type TEXT, quality TEXT
            );
            CREATE TABLE Users (
                id INTEGER, name TEXT, email TEXT, username TEXT,
                role TEXT, password TEXT, salt TEXT
            );
            CREATE TABLE Tokens (
                id INTEGER,
                token TEXT,
                user INTEGER,
                expires INTEGER
            );
            """
        )
        connection.executemany(
            "INSERT INTO Individuals VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (2, None, None, None, 10, 1, None),
                (1, "Shark One", "F", 2.5, None, 1, 1),
                (-1, "Reserved", None, None, None, -1, -1),
            ],
        )
        connection.execute(
            "INSERT INTO Media VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (10, "image-hash", "AN01", "AN01.jpeg", None, None,
             "AN", 1700000000, "topshot", "good"),
        )
        connection.execute(
            "INSERT INTO Users VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "Admin", "admin@example.test", "admin", "admin", "secret", "salt"),
        )
        connection.execute(
            "INSERT INTO Tokens VALUES (?, ?, ?, ?)",
            (1, "token-secret", 1, 2_000_604_800),
        )


def _read_csv(path):
    """Read all rows from a headerless snapshot CSV."""
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.reader(csv_file))


def test_create_snapshot_publishes_exact_public_artifacts(tmp_path):
    """Check that a snapshot contains only manifest.json and the two public CSVs."""
    database_path = tmp_path / "app.db"
    snapshot_root = tmp_path / "snapshots"
    _create_database(database_path)
    create_snapshot, _ = _snapshot_api()

    snapshot_id = create_snapshot(db_path=database_path, snapshot_root=snapshot_root)
    snapshot_dir = snapshot_root / snapshot_id

    assert sorted(path.name for path in snapshot_dir.iterdir()) == [
        "individuals.csv",
        "manifest.json",
        "media.csv",
    ]
    assert _read_csv(snapshot_dir / "individuals.csv")[0][0] == "1"
    assert _read_csv(snapshot_dir / "media.csv")[0][0] == "10"
    assert not (snapshot_dir / "users.csv").exists()
    assert not (snapshot_dir / "tokens.csv").exists()


def test_snapshot_manifest_hashes_and_counts_match_file_bytes(tmp_path):
    """Check that manifest hashes and row counts match the snapshot CSV files."""
    database_path = tmp_path / "app.db"
    snapshot_root = tmp_path / "snapshots"
    _create_database(database_path)
    create_snapshot, _ = _snapshot_api()

    snapshot_id = create_snapshot(db_path=database_path, snapshot_root=snapshot_root)
    snapshot_dir = snapshot_root / snapshot_id
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))

    for filename in ("individuals.csv", "media.csv"):
        file_path = snapshot_dir / filename
        assert manifest["files"][filename]["sha256"] == hashlib.sha256(
            file_path.read_bytes()
        ).hexdigest()
        assert manifest["files"][filename]["rows"] == len(_read_csv(file_path))


def test_snapshot_csvs_match_the_shared_public_export(tmp_path):
    """Check that snapshot CSVs match the output of db_dump_tables."""
    database_path = tmp_path / "app.db"
    public_export = tmp_path / "public-export"
    snapshot_root = tmp_path / "snapshots"
    _create_database(database_path)

    with sqlite3.connect(database_path) as connection:
        db_dump.db_dump_tables(
            tables=["Individuals", "Media"],
            output_directory=public_export,
            export_mode=db_dump.ExportMode.PUBLIC,
            connection=connection,
            durable=True,
        )

    create_snapshot, _ = _snapshot_api()
    snapshot_id = create_snapshot(
        db_path=database_path,
        snapshot_root=snapshot_root,
    )
    snapshot_directory = snapshot_root / snapshot_id

    for filename in ("individuals.csv", "media.csv"):
        assert (snapshot_directory / filename).read_bytes() == (
            public_export / filename
        ).read_bytes()


def test_snapshot_uses_one_explicit_read_transaction(tmp_path, monkeypatch):
    """Check that both tables are read in one transaction on one connection."""
    create_snapshot, _ = _snapshot_api()

    class Cursor:  # pylint: disable=too-few-public-methods
        """Stub cursor returning fixed rows."""

        def __init__(self, rows):
            """Store rows that ``fetchall`` will return."""
            self._rows = rows

        def fetchall(self):
            """Return the stored rows."""
            return self._rows

    class RecordingConnection:
        """Stub connection that records SQL statements and cleanup calls."""

        def __init__(self):
            """Initialize recorded statements and connection flags."""
            self.statements = []
            self.rolled_back = False
            self.closed = False

        def execute(self, statement, _data=None):
            """Record a statement and return its matching rows."""
            self.statements.append(" ".join(statement.split()))
            if "FROM Individuals" in statement:
                return Cursor([(1, "Shark One", "F", 2.5, None, 1, 1)])
            if "FROM Media" in statement:
                return Cursor([(10, "hash", "AN01", "AN01.jpeg", None,
                                None, "AN", 1700000000, "topshot", "good")])
            return Cursor([])

        def rollback(self):
            """Record the end of the read transaction."""
            self.rolled_back = True

        def close(self):
            """Record connection cleanup."""
            self.closed = True

    connection = RecordingConnection()
    snapshot_module = __import__("backend.inference_snapshot", fromlist=["app_utils"])
    monkeypatch.setattr(
        snapshot_module.app_utils,
        "db_connect",
        lambda *args, **kwargs: connection,
    )
    build_snapshot_manifest = snapshot_module.build_snapshot_manifest

    def build_manifest_after_transaction(**kwargs):
        """Check that the database connection is closed before building the manifest."""
        assert connection.rolled_back
        assert connection.closed
        return build_snapshot_manifest(**kwargs)

    monkeypatch.setattr(
        snapshot_module,
        "build_snapshot_manifest",
        build_manifest_after_transaction,
    )

    create_snapshot(db_path=tmp_path / "app.db", snapshot_root=tmp_path / "snapshots")

    assert any(statement == "BEGIN" for statement in connection.statements)
    assert sum("FROM Individuals" in statement for statement in connection.statements) == 1
    assert sum("FROM Media" in statement for statement in connection.statements) == 1
    assert connection.rolled_back
    assert connection.closed


def test_failed_snapshot_publication_removes_its_staging_directory(tmp_path):
    """Check that a failed snapshot leaves no output or staging directory behind."""
    database_path = tmp_path / "app.db"
    snapshot_root = tmp_path / "snapshots"
    _create_database(database_path)
    create_snapshot, _ = _snapshot_api()
    snapshot_module = __import__("backend.inference_snapshot", fromlist=["build_snapshot_manifest"])

    with patch.object(snapshot_module, "build_snapshot_manifest", side_effect=RuntimeError("stop")):
        with pytest.raises(RuntimeError, match="stop"):
            create_snapshot(db_path=database_path, snapshot_root=snapshot_root)

    assert not list(snapshot_root.iterdir())


def test_completed_snapshots_are_not_automatically_deleted(tmp_path):
    """Check that creating a new snapshot does not delete previous snapshots."""
    database_path = tmp_path / "app.db"
    snapshot_root = tmp_path / "snapshots"
    _create_database(database_path)
    create_snapshot, _ = _snapshot_api()

    first_id = create_snapshot(db_path=database_path, snapshot_root=snapshot_root)
    second_id = create_snapshot(db_path=database_path, snapshot_root=snapshot_root)

    assert (snapshot_root / first_id).is_dir()
    assert (snapshot_root / second_id).is_dir()


def test_cleanup_removes_only_old_staging_directories(tmp_path):
    """Check that cleanup deletes only old staging directories."""
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    old_staging = snapshot_root / ".staging-old"
    recent_staging = snapshot_root / ".staging-recent"
    completed = snapshot_root / "11111111-1111-4111-8111-111111111111"
    for directory in (old_staging, recent_staging, completed):
        directory.mkdir()
    old_time = time.time() - 7200
    os.utime(old_staging, (old_time, old_time))
    _, cleanup_staging = _snapshot_api()

    removed = cleanup_staging(snapshot_root=snapshot_root, older_than_seconds=3600)

    assert removed == 1
    assert not old_staging.exists()
    assert recent_staging.is_dir()
    assert completed.is_dir()


def test_cleanup_does_not_follow_staging_symlinks(tmp_path):
    """Check that cleanup ignores staging symlinks and leaves their targets intact."""
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    marker = outside_directory / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    (snapshot_root / ".staging-link").symlink_to(
        outside_directory, target_is_directory=True
    )
    _, cleanup_staging = _snapshot_api()

    removed = cleanup_staging(
        snapshot_root=snapshot_root,
        older_than_seconds=0,
    )

    assert removed == 0
    assert marker.read_text(encoding="utf-8") == "keep"


def test_snapshot_created_at_reads_a_published_manifest(tmp_path):
    """Check that snapshot_created_at returns the manifest timestamp as epoch seconds."""
    from backend.inference_snapshot import snapshot_created_at  # pylint: disable=import-outside-toplevel
    database_path = tmp_path / "app.db"
    snapshot_root = tmp_path / "snapshots"
    _create_database(database_path)
    create_snapshot, _ = _snapshot_api()
    before = time.time()
    snapshot_id = create_snapshot(db_path=database_path, snapshot_root=snapshot_root)
    created_at = snapshot_created_at(snapshot_id, snapshot_root=snapshot_root)
    assert before - 1 <= created_at <= time.time() + 1


@pytest.mark.parametrize("snapshot_id", [
    "33333333-3333-4333-8333-333333333333", "../escape", None, "not-a-uuid"])
def test_snapshot_created_at_is_none_for_unknown_or_invalid_ids(tmp_path, snapshot_id):
    """Check that missing or invalid snapshot IDs return None."""
    from backend.inference_snapshot import snapshot_created_at  # pylint: disable=import-outside-toplevel
    assert snapshot_created_at(snapshot_id, snapshot_root=tmp_path) is None
