"""Tests for matcher-side inference snapshot validation."""

import csv
import hashlib
import json
import sqlite3

import pytest


SNAPSHOT_ID = "11111111-1111-4111-8111-111111111111"
OTHER_SNAPSHOT_ID = "22222222-2222-4222-8222-222222222222"
INDIVIDUAL_ROWS = [
    [1, "Shark One", "F", 2.5, "", 1, 1],
    [2, "", "", "", 10, 1, ""],
]
MEDIA_ROWS = [
    [10, "image-hash", "AN01", "AN01.jpeg", "", "", "AN", 1700000000, "topshot", "good"],
]


def _write_csv(path, rows):
    """Write headerless CSV rows using the production encoding and newline mode."""
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)


def _digest(path):
    """Return the lowercase SHA-256 digest for a test file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_snapshot(root, snapshot_id=SNAPSHOT_ID):
    """Create a valid on-disk snapshot fixture and return its directory."""
    snapshot_dir = root / snapshot_id
    snapshot_dir.mkdir(parents=True)
    individuals_path = snapshot_dir / "individuals.csv"
    media_path = snapshot_dir / "media.csv"
    _write_csv(individuals_path, INDIVIDUAL_ROWS)
    _write_csv(media_path, MEDIA_ROWS)
    manifest = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "created_at": "2026-08-31T12:00:00Z",
        "files": {
            "individuals.csv": {
                "rows": len(INDIVIDUAL_ROWS),
                "sha256": _digest(individuals_path),
            },
            "media.csv": {
                "rows": len(MEDIA_ROWS),
                "sha256": _digest(media_path),
            },
        },
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return snapshot_dir


def _snapshot_api():
    """Import the snapshot-validation interface used by each test."""
    from src.retrieval.inference_snapshot import (  # pylint: disable=import-outside-toplevel
        SnapshotNotFoundError,
        SnapshotValidationError,
        load_inference_snapshot,
    )
    return load_inference_snapshot, SnapshotNotFoundError, SnapshotValidationError


def test_valid_snapshot_returns_resolved_read_only_paths(tmp_path):
    """A valid snapshot resolves to its two validated CSV paths."""
    snapshot_dir = _create_snapshot(tmp_path)
    load_snapshot, _, _ = _snapshot_api()
    snapshot = load_snapshot(SNAPSHOT_ID, snapshot_root=tmp_path)

    assert snapshot.snapshot_id == SNAPSHOT_ID
    assert snapshot.directory == snapshot_dir.resolve()
    assert snapshot.individuals_csv == snapshot_dir.resolve() / "individuals.csv"
    assert snapshot.media_csv == snapshot_dir.resolve() / "media.csv"
    assert snapshot.manifest.snapshot_id == SNAPSHOT_ID


def test_snapshot_loading_does_not_open_sqlite(tmp_path, monkeypatch):
    """Matcher snapshot validation reads files without opening the app database."""
    _create_snapshot(tmp_path)
    load_snapshot, _, _ = _snapshot_api()

    def reject_database_access(*_args, **_kwargs):
        """Fail if snapshot validation attempts to open any SQLite database."""
        raise AssertionError("matcher snapshot validation must not open SQLite")

    monkeypatch.setattr(sqlite3, "connect", reject_database_access)

    snapshot = load_snapshot(SNAPSHOT_ID, snapshot_root=tmp_path)

    assert snapshot.snapshot_id == SNAPSHOT_ID


@pytest.mark.parametrize(
    "snapshot_id_kind",
    ["parent traversal", "absolute path", "non-uuid"],
)
def test_snapshot_identifier_cannot_be_a_path(tmp_path, snapshot_id_kind):
    """Only canonical UUID identifiers may participate in path resolution."""
    invalid_ids = {
        "parent traversal": "../outside",
        "absolute path": str(tmp_path / "outside"),
        "non-uuid": "not-a-uuid",
    }
    load_snapshot, _, validation_error = _snapshot_api()
    with pytest.raises(validation_error):
        load_snapshot(invalid_ids[snapshot_id_kind], snapshot_root=tmp_path)


def test_missing_snapshot_has_a_distinct_error(tmp_path):
    """A valid but absent snapshot ID raises the not-found exception."""
    load_snapshot, not_found_error, _ = _snapshot_api()
    with pytest.raises(not_found_error):
        load_snapshot(SNAPSHOT_ID, snapshot_root=tmp_path)


def test_manifest_identifier_must_match_requested_snapshot(tmp_path):
    """The request, directory, and manifest identifiers must agree."""
    snapshot_dir = _create_snapshot(tmp_path)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot_id"] = OTHER_SNAPSHOT_ID
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    load_snapshot, _, validation_error = _snapshot_api()
    with pytest.raises(validation_error):
        load_snapshot(SNAPSHOT_ID, snapshot_root=tmp_path)


def test_snapshot_rejects_tampered_file_hash(tmp_path):
    """Changing CSV bytes after publication invalidates the snapshot."""
    snapshot_dir = _create_snapshot(tmp_path)
    with (snapshot_dir / "media.csv").open("a", encoding="utf-8") as media_file:
        media_file.write("tampered")

    load_snapshot, _, validation_error = _snapshot_api()
    with pytest.raises(validation_error):
        load_snapshot(SNAPSHOT_ID, snapshot_root=tmp_path)


def test_snapshot_rejects_row_count_mismatch(tmp_path):
    """Manifest row counts must match parsed data records."""
    snapshot_dir = _create_snapshot(tmp_path)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["media.csv"]["rows"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    load_snapshot, _, validation_error = _snapshot_api()
    with pytest.raises(validation_error):
        load_snapshot(SNAPSHOT_ID, snapshot_root=tmp_path)


def test_snapshot_rejects_malformed_csv_columns(tmp_path):
    """Version-1 CSV rows require their exact documented column counts."""
    snapshot_dir = _create_snapshot(tmp_path)
    media_path = snapshot_dir / "media.csv"
    _write_csv(media_path, [[10, "too", "few"]])
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["media.csv"] = {"rows": 1, "sha256": _digest(media_path)}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    load_snapshot, _, validation_error = _snapshot_api()
    with pytest.raises(validation_error):
        load_snapshot(SNAPSHOT_ID, snapshot_root=tmp_path)


def test_snapshot_rejects_symlinked_required_file(tmp_path):
    """Required files must be regular files physically inside the snapshot."""
    snapshot_dir = _create_snapshot(tmp_path)
    outside_file = tmp_path / "outside.csv"
    outside_file.write_text("outside", encoding="utf-8")
    media_path = snapshot_dir / "media.csv"
    media_path.unlink()
    media_path.symlink_to(outside_file)

    load_snapshot, _, validation_error = _snapshot_api()
    with pytest.raises(validation_error):
        load_snapshot(SNAPSHOT_ID, snapshot_root=tmp_path)


def test_snapshot_rejects_symlinked_manifest(tmp_path):
    """The manifest cannot redirect validation to a file outside the snapshot."""
    snapshot_dir = _create_snapshot(tmp_path)
    manifest_path = snapshot_dir / "manifest.json"
    outside_manifest = tmp_path / "outside-manifest.json"
    manifest_path.rename(outside_manifest)
    manifest_path.symlink_to(outside_manifest)

    load_snapshot, _, validation_error = _snapshot_api()
    with pytest.raises(validation_error):
        load_snapshot(SNAPSHOT_ID, snapshot_root=tmp_path)


def test_snapshot_rejects_symlinked_snapshot_directory(tmp_path):
    """A snapshot UUID directory cannot redirect outside the configured root."""
    outside_root = tmp_path / "outside"
    outside_snapshot = _create_snapshot(outside_root)
    link_root = tmp_path / "snapshots"
    link_root.mkdir()
    (link_root / SNAPSHOT_ID).symlink_to(outside_snapshot, target_is_directory=True)

    load_snapshot, _, validation_error = _snapshot_api()
    with pytest.raises(validation_error):
        load_snapshot(SNAPSHOT_ID, snapshot_root=link_root)
