"""Tests for inference snapshot manifest validation and data path settings."""

from datetime import datetime, timezone
import json
import os
import subprocess
import sys

import pytest

from core import app_constants
from backend.inference_snapshot_contract import SnapshotFile, build_snapshot_manifest


SNAPSHOT_ID = "11111111-1111-4111-8111-111111111111"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _files():
    """Return valid snapshot file metadata for tests."""
    return {
        "individuals.csv": SnapshotFile(rows=2, sha256=SHA_A),
        "media.csv": SnapshotFile(rows=1, sha256=SHA_B),
    }


def test_build_snapshot_manifest_matches_the_v1_wire_shape():
    """Check that build_snapshot_manifest returns the expected JSON-serializable dict."""
    manifest = build_snapshot_manifest(
        SNAPSHOT_ID,
        datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
        _files(),
    )
    assert manifest == {
        "schema_version": 1,
        "snapshot_id": SNAPSHOT_ID,
        "created_at": "2026-08-31T12:00:00Z",
        "files": {
            "individuals.csv": {"rows": 2, "sha256": SHA_A},
            "media.csv": {"rows": 1, "sha256": SHA_B},
        },
    }
    assert json.loads(json.dumps(manifest)) == manifest


@pytest.mark.parametrize(
    "invalid_id",
    ["../snapshot", "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA", "not-a-uuid"],
)
def test_build_snapshot_manifest_rejects_noncanonical_identifiers(invalid_id):
    """Paths, uppercase UUIDs, and malformed IDs are rejected."""
    with pytest.raises(ValueError):
        build_snapshot_manifest(
            invalid_id,
            datetime.now(timezone.utc),
            _files(),
        )


def test_build_snapshot_manifest_rejects_sensitive_or_missing_files():
    """Check that manifests reject sensitive or missing CSV files."""
    files = _files()
    files.pop("media.csv")
    files["users.csv"] = SnapshotFile(rows=1, sha256=SHA_B)
    with pytest.raises(ValueError):
        build_snapshot_manifest(SNAPSHOT_ID, datetime.now(timezone.utc), files)


def test_snapshot_file_rejects_invalid_counts_and_hashes():
    """Check that SnapshotFile rejects negative row counts and invalid SHA-256 hashes."""
    with pytest.raises(ValueError):
        SnapshotFile(rows=-1, sha256=SHA_A)
    with pytest.raises(ValueError):
        SnapshotFile(rows=1, sha256="A" * 64)


def test_snapshot_root_defaults_to_shared_data_directory(monkeypatch):
    """Check that the default snapshot directory is under BASE_DATA_DIR."""
    monkeypatch.delenv("SHARK_INFERENCE_SNAPSHOT_ROOT", raising=False)
    expected = os.path.abspath(os.path.join(app_constants.BASE_DATA_DIR, "inference-snapshots"))
    assert app_constants.resolve_inference_snapshot_root() == expected


def test_snapshot_root_honors_explicit_environment_path(monkeypatch, tmp_path):
    """Check that SHARK_INFERENCE_SNAPSHOT_ROOT overrides the default snapshot directory."""
    configured = tmp_path / "snapshots"
    monkeypatch.setenv("SHARK_INFERENCE_SNAPSHOT_ROOT", str(configured))
    assert app_constants.resolve_inference_snapshot_root() == str(configured)


def test_suite_runs_against_a_disposable_data_root():
    """Check that tests use a temporary data directory instead of sm-data."""
    data_root = os.environ["SHARK_DATA_ROOT"]
    assert os.path.basename(data_root).startswith("shark-matcher-app-tests-")
    assert app_constants.BASE_DATA_DIR == data_root
    assert app_constants.DB_PATH.startswith(data_root + os.sep)


def test_data_root_defaults_to_sibling_sm_data(monkeypatch):
    """Check that BASE_DATA_DIR defaults to the sibling sm-data directory."""
    monkeypatch.delenv("SHARK_DATA_ROOT", raising=False)
    assert app_constants.resolve_base_data_dir() == os.path.join(app_constants.BASE_DIR, "sm-data")


def test_data_root_honors_an_absolute_override(monkeypatch, tmp_path):
    """Check that an absolute SHARK_DATA_ROOT overrides BASE_DATA_DIR."""
    monkeypatch.setenv("SHARK_DATA_ROOT", str(tmp_path / "data"))
    assert app_constants.resolve_base_data_dir() == str(tmp_path / "data")


def test_data_root_rejects_a_relative_override(monkeypatch):
    """Check that a relative SHARK_DATA_ROOT raises ValueError."""
    monkeypatch.setenv("SHARK_DATA_ROOT", os.path.join("relative", "data"))
    with pytest.raises(ValueError, match="absolute"):
        app_constants.resolve_base_data_dir()


def test_data_root_override_moves_every_derived_data_directory(tmp_path):
    """Check that all derived data paths use SHARK_DATA_ROOT."""
    data_root = tmp_path / "data"
    names = ["ANNOTATIONS_DIR", "DB_DATA_DIR", "DB_PATH", "IMAGES_DIR", "INFERENCE_SNAPSHOT_ROOT", "OUT_DATA_DIR"]
    environment = {
        key: value for key, value in os.environ.items()
        if key not in {"SHARK_TEST_MODE", "SHARK_INFERENCE_SNAPSHOT_ROOT"}
    }
    environment["SHARK_DATA_ROOT"] = str(data_root)
    script = (
        "import json, sys; from core import app_constants; "
        "print(json.dumps({name: getattr(app_constants, name) for name in sys.argv[1:]}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, *names],
        cwd=app_constants.THIS_DIR,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = json.loads(result.stdout)
    assert all(path.startswith(str(data_root) + os.sep) for path in paths.values()), paths
