"""Create immutable, public inference snapshots from the app database.

Reads inference data, writes a strict manifest, and atomically publishes completed snapshots beneath the configured snapshot root.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time
from uuid import uuid4

from core import app_constants
from core import app_utils
from backend.inference_snapshot_contract import (
    SNAPSHOT_MANIFEST_FILENAME,
    SnapshotFile,
    build_snapshot_manifest,
    canonical_snapshot_id,
)
from db.scripts.db_dump import ExportMode, db_dump_tables


_DEFAULT_STAGING_MAX_AGE_SECONDS = 24 * 60 * 60


def _sha256(path):
    """Return the lowercase SHA-256 digest of one completed file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_file(path):
    """Build manifest metadata from one CSV written by the shared exporter."""
    rows = app_utils.load_csv_file(path)
    if rows is None:
        raise FileNotFoundError(f"snapshot CSV was not written: {path.name}")
    return SnapshotFile(rows=len(rows), sha256=_sha256(path))


def _write_manifest(path, manifest):
    """Write and sync the manifest after all snapshot data files are complete."""
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def create_inference_snapshot(db_path=None, snapshot_root=None):
    """Create and atomically publish one app-owned inference snapshot from public Individuals and Media tables.

    Returns the canonical UUID identifying the immutable completed snapshot.
    """
    database_path = Path(db_path or app_constants.DB_PATH)
    root = Path(snapshot_root or app_constants.INFERENCE_SNAPSHOT_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    snapshot_id = str(uuid4())
    staging_directory = root / f".staging-{snapshot_id}"
    completed_directory = root / snapshot_id
    if completed_directory.exists() or completed_directory.is_symlink():
        raise FileExistsError(f"snapshot already exists: {snapshot_id}")

    staging_directory.mkdir()
    try:
        connection = app_utils.db_connect(database_path)
        try:
            connection.execute("BEGIN")
            db_dump_tables(
                tables=["Individuals", "Media"],
                output_directory=staging_directory,
                export_mode=ExportMode.PUBLIC,
                connection=connection,
                durable=True,
            )
        finally:
            try:
                connection.rollback()
            finally:
                connection.close()

        files = {
            filename: _snapshot_file(staging_directory / filename)
            for filename in ("individuals.csv", "media.csv")
        }
        manifest = build_snapshot_manifest(
            snapshot_id=snapshot_id,
            created_at=datetime.now(timezone.utc),
            files=files,
        )
        _write_manifest(staging_directory / SNAPSHOT_MANIFEST_FILENAME, manifest)

        if completed_directory.exists() or completed_directory.is_symlink():
            raise FileExistsError(f"snapshot already exists: {snapshot_id}")
        staging_directory.rename(completed_directory)
    except BaseException:
        if staging_directory.exists() and not staging_directory.is_symlink():
            shutil.rmtree(staging_directory)
        raise

    return snapshot_id


def snapshot_created_at(snapshot_id, snapshot_root=None):
    """Return a completed snapshot's creation time in epoch seconds, or None when it is not a readable app snapshot."""
    try:
        directory = Path(snapshot_root or app_constants.INFERENCE_SNAPSHOT_ROOT) / canonical_snapshot_id(snapshot_id)
        with open(directory / SNAPSHOT_MANIFEST_FILENAME, encoding="utf-8") as manifest_file:
            created_at = json.load(manifest_file)["created_at"]
        return datetime.fromisoformat(created_at).timestamp()
    except (OSError, ValueError, KeyError, TypeError):
        return None


def cleanup_abandoned_snapshot_staging(
        snapshot_root=None, older_than_seconds=None):
    """Remove abandoned staging directories older than the cutoff age without touching completed snapshots."""
    root = Path(snapshot_root or app_constants.INFERENCE_SNAPSHOT_ROOT)
    maximum_age = (
        _DEFAULT_STAGING_MAX_AGE_SECONDS
        if older_than_seconds is None
        else older_than_seconds
    )
    if maximum_age < 0:
        raise ValueError("older_than_seconds must be non-negative")
    if not root.exists():
        return 0

    cutoff = time.time() - maximum_age
    removed = 0
    for candidate in root.iterdir():
        if not candidate.name.startswith(".staging-"):
            continue
        try:
            candidate_stat = candidate.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(candidate_stat.st_mode)
                or not stat.S_ISDIR(candidate_stat.st_mode)
                or candidate_stat.st_mtime >= cutoff):
            continue
        try:
            shutil.rmtree(candidate)
        except FileNotFoundError:
            continue
        removed += 1
    return removed
