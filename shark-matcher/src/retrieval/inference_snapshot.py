"""Validate app-owned inference snapshots at the matcher file boundary."""

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import stat

from pydantic import ValidationError

from src.core import constants
from src.contracts import SnapshotManifestV1, canonical_uuid4

_CSV_COLUMN_COUNTS = {
    "individuals.csv": 7,
    "media.csv": 10,
}


class SnapshotNotFoundError(FileNotFoundError):
    """Indicate that a valid snapshot identifier has not been published."""


class SnapshotValidationError(ValueError):
    """Indicate that a snapshot is malformed or violates its file boundary."""


@dataclass(frozen=True)
class ValidatedInferenceSnapshot:
    """Resolved paths and manifest for one fully validated snapshot."""

    snapshot_id: str
    directory: Path
    individuals_csv: Path
    media_csv: Path
    manifest: SnapshotManifestV1


def load_inference_snapshot(snapshot_id, snapshot_root=None):
    """Validate one snapshot and return immutable resolved input paths."""
    try:
        canonical_id = canonical_uuid4(snapshot_id, "snapshot_id")
    except ValueError as exc:
        raise SnapshotValidationError(str(exc)) from exc
    root = Path(snapshot_root or constants.INFERENCE_SNAPSHOT_ROOT).resolve()
    snapshot_path = root / canonical_id
    _require_directory(snapshot_path, canonical_id)

    resolved_directory = snapshot_path.resolve(strict=True)
    if resolved_directory.parent != root:
        raise SnapshotValidationError(
            "snapshot directory must be a direct child of the configured root"
        )

    manifest_path = snapshot_path / "manifest.json"
    _require_regular_file(manifest_path)
    manifest = _load_manifest(manifest_path)
    if manifest.snapshot_id != canonical_id:
        raise SnapshotValidationError(
            "snapshot manifest identifier does not match requested snapshot"
        )

    resolved_files = {}
    for filename, expected_columns in _CSV_COLUMN_COUNTS.items():
        file_path = snapshot_path / filename
        _require_regular_file(file_path)
        resolved_path = file_path.resolve(strict=True)
        if resolved_path.parent != resolved_directory:
            raise SnapshotValidationError(
                f"snapshot file is outside its directory: {filename}"
            )
        file_manifest = manifest.files[filename]
        if _sha256(resolved_path) != file_manifest.sha256:
            raise SnapshotValidationError(
                f"snapshot file hash does not match manifest: {filename}"
            )
        _validate_csv(
            resolved_path,
            expected_columns=expected_columns,
            expected_rows=file_manifest.rows,
        )
        resolved_files[filename] = resolved_path

    return ValidatedInferenceSnapshot(
        snapshot_id=canonical_id,
        directory=resolved_directory,
        individuals_csv=resolved_files["individuals.csv"],
        media_csv=resolved_files["media.csv"],
        manifest=manifest,
    )


def _require_directory(path, snapshot_id):
    """Require a real, nonsymlinked snapshot directory."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise SnapshotNotFoundError(
            f"inference snapshot not found: {snapshot_id}"
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise SnapshotValidationError(
            "snapshot path must be a nonsymlinked directory"
        )


def _require_regular_file(path):
    """Require one existing, nonsymlinked regular snapshot file."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise SnapshotValidationError(
            f"required snapshot file is missing: {path.name}"
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise SnapshotValidationError(
            f"snapshot file must be a nonsymlinked regular file: {path.name}"
        )


def _sha256(path):
    """Return a streaming lowercase SHA-256 digest for one snapshot file."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SnapshotValidationError(
            f"could not read snapshot file: {path.name}"
        ) from exc
    return digest.hexdigest()


def _validate_csv(path, expected_columns, expected_rows):
    """Validate the exact column count and data-row count of a snapshot CSV."""
    row_count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as input_file:
            for row in csv.reader(input_file):
                if len(row) != expected_columns:
                    raise SnapshotValidationError(
                        f"snapshot CSV has an invalid column count: {path.name}"
                    )
                row_count += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SnapshotValidationError(
            f"could not parse snapshot CSV: {path.name}"
        ) from exc
    if row_count != expected_rows:
        raise SnapshotValidationError(
            f"snapshot CSV row count does not match manifest: {path.name}"
        )


def _load_manifest(path):
    """Read and validate the strict version-1 snapshot manifest."""
    try:
        return SnapshotManifestV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise SnapshotValidationError("snapshot manifest is invalid") from exc
