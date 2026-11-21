"""Build and validate the JSON manifest for an inference snapshot."""

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Mapping
from uuid import UUID


SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_MANIFEST_FILENAME = "manifest.json"
SNAPSHOT_FILENAMES = frozenset({"individuals.csv", "media.csv"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_snapshot_id(value):
    """Validate that a snapshot ID is a lowercase UUIDv4 string."""
    if not isinstance(value, str):
        raise ValueError("snapshot_id must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("snapshot_id must be a canonical UUID string") from exc
    if parsed.version != 4 or value != str(parsed):
        raise ValueError("snapshot_id must be a canonical lowercase UUIDv4 string")
    return value


def format_utc_timestamp(value):
    """Convert a timezone-aware datetime to a UTC timestamp string ending in Z."""
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("created_at must be a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SnapshotFile:
    """Row count and SHA-256 checksum for a single snapshot CSV file."""

    rows: int
    sha256: str

    def __post_init__(self):
        """Validate the row count and SHA-256 checksum."""
        if isinstance(self.rows, bool) or not isinstance(self.rows, int) or self.rows < 0:
            raise ValueError("rows must be a non-negative integer")
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("sha256 must be a lowercase hexadecimal SHA-256 digest")

    def as_dict(self):
        """Return the file metadata as a plain dictionary."""
        return {"rows": self.rows, "sha256": self.sha256}


def build_snapshot_manifest(snapshot_id, created_at, files: Mapping[str, SnapshotFile]):
    """Build and validate the manifest dictionary for a snapshot."""
    canonical_id = canonical_snapshot_id(snapshot_id)
    if set(files) != SNAPSHOT_FILENAMES:
        raise ValueError("files must contain exactly individuals.csv and media.csv")
    if not all(isinstance(file_info, SnapshotFile) for file_info in files.values()):
        raise TypeError("files values must be SnapshotFile instances")

    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": canonical_id,
        "created_at": format_utc_timestamp(created_at),
        "files": {
            filename: files[filename].as_dict()
            for filename in sorted(SNAPSHOT_FILENAMES)
        },
    }
    return manifest
