"""Standard-library builder for the app-owned V1 snapshot manifest.

Database reads and filesystem publication belong to the V1.2 implementation;
this module only constructs the strict cross-repository wire representation.
"""

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
    """Validate and return a canonical lowercase UUIDv4 snapshot ID."""
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
    """Format an aware datetime as an RFC 3339 UTC timestamp ending in Z."""
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("created_at must be a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SnapshotFile:
    """Row count and content digest for one snapshot CSV file."""

    rows: int
    sha256: str

    def __post_init__(self):
        """Validate the immutable file entry after construction."""
        if isinstance(self.rows, bool) or not isinstance(self.rows, int) or self.rows < 0:
            raise ValueError("rows must be a non-negative integer")
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("sha256 must be a lowercase hexadecimal SHA-256 digest")

    def as_dict(self):
        """Return the JSON-serializable file entry."""
        return {"rows": self.rows, "sha256": self.sha256}


def build_snapshot_manifest(snapshot_id, created_at, files: Mapping[str, SnapshotFile]):
    """Build a strict version-1 manifest without performing any I/O."""
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
