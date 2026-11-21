"""Pydantic schemas for snapshot manifests, embedding generations, and model status."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

EMBEDDINGS_METADATA_FILENAME = "embeddings_metadata.json"
GENERATION_SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 1

SNAPSHOT_FILENAMES = frozenset({"individuals.csv", "media.csv"})

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]
RetrievalMode = Literal["baseline", "psl", "multimodal", "stacked"]
Rerank = Literal["none", "kreciprocal"]


class _StrictContractModel(BaseModel):
    """Base model that rejects unknown fields and prevents mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SnapshotFileV1(_StrictContractModel):
    """Row count and SHA-256 checksum for one snapshot CSV."""

    rows: int = Field(ge=0)
    sha256: Sha256


class SnapshotManifestV1(_StrictContractModel):
    """Manifest for an inference snapshot exported by the web app."""

    schema_version: Literal[SNAPSHOT_SCHEMA_VERSION]
    snapshot_id: str
    created_at: str
    files: dict[str, SnapshotFileV1]

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value):
        """Require a lowercase UUIDv4 snapshot ID."""
        return canonical_uuid4(value, "snapshot_id")

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value):
        """Require a UTC timestamp ending in Z."""
        return _rfc3339_utc(value, "created_at")

    @field_validator("files")
    @classmethod
    def validate_files(cls, value):
        """Require exactly individuals.csv and media.csv."""
        if set(value) != SNAPSHOT_FILENAMES:
            raise ValueError("files must contain exactly individuals.csv and media.csv")
        return value


class ArtifactFileV1(_StrictContractModel):
    """Byte size and SHA-256 checksum for one generation file."""

    size_bytes: int = Field(ge=0)
    sha256: Sha256


class EmbeddingsMetadataV1(_StrictContractModel):
    """Timing and ID metadata stored inside an embedding generation."""

    num_images: int = Field(ge=0)
    start_time: float
    start_time_readable: NonEmptyText
    update_time: float = Field(ge=0)
    update_time_readable: NonEmptyText
    generation_id: str
    snapshot_id: str | None

    @field_validator("generation_id")
    @classmethod
    def validate_generation_id(cls, value):
        """Require a lowercase UUIDv4 generation ID."""
        return canonical_uuid4(value, "generation_id")

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value):
        """Validate the optional snapshot ID."""
        if value is None:
            return value
        return canonical_uuid4(value, "snapshot_id")


class GenerationManifestV1(_StrictContractModel):
    """Manifest for a completed embedding generation."""

    schema_version: Literal[GENERATION_SCHEMA_VERSION]
    generation_id: str
    created_at: str
    embedding_set: NonEmptyText
    model_name: NonEmptyText
    source: Literal["snapshot", "offline_annotations"]
    snapshot_id: str | None
    files: dict[str, ArtifactFileV1]

    @field_validator("generation_id")
    @classmethod
    def validate_generation_id(cls, value):
        """Require a lowercase UUIDv4 generation ID."""
        return canonical_uuid4(value, "generation_id")

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value):
        """Require a UTC timestamp ending in Z."""
        return _rfc3339_utc(value, "created_at")

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value):
        """Validate the optional snapshot ID."""
        if value is None:
            return value
        return canonical_uuid4(value, "snapshot_id")

    @field_validator("files")
    @classmethod
    def validate_files(cls, value):
        """Require flat filenames and include embeddings_metadata.json."""
        for filename in value:
            if (not filename or filename in {".", "..", "manifest.json"}
                    or "/" in filename or "\\" in filename):
                raise ValueError("generation artifact names must be flat safe filenames")
        if EMBEDDINGS_METADATA_FILENAME not in value:
            raise ValueError("files must include embeddings_metadata.json")
        return value

    @model_validator(mode="after")
    def validate_source(self):
        """Ensure snapshot_id is set if and only if source is snapshot."""
        if self.source == "snapshot" and self.snapshot_id is None:
            raise ValueError("snapshot_id is required when source is snapshot")
        if self.source == "offline_annotations" and self.snapshot_id is not None:
            raise ValueError("snapshot_id must be null when source is offline_annotations")
        return self


class GenerationPointerV1(_StrictContractModel):
    """Contents of current.json pointing to the active embedding generation."""

    schema_version: Literal[GENERATION_SCHEMA_VERSION]
    generation_id: str
    published_at: str

    @field_validator("generation_id")
    @classmethod
    def validate_generation_id(cls, value):
        """Require a lowercase UUIDv4 generation ID."""
        return canonical_uuid4(value, "generation_id")

    @field_validator("published_at")
    @classmethod
    def validate_published_at(cls, value):
        """Require a UTC timestamp ending in Z."""
        return _rfc3339_utc(value, "published_at")


class ModelState(str, Enum):
    """Load states reported by the health and status endpoints."""

    NOT_ATTEMPTED = "not_attempted"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    DISABLED = "disabled"


class ModelDiagnostic(_StrictContractModel):
    """Safe machine-readable diagnostic for a model load failure."""

    code: NonEmptyText
    message: NonEmptyText


class ModelStatus(_StrictContractModel):
    """Readiness state for one configured inference model."""

    display_name: NonEmptyText
    embedding_set: NonEmptyText
    required: bool
    state: ModelState
    error: ModelDiagnostic | None = None

    @model_validator(mode="after")
    def validate_state(self):
        """Require diagnostics only for failures and valid disabled states."""
        if self.state == ModelState.FAILED and self.error is None:
            raise ValueError("failed model state requires an error diagnostic")
        if self.state != ModelState.FAILED and self.error is not None:
            raise ValueError("only failed model state may include an error diagnostic")
        if self.state == ModelState.DISABLED and self.required:
            raise ValueError("a disabled model cannot be required")
        return self


class ModelSpec(_StrictContractModel):
    """Specification for one configured inference and retrieval model."""

    display_name: NonEmptyText
    embedding_set: NonEmptyText
    checkpoint: NonEmptyText
    retrieval_mode: RetrievalMode = "multimodal"
    rerank: Rerank = "kreciprocal"
    required: bool = True

    @model_validator(mode="before")
    @classmethod
    def _remap_legacy_fields(cls, data):
        """Map older config keys (name, dir, model) to current field names."""
        if isinstance(data, dict):
            mapped = dict(data)
            if "display_name" not in mapped and "name" in mapped:
                mapped["display_name"] = mapped.pop("name")
            if "embedding_set" not in mapped and "dir" in mapped:
                mapped["embedding_set"] = mapped.pop("dir")
            if "checkpoint" not in mapped and "model" in mapped:
                mapped["checkpoint"] = mapped.pop("model")
            if "embedding_set" not in mapped and "display_name" in mapped:
                mapped["embedding_set"] = mapped["display_name"]
            if "display_name" not in mapped and "embedding_set" in mapped:
                mapped["display_name"] = mapped["embedding_set"]
            if "checkpoint" not in mapped and "embedding_set" in mapped:
                mapped["checkpoint"] = mapped["embedding_set"]
            return mapped
        return data

    @property
    def dir(self) -> str:
        """Alias for embedding_set matching legacy config callers."""
        return self.embedding_set

    @property
    def model(self) -> str:
        """Alias for checkpoint matching legacy config callers."""
        return self.checkpoint

    def get(self, key, default=None):
        """Map legacy dict-style access to attributes."""
        if key == "dir":
            return self.embedding_set
        if key == "model":
            return self.checkpoint
        return getattr(self, key, default)

    def __getitem__(self, key):
        """Map legacy subscript access to attributes."""
        val = self.get(key)
        if val is None and not hasattr(self, key):
            raise KeyError(key)
        return val


def canonical_uuid4(value, field_name):
    """Validate that value is a lowercase UUIDv4 string."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a canonical UUID string") from exc
    if parsed.version != 4 or value != str(parsed):
        raise ValueError(f"{field_name} must be a canonical lowercase UUIDv4 string")
    return value


def _rfc3339_utc(value, field_name):
    """Validate an RFC 3339 timestamp expressed explicitly in UTC."""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field_name} must be an RFC 3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an RFC 3339 UTC timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field_name} must use UTC")
    return value
