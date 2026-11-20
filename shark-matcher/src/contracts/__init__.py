"""Stable cross-repository contracts used by the inference service."""

from src.contracts.artifacts import (
    ArtifactFileV1,
    canonical_uuid4,
    EMBEDDINGS_METADATA_FILENAME,
    EmbeddingsMetadataV1,
    GENERATION_SCHEMA_VERSION,
    GenerationManifestV1,
    GenerationPointerV1,
    ModelDiagnostic,
    ModelSpec,
    ModelState,
    ModelStatus,
    Rerank,
    RetrievalMode,
    SnapshotFileV1,
    SnapshotManifestV1,
)

__all__ = [
    "ArtifactFileV1",
    "canonical_uuid4",
    "EMBEDDINGS_METADATA_FILENAME",
    "EmbeddingsMetadataV1",
    "GENERATION_SCHEMA_VERSION",
    "GenerationManifestV1",
    "GenerationPointerV1",
    "ModelDiagnostic",
    "ModelSpec",
    "ModelState",
    "ModelStatus",
    "Rerank",
    "RetrievalMode",
    "SnapshotFileV1",
    "SnapshotManifestV1",
]
