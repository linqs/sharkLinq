# Contracts

The `src/contracts` package defines the strict version-1 data models that fix the shapes shared between the matcher and `shark-matcher-app`.

## Modules

| File | Responsibility |
| --- | --- |
| `artifacts.py` | The version-1 models and constants listed below |
| `__init__.py` | Re-exports the public names from `artifacts.py` |

### Models in `artifacts.py`

| Name | Describes |
| --- | --- |
| `SnapshotManifestV1`, `SnapshotFileV1` | An app-owned inference snapshot: identifier, creation time, and the two hashed CSV files |
| `GenerationManifestV1`, `ArtifactFileV1` | One complete embedding generation and its hashed artifact files |
| `GenerationPointerV1` | The atomic pointer to the active generation |
| `EmbeddingsMetadataV1` | Metadata stored inside a completed generation |
| `ModelSpec` | One configured model: display name, embedding set, checkpoint, retrieval mode, rerank, and whether it is required |
| `ModelStatus`, `ModelState`, `ModelDiagnostic` | Readiness state of one model and the safe diagnostic for a load failure |
| `canonical_uuid4` | Validates a lowercase UUIDv4 identifier |

## Entry Points

None. This package is imported, never run.

## Boundaries and Dependencies

- Every model rejects unknown fields and is immutable.
- Identifiers must be canonical lowercase UUIDv4 strings, and manifest and pointer timestamps must be RFC 3339 UTC ending in `Z`.
- `ModelSpec` still accepts the older field names `name`, `dir`, and `model` and maps them to the current ones.
- The package imports only Pydantic and the standard library.
- It is used by `src/api/server.py`, `src/retrieval/retrieval.py`, `src/retrieval/generation_store.py`, and `src/retrieval/inference_snapshot.py`.
- Changing a version-1 shape changes the contract with `shark-matcher-app`; `tests/contracts` pins these shapes.
