# Backend

Application-domain and data-access logic, plus the publication of inference snapshots for the matcher.

## Modules

| Path | Responsibility |
| --- | --- |
| `db/` | Record classes and SQLite access for individuals, media, users, actions, and proposals; see `backend/db/README.md`. |
| `inference_snapshot.py` | Creates an immutable snapshot of the public Individuals and Media tables and publishes it atomically under the snapshot root. |
| `inference_snapshot_contract.py` | Builds and validates the snapshot manifest (identifier, timestamp, file names, row counts, SHA-256 digests) without any I/O. |
| `convert_images.py` | Image-conversion helper with no importers in this repository; see Image Conversion Helper below. |
| `__init__.py` | Marks the package. |

## Inference Snapshots

The embedding-update handler in `frontend/project.py` calls `create_inference_snapshot()` once per request. The returned identifier is sent to the matcher, which validates the snapshot before generating embeddings.

- Staging directories are named `.staging-` plus the snapshot identifier and are renamed to the identifier when complete.
- `cleanup_abandoned_snapshot_staging()` removes staging directories older than a cutoff (24 hours by default). Only the tests call it in this repository.
- The snapshot root comes from `INFERENCE_SNAPSHOT_ROOT` in `core/app_constants.py`.
- The `db/README.md` section Inference Snapshots describes the workflow from the database side.

## Image Conversion Helper

`backend/convert_images.py` converts image directories to standardised JPEGs. It is a copy kept in sync with `src/cli/convert_images.py` in the matcher repository (`shark-matcher`), and no module in this repository imports it.

Change the copy and the original together, or change neither.

## Running

The package has no command-line entry point. `inference_snapshot.py` has no `__main__` block and runs only when the web server's embedding update calls it.

## Boundaries

- Imports `core` for constants and utilities.
- `inference_snapshot.py` reuses the public table export in `db/scripts/db_dump.py`, which in turn reads through `backend/db/`.
- Never imports `frontend`; HTTP handlers call into `backend`, not the reverse.
- Naming pairing: `backend/db/` is data access, `frontend/` is HTTP handlers, and top-level `db/` is SQL and administration. See the Layout and Naming Pairing section of `db/README.md`.
- Tests live in `tests/backend/`.
