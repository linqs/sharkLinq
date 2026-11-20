# Core

Shared constants and helper functions that every other package in the app imports.

## Modules

| File | Responsibility |
| --- | --- |
| `app_constants.py` | Ports, size limits, inference-service settings, and every data path (database, images, exports, snapshots). |
| `app_utils.py` | Console banners and timing, input and name validation, SQLite helpers, CSV reading and writing, safe file-name resolution, and media-file name and metadata parsing. |
| `__init__.py` | Marks the package. |

## Configuration

`app_constants.py` is the single authority for paths and limits. Other documents point here instead of repeating values.

| Setting | Source | Effect |
| --- | --- | --- |
| `SHARK_DATA_ROOT` | Environment | Absolute path of the data directory; defaults to the sibling `sm-data` directory. |
| `SHARK_TEST_MODE` | Environment | Set to `true` to use `test_sharkMatcher.db` instead of the live database. |
| `SHARK_INFERENCE_SNAPSHOT_ROOT` | Environment | Overrides where inference snapshots are published. |
| `SHARK_INFERENCE_API_KEY` | Environment | Shared credential for the inference service. |
| `INFERENCE_SERVICE_URL` | Environment | Inference service address. |
| `SHARK_MAX_JSON_BYTES`, `SHARK_MAX_UPLOAD_BYTES`, `SHARK_MAX_IMAGE_BYTES`, `SHARK_MAX_IMAGE_PIXELS`, `SHARK_MAX_UPLOAD_ITEMS`, `SHARK_MAX_PAGE_SIZE` | Environment | Request and image limits. |
| `SHARK_LISTEN_BACKLOG` | Environment | Queued connections the server accepts before refusing new ones; default `128`. |
| `DEFAULT_PORT`, `DEFAULT_HOST` | Constant | Web server defaults. |

Read `app_constants.py` for the current default of each setting.

## Usage

The package has no command-line entry point. Import it from other modules:

```python
from core import app_constants
from core import app_utils
```

The media-metadata helpers in `app_utils.py` (`get_all_metadata`, `get_key_metadata`) shell out to the `ffprobe` binary. No other module in the app calls them.

## Boundaries

- `core` imports nothing else from this repository.
- `backend`, `frontend`, `db`, `scripts`, and `demo` all import `core`.
- Database record logic belongs in `backend/db/`, not in `app_utils.py`.
- Tests for these modules live in `tests/core/`.
