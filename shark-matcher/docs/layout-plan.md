# Repository Layout

> **Status:** Maintainer record covering `shark-matcher` and `shark-matcher-app`.
> The layout work described here is complete and committed.

## Root Convention

A repository root holds only entry documents, packaging entry points, and top-level folders.

| Item | Location |
| --- | --- |
| Entry doc | `README.md` |
| Packaging | `setup.py`, `MANIFEST.in` (setuptools requires the root) |
| Requirements | `requirements/` |
| Lint and tool config | `config/pylintrc`; app JavaScript tooling in `frontend/` |
| Operator scripts | `scripts/` (app only) |
| Release notes, audits, maintainer notes | `docs/`, lowercase-kebab file names |
| Shared modules | `src/core/` (matcher), `core/` (app) |

## Matcher Layout

- `src/core/` holds `constants.py`, `utils.py`, `path_utils.py`; import with `from src.core import constants`.
- `src/cli/` holds `run_pipeline`, `run_tensorboard`, `convert_data`, `convert_images`.
- `src/config/` holds `config.py` and `config_schema.py`.
- `src/individual_dataset.py` stays in `src/` because stored datasets pickle its module path.
- `src/data_scripts/` and `experimental/data_scripts/` replace the hyphenated folders.
- `experimental/thesis/` holds the thesis evaluation scripts; research tests live in `tests/experimental/`.
- `tests/unit/` mirrors `src/`; `tests/contracts/` holds cross-cutting checks; `tests/tools/` holds the runner and style gate.

## App Layout

- `core/` holds `app_constants.py` and `app_utils.py`; import with `from core import app_constants`.
- `scripts/` holds `shark_identification.py`, `setup_cron.sh` and the generated cron wrapper; it is not packaged.
- `db/sql/` holds the schema and index SQL; `db/database.py` is the administration CLI (`python3 -m db.database`).
- `db/scripts/` holds runnable scripts; `db/scripts/migrations/` holds one-off migrations.
- `frontend/cli/serve.py` replaces `frontend/cli/server.py`.
- `frontend/static/images/` and `frontend/static/js/modules/shark_matcher/` replace the old names.
- `tests/` is grouped by area: `core`, `backend`, `db`, `frontend`, `demo`, `packaging`, `scripts`, `style`, `tools`.

## Decisions to Remember

- Old import names (`constants`, `utils`, `app_constants`) have no compatibility shims.
- The installed app wheel no longer contains `shark_identification`; run it from a checkout.
- `backend/convert_images.py`, the one-off migrations and `db/sql/add_indexes.sql` were kept and documented.
- Large modules were not split; the audits list them as future candidates.

## Open Items

- A root allowlist check is not implemented; link, emoji and path checks are.
