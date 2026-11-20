# Database Scripts

Runnable Python modules for database export, scheduled backups, sample-dataset creation, and one-off data migrations.

This file is the authoritative reference for these scripts. The `db.database` administration commands are in `db/README.md`.

## Modules

| File | Responsibility |
| --- | --- |
| `db_dump.py` | Exports the database to table CSV files and an annotations file; `ExportMode.PUBLIC` omits authentication tables and `ExportMode.TRUSTED` includes them. |
| `backup_runner.py` | Background loop that writes a complete trusted backup each Sunday between 00:00 and 01:00. |
| `create_sample_dataset.py` | Builds a smaller cutoff-year sample dataset (images, table CSVs, annotations) from existing archives. |
| `migrations/` | One-off data migrations; see the Migrations section. |
| `__init__.py` | Marks the package. |

## Running

Run every script from the repository root as a module.

| Script | Command | Notes |
| --- | --- | --- |
| `db_dump` | `python3 -m db.database dump` | Runs `db_dump.py` for the live database; add `-t` before `dump` for the test database. Writes under `data-out/` in the data root. |
| `backup_runner` | Started by `python3 -m frontend.cli.serve` | Has no options; `python3 -m db.scripts.backup_runner` starts the loop and waits for Sunday. |
| `create_sample_dataset` | `python3 -m db.scripts.create_sample_dataset --help` | Read the options before running; use `--dry-run` first. |

The `dump` subcommand and its trusted-output warning are described in the Dump Database section of `db/README.md`.

### Dump Output

`db_dump.py` writes a new timestamped directory on each run:

- `data-out/annotations/TIMESTAMP/raw_annotations.csv` for the annotations format.
- `data-out/database-format/TIMESTAMP/` for the table CSV files.

Both paths are relative to the data root and are built from `OUT_DATA_DIR` in `core/app_constants.py`.

### Sample Dataset Options

`create_sample_dataset.py` copies and filters data without deleting inputs.

| Option | Effect |
| --- | --- |
| `--input-images-dir`, `--output-images-dir` | Source and destination image directories. |
| `--input-csv-dir`, `--output-csv-dir` | Source and destination table CSV directories. |
| `--input-annotations-dir`, `--output-annotations-dir` | Source and destination annotations split directories. |
| `--cutoff-year` | Inclusive cutoff year; default `2018`. |
| `--reference-policy` | `latest_pre`, `oldest`, or `lowest_id` for replacement reference individuals. |
| `--no-copy-images` | Write CSV files only. |
| `--annotations-only`, `--no-annotations` | Restrict the run to, or skip, the annotations split. |
| `--dry-run` | Preview counts and integrity without writing files. |

Defaults for every directory are set in the script and resolve under the data root.

## Migrations

`migrations/` holds one-off data migrations that run once per database. Run each as `python3 -m db.scripts.migrations.NAME` from the repository root.

| Name | Responsibility | Options |
| --- | --- | --- |
| `backfill_media_dates` | Fills the date of `Media` rows that have none by parsing the timestamp in the media name; unparseable names are skipped. | `--dry-run` previews the changes without writing. |
| `add_cwsp_labels` | Adds a `CWSP` label to individuals linked to media from the `FAR` or `PR` locations dated on or after 2020-01-01; individuals that already have the label are skipped. | None. |

```bash
python3 -m db.scripts.migrations.backfill_media_dates --help
python3 -m db.scripts.migrations.backfill_media_dates --dry-run
```

> [!WARNING]
> `add_cwsp_labels` has no option parser, so even `--help` starts the migration immediately. Both migrations write to the configured database; back it up first and set `SHARK_TEST_MODE=true` to target the test database.

## Boundaries

- `db_dump`, `backup_runner`, and the migrations use the database selected in `core/app_constants.py`; none takes a database path option.
- `create_sample_dataset` reads and writes files only and never opens the database.
- Table reads and writes go through `backend/db/`, apart from `backfill_media_dates`, which issues its own SQL through `core.app_utils`.
- `backend/inference_snapshot.py` and `frontend/project.py` import `db_dump.py`; `frontend/cli/serve.py` imports `backup_runner.py`.
- Naming pairing: top-level `db/` holds SQL (`db/sql/`), the administration CLI (`db/database.py`), and these scripts. See the Layout and Naming Pairing section of `db/README.md`.
