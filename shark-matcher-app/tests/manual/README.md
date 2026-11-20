# Manual Tests

A data-integrity audit for the application database, kept outside the deterministic test run.

## Modules

| File | Responsibility |
| --- | --- |
| `test_data_integrity.py` | `unittest` checks of individual and media linkage, reference constraints, media validity, orphaned associated objects, and authentication-table relationships, followed by a row count per table. |

## Running

Run it from the repository root. It reads the configured live database and writes nothing.

```bash
python3 -m tests.manual.test_data_integrity
```

| Setting | Effect |
| --- | --- |
| `SHARK_INTEGRITY_DB_PATH` | Audit this database file instead of `DB_PATH` from `core/app_constants.py`. |
| `SHARK_TEST_MODE=true` | Audit the test database. |
| `SHARK_DATA_ROOT` | Select another data directory. |

The report lists any issues it found and ends with the statistics. Use it only when intentionally auditing a database.

## Where It Also Runs

`python3 -m db.database populate` runs this module against its temporary database and aborts the replacement if it fails, unless `--force` is given. The module therefore must stay importable and runnable from a checkout.

## Boundaries

- Excluded from `python3 -m tests.tools.run_tests` by default; the runner ignores `tests/manual`.
- Run it explicitly with `python3 -m tests.tools.run_tests --run tests/manual`, or as the module above.
- Reads through `backend/db/` and direct SQLite queries; it never modifies data.
- The `db/README.md` Integrity Tests section describes the audit from the database side.
