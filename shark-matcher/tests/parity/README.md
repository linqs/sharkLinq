# Parity Tests

The `tests/parity` folder holds opt-in tests that check the CSV store returns the same results as the live SQLite database for one dataset.

## Modules

| File | Responsibility |
| --- | --- |
| `conftest.py` | Reads `PARITY_DB_PATH` and `PARITY_CSV_DIR`, skips the suite when either is missing, points the CSV store at the CSV directory, and opens the database |
| `test_csv_store_parity.py` | Compares `src.csv_store` media and individual queries with the same queries on the database |

## Entry Points

The root `README.md` section "Parity Tests" gives the command and the two variables it needs, `PARITY_DB_PATH` and `PARITY_CSV_DIR`.

Without both variables the suite skips with a message naming what is missing.

## Boundaries and Dependencies

- Parity tests are outside the standard runner; run them when deploying a new dataset.
- The tests query the database through `sqlite3`; the matcher itself has no database dependency.
- The root `README.md` section "Parity Tests" owns the workflow; `tests/pytest.ini` registers the `parity` marker and leaves this folder out of the default paths.
- The database file belongs to `shark-matcher-app`, and the tests only issue SELECT queries against it.
