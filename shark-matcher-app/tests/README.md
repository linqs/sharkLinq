# Shark Matcher App Tests

The application suite verifies the app-owned database, HTTP handlers, browser
support code, inference-service client boundary, snapshot publication, and
release packaging. Machine-learning training and visualization tests belong to
the separate `shark-match-model` repository.

## Running Tests

Use the repository-owned runner from the `shark-matcher-app/` root:

```bash
# Run the standard deterministic suite.
python3 -m tests.tools.run_tests

# Run only the app backend tests.
python3 -m tests.tools.run_tests --run tests/backend

# Run the existing language-specific style gates.
python3 -m tests.tools.run_tests --py-style
python3 -m tests.tools.run_tests --js-style
```

The runner, `tests/tools/run_tests.py`, owns test selection and options. Use
its `--help` output for the current interface.

## Layout

| Folder | Contents | In standard run |
| --- | --- | :---: |
| `core/` | Tests for `core/` (name and date helpers) | yes |
| `backend/` | Database access modules (`backend/db/`), snapshot publication and its schema contract | yes |
| `db/` | Tests for the `db/database.py` administration commands | yes |
| `frontend/` | HTTP handlers, server behavior, authentication, export boundaries, uploads, pagination contracts (`fixtures/`), inference client | yes |
| `demo/` | Shared matching demo | yes |
| `packaging/` | Publication, dependency, portability, and immutable-fixture contracts | yes |
| `scripts/` | The cron installer | yes |
| `style/` | The Python style gate test | yes |
| `ci-test-data/` | Immutable CI fixture | n/a |
| `tools/` | The test runner and the Python and JavaScript style gates | n/a |
| `browser/` | Browser workflow tests | `--browser` |
| `manual/` | Live-database integrity audit | manual |

This overview follows stable ownership boundaries instead of maintaining a
file-by-file test list.

## Immutable CI Fixture

The deterministic suite rebuilds a disposable test database from
`tests/ci-test-data/`. Its
`README.md` is authoritative: the committed CSV
and TSV content must never be edited, regenerated, replaced, or written by a
test. Add cases with temporary inputs or mocks instead.

The separately provided `sm-data` workspace is sample and reference data for
manual workflows. It is not an alternative input to the standard CI suite. Use
the documented manual workflows when intentionally validating sample data
rather than adding it to the deterministic runner.

## Manual Integrity Check

The live-database integrity audit is deliberately outside the deterministic
suite:

```bash
python3 -m tests.manual.test_data_integrity
```

It reads the configured application database. Use it only when intentionally
auditing that database; normal tests use the disposable CI database instead.
