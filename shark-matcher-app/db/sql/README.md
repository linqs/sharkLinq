# Database SQL

The SQLite schema and performance indexes for the application database.

## Files

| File | Responsibility |
| --- | --- |
| `create_database.sql` | Drops and recreates every table: `Media`, `Individuals`, `AlternateNames`, `Tags`, `Labels`, `Users`, `Tokens`, `Actions`, and `Proposals`. |
| `add_indexes.sql` | Creates the performance indexes with `CREATE INDEX IF NOT EXISTS`, so it is safe to apply repeatedly. |

## Running

Do not run these files by hand against the live database. `python3 -m db.database init` reads them in order through the `sqlite3` command-line tool, and `create_database.sql` deletes existing data.

```bash
python3 -m db.database init --help
```

Use `python3 -m db.database -t init` to build the disposable test database instead. The `init` section of `db/README.md` lists the warnings and the `--no-indexes` option.

## Boundaries

- `DB_SQL_DIR` in `core/app_constants.py` locates this folder.
- `create_database.sql` is the authority for columns and constraints; `backend/db/` mirrors them in Python record classes.
- Changing a table here also means updating the matching `backend/db/` module and `db/database.py` if migration or population is affected.
- Tests rebuild a disposable database from these files through `tests/conftest.py`.
