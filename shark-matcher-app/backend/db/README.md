# Backend Database Access

Record classes and SQLite query functions that give the rest of the app create, read, update, and delete access to each table.

## Modules

| File | Tables | Responsibility |
| --- | --- | --- |
| `individual.py` | `Individuals`, `AlternateNames`, `Tags`, `Labels` | `Individual` and `AssociatedObject` records, source and media individual relationships, reference selection, filtering, sorting, and counting. |
| `media.py` | `Media` | `Media` records, filtering, and the media-to-individual lookup. |
| `user.py` | `Users`, `Tokens` | `User` and `Token` records, Argon2 password storage, session-token digests and expiry, login, logout, and password updates. |
| `action.py` | `Actions` | `Action` audit-log records and filtering. |
| `proposal.py` | `Proposals` | `Proposal` records for suggested changes and filtering. |
| `__init__.py` | none | Imports `individual` and `media`. |

## Conventions

Each module follows the same shape:

- A record class with `to_dict()` and a `db_to_*` converter from a database row.
- `upsert_*`, `get_*`, `get_matching_*`, and `delete_*` functions that accept an optional open `connection` so callers can group writes in one transaction.
- Upserts return the stored record, deletes return `app_constants.SUCCESS`, and failures return `app_constants.ERROR`.

`create_database.sql` in `db/sql/` is the authority for columns and constraints.

## Running

These modules have no command-line entry point. Import them, for example:

```python
import backend.db.media
```

Calls use the database selected by `app_utils.db_connect()`, which is `DB_PATH` in `core/app_constants.py` unless a `database_scope` is active. Do not call write functions against the live database from an ad hoc shell.

## Boundaries

- Imports `core` only, plus other modules in this package (`individual` and `media` import each other).
- `frontend/` handlers, `db/scripts/`, `db/database.py`, and `demo/` call these modules.
- SQL files and the administration CLI live in top-level `db/`; HTTP handlers live in `frontend/`.
- Tests live in `tests/backend/db/`.
