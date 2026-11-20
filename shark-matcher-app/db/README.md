# Shark Matcher Database

This directory contains schema definitions and utility scripts for the
application-owned SQLite database. The database itself lives in the configured
shared data workspace; its authoritative path is defined in
[`core/app_constants.py`](../core/app_constants.py).

## Layout and Naming Pairing

The same layers appear under parallel names across the repository:

| Location | Holds |
| --- | --- |
| `frontend/` | HTTP handlers: one module per API area, plus the server and the browser interface. |
| `backend/db/` | Data access: record classes and SQLite queries per table. |
| `db/sql/` | The schema and index SQL; see `db/sql/README.md`. |
| `db/database.py` | The administration CLI, run as `python3 -m db.database`. |
| `db/scripts/` | Runnable scripts: export, backup, sample data, and migrations; see `db/scripts/README.md`. |

Handlers in `frontend/` call `backend/db/` for data. Only `db/` and its scripts change the schema.

## Scripts

> [!TIP]
> Database management is orchestrated through
> [`db/database.py`](database.py). The commands below illustrate
> its principal workflows. Run `python3 -m db.database --help`, followed
> by the selected subcommand's `--help`, for current options and defaults.

A new database is set up with `init`, then `populate`, then `provision-owner`.
A database created before expiring sessions were introduced runs `migrate-auth`
instead of `init`.

### 1. Initialize Database (`init`)

Initialize or completely reset the configured database structure:

```bash
python3 -m db.database init
```

> [!WARNING]
> `init` is destructive. It drops existing tables and recreates an empty schema
> from [`create_database.sql`](sql/create_database.sql), then applies the indexes
> owned by [`sql/add_indexes.sql`](sql/add_indexes.sql). Back up any
> database that must be retained before running it.

`init` stops without changing anything if the database already holds media,
users, actions, proposals, or sessions. Pass `--force` to reset it anyway, and
`--no-indexes` to skip the performance indexes when benchmarking.

### 2. Upgrade Authentication Schema (`migrate-auth`)

Existing databases created before expiring sessions were introduced must be
upgraded before the app server will start:

```bash
python3 -m db.database migrate-auth
```

- The migration preserves users and invalidates legacy sessions, so every user
  must log in again.
- It is safe to rerun.
- Fresh databases created by `init` already have the current schema.
- The migration implementation in [`database.py`](database.py) is
  authoritative for compatibility checks.

### 3. Populate Database (`populate`)

This command reads prepared CSV files and imports them into the SQLite database.

- It performs the update against a temporary database.
- It runs the existing data integrity suite.
- It replaces the configured database only after those checks succeed.

```bash
# Populate application dataset tables from one dataset directory.
python3 -m db.database populate --data-path DATASET

# Restore every table from a complete trusted snapshot.
python3 -m db.database populate --all --data-path SNAPSHOT_DIRECTORY
```

- `populate` stops without changing anything if the database already holds
  data; pass `--overwrite` to replace it.
- `--force` is separate: it publishes even when the integrity checks fail.

#### The Data Path

- `--data-path` accepts an existing directory of table CSVs, absolute or
  relative to the current directory.
- Any other value names a directory under the data root's `database-format/`
  directory.
- It is required; the command stops with an error when it is missing.

#### Matching Export

Populate from the same database-format export that the matcher's active
embedding set was generated from. The matcher returns media identifiers from
that set, so a database populated from a different export shows missing or
wrong matches.

#### Restore Modes

- Without the full-restore option, population replaces application dataset rows
  while preserving authentication and operational records.
- A full restore truncates every table and imports authentication, session, and
  operational data from a trusted administrative snapshot.
- Selective administrative restores and the integrity-bypass option are
  available only through this local command; review their current contract with
  `populate --help` before use.
- [`handle_populate()`](database.py) owns staging, integrity validation,
  and replacement behavior. Do not reproduce its table choices or parser
  defaults in other documentation.

### 4. Provision First Owner (`provision-owner`)

After initializing and populating a new database, provision its first owner
through the trusted local command. A default `populate` does not touch the user
tables; a full `--all` restore brings its own users, so this step is not needed
after one. Replace the uppercase values with the intended account
details:

```bash
python3 -m db.database provision-owner \
    --display-name "DISPLAY_NAME" \
    --email "EMAIL_ADDRESS" \
    --username "USERNAME"
```

The command prompts for the password and confirmation without accepting a
password argument. It refuses to run when the database already contains any
user and delegates credential storage to the existing application user
implementation. The authoritative provisioning and password-derivation
behavior lives in [`database.py`](database.py) and
[`backend/db/user.py`](../backend/db/user.py).

#### Reset User Password

To reset an existing user's password directly from the `shark-matcher-app`
directory, invoke the existing password-derivation and update helpers:

```bash
python3 -c 'from backend.db.user import derive_client_password, update_user_password; from core import app_constants; status = update_user_password("USERNAME", derive_client_password("NEW_PASSWORD")); print("Success" if status == app_constants.SUCCESS else "Failed")'
```

This derives the SHA-256 client representation, rehashes it with Argon2, and
updates the user record in the configured database.

### 5. Debug Database (`print`)

Print database diagnostics to the console:

```bash
python3 -m db.database print
```

The command displays row counts and diagnostic samples for application data,
followed by index and image-sync summaries. Authentication tables are
count-only, so password hashes and stored session digests are not printed.

### 6. Dump Database (`dump`)

Write the current database to table CSV files and the supported annotations
format through the existing dump implementation:

```bash
python3 -m db.database dump
```

The script writes beneath the configured application data workspace. Output
locations and filenames are owned by [`core/app_constants.py`](../core/app_constants.py)
and [`db_dump.py`](scripts/db_dump.py).

#### Trusted Output

> [!IMPORTANT]
> The local `dump` command is a trusted server-side administrative operation.
> It intentionally includes stored password representations, session digests
> and expirations, and operational records needed for a complete backup.
> Protect its output like the live database.

#### Browser Exports

Browser-triggered exports cannot select this trusted mode:

- Both supported browser export paths force the public mode in
  [`frontend/project.py`](../frontend/project.py).
- Public table extraction in [`db_dump.py`](scripts/db_dump.py) excludes
  authentication and session tables.
- `ExportMode` is selected by server code rather than accepted from request data.

### 7. Remove Unreferenced Images (`clear-images`)

Delete files in the configured image directory that no `Media` record
references:

```bash
python3 -m db.database clear-images
```

> [!WARNING]
> `clear-images` deletes files without asking, including images that belong to
> other datasets sharing the folder. Run `print` first; its image-sync summary
> counts the same unreferenced files and names the first ten.

### Inference Snapshots

The application owns creation of immutable inference snapshots for the matcher.
A snapshot contains only the public individual and media projections plus a
validation manifest. Snapshot publication reuses the public table-extraction
path while adding one consistent database transaction, staging cleanup, and
atomic publication.

The publisher and cleanup behavior live in
[`backend/inference_snapshot.py`](../backend/inference_snapshot.py); its schema
contract lives in
[`backend/inference_snapshot_contract.py`](../backend/inference_snapshot_contract.py).
The embedding update handler creates one snapshot per request and sends its
identifier to the matcher for every selected model. The matcher validates that
snapshot before generation and does not import this database layer.

### Other Scripts

Sample-dataset creation, the export and backup modules, and the one-off migrations
are documented in the `db/scripts/README.md` Modules, Running, and Migrations
sections. Run them from the repository root with `python3 -m db.scripts.NAME`.

## Database Schema

[`create_database.sql`](sql/create_database.sql) is the authoritative table,
column, constraint, and relationship definition. The modules under
`backend/db/` own application-level record behavior. Refer to
those sources instead of maintaining a second schema inventory here.

At a conceptual level, the database separates media assets, observed
individual records, grouped source individuals, descriptive attributes,
authentication and session state, and curation history. That separation
supports the source/media relationship below while keeping privileged
authentication data out of public exports and inference snapshots.

## Example and CI Data

The separately provided `sm-data` sample workspace holds one dataset,
`pre2019-sample-data`, under `database-format/`; it matches the bundled
embedding set.

The test suite instead uses `tests/ci-test-data/`,
whose documented CSV and TSV contents are immutable and must never be edited,
written, or regenerated. `python3 -m scripts.shark_identification --test --repopulate` reads it
the same way to rebuild the disposable test database. Operational examples and
manual walkthroughs should use the external sample workspace, not that CI
fixture.

## Concepts: Individuals vs Media

The database distinguishes between **Source Individuals** and **Media
Individuals** to represent a real-world animal separately from its recorded
observations.

### Source Individuals

A **Source Individual** represents a distinct, real-world shark and aggregates
its observation-specific individual records.

### Media Individuals

A **Media Individual** represents one observation or sighting, links that
observation to its media record, and groups it beneath a Source Individual.
Multiple observations can therefore belong to one real-world shark.

> [!NOTE]
> Several photographs of one shark produce one Source Individual, one Media
> record per photograph, and one Media Individual per observation. The schema
> and backend modules define the exact stored fields and relationships.

## Performance Indexes

Performance indexes are defined in
[`sql/add_indexes.sql`](sql/add_indexes.sql). The `init` command applies
that file after the schema, and its idempotent definitions can also be applied
to an existing configured database without recreating it. The SQL file and
queries under `backend/db/` are the authorities for exact
index names, columns, and optimized access paths.

## Integrity Tests

A manual integrity suite checks the configured database for relationship and
authorship errors. The population workflow already runs it against the staged
database before publication; run it directly when auditing the live database:

```bash
python3 -m tests.manual.test_data_integrity
```

The suite reports its current checks and database statistics. Its implementation
in
[`tests/manual/test_data_integrity.py`](../tests/manual/test_data_integrity.py)
is authoritative for the enforced rules.
