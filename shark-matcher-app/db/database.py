"""
Orchestration script for database initialization, migration, provisioning,
population, backup, and diagnostics.
"""

import argparse
import contextlib
import getpass
import os
import pathlib
import sqlite3
import subprocess
import sys

# Catch test flag early to set environment variable before importing constants.
if "-t" in sys.argv or "--test" in sys.argv:
    os.environ["SHARK_TEST_MODE"] = "true"

from core import app_constants
from core import app_utils
import backend.db.user


def _configured_database_path(args):
    """Return the configured database path, honoring a subcommand name override."""
    database_name = getattr(args, "name", None)
    if database_name:
        return os.path.join(app_constants.DB_DIR, database_name)
    return app_constants.DB_PATH


def _database_is_initialized(db_path):
    """Return whether db_path exists with the application schema, without creating the file."""
    if not os.path.exists(db_path):
        return False
    read_only_uri = pathlib.Path(db_path).resolve().as_uri() + "?mode=ro"
    with contextlib.closing(sqlite3.connect(read_only_uri, uri=True)) as connection:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'Media'"
        ).fetchone()
    return row is not None


def _database_has_data(db_path):
    """Return whether an initialised database holds any rows that init would destroy."""
    if not _database_is_initialized(db_path):
        return False
    read_only_uri = pathlib.Path(db_path).resolve().as_uri() + "?mode=ro"
    with contextlib.closing(sqlite3.connect(read_only_uri, uri=True)) as connection:
        return any(
            connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None
            for table in ("Media", "Users", "Actions", "Proposals", "Tokens"))


def _database_has_users(db_path):
    """Return whether an initialised database already holds a user account."""
    if not _database_is_initialized(db_path):
        return False
    read_only_uri = pathlib.Path(db_path).resolve().as_uri() + "?mode=ro"
    with contextlib.closing(sqlite3.connect(read_only_uri, uri=True)) as connection:
        return connection.execute("SELECT 1 FROM Users LIMIT 1").fetchone() is not None


def _sqlite_shell_arg(path):
    """Quote a path as a double-quoted sqlite3 dot-command argument so spaces, quotes and backslashes survive."""
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def run_sqlite_cmd(db_path, cmd_string):
    """Run a pipeline of commands in sqlite3 CLI."""
    try:
        result = subprocess.run(["sqlite3", db_path], input=cmd_string, text=True, check=False, capture_output=True)
    except FileNotFoundError:
        print("[ERROR] The sqlite3 command-line tool is required; install it and make sure it is on PATH.")
        sys.exit(1)
    
    if result.stdout:
        print(result.stdout.strip())
        
    if result.returncode != 0:
        if result.stderr:
            print(f"[STDERR sqlite3]: {result.stderr.strip()}")
        print(f"[ERROR] sqlite3 command failed.")
        sys.exit(result.returncode)

def run_python_module(module_name, args=None, env=None):
    """Run a python module as a subprocess."""
    cmd = [sys.executable, "-m", module_name]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        print(f"[ERROR] Python module {module_name} failed.")
        sys.exit(result.returncode)

def handle_dump(args):
    """Handle the dump command."""
    print(f"Dumping {'Test' if args.test else 'Production'} Database...")
    
    env = os.environ.copy()
    if args.test:
        env["SHARK_TEST_MODE"] = "true"
        
    run_python_module("db.scripts.db_dump", env=env)

def get_image_sync_status():
    """Return (orphaned, missing): image-directory files absent from Media, and Media filenames absent from the directory."""
    images_dir = app_constants.IMAGES_DIR
    
    # Get filenames from the frontend image directory.
    if os.path.isdir(images_dir):
        dir_files = {f for f in os.listdir(images_dir)
                     if os.path.isfile(os.path.join(images_dir, f)) and not f.startswith(".")}
    else:
        dir_files = set()

    # Get filenames from the Media table.
    result = app_utils.execute_sql("SELECT filename FROM Media WHERE filename IS NOT NULL")
    db_files = {row[0] for row in result} if result and result != app_constants.ERROR else set()

    orphaned_files = dir_files - db_files   # In folder, not in DB.
    missing_files = db_files - dir_files    # In DB, not in folder.
    return orphaned_files, missing_files

def print_indexes():
    """Print a summary table of all performance indexes present in the database."""
    result = app_utils.execute_sql(
        "SELECT name, tbl_name FROM sqlite_master "
        "WHERE type='index' AND name LIKE 'idx_%' "
        "ORDER BY tbl_name, name;"
    )

    EXPECTED_INDEXES = {
        "idx_individuals_source", "idx_individuals_media",
        "idx_individuals_reference", "idx_individuals_name",
        "idx_media_quality", "idx_media_location",
        "idx_media_time", "idx_media_name",
        "idx_alternatenames_individual", "idx_alternatenames_name",
        "idx_tags_individual", "idx_tags_name",
        "idx_labels_individual", "idx_labels_name",
    }

    if not result or result == app_constants.ERROR:
        print("  (no performance indexes found)")
        return

    present = {row[0] for row in result}
    missing = EXPECTED_INDEXES - present

    col_w = (32, 16, 8)
    header = f"  {'Index':<{col_w[0]}}  {'Table':<{col_w[1]}}  {'Status':<{col_w[2]}}"
    divider = "  " + "-" * (sum(col_w) + 4)
    print(header)
    print(divider)
    for name, table in result:
        print(f"  {name:<{col_w[0]}}  {table:<{col_w[1]}}  {'OK':<{col_w[2]}}")
    for name in sorted(missing):
        table = name.split("_")[1].capitalize()
        print(f"  {name:<{col_w[0]}}  {table:<{col_w[1]}}  {'MISSING':<{col_w[2]}}")
    print(f"\n  {len(present)}/{len(EXPECTED_INDEXES)} expected indexes present.")

def handle_print(args):
    """Handle the print command."""
    db_path = app_constants.DB_PATH
    print(f"Using Database: {db_path}")

    table_names = ["Media", "Individuals", "AlternateNames", "Tags", "Labels", "Users", "Tokens", "Actions", "Proposals"]
    summary_only_tables = {"Users", "Tokens"}

    print("Printing Heads\n")
    for table in table_names:
        print(f"{table}:")
        command = f"SELECT COUNT(*) from {table};\n"
        if table not in summary_only_tables:
            command += f"SELECT * from {table} LIMIT 5;\n"
        run_sqlite_cmd(db_path, command)
        print()

    # Performance index summary.
    print("Performance Indexes:")
    print_indexes()
    print()

    # Image / DB sync diagnostics.
    print("Image Sync Status:")
    images_dir = app_constants.IMAGES_DIR
    print(f"  Images directory: {os.path.abspath(images_dir)}")
    orphaned, missing = get_image_sync_status()
    print(f"  Images in folder but NOT in DB (orphaned): {len(orphaned)}")
    print(f"  Images in DB but NOT in folder (missing):  {len(missing)}")
    if orphaned:
        for f in sorted(orphaned)[:10]:
            print(f"    orphaned: {f}")
        if len(orphaned) > 10:
            print(f"    ... and {len(orphaned) - 10} more")
    if missing:
        for f in sorted(missing)[:10]:
            print(f"    missing:  {f}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    print()

def handle_clear_images(args):
    """Remove images from the images directory that are not referenced in the database."""
    images_dir = app_constants.IMAGES_DIR
    print(f"Images directory: {os.path.abspath(images_dir)}")

    orphaned, _ = get_image_sync_status()

    if not orphaned:
        print("No orphaned images found. Directory is in sync with the database.")
        return

    print(f"Found {len(orphaned)} orphaned image(s) not referenced in the database.")
    for f in sorted(orphaned):
        path = os.path.join(images_dir, f)
        try:
            os.remove(path)
            print(f"  Removed: {f}")
        except OSError as e:
            print(f"  Error removing {f}: {e}")

    print(f"Done. Removed {len(orphaned)} orphaned image(s).")

def nullify_strings(db_path):
    """Convert empty strings and null-like values to NULLs in all nullable columns."""
    print("Converting empty strings to NULLs in database.")
    tables = ["Media", "Individuals", "AlternateNames", "Tags", "Labels", "Actions", "Proposals"]
    with sqlite3.connect(db_path) as conn:
        for table in tables:
            for row in conn.execute(f"PRAGMA table_info({table})"):
                # row: (cid, name, type, notnull, dflt_value, pk)
                col_name, notnull, pk = row[1], row[3], row[5]
                if pk or notnull:
                    continue
                conn.execute(
                    f"UPDATE {table} SET [{col_name}] = NULL "
                    f"WHERE [{col_name}] IS NOT NULL "
                    f"AND (TRIM([{col_name}]) = '' "
                    f"OR UPPER(TRIM([{col_name}])) IN ('NULL', 'UNKNOWN', 'U'))"
                )

def recreate_database(db_path, with_indexes=True):
    """Recreate the database from the SQL schema, optionally creating performance indexes."""
    database_parent = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(database_parent, exist_ok=True)
    print(f"Recreating database: {db_path}")
    schema_path = os.path.join(app_constants.DB_SQL_DIR, "create_database.sql")
    run_sqlite_cmd(db_path, f".read {_sqlite_shell_arg(schema_path)}\n")
    if with_indexes:
        indexes_path = os.path.join(app_constants.DB_SQL_DIR, "add_indexes.sql")
        run_sqlite_cmd(db_path, f".read {_sqlite_shell_arg(indexes_path)}\n")
        print("Performance indexes created.")
    else:
        print("Skipped performance indexes (--no-indexes).")

def migrate_authentication_schema(db_path):
    """Add persisted session expiry to the token schema while invalidating legacy session rows."""
    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(Tokens)")
        }
        if not columns:
            raise sqlite3.DatabaseError("Tokens table is missing.")
        if "expires" in columns:
            return False

        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                CREATE TABLE Tokens_auth_migration (
                    id INTEGER PRIMARY KEY,
                    token TEXT NOT NULL,
                    user INTEGER REFERENCES Users(id) ON DELETE CASCADE NOT NULL,
                    expires INTEGER NOT NULL
                )
                """
            )
            connection.execute("DROP TABLE Tokens")
            connection.execute(
                "ALTER TABLE Tokens_auth_migration RENAME TO Tokens"
            )
            connection.commit()
        except sqlite3.DatabaseError:
            connection.rollback()
            raise

    return True

def handle_migrate_authentication(args):
    """Migrate the configured database to expiring sessions."""
    db_path = _configured_database_path(args)

    print(f"Migrating authentication schema: {db_path}")
    try:
        migrated = migrate_authentication_schema(db_path)
    except sqlite3.DatabaseError as ex:
        print(f"[ERROR] Authentication schema migration failed: {ex}")
        sys.exit(1)

    if migrated:
        print("Authentication schema migrated; existing sessions were invalidated.")
    else:
        print("Authentication schema is already current.")


def provision_first_owner(db_path, display_name, email, username, password):
    """Provision the initial owner account in an empty application database."""
    if not os.path.isfile(db_path):
        raise FileNotFoundError("Initialized database not found.")

    normalized_name = display_name.strip() if isinstance(display_name, str) else ""
    normalized_email = email.strip() if isinstance(email, str) else ""
    normalized_username = username.strip() if isinstance(username, str) else ""
    valid_owner_details = (
        bool(normalized_name)
        and app_utils.valid_name(normalized_name)
        and bool(normalized_email)
        and bool(normalized_username)
    )
    if not valid_owner_details or not isinstance(password, str) or not password:
        raise ValueError("Owner details and password are required and must be valid.")

    with sqlite3.connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if backend.db.user.get_users(connection=connection):
            raise ValueError("First-owner provisioning requires an empty Users table.")

        password_derivative = backend.db.user.derive_client_password(password)
        owner = backend.db.user.create_user(
            name=normalized_name,
            email=normalized_email,
            username=normalized_username,
            role="owner",
            password=password_derivative,
            connection=connection,
        )
        if owner == app_constants.ERROR:
            raise RuntimeError("Shared user creation failed.")

    return owner


def handle_provision_owner(args):
    """Prompt securely and provision the first owner in the configured database."""
    db_path = _configured_database_path(args)
    print(f"Provisioning first owner in: {db_path}")
    if _database_has_users(db_path):
        print("[ERROR] First-owner provisioning failed: the database already has users.")
        sys.exit(1)

    password = getpass.getpass("Owner password: ")
    password_confirmation = getpass.getpass("Confirm owner password: ")
    if password != password_confirmation:
        print("[ERROR] Owner passwords do not match.")
        sys.exit(1)

    try:
        owner = provision_first_owner(
            db_path=db_path,
            display_name=args.display_name,
            email=args.email,
            username=args.username,
            password=password,
        )
    except (OSError, sqlite3.DatabaseError, RuntimeError, ValueError) as ex:
        print(f"[ERROR] First-owner provisioning failed: {ex}")
        sys.exit(1)

    print(f"First owner provisioned: {owner.username}")

def default_repopulate(db_path, data_path):
    """Repopulate the default tables."""
    print("Repopulating default tables (Individuals, Media, Labels, Tags, AlternateNames).")
    print("NOT repopulating Users, Tokens, Actions, or Proposals.")
    
    cmds = []
    for table in ["Labels", "Tags", "AlternateNames", "Individuals", "Media"]:
        cmds.append(f"DELETE FROM {table};")
        
    cmds.append("INSERT INTO Individuals VALUES (-1, -1, NULL, NULL, NULL, -1, -1);")
    run_sqlite_cmd(db_path, "\n".join(cmds) + "\n")
    
    import_cmds = []
    import_cmds.append(f".import --csv {_sqlite_shell_arg(os.path.join(data_path, 'individuals.csv'))} Individuals")
    import_cmds.append(f".import --csv {_sqlite_shell_arg(os.path.join(data_path, 'media.csv'))} Media")
    import_cmds.append(f".import --csv {_sqlite_shell_arg(os.path.join(data_path, 'alternateNames.csv'))} AlternateNames")
    import_cmds.append(f".import --csv {_sqlite_shell_arg(os.path.join(data_path, 'tags.csv'))} Tags")
    import_cmds.append(f".import --csv {_sqlite_shell_arg(os.path.join(data_path, 'labels.csv'))} Labels")
    run_sqlite_cmd(db_path, "\n".join(import_cmds) + "\n")
    
    nullify_strings(db_path)

def actions_repopulate(db_path, data_path):
    """Repopulate the Actions table."""
    print("Repopulating Actions table.")
    cmds = [
        "DELETE FROM Actions;",
        f".import --csv {_sqlite_shell_arg(os.path.join(data_path, 'actions.csv'))} Actions"
    ]
    run_sqlite_cmd(db_path, "\n".join(cmds) + "\n")
    nullify_strings(db_path)

def tokens_repopulate(db_path, data_path):
    """Repopulate the Tokens table."""
    print("Repopulating Tokens table.")
    cmds = [
        "DELETE FROM Tokens;",
        f".import --csv {_sqlite_shell_arg(os.path.join(data_path, 'tokens.csv'))} Tokens"
    ]
    run_sqlite_cmd(db_path, "\n".join(cmds) + "\n")

def users_repopulate(db_path, data_path, users_file):
    """Repopulate the Users table."""
    print("Repopulating Users table.")
    print(f"Repopulating Users table using: {users_file}")
    cmds = [
        "DELETE FROM Users;",
        f".import --csv {_sqlite_shell_arg(os.path.join(data_path, users_file))} Users"
    ]
    run_sqlite_cmd(db_path, "\n".join(cmds) + "\n")

def full_repopulate(db_path, data_path):
    """Repopulate the database from a full snapshot."""
    print("Repopulating database fully.")
    
    tables = ["Proposals", "Actions", "Tokens", "Users", "Labels", "Tags", "AlternateNames", "Individuals", "Media"]
    cmds = [f"DELETE FROM {t};" for t in tables]
    cmds.append("INSERT INTO Individuals VALUES (-1, -1, NULL, NULL, NULL, -1, -1);")
    run_sqlite_cmd(db_path, "\n".join(cmds) + "\n")
    
    import_cmds = []
    for table, filename in zip(
        reversed(tables), 
        ["media.csv", "individuals.csv", "alternateNames.csv", "tags.csv", "labels.csv", "users.csv", "tokens.csv", "actions.csv", "proposals.csv"]
    ):
        import_cmds.append(f".import --csv {_sqlite_shell_arg(os.path.join(data_path, filename))} {table}")
        
    run_sqlite_cmd(db_path, "\n".join(import_cmds) + "\n")
    nullify_strings(db_path)

def handle_init(args):
    """Handle the init command."""
    db_path = _configured_database_path(args)

    if _database_has_data(db_path) and not args.force:
        print(f"[ERROR] {db_path} already holds data and init would erase it. "
              "Back it up, then rerun with --force to reset it.")
        sys.exit(1)

    print(f"Initializing Database: {db_path}")
    recreate_database(db_path, with_indexes=not args.no_indexes)
    print("Database initialized successfully.")

def handle_populate(args):
    """Handle the populate command."""
    import shutil
    
    data_path = app_utils.resolve_directory_path(args.data_path, app_constants.DB_DATA_DIR)
    print(f"Using Data Path: {data_path}")
    
    db_path = _configured_database_path(args)
        
    print(f"Using Database: {db_path}")
    if not _database_is_initialized(db_path):
        print("[ERROR] The database has no schema yet; run "
              "`python3 -m db.database init` first.")
        sys.exit(1)
    if _database_has_data(db_path) and not args.overwrite:
        print(f"[ERROR] {db_path} already holds data and populate would replace it. "
              "Back it up, then rerun with --overwrite to replace it.")
        sys.exit(1)
    
    tmp_db_path = db_path + ".tmp"
    if os.path.exists(db_path):
        shutil.copy2(db_path, tmp_db_path)
    elif os.path.exists(tmp_db_path):
        os.remove(tmp_db_path)

    published = False
    try:
        # Execute actions on temporary database.
        if args.all:
            full_repopulate(tmp_db_path, data_path)
        elif args.tables:
            if "Actions" in args.tables:
                actions_repopulate(tmp_db_path, data_path)
            if "Tokens" in args.tables:
                tokens_repopulate(tmp_db_path, data_path)
            if "Users" in args.tables:
                users_repopulate(tmp_db_path, data_path, args.users_file)
        else:
            default_repopulate(tmp_db_path, data_path)

        print("Running data integrity tests against temporary database...")
        env = os.environ.copy()
        env["SHARK_INTEGRITY_DB_PATH"] = tmp_db_path

        result = subprocess.run([sys.executable, "-m", "tests.manual.test_data_integrity"],
                                env=env, cwd=app_constants.THIS_DIR, check=False)

        if result.returncode != 0:
            if getattr(args, "force", False):
                print("[WARNING] Data integrity tests failed but --force was used. Proceeding anyway.")
            else:
                print("[ERROR] Data integrity tests failed! Aborting database update.")
                sys.exit(result.returncode)

        os.replace(tmp_db_path, db_path)
        published = True
    finally:
        # Every failed or interrupted population discards its staging copy.
        if not published and os.path.exists(tmp_db_path):
            os.remove(tmp_db_path)
    print("Database populated successfully.")


def main():
    """Parse the command line and run the selected database subcommand."""
    parser = argparse.ArgumentParser(
        description=(
            "Database Orchestration "
            "(init, migrate, provision, populate, dump, print)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Global test flag
    parser.add_argument("-t", "--test", action="store_true", help="Use the Test Database (test_sharkMatcher.db)")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Subparser for init.
    init_parser = subparsers.add_parser("init", help="Drop and recreate database from schema")
    init_parser.add_argument("-n", "--name", type=str, default=None, help="Specify database name")
    init_parser.add_argument("--no-indexes", action="store_true", help="Skip performance indexes (useful for benchmarking)")
    init_parser.add_argument("--force", action="store_true", help="Reset a database that already holds data")

    # Subparser for authentication-schema migration.
    migrate_auth_parser = subparsers.add_parser(
        "migrate-auth",
        help="Add expiring sessions and invalidate legacy tokens",
    )
    migrate_auth_parser.add_argument(
        "-n", "--name", type=str, default=None, help="Specify database name"
    )

    # Subparser for trusted first-owner provisioning.
    provision_parser = subparsers.add_parser(
        "provision-owner",
        help="Securely provision the first owner in an empty database",
    )
    provision_parser.add_argument(
        "--display-name", required=True, help="Owner display name"
    )
    provision_parser.add_argument("--email", required=True, help="Owner email")
    provision_parser.add_argument(
        "--username", required=True, help="Owner login username"
    )
    provision_parser.add_argument(
        "-n", "--name", type=str, default=None, help="Specify database name"
    )
    
    # Subparser for dump.
    dump_parser = subparsers.add_parser("dump", help="Dump the contents of the database into annotations style files")
    
    # Subparser for print.
    print_parser = subparsers.add_parser(
        "print", help="Print database, index, and image-sync diagnostics"
    )

    # Subparser for clear-images.
    clear_parser = subparsers.add_parser("clear-images", help="Remove frontend images not referenced in the database")
    
    # Subparser for populate.
    pop_parser = subparsers.add_parser("populate", help="Populate the database with data")
    pop_parser.add_argument(
        "--data-path", type=str, required=True,
        help="CSV directory: an existing path (absolute or relative to the current directory), "
             "otherwise a dataset name under the data root's database-format/ directory")
    pop_parser.add_argument("--all", action="store_true", help="Repopulate all tables from a full snapshot")
    pop_parser.add_argument("--force", action="store_true", help="Force database update even if data integrity tests fail")
    pop_parser.add_argument("--overwrite", action="store_true", help="Replace the data in a database that already holds data")
    pop_parser.add_argument("--tables", nargs="+", choices=["Actions", "Tokens", "Users"], help="Selectively update specific tables (e.g., --tables Users Tokens)")
    pop_parser.add_argument("--users-file", type=str, default="users.csv", help="Specify a trusted users CSV when updating the Users table")
    pop_parser.add_argument("-n", "--name", type=str, default=None, help="Specify database name")

    args = parser.parse_args()
    app_utils.print_script_banner(__file__, args)
    
    # Test mode is handled globally via the early sys.argv check, but we can also set the env var here just in case.
    if args.test:
        os.environ["SHARK_TEST_MODE"] = "true"

    if args.command == "dump":
        handle_dump(args)
    elif args.command == "print":
        handle_print(args)
    elif args.command == "clear-images":
        handle_clear_images(args)
    elif args.command == "init":
        handle_init(args)
    elif args.command == "migrate-auth":
        handle_migrate_authentication(args)
    elif args.command == "provision-owner":
        handle_provision_owner(args)
    elif args.command == "populate":
        handle_populate(args)

if __name__ == "__main__":
    main()
