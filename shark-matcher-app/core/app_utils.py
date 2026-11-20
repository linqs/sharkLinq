"""
Utility functions for the application.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import csv
import datetime
import functools
import glob
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import time
from typing import Any, List, Optional, Sequence

from core import app_constants

# Constants.
SHARK_TAG_NAME = 'shark-clipper'
TABLE_NAMES = ["Individuals", "Media", "Tags", "AlternateNames", "Labels", "Users", "Tokens", "Actions", "Proposals"]

# Decorators and logging functions.
def print_script_banner(script_name, args=None):
    """
    Prints a standardized banner with the script name and arguments.
    """
    print("-" * 60)
    print(f"Running: {script_name}")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args:
        print("Arguments:")
        if hasattr(args, '__dict__'):
            args_dict = vars(args)
        elif isinstance(args, dict):
            args_dict = args
        else:
            args_dict = {"args": str(args)}

        for key, value in args_dict.items():
            print(f"  {key}: {value}")
    print("-" * 60)

def print_key_value_section(title, data):
    """
    Prints a section with a title and key-value pairs aligned.
    """
    print(f"\n=== {title} ===")
    if not data:
        print("No data.")
        return

    # Determine max key length for alignment
    max_key_len = max(len(str(k)) for k in data.keys())

    for key, value in data.items():
        print(f"{str(key):<{max_key_len}} : {value}")

def print_table(headers, rows):
    """
    Prints a formatted table with headers and rows.
    """
    if not rows:
        print("No data to display.")
        return

    # Calculate column widths
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(val)))
            else:
                 # Handle case where row has more columns than headers (shouldn't happen ideally)
                 col_widths.append(len(str(val)))

    # Create format string
    fmt = "  ".join([f"{{:<{w}}}" for w in col_widths])

    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    print(fmt.format(*headers))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))

    for row in rows:
        # Ensure row elements are strings
        row_str = [str(val) for val in row]
        # Pad row if too short
        while len(row_str) < len(headers):
            row_str.append("")
        print(fmt.format(*row_str))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))

def print_function_banner(function_name, description=None):
    """
    Prints a banner for a specific function execution.
    """
    print(f"\n=== Running: {function_name} ===")
    if description:
        print(f"{description}")
    print("-" * 40)

def timeit(func):
    """
    A decorator to time the execution of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Run the wrapped function and report its elapsed time."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"'{func.__qualname__}' executed in {end_time - start_time:.4f}s")
        return result
    return wrapper

# Data and embeddings utility functions.
def resolve_directory_path(input_path: str, fallback_base_dir: str) -> str:
    """Resolve a CLI directory path, falling back to fallback_base_dir if not found directly."""
    if input_path is None:
        return fallback_base_dir

    if os.path.exists(input_path):
        return os.path.abspath(input_path)

    fallback_path = os.path.join(fallback_base_dir, input_path)
    if os.path.exists(fallback_path):
        return os.path.abspath(fallback_path)

    raise ValueError(f"Directory not found. Tried exact match '{input_path}' and fallback '{fallback_path}'.")


def portable_csv_dir(csv_dir):
    """Return csv_dir relative to the database-format or data root with forward slashes, else its absolute path.

    Mirrors the matcher's `utils.portable_csv_dir` so annotation sidecars resolve identically in both repositories.
    """
    for root in (app_constants.DB_DATA_DIR, app_constants.BASE_DATA_DIR):
        try:
            rel = os.path.relpath(csv_dir, root)
        except ValueError:
            continue
        if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
            return rel.replace(os.sep, "/")
    return os.path.abspath(csv_dir)


def resolve_served_path(root, relative, filename_only=False):
    """Resolve a client/data filename strictly inside its configured file root.

    Absolute paths, traversal segments, and symlink descendants are rejected.
    """
    from pathlib import Path, PureWindowsPath
    if (not isinstance(relative, str) or not relative
            or relative.startswith('/') or PureWindowsPath(relative).drive
            or '\\' in relative or '%' in relative
            or any(ord(char) < 32 or ord(char) == 127 for char in relative)):
        raise ValueError('Invalid file name.')
    parts = relative.split('/')
    if any(part in ('', '.', '..') for part in parts) or (filename_only and len(parts) != 1):
        raise ValueError('Invalid file name.')
    base = Path(root).resolve()
    target = base
    for part in parts:
        target = target / part
        if target.is_symlink():
            raise ValueError('Invalid file name.')
    if not target.resolve().is_relative_to(base):
        raise ValueError('Invalid file name.')
    return target

def list_to_string(input_list, delimiter=", "):
    """Join list items into a string using the requested delimiter."""
    return_string = ""
    for item in input_list:
        return_string = return_string + str(item) + delimiter

    return_string = return_string[:-len(delimiter)]
    return return_string

def convert_date_to_unix(date_input):
    """Convert a date string in 'mm-dd-yyyy' format or numeric timestamp to a unix timestamp."""
    if isinstance(date_input, (int, float)):
        return date_input

    if isinstance(date_input, str):
        try:
            # Try parsing mm-dd-yyyy
            dt = datetime.datetime.strptime(date_input, "%m-%d-%Y")
            return int(dt.timestamp())
        except ValueError:
            # Fall back to returning unparsed input if date parsing fails.
            pass

    return date_input

def clean_input_list(input_list, remove_item=""):
    """Remove a sentinel value and leading spaces from a mutable input list."""
    if not isinstance(input_list, list):
        return None
    # Remove all instances of remove_item
    while remove_item in input_list:
         input_list.remove(remove_item)

    # Remove leading spaces (to allow commas followed by spaces in search).
    for index, item in enumerate(input_list):
        if isinstance(item, str):
            input_list[index] = item.lstrip()

    return input_list

def clean_input_bool(input_val):
    """Normalize boolean or case-insensitive 'true'/'false' string input to a boolean or None."""
    if input_val is None:
        return None

    if isinstance(input_val, bool):
        return input_val

    if isinstance(input_val, str):
        lower_val = input_val.lower()
        if lower_val == 'true':
            return True
        if lower_val == 'false':
            return False

    return None

# Match safe name characters, excluding injection characters and kept in sync with validName() in util.js.
NAME_PATTERN = re.compile(r"[\w \-'.,()/]*", re.UNICODE)
MAX_NAME_LENGTH = 128

def valid_name(name_input):
    """Return True if a candidate name is absent or contains only permitted safe characters."""
    if name_input is None:
        return True

    if not isinstance(name_input, str):
        return False

    if len(name_input) > MAX_NAME_LENGTH:
        return False

    return NAME_PATTERN.fullmatch(name_input) is not None

def convert_unknown_to_none(input_val):
    """Convert unknown sentinels ('Unknown', 'u', empty string) to None, returning other values unchanged."""
    if input_val is None:
        return None

    if isinstance(input_val, str):
        if input_val.lower() in ('unknown', 'u'):
            return None
        if input_val.strip() == "":
            return None

    return input_val

SESSION_FIELDS = frozenset({"user_id", "user_username", "user_token"})


def accepts(*fields):
    """Declare the request fields an API handler accepts, besides the session fields any request carries."""
    def decorate(handler):
        handler.accepted_fields = SESSION_FIELDS | frozenset(fields)
        return handler
    return decorate


def normalize_sex(value, unknown=None):
    """Return 'F' or 'M' in any letter case, `unknown` for U/Unknown markers, None for blanks, and raise ValueError otherwise."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        marker = value.strip().upper()
        if marker in ("F", "M"):
            return marker
        if marker in ("U", "UNKNOWN"):
            return unknown
    raise ValueError(f"Invalid sex value: {value!r}.")


def normalize_list_params(kwargs, list_params, splitter=","):
    """Normalize list-type parameters in kwargs in-place, converting empty values to None."""
    for key in list_params:
        value = kwargs.get(key)
        final_val = []

        if value is None:
            final_val = []
        elif isinstance(value, str):
            if value.strip() == "":
                final_val = []
            else:
                final_val = clean_input_list(value.split(splitter))
        elif isinstance(value, list):
            final_val = clean_input_list(value)
        elif isinstance(value, int):
            final_val = [value]
        else:
            final_val = [value]

        # If the resulting list is empty, set to None so backend treats it as "all".
        kwargs[key] = final_val if len(final_val) > 0 else None

def normalize_scalar_params(kwargs, scalar_params):
    """Normalize scalar parameters in kwargs in-place, treating empty strings as None."""
    for key in scalar_params:
        value = kwargs.get(key)
        if key == "allow_unknown":
            kwargs[key] = clean_input_bool(value)
        else:
            # For other scalars, treat empty string as None.
            if value == "":
                kwargs[key] = None
            else:
                kwargs[key] = value

# SQL Related.
def execute_sql(sql, data=None, commit=False, connection=None):
    """Execute SQL statements using an existing connection or a managed temporary connection."""
    is_local_connection = (connection is None)
    conn = connection or db_connect()

    try:
        # TODO Correct foreign key enforcement.
        # conn.execute("PRAGMA foreign_keys = 1")

        if is_local_connection:
            with conn:
                if data is None:
                    result = conn.execute(sql)
                else:
                    result = conn.execute(sql, data)
                rows = result.fetchall()

            if commit is True:
                conn.commit()
        else:
            if data is None:
                result = conn.execute(sql)
            else:
                result = conn.execute(sql, data)
            rows = result.fetchall()

        return rows
    except Exception as e:
        raise e
    finally:
        if is_local_connection:
            conn.close()

def build_sql_placeholders(values_list, basename="x") -> tuple[str, dict]:
    """
    Helper function to prepare named placedholers and value dict for sql queries.
    """
    placeholders = ", ".join([f":{basename}_{i}" for i, x in enumerate(values_list)])
    values = {f"{basename}_{i}": x for i, x in enumerate(values_list)}
    return f"({placeholders})", values

_DATABASE_PATH = ContextVar("application_database_path", default=None)


@contextmanager
def database_scope(db_path):
    """Select a database for this request without changing process configuration."""
    token = _DATABASE_PATH.set(db_path)
    try:
        yield
    finally:
        _DATABASE_PATH.reset(token)


def current_database_path():
    """Return the request's database, or the configured live database."""
    return _DATABASE_PATH.get() or app_constants.DB_PATH


def db_connect(db_path=None):
    """Open an application SQLite connection using db_path or the configured default database."""
    connection = sqlite3.connect(db_path or current_database_path())
    return (connection)

def get_all_ids(table, source=False, connection=None):
    """Return nonreserved IDs from an approved database table."""
    if table not in TABLE_NAMES:
        raise ValueError("Error, invalid database table.")

    source_clause = ""
    # Assure expected ordering for frontend.
    source_clause = ""
    # Assure expected ordering for frontend.
    if table == "Individuals":
        # For the individuals table, getting source individuals means getting current source individuals,
        # otherwise get all individuals with media.
        if source is True:
            source_clause = " AND reference is NOT null"

    # Negative IDs are reserved for utility objects, do not return these unless specifically called for.
    sql = f"""SELECT id FROM {table} WHERE id >= 0{source_clause}"""

    result = execute_sql(sql, connection=connection)
    if result is None:
        return app_constants.ERROR

    return [item[0] for item in result]

def id_exists(id, table, connection=None):
    """
    Return true if an item with the specified ID exists in a table, else false.
    """
    if table not in TABLE_NAMES:
        raise ValueError("Error, invalid database table.")

    data = ({"id": id})
    sql = f"""SELECT COUNT(*) FROM {table} WHERE id = :id;"""

    result = execute_sql(sql, data, connection=connection)
    if not result:
        return False

    count = result[0][0]
    return False if count == 0 else True

# CSV file I/O functions.
def load_csv_file(path: str, delimiter: str = ',') -> Optional[List[List[str]]]:
    """Load a CSV file from the specified path and return its rows as a list."""
    try:
        with open(path, 'r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file, delimiter=delimiter)
            return list(reader)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None

def write_csv_file(path: str, data: Sequence[Sequence[Any]], delimiter: str = ',',
                   durable: bool = False) -> None:
    """Write data to a CSV file at the specified path."""
    with open(path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerows(data)
        if durable:
            file.flush()
            os.fsync(file.fileno())

# Parsing functions.
def parse_time_from_name(name):
    """Extract and parse date components from a media filename into a unix timestamp."""
    # Extract the first sequence of 8 digits.
    match = re.search(r"(\d{8})", name)
    if not match:
        return None

    numerical = match.group(1)

    # Parse numerical information.
    year_str = numerical[0:2]
    month_str = numerical[2:4]
    day_str = numerical[4:6]

    try:
        year = int(year_str)
        month = int(month_str)
        day = int(day_str)
    except ValueError:
        return None

    # Year logic
    if year <= 50:
        full_year = 2000 + year
    else:
        full_year = 1900 + year

    # Months past 12 roll into later years (13 -> January next year, 25 -> January two years on).

    if month > 12:
        month_index = month - 1
        years_to_add = month_index // 12
        full_year += years_to_add
        month = (month_index % 12) + 1

    # Format date string for conversion
    # convert_date_to_unix expects mm-dd-yyyy
    date_str = f"{month:02d}-{day:02d}-{full_year}"

    return convert_date_to_unix(date_str)

def parse_media_medatada(media_directory):
    """Get metadata for all images."""
    for path in glob.iglob(os.path.join(media_directory, '**', '*'), recursive=True):
        filename = os.path.basename(path)
        if os.path.isfile(path):
            print(path)

            image_medatada = get_all_metadata(path)
            print(filename, image_medatada)

# Extract raw metadata dictionary from a media file using ffprobe (adapted from Shark Clipper).
def get_all_metadata(path):
    """Get the metadata from a file using ffprobe."""
    args = [
        _get_path('ffprobe'),
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        path,
    ]

    stdout, _ = _run(args, 'ffprobe')
    return json.loads(stdout)


# TODO Sort and cleanup what's below this line.


# Get metadata that we believe is important.
def get_key_metadata(path, override_with_individual_data=True):
    """
    Extracts key metadata (location, time, shark tags) from an image file.
    """
    data = {}
    all_metadata = get_all_metadata(path)

    if ('format' in all_metadata):
        tags = all_metadata['format'].get('tags', {})

        if ('location' in tags):
            text_location = tags['location'].strip()
            data['location_raw'] = text_location

            match = re.search(r'^((?:\+|-)?\d+\.\d+)((?:\+|-)?\d+\.\d+)/?$', text_location)
            if (match is not None):
                data['location'] = {
                    'latitude': match.group(1),
                    'longitude': match.group(2),
                }

        if ('creation_time' in tags):
            text_time = tags['creation_time'].strip()
            data['start_time_raw'] = text_time

            try:
                parsed_time = datetime.datetime.fromisoformat(text_time)
                data['start_time_unix'] = int(parsed_time.timestamp())
            except Exception as ex:
                print(f"Invalid start/creation time found in file metadata. | Exception: {ex}")

        if ((override_with_individual_data) and (SHARK_TAG_NAME in tags)):
            individual_data_str = tags[SHARK_TAG_NAME].strip()

            try:
                individual_data = json.loads(individual_data_str)
            except Exception as ex:
                print(f"Invalid (json) individual data found in file metadata. | Exception: {ex}")

            data.update(individual_data)

    return data, all_metadata

def _run(args, name):
    """
    Internal helper to run subprocess commands.
    """
    print(shlex.join(args))

    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode != 0:
        raise ValueError(f"{name} did not exit cleanly.\n--- stdout ---\n{result.stdout}\n"
                         f"---\n--- stderr ---{result.stderr}\n---")

    return result.stdout, result.stderr

def _get_path(name):
    """
    Internal helper to find the path of a binary.
    """
    # First, do a standard path search.
    print(name)
    path = shutil.which(name)
    if path is not None:
        return path

    # Next, check the local (project) path.
    path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(
        app_constants.THIS_DIR), '..', '..')), name)
    if os.path.isfile(path):
        return path

    return None

def handle_error(error):
    """
    Basic error handling helper.
    """
    print(error)

def main():
    """
    Main function for running utility scripts.
    """

if __name__ == "__main__":
    main()
