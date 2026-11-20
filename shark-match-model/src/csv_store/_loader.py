"""CSV loading and in-process caching for individuals and media records."""

import csv
import os

from src.core import constants
from src.core import utils

from src.csv_store.models import Individual, Media

# Module-level caches: None = not yet loaded.
_individuals: list | None = None
_media: list | None = None
_custom_db_dir: str | None = None


def load_individuals() -> list:
    """Return a flat cached list of read-only Individual objects parsed from individuals.csv."""
    global _individuals
    if _individuals is not None:
        return _individuals

    base_dir = _custom_db_dir if _custom_db_dir is not None else constants.DB_DATA_DIR
    if base_dir is not None and not os.path.isabs(base_dir):
        base_dir = utils.resolve_relative_csv_dir(base_dir) or base_dir
    path = os.path.join(base_dir, 'individuals.csv')
    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.reader(fh):
            if len(row) < 7:
                continue
            ind_id = _int(row[0])
            if ind_id is None:
                continue
            rows.append(Individual(
                id=ind_id,
                name=_str(row[1]),
                sex=_str(row[2]),
                length=_float(row[3]),
                media=_int(row[4]),
                source=_int(row[5]),
                reference=_int(row[6]),
            ))

    _individuals = rows
    return _individuals


def load_media() -> list:
    """Return a flat cached list of Media objects parsed from media.csv."""
    global _media
    if _media is not None:
        return _media

    base_dir = _custom_db_dir if _custom_db_dir is not None else constants.DB_DATA_DIR
    if base_dir is not None and not os.path.isabs(base_dir):
        base_dir = utils.resolve_relative_csv_dir(base_dir) or base_dir
    path = os.path.join(base_dir, 'media.csv')
    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.reader(fh):
            if len(row) < 10:
                continue
            media_id = _int(row[0])
            if media_id is None:
                continue
            rows.append(Media(
                id=media_id,
                hash=_str(row[1]),
                name=_str(row[2]),
                filename=_str(row[3]),
                source=_int(row[4]),
                metadata=_str(row[5]),
                location=_str(row[6]),
                time=_int(row[7]),
                type=_str(row[8]),
                quality=_str(row[9]),
            ))

    _media = rows
    return _media


def clear_cache():
    """Discard cached data so the next access re-reads from disk."""
    global _individuals, _media
    _individuals = None
    _media = None


def set_db_dir(path: str):
    """Set a custom directory to load the CSVs from instead of DB_DATA_DIR."""
    global _custom_db_dir
    if path is not None and not os.path.isabs(path):
        resolved = utils.resolve_relative_csv_dir(path)
        if resolved:
            path = resolved
    _custom_db_dir = path
    clear_cache()


def _int(value: str):
    try:
        return int(value) if value not in (None, '') else None
    except ValueError:
        return None


def _float(value: str):
    try:
        return float(value) if value not in (None, '') else None
    except ValueError:
        return None


def _str(value: str):
    return value if value not in (None, '') else None
