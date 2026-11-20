"""Convert a biology-format annotations CSV into database-format CSV files."""

import argparse
import csv
import datetime
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import constants


def main():
    """Parse converter arguments and write one database-format dataset."""
    parser = argparse.ArgumentParser(
        description="Convert biology-format annotations CSV to database-format CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 src/data_scripts/biology_to_db_format.py \\\n"
            "      --input data/finMatch_combined_2006-2024.csv \\\n"
            "      --output-name delta-v2"
        ),
    )
    parser.add_argument('--input', required=True,
                        help='Path to the biology-format annotations CSV.')
    parser.add_argument('--output-name', default=None,
                        help=(
                            'Subdirectory name under the output root. '
                            'Defaults to the input file stem.'
                        ))
    parser.add_argument(
        '--output-root',
        default=constants.DB_DATA_DIR,
        help=(
            'Parent directory for generated datasets. Defaults to the '
            'DB_DATA_DIR authority in constants.py.'
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_name = args.output_name or os.path.splitext(os.path.basename(args.input))[0]
    output_dir = os.path.join(args.output_root, output_name)

    convert(args.input, output_dir)


def convert(bio_path: str, output_dir: str):
    """Convert biology-format CSV at *bio_path* into database-format CSVs in *output_dir*."""
    # 1. Read biology CSV
    with open(bio_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        bio_rows = list(reader)
        fieldnames = reader.fieldnames or []

    id_cols = sorted(
        [c for c in fieldnames if re.match(r'^id_\d+$', c)],
        key=lambda c: int(c.split('_')[1])
    )

    # 2. Assign media IDs (deduplicated, in order of first appearance)
    photo_to_media_id: dict[str, int] = {}
    for row in bio_rows:
        for col in id_cols:
            name = row.get(col, '').strip()
            if name and name not in photo_to_media_id:
                photo_to_media_id[name] = len(photo_to_media_id)

    n_media = len(photo_to_media_id)

    # 3. Build media.csv rows
    media_rows = []
    for photo_name, media_id in sorted(photo_to_media_id.items(), key=lambda kv: kv[1]):
        location, timestamp = _parse_photo_name(photo_name)
        media_rows.append([
            media_id,
            _make_hash(photo_name),
            photo_name,
            photo_name + '.jpeg',
            media_id,           # source: self-reference
            '',                 # metadata
            location,
            timestamp if timestamp is not None else '',
            '',                 # type (unknown)
            '',                 # quality (unknown)
        ])

    # 4. Build individuals.csv + alternateNames + tags
    media_individual_rows = []   # collected first; emitted before source rows
    source_individual_rows = []
    alt_name_rows = []
    tag_rows = []

    alt_name_id = 0
    tag_id = 0

    for shark_idx, row in enumerate(bio_rows):
        source_id = n_media + shark_idx

        sex = row.get('Sex', '').strip()

        length_str = row.get('Length', '').strip()
        try:
            length = float(length_str)
        except (ValueError, TypeError):
            length = ''     # 'U' or missing -> empty (avoids float() crash in csv_store)

        # Collect this shark's photos in file order (id_0 first)
        shark_photos = [
            row.get(col, '').strip()
            for col in id_cols
            if row.get(col, '').strip() and row.get(col, '').strip() in photo_to_media_id
        ]
        if not shark_photos:
            continue

        primary_media_id = photo_to_media_id[shark_photos[0]]

        # Source individual
        source_individual_rows.append([
            source_id,
            '',             # name: empty for source individuals
            sex,
            length,
            '',             # media: None for source individuals
            source_id,      # source == id
            primary_media_id,  # reference -> primary media
        ])

        # Media individuals (one per photo)
        for photo_name in shark_photos:
            media_id = photo_to_media_id[photo_name]
            media_individual_rows.append([
                media_id,
                str(media_id),  # name: string ID, matching DB convention
                sex,
                length,
                media_id,       # media FK
                source_id,      # source FK
                '',             # reference: empty for media individuals
            ])

        # Alternate names from 'Name' column (slash-separated)
        names_str = row.get('Name', '').strip()
        for alt_name in filter(None, (n.strip() for n in names_str.split('/'))):
            alt_name_rows.append([alt_name_id, source_id, alt_name, '', ''])
            alt_name_id += 1

        # Tags from 'Tags' column (name~type/name~type/...)
        tags_str = row.get('Tags', '').strip()
        for entry in filter(None, (e.strip() for e in tags_str.split('/'))):
            if '~' in entry:
                tag_name, tag_type = entry.split('~', 1)
            else:
                tag_name, tag_type = entry, ''
            tag_name = tag_name.strip()
            if tag_name:
                tag_rows.append([tag_id, source_id, tag_name, tag_type.strip(), ''])
                tag_id += 1

    # Media individuals first, then source individuals (mirrors existing DB convention)
    individual_rows = media_individual_rows + source_individual_rows

    # 5. Write output
    os.makedirs(output_dir, exist_ok=True)

    _write_csv(os.path.join(output_dir, 'individuals.csv'), individual_rows)
    _write_csv(os.path.join(output_dir, 'media.csv'), media_rows)
    _write_csv(os.path.join(output_dir, 'alternateNames.csv'), alt_name_rows)
    _write_csv(os.path.join(output_dir, 'tags.csv'), tag_rows)

    for empty_file in ('labels.csv', 'proposals.csv', 'actions.csv', 'tokens.csv', 'users.csv'):
        _write_csv(os.path.join(output_dir, empty_file), [])

    n_sharks = len(source_individual_rows)
    print(f"Output written to: {output_dir}")
    print(f"  Sharks (source individuals): {n_sharks}")
    print(f"  Photos (media):              {n_media}")
    print(f"  Alternate names:             {alt_name_id}")
    print(f"  Tags:                        {tag_id}")


def _parse_photo_name(name: str):
    """Return location and unix timestamp parsed from a photo filename."""
    m = re.match(r'^([A-Za-z]+)(\d{6})', name)
    if not m:
        loc_m = re.match(r'^([A-Za-z]+)', name)
        return (loc_m.group(1) if loc_m else ''), None

    location = m.group(1)
    date_str = m.group(2)   # YYMMDD
    try:
        year  = int(date_str[0:2]) + 2000
        month = int(date_str[2:4])
        day   = int(date_str[4:6])
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return location, None
        dt = datetime.datetime(year, month, day, 12, 0, 0)
        return location, int(dt.timestamp())
    except (ValueError, OverflowError, OSError):
        return location, None


def _make_hash(name: str) -> str:
    """Return the SHA-256 hex digest of a string."""
    return hashlib.sha256(name.encode('utf-8')).hexdigest()


def _write_csv(path: str, rows: list):
    """Write rows to a CSV file at the specified path."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)


if __name__ == '__main__':
    main()
