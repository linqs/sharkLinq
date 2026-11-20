"""Backfill missing dates on Media records by parsing timestamps from media filenames."""


import argparse

from core import app_constants
from core import app_utils


def backfill_media_dates(dry_run=False):
    """Re-parse dates from media names for all Media rows with a NULL time."""
    app_utils.print_function_banner(
        "backfill_media_dates",
        "Backfilling missing dates on Media records from parsed names.")

    # Fetch all media rows where time is NULL.
    rows = app_utils.execute_sql(
        "SELECT id, name FROM Media WHERE time IS NULL"
    )

    if not rows:
        print("No Media records with missing dates found. Nothing to do.")
        return

    print(f"Found {len(rows)} Media record(s) with missing dates.\n")

    updated_count = 0
    skipped_count = 0

    connection = app_utils.db_connect()
    try:
        for media_id, media_name in rows:
            if not media_name:
                skipped_count += 1
                continue

            parsed_time = app_utils.parse_time_from_name(media_name)

            if parsed_time is None:
                skipped_count += 1
                continue

            if dry_run:
                print(f"  [DRY RUN] Would update Media id={media_id} "
                      f"name='{media_name}' -> time={parsed_time}")
            else:
                connection.execute(
                    "UPDATE Media SET time = :time WHERE id = :id",
                    {"time": parsed_time, "id": media_id}
                )

            updated_count += 1

        if not dry_run:
            connection.commit()

    except Exception as e:
        print(f"Error during backfill, rolling back: {e}")
        try:
            connection.rollback()
        except Exception as rb_ex:
            print(f"Warning: Rollback failed: {rb_ex}")
        raise
    finally:
        connection.close()

    # Summary.
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}Backfill complete.")
    print(f"  {prefix}Updated:  {updated_count}")
    print(f"  Skipped (unparseable or empty name): {skipped_count}")
    print(f"  Total processed: {len(rows)}")


def main():
    """Main entry point for the backfill script."""
    parser = argparse.ArgumentParser(
        description="Backfill missing Media dates from parsed media names.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing to the database.")
    args = parser.parse_args()

    app_utils.print_script_banner(__file__, args)
    backfill_media_dates(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
