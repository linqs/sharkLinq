"""Schedule complete trusted database and annotation backups."""

import datetime
import time
import multiprocessing

from core import app_utils
import db.scripts.db_dump

def backup_runner():
    """Run a weekly database backup loop each Sunday between 12:00 AM and 1:00 AM."""
    app_utils.print_function_banner("backup_runner", "Starting backup runner.")
    while True:
        try:
            now = datetime.datetime.now()
            
            # Check if it's Sunday between 12:00 AM and 1:00 AM.
            if now.weekday() == 6 and 0 <= now.hour < 1:
                print("Running backup for annotations files and database tables.")
                tables_output_directory = db.scripts.db_dump.db_dump_tables(
                    export_mode=db.scripts.db_dump.ExportMode.TRUSTED,
                    durable=True,
                )
                annotations_output_directory = db.scripts.db_dump.db_dump_annotations()
                print("Backup complete. Annotations files stored in: " + annotations_output_directory)
                print("Database tables stored in: " + tables_output_directory)
                
                # After backup, calculate sleep until next Sunday 12:00 AM.
                sleep_until_next_sunday(now)
            else:
                # Not in backup window, calculate sleep until next Sunday 12:00 AM.
                sleep_until_next_sunday(now)
                
        except Exception as e:
            print(f"Backup runner crashed with error: {e}")
            import traceback
            traceback.print_exc()
            # If we crash, sleep for a minute to avoid rapid fail loops, then retry scheduling.
            time.sleep(60)


def run_backup_worker():
    """Worker function for the backup process."""
    try:
        backup_runner()
    except KeyboardInterrupt:
        print("Backup runner stopped by user.")
    except Exception as e:
        print(f"Backup runner error: {e}")
        import traceback
        traceback.print_exc()

def start_backup_runner_process():
    """Start the backup runner in a background process and return the Process object."""
    app_utils.print_function_banner("start_backup_runner_process")
    
    # Use multiprocessing instead of threading to avoid GIL/Lock issues.
    backup_process = multiprocessing.Process(target=run_backup_worker, daemon=True)
    backup_process.start()
    print("Backup runner started in background process.")
    return backup_process


def sleep_until_next_sunday(current_time):
    """Calculate and sleep until the next Sunday at 12:00 AM."""
    app_utils.print_function_banner("sleep_until_next_sunday")
    
    # Calculate days to add to get to next Sunday (weekday 6).
    days_ahead = 6 - current_time.weekday()
    
    # Target is 00:00:00 of that day.
    target_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=days_ahead)
    
    # Target next week if the computed Sunday timestamp is in the past or current window.
    if target_date <= current_time:
        target_date += datetime.timedelta(days=7)
    
    # Calculate sleep duration.
    sleep_duration = (target_date - current_time).total_seconds()
    
    print(f"Current time: {current_time}")
    print(f"Next backup scheduled for: {target_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sleeping for {sleep_duration/3600:.2f} hours ({sleep_duration:.1f} seconds)...")
    
    # Sleep until next Sunday.
    time.sleep(sleep_duration)

def main():
    """Run the automatic weekly database backup process directly."""
    # When run directly, start the backup runner process.
    start_backup_runner_process()
    
    # Keep the main thread alive.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Backup runner main process stopped.")

if __name__ == "__main__":
    main()
