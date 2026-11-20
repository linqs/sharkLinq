"""Orchestrator script for the Shark Matcher App."""

import argparse
import datetime
import os
import signal
import subprocess
import sys

from core import app_constants
from core import app_utils

INTERRUPT_GRACE_SECONDS = 10


def main():
    """Orchestrate the Shark Matcher App server, test mode, and database operations."""
    parser = argparse.ArgumentParser(
        description="Shark Matcher App Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-r', '--run', action='store_true',
                        help="Run the server.")
    parser.add_argument('--demo', action='store_true',
                        help='Offer the shared disposable Demo mode in the app.')
    parser.add_argument('--port', type=int, default=None,
                        help="Serve the app on this port "
                             f"(default: {app_constants.DEFAULT_PORT}).")
    parser.add_argument('--host', default=None,
                        help="Listen only on this interface address, such as 127.0.0.1 "
                             "(default: every interface).")
    parser.add_argument('-t', '--test', action='store_true',
                        help="Run the server using test database paths.")
    parser.add_argument('--repopulate', action='store_true',
                        help="Repopulate the test database before running.")
    parser.add_argument('-s', '--snapshot', action='store_true',
                        help="Create a snapshot of the database.")

    # If no arguments, show help.
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    args = parser.parse_args()

    app_utils.print_script_banner(__file__, args)

    python_exec = sys.executable
    server_command = [
        python_exec,
        "-m",
        "frontend.cli.serve",
        ".",
        "--no-browser",
    ]
    if args.demo:
        server_command.append('--demo')
    if args.port is not None:
        server_command += ['--port', str(args.port)]
    if args.host is not None:
        server_command += ['--host', args.host]

    # Set up logging directory and file.
    log_dir = os.path.join(app_constants.OUT_DATA_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "shark_identification.log")

    with open(log_file, "a") as f:
        f.write(f"{datetime.datetime.now()}: Starting shark_identification.py with args: {' '.join(sys.argv[1:])}\n")

    # Execute actions based on flags.

    if args.snapshot:
        cmd_snapshot = [python_exec, "-m", "db.database", "dump"]
        run_step(cmd_snapshot, "Database Snapshot")

    if args.run and not args.test:
        print(
            "Running Shark Matcher app server "
            "(backup runner will start automatically)"
        )
        run_step(server_command, "Run Server")

    if args.test:
        print("Running Shark Matcher server with test data.")

        env = os.environ.copy()
        env["SHARK_TEST_MODE"] = "true"

        if args.repopulate:
            cmd_init = [python_exec, "-m", "db.database", "--test", "init", "--force"]
            run_step(cmd_init, "Init Test Database", env=env)

            cmd_populate = [
                python_exec, "-m", "db.database",
                "--test", "populate", "--all", "--data-path",
                app_constants.CI_TEST_DATABASE_FORMAT_DIR,
            ]
            run_step(cmd_populate, "Populate Test Database", env=env)

        run_step(server_command, "Run Server (Test Mode)", env=env)


def run_step(command, step_name, cwd=None, env=None):
    """Run a subprocess command and check for errors."""
    print(f"\n{'='*60}")
    print(f"=== {step_name} ===")
    print(f"{'='*60}")
    print(f"Running command: {' '.join(command)}")

    process = subprocess.Popen(command, cwd=cwd or app_constants.THIS_DIR, env=env)
    # Forward SIGTERM so stopping the orchestrator also stops the step it is running.
    previous_handler = signal.signal(
        signal.SIGTERM, lambda signum, _frame: process.send_signal(signum))
    try:
        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            # Terminal Ctrl-C reaches the step too; forward SIGINT only if it outlives the grace period.
            try:
                returncode = process.wait(timeout=INTERRUPT_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.send_signal(signal.SIGINT)
                returncode = process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    if returncode != 0:
        print(f"\n[ERROR] Failed at step: {step_name}")
        print(f"Command returned non-zero exit status: {returncode}")
        sys.exit(returncode)

    print(f"\n[SUCCESS] Step completed: {step_name}")


if __name__ == "__main__":
    main()
