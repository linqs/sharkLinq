"""CLI server runner with command-line argument parsing."""

import argparse
import multiprocessing
import os
import signal
import sys
import time
import webbrowser

from core import app_constants
from core import app_utils
import db.scripts.backup_runner
import frontend.server


def main():
    """Parse CLI arguments and run the server."""
    parser = argparse.ArgumentParser(description="Start the webserver for Shark Matcher.")
    parser.add_argument('base_dir', metavar='PROJECT_DIR',
                        type=str,
                        help='The base directory for the Shark Matcher project the GUI will open.')
    parser.add_argument('--port', dest='port',
                        type=int, default=app_constants.DEFAULT_PORT,
                        help='The port to start the server on (default: %(default)s)')
    parser.add_argument('--host', dest='host',
                        type=str, default=app_constants.DEFAULT_HOST,
                        help='The interface address to listen on, such as 127.0.0.1 '
                             '(default: every interface)')
    parser.add_argument('--no-browser', dest='no_browser',
                        action='store_true',
                        help='Do not try to open a web browser when launching (default: %(default)s)')
    parser.add_argument('--demo', action='store_true',
                        help='Enable the optional shared, disposable Demo mode.')
    args = parser.parse_args()

    app_utils.print_script_banner(__file__, args)

    return run_cli(**vars(args))


def run_cli(base_dir=None, port=app_constants.DEFAULT_PORT, no_browser=None, demo=False,
            host=app_constants.DEFAULT_HOST, **kwargs):
    """Run server according to the arguments provided."""
    if not os.path.isdir(base_dir):
        raise ValueError("Project directory '%s' does not exist or is not a directory." % (base_dir))

    # Start the backup runner in a background process.
    backup_process = db.scripts.backup_runner.start_backup_runner_process()

    # Unwind through the server's finally blocks; shutdown() would deadlock on the serving thread.
    def _handle_sigterm(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("Received SIGTERM, shutting down cleanly...")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # A queue that we will put something in if the server has been stopped.
    server_done_queue = multiprocessing.SimpleQueue()

    # Open a browser window in the background.
    browser_open_process = None
    if not no_browser:
        browser_open_process = multiprocessing.Process(target = _open_browser,
                                                       args = (server_done_queue,
                                                               frontend.server.local_url(host, port)))
        browser_open_process.start()

    try:
        frontend.server.run(base_dir, port=port, demo_enabled=demo, host=host)
    except KeyboardInterrupt:
        print("Received Ctrl-C, shutting down cleanly...")
    finally:
        server_done_queue.put(True)

        if browser_open_process is not None:
            browser_open_process.join()

        # Explicitly terminate the backup process to allow it to flush before exiting.
        if backup_process.is_alive():
            backup_process.terminate()
            backup_process.join(timeout=5)

    return 0


def _open_browser(server_done_queue, address, delay_secs = 0.5):
    """Wait briefly and open the web browser."""

    # Check the queue for an indication to stop.
    if not server_done_queue.empty():
        return

    time.sleep(delay_secs)

    # Check the queue (again) for an indication to stop.
    if not server_done_queue.empty():
        return

    webbrowser.open(address)


if __name__ == '__main__':
    sys.exit(main())
