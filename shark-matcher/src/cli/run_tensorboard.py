"""Launch TensorBoard to inspect training logs for one or more trained models."""
import argparse
import glob
import os
import subprocess
import sys
import threading
import time
import webbrowser

from src.core import constants
from src.core import path_utils

path_utils.ensure_shark_model_in_path()


def main():
    """Launch TensorBoard for one or more trained Shark Matcher models."""
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard for Shark Matcher trained models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m src.cli.run_tensorboard "
               "--models test-model/vit_triplet_1\n"
               "  python3 -m src.cli.run_tensorboard --models latest\n"
               "  python3 -m src.cli.run_tensorboard --list\n")
    parser.add_argument('--models', nargs='+',
                        help='Name(s) of the model(s) to visualize (directories '
                             'under sm-data/trained-models/). '
                             'Use "latest" to pick the most recent.')
    parser.add_argument('--port', type=int, default=6006,
                        help='Port to serve TensorBoard on. Default: 6006.')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not automatically open the browser.')
    parser.add_argument('--list', action='store_true',
                        help='List available trained models and exit.')

    args = parser.parse_args()

    # List mode.
    if args.list:
        list_available_runs()
        return

    # Target resolution.
    if not args.models:
        parser.print_help()
        print("\nError: Please specify --models <name> or 'latest'.")
        return

    # Resolve paths.
    target_runs = resolve_target_runs(args.models)

    # Launch TensorBoard.
    launch_tensorboard(target_runs, args.port, not args.no_browser)


def launch_tensorboard(target_runs, port, open_browser_auto=True):
    """Start TensorBoard for the specified runs."""
    if not target_runs:
        print("No valid log directories found. Exiting.")
        return

    # Construct the logdir spec.
    if len(target_runs) == 1:
        logdir_spec = target_runs[0][1]
        names_str = target_runs[0][0]
    else:
        # For multiple models, use comma-separated name:path syntax without spaces.
        logdir_spec = ",".join([f"{name}:{path}" for name, path in target_runs])
        names_str = ", ".join([name for name, _ in target_runs])

    print(f"\nStarting TensorBoard for: {names_str}")
    print(f"Logdir Spec: {logdir_spec}")

    # For multiple models, use --logdir_spec instead of --logdir.
    logdir_flag = "--logdir" if len(target_runs) == 1 else "--logdir_spec"
    cmd = ["tensorboard", logdir_flag, logdir_spec, "--port", str(port), "--bind_all"]

    # Browser thread.
    if open_browser_auto:
        threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")


def resolve_target_runs(targets):
    """Resolve a list of model names or 'latest' to their log directory paths."""
    resolved_runs = []

    for name in targets:
        path = None

        if name.lower() == "latest":
            path = get_latest_model_dir()
            if path:
                print(f"Resolved 'latest' to: {os.path.basename(path)}")
                name = os.path.basename(path) # Update name for display.
            else:
                print("Error: No trained models found to determine 'latest'.")
                continue
        else:
            path = os.path.join(constants.TRAINED_MODELS_DIR, name)

        if path and os.path.exists(path):
            # Check for "logs" subdirectory.
            logs_path = os.path.join(path, "logs")
            if os.path.exists(logs_path):
                resolved_runs.append((name, os.path.realpath(logs_path)))
        else:
            print(f"Error: Model directory not found: {path}")

    return resolved_runs


def list_available_runs():
    """Print a list of available trained models."""
    print(f"\nAvailable Trained Models in {constants.TRAINED_MODELS_DIR}:")
    if os.path.exists(constants.TRAINED_MODELS_DIR):
        for d in sorted(os.listdir(constants.TRAINED_MODELS_DIR)):
            if os.path.isdir(os.path.join(constants.TRAINED_MODELS_DIR, d)):
                print(f"  - {d}")
    else:
        print("  (Directory not found)")


def get_latest_model_dir():
    """Find the most recently modified directory in TRAINED_MODELS_DIR."""
    basedir = constants.TRAINED_MODELS_DIR
    all_dirs = [os.path.join(basedir, d) for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d))]
    if not all_dirs:
        return None
    return max(all_dirs, key=os.path.getmtime)


def open_browser(port):
    """Wait for TensorBoard to start and open the default browser."""
    time.sleep(2.0)
    url = f"http://localhost:{port}/"
    print(f"Opening {url} in default browser...")
    webbrowser.open(url)


if __name__ == "__main__":
    main()
