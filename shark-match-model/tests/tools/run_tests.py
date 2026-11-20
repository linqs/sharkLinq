"""Unified functional-test and Python-style runner for Shark Matcher."""
import argparse
import os
import subprocess
import sys

from src.core import constants
from src.core import path_utils

path_utils.ensure_shark_model_in_path()

from src.core import utils  # pylint: disable=wrong-import-position


def run_command(command):
    """Run a shell command with real-time output."""
    print(f"Running: {' '.join(command)}")
    try:
        # Use subprocess.run to execute command.
        # Check=True will raise CalledProcessError on non-zero exit code.
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as error:
        print(f"Error: Command failed with exit code {error.returncode}")
        return False


def collect_commands(args: argparse.Namespace) -> list:
    """Gather the list of test commands to run based on arguments."""
    commands = []

    # 1. Style Checks
    if args.py_style:
        commands.append((
            [sys.executable, "tests/tools/python_style_tests.py"],
            "Python Style Checks",
        ))

    # 2. Functional Tests
    test_targets = []
    if args.run:
        for item in args.run:
            if os.path.exists(item):
                test_targets.append(item)
            elif os.path.exists(os.path.join("tests", item)):
                test_targets.append(os.path.join("tests", item))
            else:
                print(f"Warning: Could not find test path '{item}'")
    elif not args.py_style:
        # Default: Run src tests only
        test_targets = ["tests/unit", "tests/contracts"]

    if test_targets:
        commands.append((
            [sys.executable, "-m", "pytest", "-ra"] + test_targets,
            "Tests (pytest)",
        ))

    return commands


def execute_steps(commands: list):
    """Execute each collected test command and exit based on success."""
    success = True
    for cmd, desc in commands:
        print(f"\n--- Running {desc} ---")
        if not run_command(cmd):
            success = False

    if not success:
        print("\nFAILURE: Some steps failed.")
        sys.exit(1)

    print("\nSUCCESS: All steps passed.")
    sys.exit(0)


def main():
    """Run style checks and functional tests for Shark Matcher."""
    parser = argparse.ArgumentParser(
        description="Unified test runner for Shark Matcher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m tests.tools.run_tests\n"
               "  python3 -m tests.tools.run_tests --py-style\n"
               "  python3 -m tests.tools.run_tests --run unit/cli/test_convert_data.py\n")

    # Flags.
    parser.add_argument('--py-style', action='store_true',
                        help='Run Python style checks (pylint).')
    parser.add_argument('--run', nargs='*',
                        help='Run specific test directories or files. '
                             'Paths are resolved relative to tests/. '
                             'Default: tests/unit and tests/contracts.')

    args = parser.parse_args()
    utils.print_script_banner(__file__, args)

    commands = collect_commands(args)
    execute_steps(commands)
    sys.exit(0)


if __name__ == "__main__":
    main()
