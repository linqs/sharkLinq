"""Unified test and style runner for the Shark Matcher application."""

import argparse
import os
import subprocess
import sys

from core import app_utils


DEFAULT_IGNORED_TEST_PATHS = (
    "tests/manual",
    "tests/browser",
)


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
    if getattr(args, 'browser', False):
        commands.append((
            [sys.executable, '-m', 'pytest', '-ra', 'tests/browser',
             '--browser', 'chromium', '--browser', 'firefox', '--browser', 'webkit',
             '--tracing', 'retain-on-failure', '--screenshot', 'only-on-failure'],
            'Browser workflows',
        ))

    # 1. Style Checks.
    if args.py_style:
        commands.append((
            [sys.executable, "tests/tools/python_style_tests.py"],
            "Python Style Checks",
        ))
    if args.js_style:
        commands.append((
            [sys.executable, "-m", "tests.tools.javascript_style_tests"],
            "JavaScript Style Checks",
        ))

    # 2. Functional Tests.
    test_targets = []
    default_suite = False
    if args.run:
        for item in args.run:
            if os.path.exists(item):
                test_targets.append(item)
            elif os.path.exists(os.path.join("tests", item)):
                test_targets.append(os.path.join("tests", item))
            else:
                print(f"Warning: Could not find test path '{item}'")
    elif not args.py_style and not args.js_style and not getattr(args, 'browser', False):
        # Run deterministic tests by default while leaving manual tests explicit.
        test_targets = ["tests"]
        default_suite = True

    if test_targets:
        pytest_command = [sys.executable, "-m", "pytest", "-ra"]
        if default_suite:
            pytest_command.extend(
                f"--ignore={path}" for path in DEFAULT_IGNORED_TEST_PATHS
            )
        commands.append((pytest_command + test_targets, "Tests (pytest)"))

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
    """Parse runner options and execute the selected app checks."""
    parser = argparse.ArgumentParser(
        description="Unified test runner for the Shark Matcher application."
    )

    # Flags.
    parser.add_argument(
        "--py-style", action="store_true", help="Run Python style checks."
    )
    parser.add_argument(
        "--js-style", action="store_true", help="Run JavaScript style checks."
    )
    parser.add_argument(
        "--run", nargs="*", help="Run specific test directories/files."
    )
    parser.add_argument('--browser', action='store_true', help='Run the required browser matrix.')

    args = parser.parse_args()
    app_utils.print_script_banner(__file__, args)

    commands = collect_commands(args)
    execute_steps(commands)
    sys.exit(0)


if __name__ == "__main__":
    main()
