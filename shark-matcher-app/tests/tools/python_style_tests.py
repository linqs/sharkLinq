"""Run the repository-owned error-level Python lint gate."""

import argparse
import os
import subprocess
import sys


PYLINT_RCFILE = "config/pylintrc"

DEFAULT_PATHS = (
    "backend",
    "core",
    "db",
    "demo",
    "frontend",
    "scripts",
    "setup.py",
    "tests",
)


def main(argv=None):
    """Parse optional lint targets and return Pylint's exit status."""
    parser = argparse.ArgumentParser(
        description="Run the repository's error-level Python lint gate."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional repository-relative files or directories to lint.",
    )
    args = parser.parse_args(argv)
    return run_pylint(args.paths or DEFAULT_PATHS)


def run_pylint(paths):
    """Run Pylint's error checks against the requested repository paths."""
    files = python_files(paths)
    command = [sys.executable, "-m", "pylint", f"--rcfile={PYLINT_RCFILE}", "--errors-only", *files]
    print(f"Running: {sys.executable} -m pylint --errors-only on {len(files)} files "
          f"from {' '.join(paths)}")
    return subprocess.run(command, check=False).returncode


def python_files(paths):
    """Expand directories into Python files, because Pylint skips modules outside packages when given a directory."""
    files = []
    for path in paths:
        if not os.path.isdir(path):
            files.append(path)
            continue
        for root, directories, names in os.walk(path):
            directories[:] = sorted(
                name for name in directories
                if name != "__pycache__" and not name.startswith(".")
            )
            files.extend(
                os.path.join(root, name) for name in sorted(names) if name.endswith(".py")
            )
    return files


if __name__ == "__main__":
    sys.exit(main())
