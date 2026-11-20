"""Run ESLint on the application's configured first-party JavaScript."""

import os
import sys

from tests.tools.run_tests import run_command


ESLINT_COMMAND = (
    "npm", "exec", "--prefix", "frontend", "--no", "--",
    "eslint", "--config", "frontend/eslint.config.js",
)

ESLINT_TARGETS = (
    "frontend/eslint.config.js",
    "frontend/static/js",
    "demo",
)


def main():
    """Run the complete JavaScript lint scope and return its status."""
    return 0 if run_eslint() else 1


def run_eslint(targets=ESLINT_TARGETS):
    """Run repository-local ESLint across the configured targets and return whether all passed."""
    missing_targets = [target for target in targets if not os.path.exists(target)]
    if missing_targets:
        for target in missing_targets:
            print(f"Lint target not found: {target}")
        return False

    print("Running eslint on", ", ".join(targets))
    command = [*ESLINT_COMMAND, *targets]
    return run_command(command)


if __name__ == "__main__":
    sys.exit(main())
