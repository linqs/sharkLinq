"""Packaging configuration for the Shark Matcher application."""

import os

from setuptools import find_packages, setup

from frontend import __version__


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_requirements(filename):
    """Read package requirements while ignoring pip include directives."""
    path = os.path.join(THIS_DIR, filename)
    requirements = []
    with open(path, encoding="utf-8") as requirements_file:
        for raw_line in requirements_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("-r ", "--requirement ")):
                continue
            requirements.append(line)
    return requirements


setup(
    name="shark-matcher-app",
    version=__version__,
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*", "scripts", "scripts.*")),
    package_data={
        "demo": ["*.js", "*.md"],
        "db": ["sql/*.sql"],
        "frontend.static": ["**/*"],
    },
    exclude_package_data={
        "frontend.static": ["**/__pycache__/*", "**/*.pyc", "**/*.pyo"],
    },
    install_requires=_read_requirements("requirements/requirements.txt"),
    extras_require={"dev": _read_requirements("requirements/requirements-dev.txt")},
)
