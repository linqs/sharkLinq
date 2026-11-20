"""Executable contract markers and strict failure expectations for the approved V1 service boundary."""

import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest
from src.core import constants
import src
from src.api import server
from tests.tools import run_tests


MATCHER_ROOT = Path(__file__).resolve().parents[2]
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
# Internal and historical records open with this banner and are exempt from current-documentation checks.
_STATUS_BANNER = re.compile(r"\A#[^\n]*\n+> \*\*Status:\*\*")
_ENVIRONMENT_NAME = re.compile(r"\bSHARK_[A-Z0-9_]+\b")
_SERVICE_KEY_ASSIGNMENT = re.compile(r"SHARK_INFERENCE_API_KEY=(\S+)")
_REPOSITORY_BLOB_LINK = re.compile(r"https://github\.com/linqs/shark-matcher(?:-app)?/blob/")
_ANGLE_PLACEHOLDER = re.compile(r"<[a-z][a-z -]*>")
# Research notes keep their own formatting outside the documentation conventions.
_RESEARCH_DIRECTORIES = ("experimental/",)
# Words that Title Case headings leave lowercase unless they come first.
_SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "nor",
                "of", "on", "or", "per", "the", "to", "via", "vs", "with"}


@pytest.fixture(scope="module", name="matcher_artifacts")
def _matcher_artifacts_fixture(tmp_path_factory):
    """Build matcher release artifacts from a repository-local temporary checkout."""
    workspace = tmp_path_factory.mktemp("matcher-artifacts")
    source = workspace / "source"
    wheelhouse = workspace / "wheelhouse"
    sdist_directory = workspace / "sdist"
    shutil.copytree(
        MATCHER_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            "__pycache__",
            "*.egg-info",
            "build",
            "dist",
            "venv",
        ),
    )
    wheelhouse.mkdir()
    sdist_directory.mkdir()
    wheel_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheelhouse),
            str(source),
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    assert wheel_result.returncode == 0, wheel_result.stderr or wheel_result.stdout
    sdist_result = subprocess.run(
        [
            sys.executable,
            "setup.py",
            "sdist",
            "--dist-dir",
            str(sdist_directory),
        ],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )
    assert sdist_result.returncode == 0, sdist_result.stderr or sdist_result.stdout
    wheels = list(wheelhouse.glob("*.whl"))
    source_archives = list(sdist_directory.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(source_archives) == 1
    return SimpleNamespace(
        source=source,
        wheel=wheels[0],
        sdist=source_archives[0],
    )


@pytest.fixture(scope="module", name="matcher_wheel")
def _matcher_wheel_fixture(matcher_artifacts):
    """Return the wheel from the shared matcher artifact build."""
    return matcher_artifacts.wheel


@pytest.fixture(scope="module", name="matcher_sdist")
def _matcher_sdist_fixture(matcher_artifacts):
    """Return the source archive from the shared matcher artifact build."""
    return matcher_artifacts.sdist


@pytest.fixture(
    params=("wheel", "sdist"),
    ids=("wheel", "source-archive"),
    name="matcher_release_artifact",
)
def _matcher_release_artifact_fixture(request, matcher_artifacts):
    """Return each matcher release artifact for common installation checks."""
    return getattr(matcher_artifacts, request.param)


def _wheel_metadata(wheel_path):
    """Return the decoded core metadata from a built wheel."""
    with zipfile.ZipFile(wheel_path) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_files) == 1
        return archive.read(metadata_files[0]).decode("utf-8")


def _source_archive_contents(source_archive):
    """Return the normalized member names from a gzipped source archive."""
    return set(_release_artifact_payloads(source_archive))


def _release_artifact_payloads(artifact_path):
    """Return file payloads from a wheel or gzipped source archive."""
    if artifact_path.suffix == ".whl":
        with zipfile.ZipFile(artifact_path) as archive:
            return {name: archive.read(name) for name in archive.namelist()}

    with tarfile.open(artifact_path, "r:gz") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }


def _git_tracked_paths(repository_root, include_untracked=False):
    """Return tracked paths, plus unignored new files when requested, that remain in the prospective working tree."""
    command = ["git", "ls-files"]
    if include_untracked:
        command += ["--cached", "--others", "--exclude-standard"]
    result = subprocess.run(
        command,
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return [
        path for path in result.stdout.splitlines()
        if (repository_root / path).is_file()
    ]


def _path_portability_violations(payloads, exact_roots=()):
    """Return files containing developer-specific or placeholder rooted paths."""
    patterns = {
        "macOS user home": re.compile(rb"/" + rb"Users/[A-Za-z0-9._-]+/"),
        "Linux user home": re.compile(rb"/" + rb"home/[A-Za-z0-9._-]+/"),
        "literal temporary path": re.compile(rb"/" + rb"(?:private/)?tmp/"),
        "Homebrew installation prefix": re.compile(rb"/" + rb"opt/homebrew/"),
        "placeholder absolute path": re.compile(rb"/" + rb"path/to/"),
        "home-directory shorthand": re.compile(rb"~" + rb"/"),
        "Windows drive root": re.compile(rb"(?<![A-Za-z])[A-Za-z]:[\\/]"),
    }
    encoded_roots = {
        str(root).encode("utf-8")
        for root in exact_roots
        if str(root)
    }
    violations = []
    for name, payload in payloads.items():
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for description, pattern in patterns.items():
            if pattern.search(payload):
                violations.append(f"{name}: {description}")
        if any(root in payload for root in encoded_roots):
            violations.append(f"{name}: local build or checkout root")
    return violations


def _requirement_name(requirement):
    """Return a normalized distribution name from a requirement specifier."""
    return re.split(r"[ <>=;\[(]", requirement, maxsplit=1)[0].lower()


def _requirement_names(metadata):
    """Return normalized mandatory dependency names from wheel metadata."""
    names = set()
    for line in metadata.splitlines():
        if not line.lower().startswith("requires-dist:") or "; extra ==" in line:
            continue
        requirement = line.split(":", 1)[1].strip()
        names.add(_requirement_name(requirement))
    return names


def _extra_requirement_names(metadata, extra):
    """Return normalized dependency names assigned to one wheel extra."""
    marker = f'extra == "{extra}"'
    names = set()
    for line in metadata.splitlines():
        if not line.lower().startswith("requires-dist:") or marker not in line.lower():
            continue
        requirement = line.split(":", 1)[1].strip()
        names.add(_requirement_name(requirement))
    return names


def _requirement_file_names(path):
    """Return normalized dependency names declared directly in a requirements file."""
    names = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        requirement = raw_line.strip()
        if not requirement or requirement.startswith(("#", "-")):
            continue
        names.add(_requirement_name(requirement))
    return names


def _model_field_names(model):
    """Return Pydantic v1 or v2 model field names."""
    fields = getattr(model, "model_fields", None)
    if fields is None:
        fields = model.__fields__
    return set(fields)


def _markdown_documents(repository_root, include_internal=True):
    """Return the text of each Markdown document in the prospective working tree, keyed by path."""
    documents = {}
    for path in _git_tracked_paths(repository_root, include_untracked=True):
        if path.endswith(".md"):
            text = (repository_root / path).read_text(encoding="utf-8")
            if include_internal or not _STATUS_BANNER.match(text):
                documents[path] = text
    return documents


def _prose(text):
    """Return Markdown text without blockquote markers or line wrapping."""
    return re.sub(r"\s+", " ", re.sub(r"(?m)^\s*>\s?", "", text))


def _heading_anchors(text):
    """Return the GitHub anchor of each heading outside fenced code."""
    anchors = set()
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("#"):
            heading = re.sub(r"[^\w\- ]", "", line.lstrip("#").strip().lower())
            anchors.add(heading.replace(" ", "-"))
    return anchors


def _documentation_link_violations(repository_root, documents, forbidden_directories=()):
    """Return relative links that are missing, leave the repository, name another document, miss their anchor, or enter a forbidden directory."""
    violations = []
    for path, text in documents.items():
        source = repository_root / path
        for target in _MARKDOWN_LINK.findall(text):
            if re.match(r"[a-z][a-z0-9+.-]*:", target):
                continue
            relative, _, anchor = target.partition("#")
            destination = (source.parent / relative).resolve() if relative else source
            if (not destination.is_relative_to(repository_root) or not destination.exists()
                    or destination.relative_to(repository_root).parts[:1] in
                    {(directory,) for directory in forbidden_directories}):
                violations.append(f"{path}: {target}")
            elif destination != source and (
                    destination.suffix in {".md", ".rst"}
                    or (destination.is_dir() and (destination / "README.md").exists())):
                violations.append(f"{path}: {target} (name the document in plain text)")
            elif (anchor and destination.suffix == ".md"
                    and anchor not in _heading_anchors(destination.read_text(encoding="utf-8"))):
                violations.append(f"{path}: {target}")
    return violations


def _fixed_service_key_documents(documents):
    """Return documents that assign the service key a literal instead of a substitution or uppercase placeholder."""
    return sorted(
        path for path, text in documents.items()
        for value in _SERVICE_KEY_ASSIGNMENT.findall(text)
        if not value.strip("\"'").startswith("$")
        and not re.fullmatch(r"[A-Z][A-Z0-9_]*", value.strip("\"'"))
    )


def _source_environment_names(repository_root):
    """Return SHARK_* names that non-test Python and shell sources mention."""
    names = set()
    for path in _git_tracked_paths(repository_root, include_untracked=True):
        if path.endswith((".py", ".sh")) and not path.startswith("tests/"):
            names.update(_ENVIRONMENT_NAME.findall(
                (repository_root / path).read_text(encoding="utf-8")))
    return names

def _markdown_convention_violations(documents):
    """Return headings, rules, fences, placeholders, and long paragraphs that break the Markdown conventions."""
    violations = []
    for path, text in documents.items():
        in_fence = False
        for number, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if _ANGLE_PLACEHOLDER.search(line):
                violations.append(f"{path}:{number}: use an UPPER_CASE placeholder")
            if stripped.startswith("```"):
                if not in_fence and stripped[3:].strip() in ("", "sh"):
                    violations.append(f"{path}:{number}: tag the fence (bash for shell, text otherwise)")
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if stripped == "---":
                violations.append(f"{path}:{number}: remove the horizontal rule")
            heading = re.match(r"#{1,6} (.*)", line)
            if heading:
                words = re.findall(r"[A-Za-z][\w'-]*", re.sub(
                    r"`[^`]*`|\[[^\]]*\]\([^)]*\)", " ", heading.group(1)))
                if any(word[0].islower() and not any(ch.isdigit() for ch in word)
                       and (index == 0 or word.lower() not in _SMALL_WORDS)
                       for index, word in enumerate(words)):
                    violations.append(f"{path}:{number}: use Title Case")
        violations.extend(_paragraph_violations(path, text))
    return violations


def _paragraph_violations(path, text):
    """Return paragraphs over three sentences and runs of three paragraphs with no heading, list, table, or code between them."""
    violations, paragraph, run, in_fence = [], [], 0, False

    def close():
        nonlocal paragraph, run
        if not paragraph:
            return
        flat = re.sub(r"`[^`]*`", "X", re.sub(r"\b(e\.g|i\.e|etc|vs)\.", r"\1", " ".join(
            line for _, line in paragraph)))
        sentences = [part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[X])", flat.strip()) if part]
        run += 1
        if len(sentences) > 3:
            violations.append(f"{path}:{paragraph[0][0]}: split the paragraph (over three sentences)")
        if run == 3:
            violations.append(f"{path}:{paragraph[0][0]}: add a heading, list, or code block (three paragraphs in a row)")
        paragraph = []

    previous_structural = False
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            close()
            run, in_fence, previous_structural = 0, not in_fence, True
            continue
        if in_fence:
            continue
        if not stripped:
            close()
            continue
        structural = (stripped.startswith(("#", "|", ">", "<", "![")) or re.match(r"([-*+]|\d+[.)])\s", stripped))
        if structural:
            close()
            run, previous_structural = 0, True
        elif line[0].isspace() and previous_structural and not paragraph:
            continue
        else:
            paragraph.append((number, stripped))
            previous_structural = False
    close()
    return violations


def test_embedding_request_is_snapshot_identifier_only():
    """The generation request carries a snapshot ID and a background flag, never a path."""
    assert _model_field_names(server.EmbeddingsRequest) == {
        "model_name",
        "snapshot_id",
        "background",
    }


def test_health_response_is_liveness_only():
    """The health response reports process liveness only."""
    assert _model_field_names(server.HealthResponse) == {"status"}


def test_readiness_route_is_declared():
    """The API exposes a distinct readiness route."""
    route_paths = {
        route.path for route in server.protected_router.routes
    }
    assert "/ready" in route_paths


def test_matcher_has_no_app_or_database_dependencies():
    """Matcher runtime and packaging have no app/database dependencies."""
    forbidden_imports = []
    for source_path in (MATCHER_ROOT / "src").rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "db" or name.startswith("db.") or name.startswith("backend.db"):
                    forbidden_imports.append(f"{source_path.relative_to(MATCHER_ROOT)}:{name}")

    constants_source = (MATCHER_ROOT / "src" / "core" / "constants.py").read_text(encoding="utf-8")
    forbidden_constants = {
        name for name in (
            "SHARK_APP_DIR",
            "DB_DIR",
            "DB_PATH",
            "ensure_shark_app_in_path",
        )
        if name in constants_source
    }
    package_source = (MATCHER_ROOT / "src" / "__init__.py").read_text(
        encoding="utf-8"
    )
    utils_source = (MATCHER_ROOT / "src" / "core" / "utils.py").read_text(encoding="utf-8")

    assert not forbidden_imports
    assert forbidden_constants == set()
    assert "SHARK_APP_DIR" not in package_source
    assert "__path__.append" not in package_source
    assert "sqlite3.connect" not in utils_source

    # Packaging metadata must also stay repository-local and never install from a sibling checkout.
    packaging_text = "\n".join(
        (MATCHER_ROOT / filename).read_text(encoding="utf-8")
        for filename in ("setup.py", "requirements/requirements.txt")
    )
    assert "shark-matcher-app" not in packaging_text
    assert "../" not in packaging_text


def test_matcher_runtime_and_current_docs_never_use_experimental_code():
    """Packaged code neither imports nor names experimental modules, and current docs never invoke them."""
    violations = []
    runtime_files = [*(MATCHER_ROOT / "src").rglob("*.py"),
                     MATCHER_ROOT / "setup.py"]
    for source_path in runtime_files:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names = re.findall(r"\bexperimental[./]\w+", node.value)
            else:
                continue
            violations.extend(
                f"{source_path.relative_to(MATCHER_ROOT)}:{node.lineno}: {name}"
                for name in names if name == "experimental" or name.startswith(("experimental.", "experimental/"))
            )
    for path, text in _markdown_documents(MATCHER_ROOT, include_internal=False).items():
        if not path.startswith(_RESEARCH_DIRECTORIES):
            violations.extend(f"{path}: {match}" for match in re.findall(
                r"(?:-m|import|from) experimental\b[\w.]*", text))

    assert violations == []


def test_matcher_wheel_excludes_experimental_code(matcher_wheel):
    """Research code under experimental/ is never packaged."""
    with zipfile.ZipFile(matcher_wheel) as archive:
        assert [name for name in archive.namelist() if name.startswith("experimental/")] == []


def test_matcher_imports_without_exposing_the_sibling_app_package():
    """An isolated matcher import cannot add app source to its package path."""
    script = """
import json
import sys

import src

app_paths = [path for path in src.__path__ if "shark-matcher-app" in path]
app_modules = [
    name for name in sys.modules
    if name == "db" or name.startswith("db.") or name.startswith("backend.db")
]
assert app_paths == [], app_paths
assert app_modules == [], app_modules
print(json.dumps({"src_path": list(src.__path__)}))
"""
    environment = os.environ.copy()
    environment.pop("SHARK_APP_ROOT", None)
    environment["PYTHONPATH"] = str(MATCHER_ROOT)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=MATCHER_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_matcher_release_artifacts_import_without_source_checkout(
    matcher_release_artifact,
    tmp_path,
):
    """Each matcher release artifact installs and imports outside the checkout."""
    install_root = tmp_path / "installed"
    install_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(install_root),
            str(matcher_release_artifact),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert install_result.returncode == 0, (
        install_result.stderr or install_result.stdout
    )

    script = """
import sys
sys.path.insert(0, sys.argv[1])
from src.core import constants
from src.core import path_utils
import src
import src.config.config
import src.retrieval.suggestions
from src.core import utils
assert src.config.config.__file__.startswith(sys.argv[1]), src.config.config.__file__
"""
    import_result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(install_root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert import_result.returncode == 0, import_result.stderr or import_result.stdout


def test_matcher_source_archive_contains_setup_inputs_and_runtime_files(
    matcher_sdist,
    matcher_wheel,
):
    """The matcher sdist contains every setup input and packaged runtime file."""
    contents = _source_archive_contents(matcher_sdist)
    archive_roots = {Path(name).parts[0] for name in contents}
    assert len(archive_roots) == 1
    archive_root = archive_roots.pop()

    setup_source = (MATCHER_ROOT / "setup.py").read_text(encoding="utf-8")
    requirement_files = set(
        re.findall(r'_read_requirements\("([^"]+)"\)', setup_source)
    )
    with zipfile.ZipFile(matcher_wheel) as archive:
        packaged_files = {
            name for name in archive.namelist() if ".dist-info/" not in name
        }
    required_files = {
        "MANIFEST.in",
        "docs/release-notes.md",
        "setup.py",
    }
    expected_contents = {
        f"{archive_root}/{name}"
        for name in required_files.union(requirement_files, packaged_files)
    }
    assert expected_contents.issubset(contents)


def test_matcher_release_version_is_synchronized(matcher_wheel):
    """Runtime and distribution metadata use the approved V1 version."""
    metadata = _wheel_metadata(matcher_wheel).splitlines()

    assert src.__version__ == "1.0.0"
    assert "Version: 1.0.0" in metadata


def test_matcher_wheel_declares_only_packaged_modules(matcher_wheel):
    """Every top-level name declared by the matcher wheel is packaged."""
    with zipfile.ZipFile(matcher_wheel) as archive:
        contents = set(archive.namelist())
        top_level_files = [
            name
            for name in contents
            if name.endswith(".dist-info/top_level.txt")
        ]
        assert len(top_level_files) == 1
        declared_modules = archive.read(top_level_files[0]).decode("utf-8").split()

    missing_modules = [
        module_name
        for module_name in declared_modules
        if f"{module_name}.py" not in contents
        and not any(name.startswith(f"{module_name}/") for name in contents)
    ]
    assert missing_modules == []


def test_matcher_wheel_excludes_test_packages(matcher_wheel):
    """The matcher release wheel contains no repository test packages."""
    with zipfile.ZipFile(matcher_wheel) as archive:
        packaged_tests = [
            name for name in archive.namelist() if name.startswith("tests/")
        ]
    assert packaged_tests == []


def test_matcher_default_install_excludes_optional_dependencies(matcher_wheel):
    """Default matcher metadata omits development and optional experiment tools."""
    metadata = _wheel_metadata(matcher_wheel).lower()
    optional_dependencies = {
        "coverage",
        "httpx",
        "jpype1",
        "matplotlib",
        "pandas",
        "pylint",
        "pytest",
        "sentence-transformers",
        "tensorboard",
    }
    mandatory_dependencies = _requirement_names(metadata)

    assert mandatory_dependencies.isdisjoint(optional_dependencies)
    assert "provides-extra:" in metadata


def test_matcher_wheel_dependencies_match_requirement_authorities(matcher_wheel):
    """Wheel dependency groups derive from their authoritative requirement files."""
    metadata = _wheel_metadata(matcher_wheel)
    analysis = _requirement_file_names(
        MATCHER_ROOT / "requirements" / "requirements-analysis.txt"
    )
    expected_extras = {
        "dev": _requirement_file_names(MATCHER_ROOT / "requirements" / "requirements-dev.txt"),
        "training": _requirement_file_names(
            MATCHER_ROOT / "requirements" / "requirements-training.txt"
        ),
        "visualization": analysis.union(
            _requirement_file_names(
                MATCHER_ROOT / "requirements" / "requirements-visualization.txt"
            )
        ),
        "psl": analysis.union(
            _requirement_file_names(MATCHER_ROOT / "requirements" / "requirements-psl.txt")
        ),
    }

    assert _requirement_names(metadata) == _requirement_file_names(
        MATCHER_ROOT / "requirements" / "requirements.txt"
    )
    for extra, expected_requirements in expected_extras.items():
        assert _extra_requirement_names(metadata, extra) == expected_requirements
    assert "sentence-transformers" not in metadata.lower()
    assert "httpx2" in expected_extras["dev"]
    assert "httpx" not in expected_extras["dev"]


def test_matcher_runner_requests_pytest_skip_summary():
    """The matcher runner reports skips without obsolete data overrides."""
    args = SimpleNamespace(
        py_style=False,
        run=None,
    )
    commands = run_tests.collect_commands(args)
    pytest_command = next(command for command, _ in commands if "pytest" in command)
    assert "-ra" in pytest_command
    runner_source = (MATCHER_ROOT / "tests/tools/run_tests.py").read_text(
        encoding="utf-8"
    )
    assert "--test-csv-dir" not in runner_source
    assert "--test-images-dir" not in runner_source
    assert "setup_integration_env" not in runner_source


def test_matcher_workflow_uses_supported_matrix_and_owned_commands():
    """CI delegates to repository authorities across the supported V1 matrix."""
    workflow_source = (
        MATCHER_ROOT / ".github/workflows/python-tests.yml"
    ).read_text(encoding="utf-8")

    assert "actions/checkout@v4" in workflow_source
    assert "actions/setup-python@v5" in workflow_source
    assert workflow_source.count("os: [ubuntu-latest, macos-latest]") == 2
    assert "python -m tests.tools.run_tests\n" in workflow_source
    assert "python -m tests.tools.run_tests --py-style" in workflow_source
    assert "python -m pytest" not in workflow_source
    assert workflow_source.count("https://download.pytorch.org/whl/cpu") == 2
    assert "python -m pip install ." in workflow_source
    assert "working-directory: ${{ runner.temp }}" in workflow_source
    assert "python -I -c" in workflow_source
    assert "PARITY_DB_PATH" in workflow_source
    assert "CUDA and MPS" in workflow_source


def test_matcher_ignore_rules_cover_publication_artifacts():
    """Matcher ignore rules exclude generated and sensitive publication paths."""
    ignored_paths = {
        ".coverage",
        ".coverage.unit",
        ".eggs/cache",
        ".env",
        ".env.local",
        ".pytest_cache/state",
        "app.db",
        "build/lib/module.py",
        "credentials.json",
        "dist/shark_matcher.whl",
        "htmlcov/index.html",
        "model.pt",
        "package.egg",
        "package.whl",
        "run.log",
        "src/__pycache__/module.pyc",
        "token.json",
    }
    retained_paths = {
        "containerization/.env.example",
        "containerization/Dockerfile",
        "containerization/README.md",
        "tests/fixtures/example.csv",
    }
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=MATCHER_ROOT,
        input="\n".join(sorted(ignored_paths | retained_paths)) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == ignored_paths


def test_matcher_tracked_files_exclude_publication_artifacts():
    """Matcher tracked files contain no generated or sensitive artifacts."""
    tracked_paths = _git_tracked_paths(MATCHER_ROOT)

    allowed_paths = {"containerization/.env.example"}
    allowed_data_roots = ("tests/fixtures/",)
    forbidden_directories = {
        ".eggs",
        ".pytest_cache",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "venv",
    }
    forbidden_suffixes = (
        ".db",
        ".egg",
        ".log",
        ".pt",
        ".pth",
        ".pyc",
        ".pyd",
        ".pyo",
        ".tar.gz",
        ".whl",
        ".zip",
    )
    violations = []
    for tracked_path in tracked_paths:
        if tracked_path in allowed_paths:
            continue
        path = Path(tracked_path)
        basename = path.name.lower()
        forbidden_data = (
            path.suffix.lower() in {".csv", ".tsv"}
            and not tracked_path.startswith(allowed_data_roots)
        )
        sensitive_name = (
            basename == ".env"
            or basename.startswith(".env.")
            or basename.startswith(("credentials", "secret", "token"))
        )
        if (
            forbidden_directories.intersection(path.parts)
            or tracked_path.lower().endswith(forbidden_suffixes)
            or forbidden_data
            or sensitive_name
        ):
            violations.append(tracked_path)

    ignored_result = subprocess.run(
        ["git", "ls-files", "-ci", "--exclude-standard"],
        cwd=MATCHER_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored_result.returncode == 0, ignored_result.stderr
    assert ignored_result.stdout.splitlines() == []
    assert not violations


def test_matcher_tracked_files_are_path_portable():
    """Tracked matcher content contains no developer-specific rooted paths."""
    payloads = {
        tracked_path: (MATCHER_ROOT / tracked_path).read_bytes()
        for tracked_path in _git_tracked_paths(MATCHER_ROOT)
    }

    assert not _path_portability_violations(
        payloads,
        exact_roots=(Path.home(), MATCHER_ROOT),
    )


def test_matcher_documentation_links_resolve_inside_the_repository():
    """Current documentation links resolve to existing repository files and headings, never into experimental code."""
    documents = _markdown_documents(MATCHER_ROOT, include_internal=False)

    assert not _documentation_link_violations(
        MATCHER_ROOT, documents, forbidden_directories=("experimental",))


def test_matcher_documentation_names_repository_files_by_path():
    """Current documentation names Shark Matcher files by path, because branch links drift from the paired checkout."""
    documents = _markdown_documents(MATCHER_ROOT, include_internal=False)

    assert [path for path, text in documents.items() if _REPOSITORY_BLOB_LINK.search(text)] == []


def test_matcher_documentation_follows_markdown_conventions():
    """Current documentation uses Title Case headings, tagged fences, UPPER_CASE placeholders, no rules, and short paragraphs."""
    documents = _markdown_documents(MATCHER_ROOT, include_internal=False)

    assert not _markdown_convention_violations({
        path: text for path, text in documents.items()
        if not path.startswith(_RESEARCH_DIRECTORIES)
    })


def test_matcher_documentation_contains_no_fixed_service_key():
    """Documented service-key exports generate the key or name a placeholder."""
    assert not _fixed_service_key_documents(_markdown_documents(MATCHER_ROOT))


def test_matcher_documented_environment_names_are_read_by_source():
    """Every SHARK_* name in current documentation exists in matcher source."""
    documented = set()
    for text in _markdown_documents(MATCHER_ROOT, include_internal=False).values():
        documented.update(_ENVIRONMENT_NAME.findall(text))

    assert documented - _source_environment_names(MATCHER_ROOT) == set()


def test_matcher_documented_service_address_matches_defaults(monkeypatch):
    """Documented inference ports and bind host match the server defaults."""
    monkeypatch.delenv("SHARK_INFERENCE_HOST", raising=False)
    monkeypatch.delenv("SHARK_INFERENCE_PORT", raising=False)
    host, port = constants.resolve_inference_bind_address()
    prose = " ".join(
        _prose(text)
        for text in _markdown_documents(MATCHER_ROOT, include_internal=False).values()
    )
    stated_ports = re.findall(r"\bport\b[^.`]*`(\d+)`", prose)
    url_ports = re.findall(r"https?://[\w.\-]+:(\d+)", prose)
    stated_hosts = re.findall(r"listens on `([^`]+)`", prose)

    assert stated_ports and stated_hosts
    assert {int(value) for value in stated_ports + url_ports} == {port}
    assert set(stated_hosts) == {host}


def test_matcher_documented_model_entry_values_match_parser():
    """The README's SHARK_MODELS modes and reranks are exactly those the parser accepts."""
    prose = _prose((MATCHER_ROOT / "README.md").read_text(encoding="utf-8"))
    modes = re.search(r"`MODE` is (.+?)\.", prose).group(1)
    reranks = re.search(r"`RERANK` is (.+?)\.", prose).group(1)

    # pylint: disable=protected-access
    assert set(re.findall(r"`(\w+)`", modes)) == server._VALID_MODES
    assert set(re.findall(r"`(\w+)`", reranks)) == server._VALID_RERANKS


def test_matcher_release_artifacts_are_path_portable(
    matcher_release_artifact,
    matcher_artifacts,
):
    """Built matcher artifacts contain no checkout or build-machine paths."""
    payloads = _release_artifact_payloads(matcher_release_artifact)

    assert not _path_portability_violations(
        payloads,
        exact_roots=(Path.home(), MATCHER_ROOT, matcher_artifacts.source),
    )


_PICTOGRAPHIC_SYMBOL = re.compile(
    "[\u2190-\u21ff\u2300-\u23ff\u25a0-\u27bf\u2900-\u297f\u2b00-\u2bff\U0001f000-\U0001faff\ufe0f]")


def test_matcher_tracked_text_contains_no_pictographic_symbols():
    """Documentation, comments, and user-facing text use words or plain ASCII, never emojis, arrows, or icons."""
    violations = []
    for tracked_path in _git_tracked_paths(MATCHER_ROOT):
        if tracked_path.endswith((".md", ".py", ".sh", ".txt", ".yml", ".yaml", ".toml", ".cfg")):
            text = (MATCHER_ROOT / tracked_path).read_text(encoding="utf-8")
            violations.extend(
                f"{tracked_path}:{number}"
                for number, line in enumerate(text.splitlines(), 1)
                if _PICTOGRAPHIC_SYMBOL.search(line))

    assert violations == []


def test_requirement_files_pin_every_dependency():
    """Every dependency line names an exact version so a fresh install matches the tested one."""
    unpinned = [
        f"{path.name}: {line.strip()}"
        for path in sorted((MATCHER_ROOT / "requirements").glob("requirements*.txt"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("#", "-"))
        and "==" not in line]

    assert unpinned == []
