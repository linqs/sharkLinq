# pylint: disable=too-many-lines
"""Executable contracts for packaging and clean-checkout behavior."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from core import app_constants
from db import database
import frontend
from tests.tools import javascript_style_tests, run_tests


APP_ROOT = Path(__file__).resolve().parents[2]
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
# Internal and historical records open with this banner and are exempt from current-documentation checks.
_STATUS_BANNER = re.compile(r"\A#[^\n]*\n+> \*\*Status:\*\*")
_ENVIRONMENT_NAME = re.compile(r"\bSHARK_[A-Z0-9_]+\b")
_SERVICE_KEY_ASSIGNMENT = re.compile(r"SHARK_INFERENCE_API_KEY=(\S+)")
_REPOSITORY_BLOB_LINK = re.compile(r"https://github\.com/linqs/(?:shark-match-model|shark-matcher-app)/blob/")
_ANGLE_PLACEHOLDER = re.compile(r"<[a-z][a-z -]*>")
# Words that Title Case headings leave lowercase unless they come first.
_SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "nor",
                "of", "on", "or", "per", "the", "to", "via", "vs", "with"}
# Matcher settings the app guides explain; the matcher's constants and server own them.
_MATCHER_ENVIRONMENT_NAMES = {"SHARK_INFERENCE_HOST", "SHARK_INFERENCE_PORT", "SHARK_MODELS"}
# Endpoints served by the matcher, so their documented URLs use the inference port.
_MATCHER_ROUTES = ("/health", "/ready", "/stats", "/embeddings")


@pytest.fixture(scope="module", name="app_artifacts")
def _app_artifacts_fixture(tmp_path_factory):
    """Build app release artifacts from a repository-local temporary checkout."""
    workspace = tmp_path_factory.mktemp("app-artifacts")
    source = workspace / "source"
    wheelhouse = workspace / "wheelhouse"
    sdist_directory = workspace / "sdist"
    shutil.copytree(
        APP_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".pytest_cache", "__pycache__", "*.egg-info",
            "build", "dist", "node_modules", "venv",
        ),
    )
    generated_cache = source / "frontend/static/__pycache__"
    generated_cache.mkdir()
    (generated_cache / "publication-audit.pyc").write_bytes(b"generated cache")
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


@pytest.fixture(scope="module", name="app_wheel")
def _app_wheel_fixture(app_artifacts):
    """Return the wheel from the shared app artifact build."""
    return app_artifacts.wheel


@pytest.fixture(scope="module", name="app_sdist")
def _app_sdist_fixture(app_artifacts):
    """Return the source archive from the shared app artifact build."""
    return app_artifacts.sdist


@pytest.fixture(
    params=("wheel", "sdist"),
    ids=("wheel", "source-archive"),
    name="app_release_artifact",
)
def _app_release_artifact_fixture(request, app_artifacts):
    """Return each app release artifact for common installation checks."""
    return getattr(app_artifacts, request.param)



def test_app_wheel_has_repository_specific_distribution_name(app_wheel):
    """The app wheel is distinguishable from the matcher distribution."""
    metadata = _wheel_metadata(app_wheel)
    assert "Name: shark-matcher-app" in metadata.splitlines()


def test_app_wheel_declares_only_app_runtime_dependencies(app_wheel):
    """App install metadata includes its runtime stack and excludes ML packages."""
    requirement_names = _requirement_names(_wheel_metadata(app_wheel))
    required = {
        "argon2-cffi",
        "pillow",
    }
    matcher_only = {
        "numpy",
        "peft",
        "scikit-learn",
        "timm",
        "torch",
        "torchvision",
        "transformers",
    }

    assert required.issubset(requirement_names)
    assert requirement_names.isdisjoint(matcher_only)


def test_app_wheel_dependencies_match_requirement_authorities(app_wheel):
    """Wheel runtime and development metadata derive from their requirement files."""
    metadata = _wheel_metadata(app_wheel)
    assert _requirement_names(metadata) == _requirement_file_names(
        APP_ROOT / "requirements" / "requirements.txt"
    )
    assert _extra_requirement_names(metadata, "dev") == _requirement_file_names(
        APP_ROOT / "requirements" / "requirements-dev.txt"
    )


def test_app_wheel_contains_runtime_modules_and_browser_assets(app_wheel):
    """The installed artifact contains the modules and assets needed to run."""
    with zipfile.ZipFile(app_wheel) as archive:
        contents = set(archive.namelist())
    _assert_no_generated_python_artifacts(contents)
    static_root = APP_ROOT / "frontend" / "static"
    browser_assets = {
        path.relative_to(APP_ROOT).as_posix()
        for path in static_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    required_contents = {
        "core/__init__.py",
        "core/app_constants.py",
        "core/app_utils.py",
        "demo/__init__.py",
        "demo/api.py",
        "demo/runtime.py",
        "demo/demo.js",
        "demo/README.md",
        "db/sql/create_database.sql",
        "db/sql/add_indexes.sql",
    }
    assert required_contents.union(browser_assets).issubset(contents)


def test_app_source_archive_contains_setup_inputs_and_runtime_files(
    app_sdist,
    app_wheel,
):
    """The app sdist contains every setup input and packaged runtime file."""
    contents = _source_archive_contents(app_sdist)
    _assert_no_generated_python_artifacts(contents)
    archive_roots = {Path(name).parts[0] for name in contents}
    assert len(archive_roots) == 1
    archive_root = archive_roots.pop()
    setup_source = (APP_ROOT / "setup.py").read_text(encoding="utf-8")
    requirement_files = set(re.findall(r'_read_requirements\("([^"]+)"\)', setup_source))
    with zipfile.ZipFile(app_wheel) as archive:
        packaged_files = {
            name for name in archive.namelist() if ".dist-info/" not in name
        }
    required_files = {"MANIFEST.in", "docs/release-notes.md", "setup.py"}
    expected_contents = {
        f"{archive_root}/{name}"
        for name in required_files.union(requirement_files, packaged_files)
    }
    assert expected_contents.issubset(contents)


def test_app_release_version_is_synchronized(app_wheel):
    """Runtime and distribution metadata use the approved V1 version."""
    metadata = _wheel_metadata(app_wheel).splitlines()

    assert frontend.__version__ == "1.0.0"
    assert "Version: 1.0.0" in metadata


def test_app_wheel_excludes_test_packages(app_wheel):
    """The app release wheel contains no repository test packages."""
    with zipfile.ZipFile(app_wheel) as archive:
        packaged_tests = [
            name for name in archive.namelist() if name.startswith("tests/")
        ]
    assert packaged_tests == []


def test_app_release_artifacts_import_without_source_checkout(
    app_release_artifact,
    tmp_path,
):
    """Each app release artifact installs and imports outside the checkout."""
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
            str(app_release_artifact),
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
from core import app_constants
from core import app_utils
"""
    import_result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(install_root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert import_result.returncode == 0, import_result.stderr or import_result.stdout


def test_recreate_database_creates_configured_parent(tmp_path, monkeypatch):
    """Fresh database initialization creates its configured parent directory."""
    database_path = tmp_path / "new-data-root" / "db" / "app.db"
    monkeypatch.setattr(database, "run_sqlite_cmd", lambda *_args: None)

    database.recreate_database(database_path)

    assert database_path.parent.is_dir()


def test_app_runner_reports_skips_and_owns_the_deterministic_suite():
    """The app runner covers root tests with browser and manual checks selected separately."""
    args = SimpleNamespace(
        py_style=False,
        js_style=False,
        run=None,
    )
    commands = run_tests.collect_commands(args)
    pytest_command = next(command for command, _ in commands if "pytest" in command)

    assert "-ra" in pytest_command
    assert "tests" in pytest_command
    assert run_tests.DEFAULT_IGNORED_TEST_PATHS == ("tests/manual", "tests/browser")
    assert "--ignore=tests/manual" in pytest_command
    assert "--ignore=tests/browser" in pytest_command
    assert all("pagination_baseline" not in argument for argument in pytest_command)
    runner_source = (APP_ROOT / "tests/tools/run_tests.py").read_text(encoding="utf-8")
    assert "--test-csv-dir" not in runner_source
    assert "--test-images-dir" not in runner_source


def test_app_runner_keeps_explicit_manual_selection_available():
    """Explicit app test selections are not filtered by default ignores."""
    args = SimpleNamespace(
        py_style=False,
        js_style=False,
        run=[
            "tests/manual",
            "tests/frontend/test_pagination_baseline.py",
        ],
    )
    commands = run_tests.collect_commands(args)
    pytest_command = next(command for command, _ in commands if "pytest" in command)

    assert "-ra" in pytest_command
    assert "tests/manual" in pytest_command
    assert "tests/frontend/test_pagination_baseline.py" in pytest_command
    assert "--ignore=tests/manual" not in pytest_command
    assert "--ignore=tests/frontend/test_pagination_baseline.py" not in pytest_command


def test_app_runner_owns_browser_matrix():
    """Browser coverage is an explicit three-engine CI job, not a silent skip."""
    commands = run_tests.collect_commands(SimpleNamespace(
        py_style=False, js_style=False, run=None, browser=True))
    assert len(commands) == 1
    command, _ = commands[0]
    assert 'tests/browser' in command
    assert [command[index + 1] for index, value in enumerate(command)
            if value == '--browser'] == ['chromium', 'firefox', 'webkit']
    assert 'retain-on-failure' in command


def test_app_runner_does_not_filter_explicit_tests_target():
    """An explicit request for all app tests is distinct from the default suite."""
    args = SimpleNamespace(
        py_style=False,
        js_style=False,
        run=["tests"],
    )
    commands = run_tests.collect_commands(args)
    pytest_command = next(command for command, _ in commands if "pytest" in command)

    assert pytest_command[-1] == "tests"
    assert not any(argument.startswith("--ignore=") for argument in pytest_command)


def test_app_workflow_uses_supported_matrix_and_owned_commands():
    """CI delegates to repository authorities across the supported V1 matrix."""
    workflow_source = (
        APP_ROOT / ".github/workflows/python-tests.yml"
    ).read_text(encoding="utf-8")

    assert "actions/checkout@v4" in workflow_source
    assert "actions/setup-python@v5" in workflow_source
    assert workflow_source.count("os: [ubuntu-latest, macos-latest]") == 2
    assert "python -m tests.tools.run_tests\n" in workflow_source
    assert "python -m tests.tools.run_tests --browser" in workflow_source
    assert "playwright install --with-deps chromium firefox webkit" in workflow_source
    assert "python -m tests.tools.run_tests --py-style" in workflow_source
    assert "python -m tests.tools.run_tests --js-style" in workflow_source
    assert "python -m pytest" not in workflow_source
    assert "python -m pip install ." in workflow_source
    assert "working-directory: ${{ runner.temp }}" in workflow_source
    assert "python -I -c" in workflow_source
    assert "npm ci --ignore-scripts" in workflow_source
    assert "`sm-data`: optional sample/reference data" in workflow_source
    assert "CI uses the immutable committed fixture" in workflow_source
    assert "private full dataset revision" not in workflow_source
    assert "tests/manual" in workflow_source


def test_javascript_manifest_owns_eslint_dependencies():
    """The private JavaScript manifest declares every configured ESLint package."""
    manifest = json.loads((APP_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    config_source = (APP_ROOT / "frontend" / "eslint.config.js").read_text(encoding="utf-8")
    configured_packages = set(
        re.findall(r"require\(['\"]([^'\"]+)['\"]\)", config_source)
    )
    configured_packages.update({"eslint", "eslint-config-google"})

    assert manifest["private"] is True
    assert set(manifest["devDependencies"]) == configured_packages


def test_javascript_lock_matches_manifest():
    """The npm lock root exactly matches the JavaScript development manifest."""
    manifest = json.loads((APP_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((APP_ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8"))

    assert lock["lockfileVersion"] == 3
    assert lock["packages"][""]["devDependencies"] == manifest["devDependencies"]
    assert lock["packages"][""]["engines"] == manifest["engines"]
    assert (
        manifest["engines"]["node"]
        == lock["packages"]["node_modules/eslint"]["engines"]["node"]
    )


def test_javascript_manifests_are_publishable_files():
    """The repository ignore rules retain both JavaScript authority files."""
    ignored_patterns = {
        line.strip()
        for line in (APP_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "package.json" not in ignored_patterns
    assert "package-lock.json" not in ignored_patterns


def test_javascript_lint_configuration_owns_first_party_boundaries():
    """One config owns the first-party scope, style, and API naming boundary."""
    config_source = (APP_ROOT / "frontend" / "eslint.config.js").read_text(encoding="utf-8")

    assert "@eslint/js" not in config_source
    assert "globals.es6" not in config_source
    assert re.search(r"^\s*\$:", config_source, flags=re.MULTILINE) is None
    assert re.search(r"^\s*jQuery:", config_source, flags=re.MULTILINE) is None
    assert "code: 100" in config_source
    assert "ignoreUrls: true" in config_source
    assert "ignoreStrings: true" in config_source
    assert "ignoreRegExpLiterals: true" in config_source
    assert "ignoreTemplateLiterals: true" in config_source
    assert "frontend/static/js/modules/shark_matcher/**/*.js" in config_source
    assert "frontend/static/js/modules/shark_matcher/vendor/**" in config_source
    assert "camelcase: 'off'" in config_source


def test_javascript_style_runner_uses_complete_scope_and_propagates_failure(
    monkeypatch,
):
    """One ESLint invocation owns all first-party files and propagates failure."""
    checked_commands = []

    def fake_run_command(command):
        checked_commands.append(command)
        return False

    monkeypatch.setattr(
        javascript_style_tests,
        "run_command",
        fake_run_command,
    )

    assert javascript_style_tests.main() == 1
    assert checked_commands == [[
        "npm", "exec", "--prefix", "frontend", "--no", "--",
        "eslint", "--config", "frontend/eslint.config.js",
        "frontend/eslint.config.js",
        "frontend/static/js",
        "demo",
    ]]


def test_webui_utility_cleanup_has_no_dead_parallel_interfaces():
    """Utility exports and callers retain only their live responsibilities."""
    webui_root = APP_ROOT / "frontend/static/js/modules/webui"
    util_source = (webui_root / "util.js").read_text(encoding="utf-8")
    individual_source = (webui_root / "individual.js").read_text(encoding="utf-8")
    home_source = (webui_root / "home.js").read_text(encoding="utf-8")

    assert "function listToLinks(inputList)" in util_source
    assert "descriptorString" not in util_source
    assert "function deepCopy(" not in util_source
    assert "deepCopy," not in util_source
    assert "const [path] = Routes.parsePath(window.location.hash)" in util_source
    assert individual_source.count("Util.listToLinks(") == 3
    assert home_source.count("Util.listToLinks(") == 3
    single_argument_calls = r"Util\.listToLinks\([^,()]+\)"
    assert len(re.findall(single_argument_calls, individual_source)) == 3
    assert len(re.findall(single_argument_calls, home_source)) == 3


def test_disabled_review_route_uses_one_guarded_handler():
    """The dormant review UI has one route handler and safe trigger boundary."""
    review_source = (
        APP_ROOT / "frontend/static/js/modules/webui/review.js"
    ).read_text(encoding="utf-8")

    assert "const REVIEW_ENABLED = false;" in review_source
    assert "Routes.addRoute(/^review$/, handlerReview, 'review');" in review_source
    assert "handlerReviewNotFound" not in review_source
    assert review_source.count("if (!REVIEW_ENABLED)") == 2
    assert "The review page is not currently available." in review_source


def test_merge_upload_cleanup_has_one_way_reuse_and_one_scale_path():
    """Merge and upload retain one directional reuse and one scale path."""
    webui_root = APP_ROOT / "frontend/static/js/modules/webui"
    merge_source = (webui_root / "merge.js").read_text(encoding="utf-8")
    upload_source = (webui_root / "upload.js").read_text(encoding="utf-8")

    assert "import * as Upload from './upload.js';" in merge_source
    assert "Upload.renderUploadResult(" in merge_source
    assert "import * as Merge from './merge.js';" not in upload_source
    assert "import * as Log from './log.js';" not in upload_source

    upload_exports = re.search(r"export\s*\{(?P<body>.*?)\};\s*$", upload_source, re.S)
    assert upload_exports is not None
    assert "previewExecuteIndividualSave" not in upload_exports.group("body")
    assert "function previewExecuteIndividualSave(" in upload_source

    scale_expression = "const scaleFactor = Math.min("
    assert upload_source.count(scale_expression) == 1
    assert "if (imageWidth > canvasWidth || imageHeight > canvasHeight)" not in upload_source


def test_individual_route_renderer_remains_private_and_live():
    """The route renderer remains locally used without a dead public export."""
    individual_source = (
        APP_ROOT / "frontend/static/js/modules/webui/individual.js"
    ).read_text(encoding="utf-8")

    individual_exports = re.search(
        r"export\s*\{(?P<body>.*?)\};\s*$", individual_source, re.S
    )
    assert individual_exports is not None
    assert not re.search(
        r"^\s*renderIndividual,?\s*$", individual_exports.group("body"), re.M
    )
    assert (
        "executeIndividualSearch(path, params, context, renderIndividual);"
        in individual_source
    )


def test_home_media_mutations_share_navigation_and_private_interfaces():
    """Home media mutations reuse routing and expose only live interfaces."""
    home_source = (
        APP_ROOT / "frontend/static/js/modules/webui/home.js"
    ).read_text(encoding="utf-8")

    assert "function navigateToFetchedIndividual(fetchIndividual)" in home_source
    assert home_source.count("navigateToFetchedIndividual(fetchIndividual);") == 3
    assert "associatedIndividual.id === individual.id" in home_source
    assert "associated_individuals[i].id === media.id" not in home_source
    assert "fetchIndividual === currentPageName" not in home_source

    private_signatures = {
        "function executeDatasetSearch(context)",
        "function executeUpdateEmbeddings()",
        "function executeMediaSave(medias, kwargs = {})",
        "function executeMediaDelete(media)",
        "function executeUnmatch(media)",
        "function executeMediaReassign(media)",
        "function renderPopup(individuals, kwargs = {})",
    }
    assert all(signature in home_source for signature in private_signatures)
    assert "kwargs['page'] === 'merge' || kwargs['page'] === 'upload'" in home_source
    assert "kwargs['page'] === 'merge' | kwargs['page'] === 'upload'" not in home_source
    assert "// import * as Log from './log.js';" not in home_source

    home_exports = re.search(
        r"export\s*\{(?P<body>.*?)\};\s*$", home_source, re.S
    )
    assert home_exports is not None
    for function_name in (
        "executeObjectDelete",
        "renderMediaProfile",
        "selectImage",
    ):
        assert f"function {function_name}(" in home_source
        assert not re.search(
            rf"^\s*{function_name},?\s*$", home_exports.group("body"), re.M
        )


def test_browser_client_uses_one_transport_and_live_core_facade():
    """Endpoint modules delegate to one transport and the facade stays live."""
    client_root = APP_ROOT / "frontend/static/js/modules/shark_matcher"
    client_sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in client_root.glob("*.js")
    }
    core_source = client_sources["core.js"]
    base_source = client_sources["base.js"]

    transport_definitions = [
        name
        for name, source in client_sources.items()
        if "function sendRequest(" in source
    ]
    assert transport_definitions == ["core.js"]

    endpoint_modules = {
        "action.js",
        "base.js",
        "individual.js",
        "match.js",
        "media.js",
        "proposal.js",
        "review.js",
        "user.js",
    }
    for module_name in endpoint_modules:
        assert "import * as Core from './core.js';" in client_sources[module_name]
        assert "Core.sendRequest({" in client_sources[module_name]

    webui_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (APP_ROOT / "frontend/static/js/modules/webui").glob("*.js")
    )
    facade_aliases = (
        "getCached",
        "getCachedValue",
        "clearCache",
        "clearAllLocal",
        "getCredentials",
        "hasCredentials",
        "setCredentials",
        "getFilters",
        "setFilters",
        "clearFilters",
        "getTheme",
        "setTheme",
    )
    for function_name in facade_aliases:
        assert f"const {function_name} = Core.{function_name};" in base_source
        assert f"SharkMatcher.{function_name}(" in webui_source

    for namespace in (
        "Action",
        "Match",
        "Media",
        "Individual",
        "User",
        "Proposal",
        "Review",
    ):
        assert f"SharkMatcher.{namespace}." in webui_source
    for function_name in (
        "export_data",
        "get_dataset_stats",
        "get_embeddings_status",
        "update_embeddings",
    ):
        assert f"SharkMatcher.{function_name}(" in webui_source

    assert "const storageKey = CACHE_PREFIX + dataScope() + key;" in core_source
    assert "if (entry && typeof entry.time === 'number')" in core_source
    assert core_source.count("localStorage.removeItem(storageKey);") == 1
    assert "// const API_VESION" not in core_source
    assert "// let url = `/api/${API_VESION}" not in core_source
    assert "eslint-disable" not in base_source


def test_action_proposal_review_clients_expose_reviewed_operations():
    """Related endpoint clients retain only live or intentionally dormant APIs."""
    client_root = APP_ROOT / "frontend/static/js/modules/shark_matcher"
    client_sources = {
        "action": (client_root / "action.js").read_text(encoding="utf-8"),
        "proposal": (client_root / "proposal.js").read_text(encoding="utf-8"),
        "review": (client_root / "review.js").read_text(encoding="utf-8"),
    }
    expected_exports = {
        "action": {"filter_actions"},
        "proposal": {
            "filter_proposals",
            "post_proposal",
            "delete_proposal",
        },
        "review": {"filter_reviews"},
    }
    for client_name, expected_names in expected_exports.items():
        source = client_sources[client_name]
        assert _javascript_exports(source) == expected_names
        assert "Core.sendRequest({" in source
        assert "eslint-disable" not in source

    action_source = client_sources["action"]
    proposal_source = client_sources["proposal"]
    server_source = (APP_ROOT / "frontend/server.py").read_text(encoding="utf-8")
    assert "function get_actions(" not in action_source
    assert "function delete_action(" not in action_source
    assert "endpoint: 'action/get'" not in action_source
    assert "endpoint: 'action/delete'" not in action_source
    assert "function get_proposals(" not in proposal_source
    assert "endpoint: 'proposal/get'" not in proposal_source
    assert "// Not implemented." not in proposal_source
    assert "build_api_route(r'proposal/get*'" in server_source


def test_match_media_user_clients_expose_reviewed_operations():
    """Related endpoint clients retain only browser-used operations."""
    client_root = APP_ROOT / "frontend/static/js/modules/shark_matcher"
    client_sources = {
        "match": (client_root / "match.js").read_text(encoding="utf-8"),
        "media": (client_root / "media.js").read_text(encoding="utf-8"),
        "user": (client_root / "user.js").read_text(encoding="utf-8"),
    }
    expected_exports = {
        "match": {"filter_matches"},
        "media": {"get_medias", "post_media", "delete_media", "upload_batch"},
        "user": {
            "filter_users",
            "post_user",
            "delete_user",
            "login_user",
            "logout_user",
        },
    }
    for client_name, expected_names in expected_exports.items():
        source = client_sources[client_name]
        assert _javascript_exports(source) == expected_names
        assert "Core.sendRequest({" in source
        assert "eslint-disable" not in source

    match_source = client_sources["match"]
    media_source = client_sources["media"]
    user_source = client_sources["user"]
    home_source = (
        APP_ROOT / "frontend/static/js/modules/webui/home.js"
    ).read_text(encoding="utf-8")
    server_source = (APP_ROOT / "frontend/server.py").read_text(encoding="utf-8")

    assert "function get_matches(" not in match_source
    assert "endpoint: 'match/get'" not in match_source
    assert "function get_medias(ids, getFull)" in media_source
    assert "'keyword': keyword" not in media_source
    assert "function filter_medias(" not in media_source
    assert "endpoint: 'media/filter'" not in media_source
    assert "function save_media(" not in media_source
    assert "endpoint: 'media/save'" not in media_source
    assert "SharkMatcher.Media.get_medias(params.id, true)" in home_source
    assert "function get_user(" not in user_source
    assert "endpoint: 'user/get'" not in user_source
    assert "function authenticate_user(" not in user_source
    assert "endpoint: 'user/auth'" not in user_source

    retained_server_routes = {
        "match/get*",
        "media/filter*",
        "media/save*",
        "user/get",
        "user/auth",
    }
    for route in retained_server_routes:
        assert f"build_api_route(r'{route}'" in server_source


def test_individual_client_exposes_reviewed_operations():
    """The individual endpoint client retains only browser-used operations."""
    client_source = (
        APP_ROOT / "frontend/static/js/modules/shark_matcher/individual.js"
    ).read_text(encoding="utf-8")
    expected_exports = {
        "get_individuals",
        "filter_individuals",
        "post_individual",
        "merge_individuals",
        "unmatch_individual",
        "update_reference",
        "post_object",
        "delete_object",
    }
    server_source = (APP_ROOT / "frontend/server.py").read_text(encoding="utf-8")

    assert _javascript_exports(client_source) == expected_exports
    assert "Core.sendRequest({" in client_source
    assert "eslint-disable" not in client_source
    assert client_source.startswith(
        "// Endpoint functions for individual and associated objects."
    )
    assert "individual adjacent objects" not in client_source
    assert "Get and filter object functions may be needed too." not in client_source
    assert "function delete_individual(" not in client_source
    assert "endpoint: 'individual/delete'" not in client_source
    assert "build_api_route(r'individual/delete*'" in server_source


def test_browser_hash_callers_are_ready_for_native_async_digest():
    """Each current hash caller can consume the asynchronous native digest."""
    webui_root = APP_ROOT / "frontend/static/js/modules/webui"
    login_source = (webui_root / "login.js").read_text(encoding="utf-8")
    settings_source = (webui_root / "settings.js").read_text(encoding="utf-8")
    upload_source = (webui_root / "upload.js").read_text(encoding="utf-8")

    assert "async function executeLogin(context)" in login_source
    assert "await Util.sha256(cleartext)" in login_source
    assert "async function deriveSettingsPassword(" in settings_source
    assert "async function executeUserSave()" in settings_source
    assert "async function executeCreateUser()" in settings_source
    assert settings_source.count("await deriveSettingsPassword(") == 2
    assert "image.onload = async function" in upload_source
    assert re.search(
        r"await\s+Util\.sha256\(\s*previewIndividuals\[i\]\.image\.src\s*\)",
        upload_source,
    )
    assert "reject(error)" in upload_source
    assert "Error processing uploaded images:" in upload_source


def test_browser_hashing_uses_only_native_web_crypto():
    """The shared browser hash uses Web Crypto without an SJCL fallback."""
    util_source = (
        APP_ROOT / "frontend/static/js/modules/webui/util.js"
    ).read_text(encoding="utf-8")
    vendor_path = (
        APP_ROOT
        / "frontend/static/js/modules/shark_matcher/vendor/sjcl.min.js"
    )

    assert "import sjcl" not in util_source
    assert "sjcl." not in util_source
    assert "isSecureContext" in util_source
    assert re.search(r"crypto\.subtle\.digest\(\s*'SHA-256'", util_source)
    assert util_source.count("new TextEncoder()") == 1
    assert "function textToBytes(text)" in util_source
    assert not vendor_path.exists()


def test_browser_hashing_supports_insecure_http_fallback():
    """Browser hashing falls back to a portable pure JS SHA-256 implementation on HTTP."""
    util_source = (
        APP_ROOT / "frontend/static/js/modules/webui/util.js"
    ).read_text(encoding="utf-8")

    assert "function sha256Fallback(text)" in util_source
    assert "return sha256Fallback(text);" in util_source

    if not shutil.which("node"):
        return

    test_inputs = [
        "",
        "password",
        "browser-password",
        "test@example.test",
        "shark \U0001f988 identifier",
        "A" * 500,
    ]
    node_script = f"""
    const window = {{ isSecureContext: false }};
    const utilSource = {json.dumps(util_source)};
    const textToBytesMatch = utilSource.match(/function textToBytes[\\s\\S]*?\\n\\}}/);
    const fallbackMatch = utilSource.match(/function sha256Fallback[\\s\\S]*?\\n\\}}/);
    const sha256Match = utilSource.match(/async function sha256[\\s\\S]*?\\n\\}}/);
    const code = textToBytesMatch[0] + "\\n" + fallbackMatch[0] + "\\n" + sha256Match[0];
    const run = new Function("window", code + "; return sha256;");
    const sha256Fn = run(window);

    (async () => {{
      const inputs = {json.dumps(test_inputs)};
      const outputs = await Promise.all(inputs.map(sha256Fn));
      console.log(JSON.stringify(outputs));
    }})();
    """
    result = subprocess.run(
        ["node", "-e", node_script],
        capture_output=True,
        text=True,
        check=True,
    )
    digests = json.loads(result.stdout)
    for cleartext, digest in zip(test_inputs, digests):
        expected = hashlib.sha256(cleartext.encode("utf-8")).hexdigest()
        assert digest == expected


def test_app_wheel_declares_only_packaged_modules(app_wheel):
    """Every top-level name declared by the app wheel is packaged."""
    with zipfile.ZipFile(app_wheel) as archive:
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
    # Both wheels may share one environment, so the app must not ship the matcher's top-level names.
    assert set(declared_modules).isdisjoint({"constants", "path_utils", "src", "utils"})


def test_static_references_resolve_to_files():
    """Every /static/ path the browser assets request exists in the checkout."""
    static_root = APP_ROOT / "frontend" / "static"
    reference = re.compile(r"""(?:src|href)=["'](/static/[^"'?#]+)|url\(["']?(/static/[^)"'?#]+)""")
    missing = []
    for path in static_root.rglob("*"):
        if path.suffix not in {".html", ".css", ".js"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in reference.finditer(text):
            target = match.group(1) or match.group(2)
            if not (APP_ROOT / "frontend" / target.lstrip("/")).is_file():
                missing.append((path.relative_to(APP_ROOT).as_posix(), target))
    assert missing == []


def test_app_ignore_rules_cover_publication_artifacts():
    """App ignore rules exclude generated and sensitive publication paths."""
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
        "dist/shark_matcher_app.whl",
        "htmlcov/index.html",
        "node_modules/eslint/index.js",
        "package.egg",
        "package.whl",
        "run.log",
        "tests/__pycache__/module.pyc",
        "token.json",
    }
    retained_paths = {
        "containerization/.env.example",
        "containerization/Dockerfile",
        "containerization/README.md",
        "containerization/docker-compose.yml",
        "frontend/package-lock.json",
        "frontend/package.json",
        "frontend/static/images/shark-loading.png",
        "tests/ci-test-data/example.csv",
    }
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=APP_ROOT,
        input="\n".join(sorted(ignored_paths | retained_paths)) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == ignored_paths


def test_app_tracked_files_exclude_publication_artifacts():
    """App tracked files contain only reviewed publication exceptions."""
    tracked_paths = _git_tracked_paths(APP_ROOT)

    allowed_paths = {
        "containerization/.env.example",
        "tests/ci-test-data/database-format/database-format-test/tokens.csv",
    }
    allowed_data_roots = ("tests/ci-test-data/",)
    forbidden_directories = {
        ".eggs",
        ".pytest_cache",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
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
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored_result.returncode == 0, ignored_result.stderr
    assert ignored_result.stdout.splitlines() == [
        "tests/ci-test-data/database-format/database-format-test/tokens.csv"
    ]
    assert not violations


def test_app_tracked_files_are_path_portable():
    """Tracked app content contains no developer-specific rooted paths."""
    payloads = {
        tracked_path: (APP_ROOT / tracked_path).read_bytes()
        for tracked_path in _git_tracked_paths(APP_ROOT)
    }

    assert not _path_portability_violations(
        payloads,
        exact_roots=(Path.home(), APP_ROOT),
    )


def test_app_documentation_links_resolve_inside_the_repository():
    """Current documentation links resolve to existing repository files and headings."""
    documents = _markdown_documents(APP_ROOT, include_internal=False)

    assert not _documentation_link_violations(APP_ROOT, documents)


def test_app_documentation_names_repository_files_by_path():
    """Current documentation names Shark Matcher files by path, because branch links drift from the paired checkout."""
    documents = _markdown_documents(APP_ROOT, include_internal=False)

    assert [path for path, text in documents.items() if _REPOSITORY_BLOB_LINK.search(text)] == []


def test_app_documentation_follows_markdown_conventions():
    """Current documentation uses Title Case headings, tagged fences, UPPER_CASE placeholders, no rules, and short paragraphs."""
    assert not _markdown_convention_violations(
        _markdown_documents(APP_ROOT, include_internal=False))


def test_app_documentation_contains_no_fixed_service_key():
    """Documented service-key exports generate the key or name a placeholder."""
    assert not _fixed_service_key_documents(_markdown_documents(APP_ROOT))


def test_app_documented_environment_names_are_known():
    """Every SHARK_* name in current documentation is app source or a matcher setting."""
    documented = set()
    for text in _markdown_documents(APP_ROOT, include_internal=False).values():
        documented.update(_ENVIRONMENT_NAME.findall(text))

    unknown = documented - _source_environment_names(APP_ROOT) - _MATCHER_ENVIRONMENT_NAMES
    assert unknown == set()


def test_app_documented_addresses_match_configured_ports():
    """Documented app URLs use DEFAULT_PORT and matcher URLs use the inference service default port."""
    constants_source = (APP_ROOT / "core" / "app_constants.py").read_text(encoding="utf-8")
    inference_default = re.search(
        r'os\.getenv\("INFERENCE_SERVICE_URL", "([^"]+)"\)', constants_source).group(1)
    inference_port = int(inference_default.rsplit(":", 1)[1])
    violations = []
    for path, text in _markdown_documents(APP_ROOT, include_internal=False).items():
        prose = _prose(text)
        for url, port, route in re.findall(r"(https?://[\w.\-]+:(\d+))(/[\w/]*)?", prose):
            matcher_url = route.startswith(_MATCHER_ROUTES) or url == inference_default
            expected = inference_port if matcher_url else app_constants.DEFAULT_PORT
            if int(port) != expected:
                violations.append(f"{path}: {url}{route}")
        violations.extend(
            f"{path}: port {port}"
            for port in re.findall(r"\bport\b[^.`]*`(\d+)`", prose)
            if int(port) != app_constants.DEFAULT_PORT
        )

    assert not violations


def test_app_release_artifacts_are_path_portable(
    app_release_artifact,
    app_artifacts,
):
    """Built app artifacts contain no checkout or build-machine paths."""
    payloads = _release_artifact_payloads(app_release_artifact)

    assert not _path_portability_violations(
        payloads,
        exact_roots=(Path.home(), APP_ROOT, app_artifacts.source),
    )


def test_committed_ci_fixture_is_marked_and_immutable():
    """The documented CI fixture retains its exact committed data content."""
    fixture_root = APP_ROOT / "tests" / "ci-test-data"
    policy = (fixture_root / "README.md").read_text(encoding="utf-8").lower()
    assert all(statement in policy for statement in ("immutable", "must never write"))
    fixture_files = sorted(
        (*fixture_root.rglob("*.csv"), *fixture_root.rglob("*.tsv"))
    )
    assert fixture_files
    digest = hashlib.sha256()
    for path in fixture_files:
        relative_path = path.relative_to(fixture_root).as_posix().encode("utf-8")
        digest.update(relative_path + b"\0" + path.read_bytes() + b"\0")
    expected = "55af1d750fb79749948c9bcee69f99f27ee5dc1aa3aec5d1426c550b55c8335e"
    assert digest.hexdigest() == expected


def _wheel_metadata(wheel_path):
    """Return the decoded core metadata from a built wheel."""
    with zipfile.ZipFile(wheel_path) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_files) == 1
        return archive.read(metadata_files[0]).decode("utf-8")


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


def _source_archive_contents(source_archive):
    """Return the normalized member names from a gzipped source archive."""
    return set(_release_artifact_payloads(source_archive))


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


def _assert_no_generated_python_artifacts(contents):
    """Require an artifact file set to exclude generated Python caches."""
    generated = {name for name in contents if "__pycache__" in Path(name).parts
                 or name.endswith((".pyc", ".pyo"))}
    assert generated == set()


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


def _javascript_exports(source):
    """Return the names from a JavaScript module's terminal export block."""
    export_match = re.search(r"export\s*\{(?P<body>.*?)\};\s*$", source, re.S)
    assert export_match is not None
    return set(re.findall(r"^\s*(\w+),?\s*$", export_match.group("body"), re.M))


_PICTOGRAPHIC_SYMBOL = re.compile(
    "[\u2190-\u21ff\u2300-\u23ff\u25a0-\u27bf\u2900-\u297f\u2b00-\u2bff\U0001f000-\U0001faff\ufe0f]")


def test_app_tracked_text_contains_no_pictographic_symbols():
    """Documentation, comments, and user-facing text use words or plain ASCII, never emojis, arrows, or icons."""
    violations = []
    for tracked_path in _git_tracked_paths(APP_ROOT):
        if tracked_path.endswith((".md", ".py", ".sh", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".js", ".html", ".css")):
            text = (APP_ROOT / tracked_path).read_text(encoding="utf-8")
            violations.extend(
                f"{tracked_path}:{number}"
                for number, line in enumerate(text.splitlines(), 1)
                if _PICTOGRAPHIC_SYMBOL.search(line))

    assert violations == []


def test_requirement_files_pin_every_dependency():
    """Every dependency line names an exact version so a fresh install matches the tested one."""
    unpinned = [
        f"{path.name}: {line.strip()}"
        for path in sorted((APP_ROOT / "requirements").glob("requirements*.txt"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("#", "-"))
        and "==" not in line]

    assert unpinned == []


def test_generated_element_ids_contain_no_spaces():
    """An id with a space is invalid HTML and cannot be selected with #id."""
    pattern = re.compile(r'\b(?:id|for)="([^"]*)"')
    offenders = [
        f"{path.name}: {value}"
        for path in sorted(Path(app_constants.THIS_DIR, "frontend", "static").rglob("*"))
        if path.suffix in (".js", ".html") and "node_modules" not in path.parts
        for value in pattern.findall(path.read_text(encoding="utf-8"))
        if " " in re.sub(r"\$\{[^}]*\}", "", value)
    ]
    assert offenders == []
