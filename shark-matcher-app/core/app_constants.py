"""Shared configuration constants for the Shark Matcher application."""

import os

SUCCESS = 0
ERROR = -1

# --- Frontend / configuration scalars ---
API_PREFIX = '/api'
ENCODING = 'utf-8'
EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"
DEFAULT_PORT = 12345
# An empty host listens on every IPv4 interface.
DEFAULT_HOST = ''
# HTTP and image limits are configurable without changing browser workflows.
MAX_JSON_BYTES = int(os.getenv('SHARK_MAX_JSON_BYTES', str(4 * 1024 * 1024)))
MAX_UPLOAD_BYTES = int(os.getenv('SHARK_MAX_UPLOAD_BYTES', str(256 * 1024 * 1024)))
MAX_IMAGE_BYTES = int(os.getenv('SHARK_MAX_IMAGE_BYTES', str(25 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv('SHARK_MAX_IMAGE_PIXELS', '50000000'))
MAX_UPLOAD_ITEMS = int(os.getenv('SHARK_MAX_UPLOAD_ITEMS', '100'))
MAX_PAGE_SIZE = int(os.getenv('SHARK_MAX_PAGE_SIZE', '1000'))
LISTEN_BACKLOG = int(os.getenv('SHARK_LISTEN_BACKLOG', '128'))
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8000")
INFERENCE_API_KEY_ENV = "SHARK_INFERENCE_API_KEY"
INFERENCE_API_KEY = os.getenv(INFERENCE_API_KEY_ENV, "")

# Maps user-facing retrieval method names to the integer IDs expected by the inference service.
MATCH_METHOD_DICT = {
    "nearest neighbor": 0,
    "nearest centroid": 2,
}

# --- Paths ---
THIS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
BASE_DIR = os.path.join(THIS_DIR, "..")

TEST_DIR = os.path.join(THIS_DIR, "tests")


def _resolve_base_data_dir():
    """Return SHARK_DATA_ROOT when set (it must be absolute), otherwise the sibling sm-data directory."""
    configured_root = os.getenv("SHARK_DATA_ROOT")
    if not configured_root:
        return os.path.join(BASE_DIR, "sm-data")
    expanded_root = os.path.expanduser(configured_root)
    if not os.path.isabs(expanded_root):
        raise ValueError(f"SHARK_DATA_ROOT must be an absolute path, got '{configured_root}'.")
    return os.path.abspath(expanded_root)


BASE_DATA_DIR = _resolve_base_data_dir()
resolve_base_data_dir = _resolve_base_data_dir

DB_DIR = os.path.join(BASE_DATA_DIR, "db")
DB_SQL_DIR = os.path.join(THIS_DIR, "db", "sql")
if os.getenv("SHARK_TEST_MODE") == "true":
    print("WARNING: USING TEST DATABASE")
    DB_PATH = os.path.join(DB_DIR, "test_sharkMatcher.db")
else:
    DB_PATH = os.path.join(DB_DIR, "sharkMatcher.db")

OUT_DATA_DIR = os.path.join(BASE_DATA_DIR, "data-out")
ANNOTATIONS_DIR = os.path.join(BASE_DATA_DIR, "annotations")
DB_DATA_DIR = os.path.join(BASE_DATA_DIR, "database-format")
IMAGES_DIR = os.path.join(BASE_DATA_DIR, "images", "images")

TRAIN_ANNOTATIONS = "train-annotations.tsv"
VALID_ANNOTATIONS = "valid-annotations.tsv"
FULL_ANNOTATIONS = "full-annotations.tsv"

# Committed test fixture used by the application test suite.
CI_TEST_DATA_DIR = os.path.join(TEST_DIR, "ci-test-data")
CI_TEST_DATABASE_FORMAT_DIR = os.path.join(CI_TEST_DATA_DIR, "database-format", "database-format-test")

# External module paths.
SHARK_APP_DIR = THIS_DIR


def _resolve_inference_snapshot_root():
    """Return the root folder where inference snapshots are stored."""
    configured_root = os.getenv("SHARK_INFERENCE_SNAPSHOT_ROOT")
    if configured_root:
        return os.path.abspath(os.path.expanduser(configured_root))
    return os.path.abspath(os.path.join(BASE_DATA_DIR, "inference-snapshots"))


INFERENCE_SNAPSHOT_ROOT = _resolve_inference_snapshot_root()
resolve_inference_snapshot_root = _resolve_inference_snapshot_root
