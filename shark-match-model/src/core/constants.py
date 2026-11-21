"""Project-wide constants for shark-match-model: scalars, paths, and toggles."""
import os

# --- 4b. Scalar / primitive constants ---
SUCCESS = 0
ERROR = -1

# Seed applied by utils.seed_everything() and by the deterministic DataLoader.
RANDOM_SEED = 42
MINI_BATCH_SIZE = 16
# Default evaluator metric keys for best-checkpoint saving and early stopping patience.
CHECKPOINT_METRIC = "average_valid_hit_at_10"
EARLY_STOPPING_METRIC = "average_valid_hit_at_10"

# --- 4c. Configuration scalars from the environment ---
# Inference service authentication.
INFERENCE_API_KEY_ENV = "SHARK_INFERENCE_API_KEY"
INFERENCE_API_KEY = os.getenv(INFERENCE_API_KEY_ENV, "")

# --- 4d. Domain enumerations, ordered sequences, and lookup maps ---
# Media quality levels, ordered from lowest to highest.
QUALITY_ORDER = ['unmatchable', 'poor', 'medium', 'good']

# Pre-canned (non-fine-tuned) model identifiers.
PRECANNED_MODELS = {"miew", "mega", "vit"}
# Hub commit of conservationxlabs/miewid-msv2; its remote code runs on load, so it must not float.
MIEWID_REVISION = "74201a38f3167b09013c20113dee11230fd5149d"

# Default model configurations used when SHARK_MODELS is not set; they match the demo bundle.
DEFAULT_MODELS = [
    {"display_name": "MiewID Multimodal", "dir": "pre2019-sample-data",
     "model": "omega_miew_arcface_random",
     "retrieval_mode": "multimodal", "rerank": "kreciprocal"},
    # {"display_name": "MiewID Image", "dir": "pre2019-sample-data",
    #  "model": "omega_miew_arcface_random",
    #  "retrieval_mode": "baseline", "rerank": "none"},
]

# --- 4e. Private initializer functions + the constants they produce ---
def _env_toggle(name, default=False):
    """Read a boolean toggle from the environment."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Seeds each DataLoader with an explicit torch.Generator from RANDOM_SEED (env: SHARK_DETERMINISTIC_DATALOADER).
DETERMINISTIC_DATALOADER = _env_toggle("SHARK_DETERMINISTIC_DATALOADER")
# Pins deterministic algorithm selection and cuDNN state in addition to RNG state (env: SHARK_STRICT_DETERMINISM).
STRICT_DETERMINISM = _env_toggle("SHARK_STRICT_DETERMINISM")
# Reseeds RNG at the start of training so augmentation is independent of model construction draws (env: SHARK_RESEED_BEFORE_TRAIN).
RESEED_BEFORE_TRAIN = _env_toggle("SHARK_RESEED_BEFORE_TRAIN")


def _resolve_inference_bind_address():
    """Return the (host, port) for `python3 -m src.api.server` from SHARK_INFERENCE_HOST and SHARK_INFERENCE_PORT."""
    return (os.getenv("SHARK_INFERENCE_HOST", "0.0.0.0"),
            int(os.getenv("SHARK_INFERENCE_PORT", "8000")))


resolve_inference_bind_address = _resolve_inference_bind_address

# --- 4f. Path constants ---
THIS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
BASE_DIR = os.path.join(THIS_DIR, "..")
BACKEND_DIR = os.path.join(THIS_DIR, "src")


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
DATA_PROC_DIR = os.path.join(BASE_DATA_DIR, "data-proc")

ANNOTATIONS_DIR = os.path.join(BASE_DATA_DIR, "annotations")
DB_DATA_DIR = os.path.join(BASE_DATA_DIR, "database-format")
EMBEDDINGS_DIR = os.path.join(BASE_DATA_DIR, "embeddings")
TRAINED_MODELS_DIR = os.path.join(BASE_DATA_DIR, "trained-models")
BASE_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "images")
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw-data")

OUT_DATA_DIR = os.path.join(BASE_DATA_DIR, "data-out")
THESIS_RESULTS_DIR = os.path.join(OUT_DATA_DIR, "thesis-results")
VISUALIZATIONS_DIR = os.path.join(OUT_DATA_DIR, "visualizations")
PAIRX_OUTPUT_DIR = os.path.join(OUT_DATA_DIR, "pairx-output")
DATA_COMPARISONS_DIR = os.path.join(OUT_DATA_DIR, "data-comparisons")
DATA_ANALYSIS_DIR = os.path.join(OUT_DATA_DIR, "data-analysis")

IMAGES_DIR = os.path.join(BASE_IMAGES_DIR, "images")

# --- Maintainer test data ---
TEST_DATA_DIR = os.path.join(BASE_DATA_DIR, "test-data")
GOLDEN_PATH = os.path.join(TEST_DATA_DIR, "model-tests", "golden_states")
# Standard dataset annotation filenames within versioned subdirectories.
TRAIN_ANNOTATIONS = "train-annotations.tsv"
VALID_ANNOTATIONS = "valid-annotations.tsv"
FULL_ANNOTATIONS = "full-annotations.tsv"
# Sidecar file recording the source CSV directory used to generate annotations.
SOURCE_CSV_DIR_SIDECAR = ".source_csv_dir"

PSL_DIR = os.path.join(BASE_DATA_DIR, "psl-rules")


def _resolve_pairx_dir():
    """Return PAIRX_ROOT as an absolute path, otherwise the sibling pairx checkout."""
    configured_root = os.getenv("PAIRX_ROOT") or os.path.join(BASE_DIR, "pairx")
    return os.path.abspath(os.path.expanduser(configured_root))


PAIRX_DIR = _resolve_pairx_dir()
resolve_pairx_dir = _resolve_pairx_dir


def _resolve_inference_snapshot_root():
    """Return the root folder where inference snapshots are stored."""
    configured_root = os.getenv("SHARK_INFERENCE_SNAPSHOT_ROOT")
    if configured_root:
        return os.path.abspath(os.path.expanduser(configured_root))
    return os.path.abspath(os.path.join(BASE_DATA_DIR, "inference-snapshots"))


INFERENCE_SNAPSHOT_ROOT = _resolve_inference_snapshot_root()
resolve_inference_snapshot_root = _resolve_inference_snapshot_root
