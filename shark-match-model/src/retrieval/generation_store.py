"""Create, resolve, prune, and roll back versioned embedding folders."""

from contextlib import contextmanager
from dataclasses import dataclass
import datetime
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import stat
import time
import uuid

from pydantic import ValidationError
import torch

from src.contracts import (
    ArtifactFileV1,
    EMBEDDINGS_METADATA_FILENAME,
    EmbeddingsMetadataV1,
    GENERATION_SCHEMA_VERSION,
    GenerationManifestV1,
    GenerationPointerV1,
)
from src.core import constants, utils

_CURRENT_POINTER_FILENAME = "current.json"
_GENERATION_LOCK_FILENAME = ".generation.lock"
_GENERATIONS_DIRECTORY_NAME = "generations"
_MANIFEST_FILENAME = "manifest.json"
STAGING_PREFIX = ".staging-"


class GenerationValidationError(ValueError):
    """Raised when an embedding generation folder or manifest is invalid."""


@dataclass(frozen=True)
class GenerationArtifactLayout:
    """Filenames used by a given retrieval mode."""

    retrieval_mode: str
    image_embeddings: str
    image_neighbors: str
    centroid_embeddings: str
    centroid_neighbors: str
    centroid_labels: str

    @property
    def required_files(self):
        """Return the files required for this retrieval mode."""
        required = {
            "stored_dataset.pt",
            self.image_embeddings,
            self.centroid_embeddings,
            self.centroid_labels,
            EMBEDDINGS_METADATA_FILENAME,
        }
        if self.retrieval_mode in {"psl", "stacked"}:
            required.add(self.image_neighbors)
        return frozenset(required)


@dataclass(frozen=True)
class ResolvedGeneration:
    """A resolved versioned generation folder or legacy flat embeddings folder."""

    directory: Path
    generation_id: str | None
    manifest: GenerationManifestV1 | None
    pointer: GenerationPointerV1 | None

    @property
    def is_legacy(self):
        """Return True when using the unversioned flat folder layout."""
        return self.generation_id is None


_ARTIFACT_LAYOUTS = {
    "baseline": GenerationArtifactLayout(
        retrieval_mode="baseline",
        image_embeddings="stored_embeddings.pt",
        image_neighbors="stored_image_neighbors.pt",
        centroid_embeddings="stored_centroid_embeddings.pt",
        centroid_neighbors="stored_centroid_neighbors.pt",
        centroid_labels="stored_centroid_labels.pt",
    ),
    "psl": GenerationArtifactLayout(
        retrieval_mode="psl",
        image_embeddings="stored_embeddings.pt",
        image_neighbors="stored_image_neighbors_psl.pt",
        centroid_embeddings="stored_centroid_embeddings.pt",
        centroid_neighbors="stored_centroid_neighbors.pt",
        centroid_labels="stored_centroid_labels.pt",
    ),
    "multimodal": GenerationArtifactLayout(
        retrieval_mode="multimodal",
        image_embeddings="stored_embeddings_multimodal.pt",
        image_neighbors="stored_image_neighbors_multimodal.pt",
        centroid_embeddings="stored_centroid_embeddings_multimodal.pt",
        centroid_neighbors="stored_centroid_neighbors_multimodal.pt",
        centroid_labels="stored_centroid_labels_multimodal.pt",
    ),
    "stacked": GenerationArtifactLayout(
        retrieval_mode="stacked",
        image_embeddings="stored_embeddings_multimodal.pt",
        image_neighbors="stored_image_neighbors_multimodal_psl.pt",
        centroid_embeddings="stored_centroid_embeddings_multimodal.pt",
        centroid_neighbors="stored_centroid_neighbors_multimodal.pt",
        centroid_labels="stored_centroid_labels_multimodal.pt",
    ),
}


# pylint: disable-next=too-many-arguments,too-many-locals
def publish_generation(
        *, embedding_set_root, embedding_set, model_name, source, snapshot_id,
        start_time, retrieval_mode, build, seed_previous=True,
        seed_neighbors=True, inherit_previous_identity=False):
    """Build and validate a new embedding generation in staging, then switch current.json to it."""
    root = Path(embedding_set_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    get_generation_artifact_layout(retrieval_mode)

    with _locked_generation_root(root):
        previous = resolve_active_generation(root)
        previous_directory = previous.directory if previous.directory.is_dir() else None
        publication_embedding_set = embedding_set
        publication_model_name = model_name
        publication_source = source
        publication_snapshot_id = snapshot_id
        if inherit_previous_identity and previous.manifest is not None:
            publication_embedding_set = previous.manifest.embedding_set
            publication_model_name = previous.manifest.model_name
            publication_source = previous.manifest.source
            publication_snapshot_id = previous.manifest.snapshot_id
        generations_directory = root / _GENERATIONS_DIRECTORY_NAME
        generations_directory.mkdir(exist_ok=True)
        _require_directory(generations_directory, "generations directory")

        generation_id = str(uuid.uuid4())
        staging_directory = generations_directory / f"{STAGING_PREFIX}{generation_id}"
        staging_directory.mkdir()
        final_directory = None
        try:
            if seed_previous:
                _seed_staging_generation(
                    previous_directory,
                    staging_directory,
                    root,
                    include_neighbors=seed_neighbors,
                )
            num_images = build(staging_directory, previous_directory)
            if isinstance(num_images, bool) or not isinstance(num_images, int):
                raise GenerationValidationError(
                    "generation builder must return an integer image count"
                )
            end_time = time.time()
            _write_embeddings_metadata(
                staging_directory,
                publication_embedding_set,
                start_time,
                end_time,
                num_images,
                generation_id,
                publication_snapshot_id,
            )
            _validate_staged_generation(
                staging_directory,
                retrieval_mode,
                generation_id,
                publication_snapshot_id,
                num_images,
            )
            created_at = _rfc3339_utc(end_time)
            manifest = _write_generation_manifest(
                staging_directory,
                generation_id,
                publication_embedding_set,
                publication_model_name,
                publication_source,
                publication_snapshot_id,
                created_at,
            )
            final_directory = _finalize_generation_directory(
                staging_directory, generations_directory, generation_id
            )
            pointer = _replace_current_pointer(
                root, generation_id, _rfc3339_utc(time.time())
            )
        except Exception:
            shutil.rmtree(staging_directory, ignore_errors=True)
            if final_directory is not None:
                shutil.rmtree(final_directory, ignore_errors=True)
            raise

        retained_generation_ids = {generation_id}
        if previous.generation_id is not None:
            retained_generation_ids.add(previous.generation_id)
        try:
            _prune_generations(
                generations_directory, retained_generation_ids
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            print("WARNING: Generation retention failed after publication: "
                  f"{error}")

        return ResolvedGeneration(
            directory=final_directory,
            generation_id=generation_id,
            manifest=manifest,
            pointer=pointer,
        )


def resolve_active_generation(embedding_set_root):
    """Return the active generation folder for an embedding set, or the flat root if unversioned."""
    root = Path(embedding_set_root).resolve()
    pointer_path = root / _CURRENT_POINTER_FILENAME

    try:
        pointer_mode = pointer_path.lstat().st_mode
    except FileNotFoundError:
        return ResolvedGeneration(
            directory=root,
            generation_id=None,
            manifest=None,
            pointer=None,
        )

    if stat.S_ISLNK(pointer_mode) or not stat.S_ISREG(pointer_mode):
        raise GenerationValidationError(
            "active generation pointer must be a nonsymlinked regular file"
        )
    try:
        pointer = _load_pointer(pointer_path)
        return _resolve_versioned_generation(
            root, pointer.generation_id, pointer=pointer
        )
    except GenerationValidationError as exc:
        raise GenerationValidationError(
            f"{exc}; to repair, restore the generation named in {pointer_path.name} "
            f"or delete {pointer_path.name} and regenerate the embeddings"
        ) from exc


def resolve_active_directory(embedding_set_name):
    """Return the active generation directory of a named embedding set under the embeddings root."""
    return resolve_active_generation(
        os.path.join(constants.EMBEDDINGS_DIR, embedding_set_name)
    ).directory


def rollback_generation(embedding_set_root):
    """Switch current.json back to the most recent valid previous generation."""
    root = Path(embedding_set_root).resolve()
    _require_directory(root, "embedding set root")
    with _locked_generation_root(root):
        active = resolve_active_generation(root)
        if active.is_legacy:
            raise GenerationValidationError(
                "rollback requires an active versioned generation"
            )
        candidates = _valid_rollback_candidates(root, active.generation_id)
        if not candidates:
            raise GenerationValidationError(
                "no valid prior generation is available for rollback"
            )
        prior = max(
            candidates,
            key=lambda candidate: (
                candidate.manifest.created_at, candidate.generation_id
            ),
        )
        pointer = _replace_current_pointer(
            root, prior.generation_id, _rfc3339_utc(time.time())
        )
        return ResolvedGeneration(
            directory=prior.directory,
            generation_id=prior.generation_id,
            manifest=prior.manifest,
            pointer=pointer,
        )


def resolve_retrieval_mode(*, use_psl=False, use_multimodal=False,
                           use_stacked=False):
    """Map boolean feature flags to a single retrieval mode name."""
    if use_stacked:
        return "stacked"
    if use_multimodal:
        return "multimodal"
    if use_psl:
        return "psl"
    return "baseline"


def retrieval_mode_cli_flags(retrieval_mode):
    """Return the CLI flags corresponding to a retrieval mode."""
    return {
        "multimodal": ["--use-multimodal"],
        "psl": ["--use-psl"],
        "stacked": ["--stacked"],
    }.get(retrieval_mode, [])


def get_generation_artifact_layout(retrieval_mode):
    """Return the expected file layout for a retrieval mode."""
    try:
        return _ARTIFACT_LAYOUTS[retrieval_mode]
    except KeyError as exc:
        raise GenerationValidationError(
            f"unsupported retrieval mode: {retrieval_mode}"
        ) from exc


def _rfc3339_utc(timestamp):
    """Return an RFC 3339 UTC timestamp for a Unix timestamp."""
    return datetime.datetime.fromtimestamp(
        timestamp, tz=datetime.timezone.utc
    ).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _require_regular_file(path, description):
    """Require an existing, nonsymlinked regular generation file."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise GenerationValidationError(f"{description} is missing") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise GenerationValidationError(
            f"{description} must be a nonsymlinked regular file"
        )


def _require_directory(path, description):
    """Require an existing, nonsymlinked generation directory."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise GenerationValidationError(f"{description} is missing") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise GenerationValidationError(
            f"{description} must be a nonsymlinked directory"
        )


def _load_pointer(path):
    """Read and validate current.json."""
    try:
        return GenerationPointerV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise GenerationValidationError(
            "active generation pointer is invalid"
        ) from exc


def _load_manifest(path):
    """Read and validate a generation's manifest.json."""
    try:
        return GenerationManifestV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise GenerationValidationError("generation manifest is invalid") from exc


def _sha256(path):
    """Compute the lowercase SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_torch_artifact(path, description):
    """Load a PyTorch file on CPU for validation."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise GenerationValidationError(f"{description} is invalid") from exc


def _is_generation_directory(path):
    """Return True if path is a real folder named with a lowercase UUIDv4."""
    if path.is_symlink() or not path.is_dir():
        return False
    try:
        parsed_id = uuid.UUID(path.name)
        return parsed_id.version == 4 and str(parsed_id) == path.name
    except ValueError:
        return False


def _sync_directory(directory):
    """Flush directory entry changes to disk with fsync."""
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sync_generation_files(directory):
    """Flush every file in a staging folder to disk before renaming it."""
    for artifact_path in sorted(directory.iterdir()):
        _require_regular_file(
            artifact_path, f"generation artifact {artifact_path.name}"
        )
        with artifact_path.open("rb") as artifact_file:
            os.fsync(artifact_file.fileno())


def _finalize_generation_directory(
        staging_directory, generations_directory, generation_id):
    """Flush staging files and rename the staging folder to its final UUID name."""
    final_directory = generations_directory / generation_id
    if final_directory.exists():
        raise GenerationValidationError("generation identifier already exists")
    _sync_generation_files(staging_directory)
    _sync_directory(staging_directory)
    os.rename(staging_directory, final_directory)
    try:
        _sync_directory(generations_directory)
    except OSError:
        shutil.rmtree(final_directory, ignore_errors=True)
        try:
            _sync_directory(generations_directory)
        except OSError:
            pass
        raise
    return final_directory


def _replace_current_pointer(root, generation_id, published_at):
    """Write current.json to a temp file, flush it, and rename it into place."""
    pointer = GenerationPointerV1(
        schema_version=GENERATION_SCHEMA_VERSION,
        generation_id=generation_id,
        published_at=published_at,
    )
    temporary_pointer = root / f".{_CURRENT_POINTER_FILENAME}.{uuid.uuid4().hex}.tmp"
    pointer_replaced = False
    try:
        with temporary_pointer.open("x", encoding="utf-8") as pointer_file:
            pointer_file.write(pointer.model_dump_json(indent=2))
            pointer_file.flush()
            os.fsync(pointer_file.fileno())
        os.replace(temporary_pointer, root / _CURRENT_POINTER_FILENAME)
        pointer_replaced = True
        try:
            _sync_directory(root)
        except OSError as error:
            print("WARNING: Active generation pointer was replaced but its "
                  f"directory sync failed: {error}")
    finally:
        try:
            temporary_pointer.unlink(missing_ok=True)
        except OSError:
            if not pointer_replaced:
                raise
    return pointer


@contextmanager
def _locked_generation_root(root):
    """Acquire an exclusive file lock on the embedding set directory."""
    lock_path = root / _GENERATION_LOCK_FILENAME
    open_flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    try:
        lock_fd = os.open(lock_path, open_flags, 0o600)
    except OSError as exc:
        raise GenerationValidationError(
            "generation lock could not be opened safely"
        ) from exc
    with os.fdopen(lock_fd, "r+b", closefd=True) as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _seed_staging_generation(
        previous_directory, staging_directory, root, include_neighbors=True):
    """Copy prior artifacts without linking mutable staging files to active data."""
    if previous_directory is None or not previous_directory.is_dir():
        return
    excluded = {
        _CURRENT_POINTER_FILENAME,
        _GENERATION_LOCK_FILENAME,
        _MANIFEST_FILENAME,
    }
    for source_path in sorted(previous_directory.iterdir()):
        if source_path.name in excluded or not source_path.is_file():
            continue
        if source_path.parent == root and source_path.name.startswith("."):
            continue
        if not include_neighbors and source_path.name.startswith((
                "stored_image_neighbors",
                "stored_centroid_neighbors",
        )):
            continue
        destination_path = staging_directory / source_path.name
        shutil.copy2(source_path, destination_path)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _write_embeddings_metadata(
        staging_directory, embedding_set, start_time, end_time, num_images,
        generation_id, snapshot_id):
    """Write and validate metadata through the existing matcher utility."""
    metadata = utils.record_embeddings_metadata(
        embedding_set,
        start_time,
        end_time,
        num_images,
        output_dir=staging_directory,
        generation_id=generation_id,
        snapshot_id=snapshot_id,
    )
    try:
        validated = EmbeddingsMetadataV1.model_validate(metadata)
    except ValidationError as exc:
        raise GenerationValidationError(
            "embedding generation metadata is invalid"
        ) from exc
    metadata_path = staging_directory / EMBEDDINGS_METADATA_FILENAME
    with metadata_path.open("rb") as metadata_file:
        os.fsync(metadata_file.fileno())
    return validated


def _resolve_versioned_generation(root, generation_id, pointer=None):
    """Resolve one versioned generation through the shared safety checks."""
    generations_directory = root / _GENERATIONS_DIRECTORY_NAME
    _require_directory(generations_directory, "generations directory")
    generation_directory = generations_directory / generation_id
    _require_directory(generation_directory, "generation directory")

    resolved_generations_directory = generations_directory.resolve(strict=True)
    resolved_generation_directory = generation_directory.resolve(strict=True)
    if resolved_generation_directory.parent != resolved_generations_directory:
        raise GenerationValidationError(
            "generation must be a direct child of the generations directory"
        )

    manifest_path = generation_directory / _MANIFEST_FILENAME
    _require_regular_file(manifest_path, "generation manifest")
    resolved_manifest_path = manifest_path.resolve(strict=True)
    if resolved_manifest_path.parent != resolved_generation_directory:
        raise GenerationValidationError(
            "generation manifest must be inside the generation directory"
        )
    manifest = _load_manifest(resolved_manifest_path)
    if manifest.generation_id != generation_id:
        raise GenerationValidationError(
            "generation manifest identifier does not match generation directory"
        )

    return ResolvedGeneration(
        directory=resolved_generation_directory,
        generation_id=generation_id,
        manifest=manifest,
        pointer=pointer,
    )


def _validate_manifest_artifacts(resolved_generation):
    """Validate every retained artifact against its manifest record."""
    generation_directory = resolved_generation.directory
    manifest = resolved_generation.manifest
    expected_filenames = set(manifest.files)
    actual_filenames = {
        artifact_path.name
        for artifact_path in generation_directory.iterdir()
        if artifact_path.name != _MANIFEST_FILENAME
    }
    if actual_filenames != expected_filenames:
        raise GenerationValidationError(
            "generation artifacts do not match manifest file list"
        )

    for filename, artifact_record in manifest.files.items():
        artifact_path = generation_directory / filename
        _require_regular_file(
            artifact_path, f"generation artifact {filename}"
        )
        resolved_artifact_path = artifact_path.resolve(strict=True)
        if resolved_artifact_path.parent != generation_directory:
            raise GenerationValidationError(
                "generation artifact must remain inside its generation directory"
            )
        if (artifact_path.stat().st_size != artifact_record.size_bytes
                or _sha256(artifact_path) != artifact_record.sha256):
            raise GenerationValidationError(
                f"generation artifact {filename} does not match its manifest"
            )


# pylint: disable-next=too-many-locals
def _validate_staged_generation(
        staging_directory, retrieval_mode, generation_id, snapshot_id,
        num_images):
    """Validate required artifacts, identities, and aligned row counts."""
    layout = get_generation_artifact_layout(retrieval_mode)
    for required_filename in layout.required_files:
        _require_regular_file(
            staging_directory / required_filename,
            f"required generation artifact {required_filename}",
        )

    metadata_path = staging_directory / EMBEDDINGS_METADATA_FILENAME
    try:
        metadata = EmbeddingsMetadataV1.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise GenerationValidationError(
            "embedding generation metadata is invalid"
        ) from exc
    if (metadata.generation_id != generation_id
            or metadata.snapshot_id != snapshot_id
            or metadata.num_images != num_images):
        raise GenerationValidationError(
            "embedding generation metadata identity or image count is inconsistent"
        )

    dataset = _load_torch_artifact(
        staging_directory / "stored_dataset.pt", "stored dataset")
    try:
        filenames = list(dataset.get_filenames())
        ids = list(dataset.get_ids())
        labels = dataset.get_labels()
    except (AttributeError, TypeError) as exc:
        raise GenerationValidationError("stored dataset interface is invalid") from exc
    if len(filenames) != num_images or len(ids) != num_images or len(labels) != num_images:
        raise GenerationValidationError(
            "stored dataset rows do not match generation image count"
        )

    embeddings = _load_torch_artifact(
        staging_directory / layout.image_embeddings, "image embeddings")
    if not hasattr(embeddings, "shape") or embeddings.shape[0] != num_images:
        raise GenerationValidationError(
            "image embedding rows do not match generation image count"
        )

    image_neighbors_path = staging_directory / layout.image_neighbors
    if layout.retrieval_mode in {"psl", "stacked"} or image_neighbors_path.exists():
        image_neighbors = _load_torch_artifact(
            image_neighbors_path, "image neighbors")
        if (not isinstance(image_neighbors, dict)
                or set(image_neighbors) != set(filenames)):
            raise GenerationValidationError(
                "image neighbors do not match stored dataset filenames"
            )

    centroid_neighbors_path = staging_directory / layout.centroid_neighbors
    if centroid_neighbors_path.exists():
        centroid_neighbors = _load_torch_artifact(
            centroid_neighbors_path, "centroid neighbors")
        if (not isinstance(centroid_neighbors, dict)
                or set(centroid_neighbors) != set(filenames)):
            raise GenerationValidationError(
                "centroid neighbors do not match stored dataset filenames"
            )

    centroid_embeddings = _load_torch_artifact(
        staging_directory / layout.centroid_embeddings,
        "centroid embeddings",
    )
    centroid_labels = _load_torch_artifact(
        staging_directory / layout.centroid_labels, "centroid labels")
    if (not hasattr(centroid_embeddings, "shape")
            or not hasattr(centroid_labels, "shape")
            or centroid_embeddings.shape[0] != centroid_labels.shape[0]):
        raise GenerationValidationError(
            "centroid embedding rows do not match centroid labels"
        )


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _write_generation_manifest(
        staging_directory, generation_id, embedding_set, model_name, source,
        snapshot_id, created_at):
    """Hash every file in staging and write manifest.json."""
    files = {}
    for artifact_path in sorted(staging_directory.iterdir()):
        _require_regular_file(
            artifact_path, f"generation artifact {artifact_path.name}"
        )
        files[artifact_path.name] = ArtifactFileV1(
            size_bytes=artifact_path.stat().st_size,
            sha256=_sha256(artifact_path),
        )
    try:
        manifest = GenerationManifestV1(
            schema_version=GENERATION_SCHEMA_VERSION,
            generation_id=generation_id,
            created_at=created_at,
            embedding_set=embedding_set,
            model_name=model_name,
            source=source,
            snapshot_id=snapshot_id,
            files=files,
        )
    except ValidationError as exc:
        raise GenerationValidationError(
            "generation manifest identity is invalid"
        ) from exc
    manifest_path = staging_directory / _MANIFEST_FILENAME
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        manifest_file.write(manifest.model_dump_json(indent=2))
        manifest_file.flush()
        os.fsync(manifest_file.fileno())
    return manifest


def _prune_generations(generations_directory, retained_generation_ids):
    """Remove finalized generations outside the requested retention set."""
    retained = set(retained_generation_ids)
    removed_any = False
    for generation_path in sorted(generations_directory.iterdir()):
        if (generation_path.name in retained
                or not _is_generation_directory(generation_path)):
            continue
        resolved_path = generation_path.resolve(strict=True)
        if resolved_path.parent != generations_directory.resolve(strict=True):
            continue
        shutil.rmtree(resolved_path)
        removed_any = True
    if removed_any:
        _sync_directory(generations_directory)


def _valid_rollback_candidates(root, active_generation_id):
    """Return manifest-valid generations other than the active generation."""
    generations_directory = root / _GENERATIONS_DIRECTORY_NAME
    candidates = []
    for generation_path in sorted(generations_directory.iterdir()):
        if (generation_path.name == active_generation_id
                or not _is_generation_directory(generation_path)):
            continue
        try:
            candidate = _resolve_versioned_generation(
                root, generation_path.name
            )
            _validate_manifest_artifacts(candidate)
        except (GenerationValidationError, OSError):
            continue
        candidates.append(candidate)
    return candidates
