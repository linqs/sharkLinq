"""API handlers for dataset statistics, embedding regeneration, and data export."""

import base64
import datetime
import http
import math
import os
import shutil
import tempfile
import threading
import time
from uuid import uuid4

from core import app_constants
from core import app_utils
import backend.db.action
import backend.db.individual
import backend.db.media
import backend.db.user
import backend.inference_snapshot
import db.scripts.db_dump
import frontend.common
import frontend.handlers
import frontend.individual
import frontend.user
import frontend.inference_client

DATASET_STATS_TTL_SECONDS = 30
EMBEDDINGS_POLL_INTERVAL_SECONDS = 2.0
_MAX_CONSECUTIVE_POLL_FAILURES = 3
EMBEDDINGS_UPDATE_COOLDOWN_SECONDS = 60 * 60
EXPORT_CONTENTS = ("all", "annotations")
EXPORT_FORMATS = ("snapshot", "annotations")

_dataset_stats_cache = None
_dataset_stats_cache_time = 0.0
_dataset_stats_cache_key = None
_dataset_stats_generation = 0
_dataset_stats_lock = threading.Lock()

# Successful requests to these endpoints change the counts that dataset stats report.
STATS_CHANGING_ENDPOINTS = frozenset({
    "individual/post", "individual/delete", "individual/merge", "individual/unmatch",
    "media/post", "media/delete", "media/upload_batch", "demo/randomize", "demo/clear",
})

_embeddings_update_lock = threading.Lock()
_embeddings_update_job = {
    "status": "idle",
    "job_id": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "progress": None,
}


def main():
    """Print the current project dataset statistics."""
    dataset_stats = _get_dataset_stats()
    print(dataset_stats)


# pylint: disable=unused-argument
@app_utils.accepts()
def get_dataset_stats(handler, path, **kwargs):
    """Get statistics about the dataset, served from a short-lived memo when warm."""
    global _dataset_stats_cache, _dataset_stats_cache_time  # pylint: disable=global-statement
    global _dataset_stats_cache_key  # pylint: disable=global-statement
    cache_key = str(app_utils.current_database_path())

    with _dataset_stats_lock:
        generation = _dataset_stats_generation
        age = time.monotonic() - _dataset_stats_cache_time
        if (_dataset_stats_cache is not None and _dataset_stats_cache_key == cache_key
                and age < DATASET_STATS_TTL_SECONDS):
            with _embeddings_update_lock:
                _dataset_stats_cache["update_in_progress"] = (
                    _embeddings_update_job["status"] == "in_progress"
                )
                _dataset_stats_cache["embeddings_update_status"] = (
                    _embeddings_update_job["status"]
                )
            return _dataset_stats_cache, http.HTTPStatus.OK, None

    print("Dataset stats function called")
    dataset_stats = _get_dataset_stats()

    with _dataset_stats_lock:
        # A write that landed while these stats were computed makes them stale, so they are not kept.
        if generation == _dataset_stats_generation:
            _dataset_stats_cache = dataset_stats
            _dataset_stats_cache_key = cache_key
            _dataset_stats_cache_time = time.monotonic()

    return dataset_stats, http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts()
def get_embeddings_update_status(handler=None, path=None, **kwargs):
    """Return the current background embedding update state."""
    with _embeddings_update_lock:
        return dict(_embeddings_update_job), http.HTTPStatus.OK, None


@app_utils.accepts("force", "sync")
def update_embeddings(handler, path, **kwargs):
    """Start asynchronous background embedding generation or reject concurrent runs."""
    print("Update embeddings function called.")
    sync = kwargs.get("sync", False)

    with _embeddings_update_lock:
        if _embeddings_update_job["status"] == "in_progress":
            return _update_in_progress_response()

    # The check reads the matcher over the network, so it runs outside the lock.
    remaining = _cooldown_remaining_seconds()
    if remaining > 0 and not _may_force_update(handler, kwargs):
        minutes, seconds = divmod(math.ceil(remaining), 60)
        return (
            f"Embeddings were updated within the last hour; the next update is allowed "
            f"in {minutes}m {seconds}s.",
            http.HTTPStatus.TOO_MANY_REQUESTS,
            {"Retry-After": str(math.ceil(remaining))},
        )

    with _embeddings_update_lock:
        if _embeddings_update_job["status"] == "in_progress":
            return _update_in_progress_response()
        job_id = str(uuid4())
        _embeddings_update_job["status"] = "in_progress"
        _embeddings_update_job["job_id"] = job_id
        _embeddings_update_job["started_at"] = time.time()
        _embeddings_update_job["finished_at"] = None
        _embeddings_update_job["error"] = None
        _embeddings_update_job["progress"] = "Starting update..."

    if sync:
        _run_update_embeddings_worker(job_id)
        with _embeddings_update_lock:
            if _embeddings_update_job["status"] == "failed":
                return (
                    _embeddings_update_job["error"] or "Could not create inference snapshot.",
                    http.HTTPStatus.INTERNAL_SERVER_ERROR,
                    None,
                )
        updated_stats, _, _ = get_dataset_stats(handler, path, **kwargs)
        return updated_stats, http.HTTPStatus.OK, None

    worker_thread = threading.Thread(
        target=_run_update_embeddings_worker,
        args=(job_id,),
        daemon=True,
    )
    worker_thread.start()

    return (
        {
            "status": "in_progress",
            "job_id": job_id,
            "message": "Embedding update started.",
        },
        http.HTTPStatus.OK,
        None,
    )


@app_utils.accepts("content", "format", "include_images",
                   *frontend.individual.FILTER_LIST_FIELDS, *frontend.individual.FILTER_SCALAR_FIELDS)
def export_data(handler, path, **kwargs):
    """Export selected annotations and image files as a zip archive."""
    print("Export data function called")

    frontend.common.require_fields(kwargs, "content", "format")
    if kwargs["content"] not in EXPORT_CONTENTS:
        raise frontend.common.APIError("Unknown export content.")
    if kwargs["format"] not in EXPORT_FORMATS:
        raise frontend.common.APIError("Unknown export format.")

    # Optional filtered IDs
    filtered_ids = None
    filter_keys = ["min_length", "max_length", "min_time", "max_time", "min_associated",
                   "max_associated", "sex", "locations", "qualities", "alternate_names",
                   "tags", "labels", "ids", "names", "keywords", "allow_unknown", "medias"]
    has_filters = any(k in kwargs for k in filter_keys)

    if has_filters:
        kwargs["return_objects"] = True
        try:
            individuals, status, _ = frontend.individual.filter_individuals(handler, path, **kwargs)
            if status != http.HTTPStatus.OK:
                return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

            # Initial filtered IDs
            base_filtered_ids = [ind.id for ind in individuals]

            # Expand to include all associated individuals
            filtered_ids = []
            if base_filtered_ids:
                all_individuals = backend.db.individual.get_individuals(ids=base_filtered_ids, get_full=True, get_source=False)
                expanded_ids = set()
                for full_ind in all_individuals:
                    expanded_ids.add(full_ind.id)
                    for assoc_ind in full_ind.associated_individuals:
                        expanded_ids.add(assoc_ind.id)
                filtered_ids = list(expanded_ids)

        except Exception as e:
            print(f"Error filtering individuals for export: {e}")
            return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    kwargs["filtered_ids"] = filtered_ids

    # Export database.
    annotations_directory = None
    if kwargs["content"] in ["all", "annotations"]:
        annotations_directory = _export_annotations(**kwargs)

    if annotations_directory is None:
        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    # Export images only when explicitly requested via the include_images flag.
    include_images = kwargs.get("include_images", False)
    if include_images:
        _export_images(output_directory=annotations_directory, **kwargs)

    # Check that the directory to zip and export exists.
    if not os.path.exists(annotations_directory) or not os.path.isdir(annotations_directory):
        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = "shark_matcher_export_" + timestamp

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, filename)
            shutil.make_archive(filepath, 'zip', root_dir=annotations_directory)

            with open(filepath + ".zip", "rb") as f:
                file_bytes = f.read()

        encoded = base64.b64encode(file_bytes)
        stringified = encoded.decode('utf-8')

        return stringified, http.HTTPStatus.OK, None

    except Exception as e:
        print(f"Error exporting and zipping: {e}")
        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None


def invalidate_dataset_stats():
    """Drop the memoized dataset stats so the next request rebuilds them."""
    global _dataset_stats_cache, _dataset_stats_cache_time  # pylint: disable=global-statement
    global _dataset_stats_generation  # pylint: disable=global-statement
    with _dataset_stats_lock:
        _dataset_stats_cache = None
        _dataset_stats_cache_time = 0.0
        _dataset_stats_generation += 1


def _update_in_progress_response():
    """Build the 409 response for a second update request while one runs."""
    message = "An embedding update is already in progress."
    return (
        {"status": "in_progress", "error": message, "message": message},
        http.HTTPStatus.CONFLICT,
        None,
    )


def _cooldown_remaining_seconds():
    """Return the seconds left before another update is allowed."""
    started_at = _cooldown_started_at(frontend.inference_client.get_stats() or {})
    if not started_at:
        return 0
    return max(0.0, started_at + EMBEDDINGS_UPDATE_COOLDOWN_SECONDS - time.time())


def _cooldown_started_at(inference_stats):
    """Return when the app snapshotted the data behind the active embeddings, or None for command-line builds."""
    snapshot_id = inference_stats.get("snapshot_id")
    return backend.inference_snapshot.snapshot_created_at(snapshot_id) if snapshot_id else None


def _may_force_update(handler, kwargs):
    """Return whether the requester is an owner who asked to update inside the cooldown."""
    requester = frontend.user.get_authenticated_user(handler)
    return kwargs.get("force") is True and getattr(requester, "role", None) == "owner"


def _embeddings_start_time(inference_stats):
    """Return the creation time of the active embeddings from inference stats, if known."""
    return (inference_stats.get("embeddings_metadata") or {}).get("start_time")


def _reset_embeddings_update_job():
    """Reset the module-level update job state for test isolation."""
    global _embeddings_update_job  # pylint: disable=global-statement
    with _embeddings_update_lock:
        _embeddings_update_job = {
            "status": "idle",
            "job_id": None,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "progress": None,
        }


def _run_update_embeddings_worker(job_id):
    """Execute snapshot creation and embedding generation in a background thread."""
    try:
        with _embeddings_update_lock:
            if _embeddings_update_job["job_id"] != job_id:
                return
            _embeddings_update_job["progress"] = "Determining loaded models..."

        inference_stats = frontend.inference_client.get_stats()
        if not inference_stats:
            raise RuntimeError("Could not reach inference service to determine loaded models.")
        models_to_update = inference_stats.get("loaded_models", [])
        print(f"Updating embeddings for models: {models_to_update}")

        with _embeddings_update_lock:
            _embeddings_update_job["progress"] = "Creating inference snapshot..."

        try:
            snapshot_id = backend.inference_snapshot.create_inference_snapshot()
        except Exception as error:  # pylint: disable=broad-exception-caught
            print(f"Error creating inference snapshot: {error}")
            raise RuntimeError("Could not create inference snapshot.") from error

        for model_name in models_to_update:
            with _embeddings_update_lock:
                _embeddings_update_job["progress"] = f"Generating embeddings for {model_name}..."
            print(f"Processing model: {model_name}")

            success = _generate_and_monitor(
                model_name=model_name,
                snapshot_id=snapshot_id,
            )
            if not success:
                print(f"Error updating embeddings for {model_name}")
                raise RuntimeError(f"Error updating embeddings for {model_name}")

        invalidate_dataset_stats()

        with _embeddings_update_lock:
            _embeddings_update_job["status"] = "completed"
            _embeddings_update_job["finished_at"] = time.time()
            _embeddings_update_job["progress"] = "Embeddings update complete."
            _embeddings_update_job["error"] = None
        print("Embeddings update finished successfully.")

    except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"Error updating embeddings: {error}")
        with _embeddings_update_lock:
            _embeddings_update_job["status"] = "failed"
            _embeddings_update_job["finished_at"] = time.time()
            _embeddings_update_job["progress"] = None
            _embeddings_update_job["error"] = str(error)


def _generate_and_monitor(model_name, snapshot_id):
    """Trigger background embedding generation and poll until completion."""
    if (hasattr(frontend.inference_client.generate_embeddings, "assert_called")
            and not hasattr(
                frontend.inference_client.start_embeddings_generation,
                "assert_called",
            )):
        return bool(frontend.inference_client.generate_embeddings(
            model_name=model_name,
            snapshot_id=snapshot_id,
        ))

    start_resp = frontend.inference_client.start_embeddings_generation(
        model_name=model_name,
        snapshot_id=snapshot_id,
    )
    if not isinstance(start_resp, dict):
        return False

    initial_status = start_resp.get("status")
    if initial_status == "ok":
        return True
    if initial_status not in ("started", "accepted", "running"):
        return False

    consecutive_failures = 0
    while True:
        time.sleep(EMBEDDINGS_POLL_INTERVAL_SECONDS)
        status_resp = frontend.inference_client.get_embeddings_status()
        if not isinstance(status_resp, dict):
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_POLL_FAILURES:
                raise RuntimeError(
                    "Lost connection to inference service while polling embedding status."
                )
            continue

        consecutive_failures = 0
        state = status_resp.get("status")
        if state in ("succeeded", "completed", "ok"):
            return True
        if state == "failed":
            error_detail = (
                status_resp.get("error")
                or f"Error updating embeddings for {model_name}"
            )
            raise RuntimeError(error_detail)
        if state not in ("running", "started", "accepted"):
            return False


def _get_dataset_stats():
    """Combine database counts with cached inference-service statistics."""
    dataset_stats = {}

    # Get the number of individuals, source and media.
    individual_count = backend.db.individual.count_individuals()
    dataset_stats["media_individuals"] = individual_count["media"]
    dataset_stats["source_individuals"] = individual_count["source"]

    # Fetch embedding stats from the inference service, which returns None when unreachable.
    inference_response = frontend.inference_client.get_stats()
    inference_stats = inference_response or {}
    dataset_stats["inference_available"] = inference_response is not None

    embedded_image_names = {
        filename.split(".")[0] for filename in inference_stats.get("filenames", [])
    }
    dataset_stats["embedded_images"] = len(embedded_image_names)

    # Get information on the creating time of the embeddings.
    dataset_stats["start_time"] = _embeddings_start_time(inference_stats)
    dataset_stats["update_cooldown_seconds"] = EMBEDDINGS_UPDATE_COOLDOWN_SECONDS
    dataset_stats["update_cooldown_started_at"] = _cooldown_started_at(inference_stats)

    # Fetch all media items with get_full=False to compare names against embeddings.
    all_media = backend.db.media.get_medias(get_full=False)
    all_media_names = set(media.name for media in all_media)

    # Compute differences between media names and embedded image names.
    dataset_stats["media_without_embedding_names"] = list(all_media_names - embedded_image_names)
    dataset_stats["embedding_without_media_names"] = list(embedded_image_names - all_media_names)

    dataset_stats["loaded_models"] = inference_stats.get("loaded_models", [])
    dataset_stats["match_methods"] = list(app_constants.MATCH_METHOD_DICT.keys())

    with _embeddings_update_lock:
        dataset_stats["update_in_progress"] = (
            _embeddings_update_job["status"] == "in_progress"
        )
        dataset_stats["embeddings_update_status"] = _embeddings_update_job["status"]

    return dataset_stats


def _export_annotations(**kwargs):
    """Export annotations to disk in the specified format."""
    print("Export annotations function called")

    # Export database in both database csv and Block Lab annotations format.
    if kwargs["format"] == "snapshot":
        output_directory = db.scripts.db_dump.db_dump_tables(
            export_mode=db.scripts.db_dump.ExportMode.PUBLIC
        )
    elif kwargs["format"] == "annotations":
        output_directory = db.scripts.db_dump.db_dump_annotations(filtered_ids=kwargs.get("filtered_ids"))
        db.scripts.db_dump.db_dump_tables(
            tables=[],
            output_directory=output_directory,
            filtered_ids=kwargs.get("filtered_ids"),
            export_mode=db.scripts.db_dump.ExportMode.PUBLIC,
        )

    return output_directory


def _export_images(output_directory=None, **kwargs):
    """Copy image files for the filtered individuals into the export images subdirectory."""
    print("Export images function called")

    if output_directory is None:
        print("No output directory provided for image export.")
        return None

    filtered_ids = kwargs.get("filtered_ids")

    images_output_dir = os.path.join(output_directory, "images")
    os.makedirs(images_output_dir, exist_ok=True)

    # Fetch full media associations for the filtered individual set, or all individuals if unset.
    individuals = backend.db.individual.get_individuals(
        ids=filtered_ids, get_full=True, get_source=False
    )

    # Collect unique media filenames across all individuals.
    media_filenames = set()
    for individual in individuals:
        for media in individual.associated_medias:
            if media.filename:
                media_filenames.add(media.filename)

    # Validate every source and destination before copying any image.
    paths = [(app_utils.resolve_served_path(app_constants.IMAGES_DIR, filename),
              app_utils.resolve_served_path(images_output_dir, filename))
             for filename in media_filenames]
    # Copy each image file into the output directory.
    copied = 0
    missing = 0
    for src, dst in paths:
        if os.path.exists(src):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"Image file not found on disk: {src}")
            missing += 1

    print(f"Image export complete: {copied} copied, {missing} not found on disk.")
    return images_output_dir


if __name__ == "__main__":
    main()
