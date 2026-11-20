"""API handlers for media creation, retrieval, file storage, and batch uploads."""

import base64
import datetime
import http
import io
import json
import mimetypes
import os
import traceback

import PIL.ExifTags
import PIL.Image

from core import app_constants
from core import app_utils
import backend.db.individual
import backend.db.media
import demo.runtime
import frontend.action
import frontend.common


# pylint: disable=unused-argument, too-many-locals, too-many-return-statements
# pylint: disable=too-many-branches
@app_utils.accepts("id", "hash", "name", "source", "individual", "metadata", "sex", "length",
                   "location", "time", "type", "quality", "alternate_names", "tags", "labels")
def post_media(handler, path, **kwargs):
    """Create or update a media record and synchronize its associated individual."""
    print("Post Media function called")

    scalar_params = ["id", "hash", "name", "source", "individual", "metadata", "location", "time", "type", "quality", "length", "sex", "user_id"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)
    frontend.common.require_fields(kwargs, "name", "hash")

    # The name becomes the on-disk filename below, so reject bad names before any write.
    if not app_utils.valid_name(kwargs["name"]):
        return "Invalid name.", http.HTTPStatus.BAD_REQUEST, None
    new_image_path = _image_path(kwargs['name'] + '.jpeg', filename_only=True)

    # Check for new creation, if no id is given.
    original_image_path = None
    if kwargs["id"] is None:
        media_id = None
        original_media = None
        original_media_individual = None
    else:
        # If updating, get original media and individual.
        media_id = kwargs["id"]
        original_media = backend.db.media.get_medias(ids=[kwargs["id"]])[0]
        original_image_path = _image_path(original_media.filename)
        original_media_individual = backend.db.individual.get_individuals(ids=[kwargs["individual"]])[0]

    # Build the media object to create or update.
    updated_media = backend.db.media.Media(
        id=media_id, hash=kwargs["hash"], name=kwargs["name"],
        filename=(kwargs["name"] + ".jpeg"), source=kwargs["source"], individual=kwargs["individual"],
        metadata=kwargs["metadata"], location=kwargs["location"], time=kwargs["time"],
        type=kwargs["type"], quality=kwargs["quality"]
    )

    # Atomically update the media record and its media individual together.
    if media_id is not None:
        connection = app_utils.db_connect()
        try:
            # 1. Update media record.
            posted_media = backend.db.media.upsert_media(updated_media, connection)
            if posted_media == app_constants.ERROR:
                raise RuntimeError("Failed to update media record.")

            # 2. Update media individual's biometric properties.
            original_media_individual.length = kwargs["length"]
            original_media_individual.sex = kwargs["sex"]
            updated_media_individual = backend.db.individual.upsert_individual(
                original_media_individual, connection)
            if updated_media_individual == app_constants.ERROR:
                raise RuntimeError("Failed to update media individual.")

            connection.commit()

        except Exception as ex:
            print(f"Post media (update) failed, rolling back: {ex}")
            try:
                connection.rollback()
            except Exception as rb_ex:
                print(f"Warning: Rollback failed: {rb_ex}")
            finally:
                connection.close()
            return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

        connection.close()

        # If media was renamed, update the stored image's filename to match.
        if demo.runtime.current() is None and original_media.filename != posted_media.filename:
            os.rename(original_image_path, new_image_path)

        # DB successful -> Log action.
        media_action_status = frontend.action.log_action(
            kwargs["user_id"], "update", "media", original_media, posted_media).to_dict()
        individual_action_status = frontend.action.log_action(
            kwargs["user_id"], "update", "individual",
            original_media_individual, updated_media_individual).to_dict()
        return {"media_action_status": media_action_status,
                "individual_action_status": individual_action_status,
                "fetch_individual": None}, http.HTTPStatus.OK, None

    # Atomically create the media record, individual hierarchy, and associated objects.
    # pylint: disable=protected-access
    individual_name = str(backend.db.individual._get_next_individual_name())
    connection = app_utils.db_connect()
    try:
        # 1. Create media record.
        posted_media = backend.db.media.upsert_media(updated_media, connection)
        if posted_media == app_constants.ERROR:
            raise RuntimeError("Failed to insert media record.")

        # 2. Create media individual.
        new_individual = backend.db.individual.Individual(
            id=None, name=individual_name, length=kwargs["length"],
            sex=kwargs["sex"], media=posted_media.id, source=None, reference=None
        )
        posted_individual = backend.db.individual.upsert_individual(new_individual, connection)
        if posted_individual == app_constants.ERROR:
            raise RuntimeError("Failed to create media individual.")

        # 3. Create source individual.
        new_source = backend.db.individual.Individual(
            id=None, name=None, length=kwargs["length"],
            sex=kwargs["sex"], source=None, reference=posted_individual.id
        )
        posted_source = backend.db.individual.upsert_individual(new_source, connection)
        if posted_source == app_constants.ERROR:
            raise RuntimeError("Failed to create source individual.")

        # 4. Update pointers: media individual -> source, source -> itself.
        posted_individual.source = posted_source.id
        posted_individual = backend.db.individual.upsert_individual(posted_individual, connection)
        posted_source.source = posted_source.id
        posted_source = backend.db.individual.upsert_individual(posted_source, connection)
        if app_constants.ERROR in (posted_individual, posted_source):
            raise RuntimeError("Failed to update individual pointers.")

        # 5. Update media source to itself.
        posted_media.source = posted_media.id
        posted_media = backend.db.media.upsert_media(posted_media, connection)
        if posted_media == app_constants.ERROR:
            raise RuntimeError("Failed to update media source.")

        # 6. Add "fresh" label to source individual.
        fresh_label = backend.db.individual.AssociatedObject(individual=posted_source.id, name="fresh")
        if backend.db.individual.upsert_associated_object(fresh_label, "Labels", connection) == app_constants.ERROR:
            raise RuntimeError("Failed to add 'fresh' label.")

        # 7. Add preview associated objects to media individual.
        for object_pair in [("alternate_names", "AlternateNames"), ("tags", "Tags"),
                            ("labels", "Labels")]:
            for preview_object in kwargs[object_pair[0]]:
                if preview_object.get("time") is None:
                    preview_object["time"] = None
                created_object = backend.db.individual.AssociatedObject(
                    None, individual=posted_individual.id, name=preview_object["name"],
                    type=preview_object["type"], time=preview_object["time"])
                if backend.db.individual.upsert_associated_object(created_object, object_pair[1], connection) == app_constants.ERROR:
                    raise RuntimeError(f"Failed to create associated object '{preview_object['name']}'.")

        connection.commit()

    except Exception as ex:
        print(f"Post media (create) failed, rolling back: {ex}")
        try:
            connection.rollback()
        except Exception as rb_ex:
            print(f"Warning: Rollback failed: {rb_ex}")
        finally:
            connection.close()
        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    connection.close()

    # DB successful -> Log action.
    try:
        log_individual = backend.db.individual.get_individuals(
            ids=[posted_individual.id], get_full=True, get_source=True)[0]
        media_action_status = frontend.action.log_action(
            kwargs["user_id"], "create", "media", None, posted_media).to_dict()
        individual_action_status = frontend.action.log_action(
            kwargs["user_id"], "create", "individual", None, log_individual).to_dict()
    except Exception as log_ex:
        print(f"Warning: logging failed during post media: {log_ex}")
        media_action_status = {}
        individual_action_status = {}
        log_individual = None

    return {"media_action_status": media_action_status, "individual_action_status": individual_action_status,
            "fetch_individual": log_individual.to_dict() if log_individual else None}, http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts("ids", "get_full")
def get_medias(handler, path, **kwargs):
    """Get media records by IDs, optionally retrieving full associated details."""
    print("Get Media function called")

    list_params = ["ids"]
    scalar_params = ["get_full"]
    app_utils.normalize_list_params(kwargs, list_params)
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    media_list = backend.db.media.get_medias(ids=kwargs["ids"], get_full=kwargs["get_full"])
    media_list = [media.to_dict() for media in media_list]

    return media_list, http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts("ids", "keywords", "locations", "max_time", "min_time", "names", "qualities",
                   "sources", "types")
def filter_medias(handler, path, **kwargs):
    """Filter media records based on query parameters."""
    print("Filter Media function called")

    list_params = ["ids", "locations", "qualities", "types", "names", "sources", "keywords"]
    scalar_params = ["min_time", "max_time"]
    app_utils.normalize_list_params(kwargs, list_params)
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    candidate_ids = None
    if kwargs["ids"] is not None:
        candidate_ids = set(kwargs["ids"])

    # Check if any database filters are active.
    db_filters = ["min_time", "max_time", "locations", "qualities", "types", "names", "sources", "keywords"]
    has_db_filters = any(kwargs.get(key) is not None for key in db_filters)

    connection = app_utils.db_connect()
    try:
        # Only call the database filter if necessary.
        if has_db_filters:
            candidate_ids = backend.db.media.get_matching_media_ids(
                ids=candidate_ids, names=kwargs.get("names"), sources=kwargs.get("sources"),
                types=kwargs.get("types"), locations=kwargs.get("locations"),
                min_time=kwargs.get("min_time"), max_time=kwargs.get("max_time"),
                qualities=kwargs.get("qualities"), keywords=kwargs.get("keywords"),
                connection=connection)

        # For all candidate ids (or all media if None), get the media objects.
        if candidate_ids is not None:
            media_list = backend.db.media.get_medias(ids=list(candidate_ids), get_full=True, connection=connection)
        else:
            media_list = backend.db.media.get_medias(ids=None, get_full=True, connection=connection)
    finally:
        connection.close()

    media_list = [media.to_dict() for media in media_list]

    return media_list, http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts("id")
def delete_media(handler, path, **kwargs):
    """Delete a media record, its associated individual, and its image file atomically."""
    print("Delete Media function called")

    scalar_params = ["id", "user_id"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    media_id = kwargs["id"]
    user_id = kwargs.get("user_id")

    # Pre-fetch needed data before transaction
    individual_ids = backend.db.media.get_individual_ids_by_media_ids([media_id])
    if len(individual_ids) == 0:
        return app_constants.ERROR, http.HTTPStatus.OK, None

    delete_individual_id = individual_ids[0]

    original_individuals = backend.db.individual.get_individuals(
        ids=[delete_individual_id], get_full=True, get_source=False)
    if not original_individuals:
        return app_constants.ERROR, http.HTTPStatus.OK, None
    original_individual = original_individuals[0]

    original_media_list = backend.db.media.get_medias(ids=[media_id])
    if not original_media_list:
        return app_constants.ERROR, http.HTTPStatus.OK, None
    original_media = original_media_list[0]
    original_image_path = _image_path(original_media.filename)

    source_individual = backend.db.individual.get_individuals(
        ids=[delete_individual_id], get_full=True, get_source=True)[0]
    fetch_individual = None

    connection = app_utils.db_connect()
    try:
        # 1. If this is a reference individual, update the source's reference pointer.
        if backend.db.individual.is_reference_individual(original_individual):
            # Find a new reference from the remaining media individuals (exclude source itself).
            media_individuals = [ind for ind in source_individual.associated_individuals
                                 if ind.id != source_individual.id and ind.id != delete_individual_id]

            if media_individuals:
                # Sort remaining media by time to pick the oldest as the new reference.
                sorted_media = backend.db.individual.sort_individuals_except_source(
                    media_individuals, "media_time")
                new_reference_individual_id = sorted_media[0].id
            else:
                new_reference_individual_id = None

            # Update the source's pointer.
            source_individual.reference = new_reference_individual_id
            updated_source = backend.db.individual.upsert_individual(source_individual, connection)
            if updated_source == app_constants.ERROR:
                raise RuntimeError("Failed to update source individual reference pointer")

        # 2. Delete media individual's associated objects.
        for assoc_object in original_individual.alternate_names:
            if backend.db.individual.delete_associated_object(assoc_object.id, "AlternateNames", connection) == app_constants.ERROR:
                raise RuntimeError("Failed to delete alternate name")
        for assoc_object in original_individual.tags:
            if backend.db.individual.delete_associated_object(assoc_object.id, "Tags", connection) == app_constants.ERROR:
                raise RuntimeError("Failed to delete tag")
        for assoc_object in original_individual.labels:
            if backend.db.individual.delete_associated_object(assoc_object.id, "Labels", connection) == app_constants.ERROR:
                raise RuntimeError("Failed to delete label")

        # 3. Delete media individual.
        if backend.db.individual.delete_individual(original_individual.id, connection) == app_constants.ERROR:
            raise RuntimeError("Failed to delete media individual")

        # 4. Re-fetch the source individual to check for remaining media individuals.
        updated_source_list = backend.db.individual.get_individuals(
            ids=[source_individual.id], get_full=True, get_source=True, connection=connection)
        remaining_media = []
        if updated_source_list:
            updated_source = updated_source_list[0]
            remaining_media = [ind for ind in updated_source.associated_individuals
                               if ind.id != source_individual.id]

        if len(remaining_media) > 0:
            # Other sightings remain. Return the current reference individual for the UI.
            ref_ind = backend.db.individual.individual_get_reference_individual(
                updated_source, connection=connection)
            fetch_individual = ref_ind.to_dict() if ref_ind else None
        else:
            # No sightings remain. Delete the source individual and all its associated objects.
            fetch_individual = None

            for table in ["AlternateNames", "Tags", "Labels"]:
                source_objects = backend.db.individual.get_individual_associated_objects(
                    source_individual.id, table, connection=connection)
                if source_objects and source_objects != app_constants.ERROR:
                    for assoc_object in source_objects:
                        if backend.db.individual.delete_associated_object(assoc_object.id, table, connection) == app_constants.ERROR:
                            raise RuntimeError(f"Failed to delete source {table}")

            # Delete the source individual itself.
            if backend.db.individual.delete_individual(source_individual.id, connection) == app_constants.ERROR:
                raise RuntimeError("Failed to delete source individual")

        # 5. Delete the media itself
        if backend.db.media.delete_media(media_id, connection) == app_constants.ERROR:
            raise RuntimeError("Failed to delete media object")

        # 6. Commit transaction
        connection.commit()

    except Exception as ex:
        print(f"Delete media failed, rolling back: {ex}")
        try:
            connection.rollback()
        except Exception as rb_ex:
            print(f"Warning: Rollback failed: {rb_ex}")
        finally:
            connection.close()

        error_response = {
            "errors": [{"filename": "delete", "reason": "The deletion could not be completed."}],
            "message": "Delete failed. No database changes were committed."
        }
        return json.dumps(error_response), http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    connection.close()

    # DB successful -> Log action.
    try:
        action_status = frontend.action.log_action(
            user_id, "delete", "media", original_media, None).to_dict()
        frontend.action.log_action(
            user_id, "delete", "individual", original_individual, None)
    except Exception as log_ex:
        print(f"Warning: logging failed during delete: {log_ex}")
        action_status = {}

    # Image files and embeddings deletion
    try:
        os.remove(original_image_path)
    except OSError:
        print("Warning, deleting media from DB, image file not present.", original_media.filename)

    # Embedding cache entries for deleted media are left in place; they become
    # stale until the next /embeddings/generate run refreshes them.

    return {"action_status": action_status,
            "fetch_individual": fetch_individual}, http.HTTPStatus.OK, None


# Save media files to the frontend media directory.
# pylint: disable=unused-argument
@app_utils.accepts("data_url", "filename")
def save_media(handler, path, **kwargs):
    """Save uploaded media files to the frontend images directory."""
    print("Save Media function called")

    scalar_params = ["data_url", "filename"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    # Process bytes.
    path = _image_path(kwargs['filename'], filename_only=True)
    image_bytes, _ = _data_url_to_bytes(kwargs["data_url"])

    # Save media file.
    image = PIL.Image.open(io.BytesIO(image_bytes))

    # Convert image to RGB if it has an alpha channel (RGBA) or is in another unsupported mode for JPEG.
    if image.mode == 'RGBA':
        # Create a new RGB image with a white background
        new_image = PIL.Image.new("RGB", image.size, (255, 255, 255))
        new_image.paste(image, (0, 0), image)  # Paste the RGBA image using its alpha channel as mask
        image = new_image
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    image.save(path)

    return None, http.HTTPStatus.OK, None


# pylint: disable=unused-argument, too-many-locals, too-many-branches, too-many-statements, too-many-return-statements
@app_utils.accepts("items")
def upload_batch(handler, path, **kwargs):
    """Atomically validate, database-persist, and file-store a batch of uploaded images."""
    print("Upload Batch function called")

    items = kwargs.get("items", [])
    user_id = kwargs.get("user_id")

    if not items:
        return "No items provided for upload.", http.HTTPStatus.BAD_REQUEST, None
    if not isinstance(items, list) or len(items) > app_constants.MAX_UPLOAD_ITEMS:
        raise frontend.common.APIError('Upload exceeds the configured batch limit.', 413)
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('name'), str):
            raise frontend.common.APIError('Each image needs a name.')
        if not isinstance(item.get('hash'), str) or not item['hash']:
            raise frontend.common.APIError('Each image needs a hash.')
        try:
            app_utils.normalize_sex(item.get('sex'))
        except ValueError as error:
            raise frontend.common.APIError(f'Invalid value for field: sex. {error}') from error
        for key in ('labels', 'tags', 'alternate_names'):
            objects = item.get(key, [])
            if not isinstance(objects, list) or not all(isinstance(entry, dict) for entry in objects):
                raise frontend.common.APIError(f'{key} must be a list of objects.')
        _image_path(item['name'] + '.jpeg', filename_only=True)
        if item.get('data_url'):
            _data_url_to_bytes(item['data_url'])

    # -----------------------------------------------------------------------
    # Phase 1: Pre-validation — check for duplicates before any writes.
    # -----------------------------------------------------------------------
    errors = []

    # Collect batch hashes/names/filenames for within-batch duplicate checking.
    batch_hashes = {}
    batch_names = {}
    batch_filenames = {}

    for i, item in enumerate(items):
        item_hash = item.get("hash")
        item_name = item.get("name")
        item_filename = item_name + ".jpeg" if item_name else None

        # Check within-batch duplicates.
        if item_hash in batch_hashes:
            errors.append({"filename": item_name, "reason": f"Duplicate hash within this batch (same image as '{batch_hashes[item_hash]}')"})
        elif item_hash:
            batch_hashes[item_hash] = item_name

        if item_name in batch_names:
            errors.append({"filename": item_name, "reason": f"Duplicate name within this batch (same name as '{batch_names[item_name]}')"})
        elif item_name:
            batch_names[item_name] = item_name

        if item_filename in batch_filenames:
            errors.append({"filename": item_name, "reason": f"Duplicate filename within this batch"})
        elif item_filename:
            batch_filenames[item_filename] = item_name

    # Check against existing DB records.
    existing_hashes = set()
    existing_names = set()
    existing_filenames = set()

    try:
        all_media = backend.db.media.get_medias(ids=None, get_full=False)
        for media in all_media:
            if media.hash:
                existing_hashes.add(media.hash)
            if media.name:
                existing_names.add(media.name)
            if media.filename:
                existing_filenames.add(media.filename)
    except Exception as ex:
        print(f"Warning: Could not fetch existing media for pre-validation: {ex}")

    for item in items:
        item_hash = item.get("hash")
        item_name = item.get("name")
        item_filename = item_name + ".jpeg" if item_name else None

        if item_hash and item_hash in existing_hashes:
            errors.append({"filename": item_name, "reason": "Duplicate hash — this image already exists in the database"})
        if item_name and item_name in existing_names:
            errors.append({"filename": item_name, "reason": "Duplicate name — an image with this name already exists in the database"})
        if item_filename and item_filename in existing_filenames:
            errors.append({"filename": item_name, "reason": "Duplicate filename — an image file with this name already exists"})

    if errors:
        error_count = len(errors)
        total_count = len(items)
        error_response = {
            "errors": errors,
            "message": f"Upload rejected: {error_count} issue(s) found in {total_count} image(s). No images were uploaded."
        }
        return json.dumps(error_response), http.HTTPStatus.CONFLICT, None

    # -----------------------------------------------------------------------
    # Phase 2: Atomic processing — single transaction for all DB writes.
    # -----------------------------------------------------------------------
    connection = app_utils.db_connect()
    saved_files = []  # Track saved file paths for rollback.

    try:
        # Get the next individual name counter once, then increment for each image.
        # pylint: disable=protected-access
        next_name_counter = backend.db.individual._get_next_individual_name()
        if next_name_counter == app_constants.ERROR:
            raise RuntimeError("Failed to determine next individual name")

        results = []

        for i, item in enumerate(items):
            item_name = item.get("name")
            item_hash = item.get("hash")
            item_metadata = item.get("metadata")
            item_location = item.get("location")
            item_time = item.get("time")
            item_type = item.get("type")
            item_quality = item.get("quality")
            item_length = item.get("length")
            item_sex = item.get("sex")
            item_data_url = item.get("data_url")
            item_labels = item.get("labels", [])
            item_tags = item.get("tags", [])
            item_alternate_names = item.get("alternate_names", [])

            # --- Step A: Create Media record ---
            new_media = backend.db.media.Media(
                id=None, hash=item_hash, name=item_name,
                filename=(item_name + ".jpeg"), source=None, individual=None,
                metadata=item_metadata, location=item_location, time=item_time,
                type=item_type, quality=item_quality
            )
            posted_media = backend.db.media.upsert_media(new_media, connection)
            if posted_media == app_constants.ERROR:
                raise RuntimeError(f"Failed to insert media record for '{item_name}'")

            # Set media source to itself.
            posted_media.source = posted_media.id
            posted_media = backend.db.media.upsert_media(posted_media, connection)
            if posted_media == app_constants.ERROR:
                raise RuntimeError(f"Failed to update media source for '{item_name}'")

            # --- Step B: Create media Individual ---
            individual_name = str(next_name_counter + i)
            new_individual = backend.db.individual.Individual(
                id=None, name=individual_name, length=item_length,
                sex=item_sex, media=posted_media.id, source=None, reference=None
            )
            posted_individual = backend.db.individual.upsert_individual(new_individual, connection)
            if posted_individual == app_constants.ERROR:
                raise RuntimeError(f"Failed to create individual for '{item_name}'")

            # --- Step C: Create source Individual ---
            new_source = backend.db.individual.Individual(
                id=None, name=None, length=item_length,
                sex=item_sex, source=None, reference=posted_individual.id
            )
            posted_source = backend.db.individual.upsert_individual(new_source, connection)
            if posted_source == app_constants.ERROR:
                raise RuntimeError(f"Failed to create source individual for '{item_name}'")

            # Update pointers: media individual -> source, source -> itself.
            posted_individual.source = posted_source.id
            posted_individual = backend.db.individual.upsert_individual(posted_individual, connection)
            posted_source.source = posted_source.id
            posted_source = backend.db.individual.upsert_individual(posted_source, connection)
            if app_constants.ERROR in (posted_individual, posted_source):
                raise RuntimeError(f"Failed to update individual pointers for '{item_name}'")

            # --- Step D: Create associated objects (labels, tags, alternate_names) ---
            # Always add a "fresh" label to the source individual.
            fresh_label = backend.db.individual.AssociatedObject(
                individual=posted_source.id, name="fresh")
            posted_label = backend.db.individual.upsert_associated_object(
                fresh_label, "Labels", connection)
            if posted_label == app_constants.ERROR:
                raise RuntimeError(f"Failed to add 'fresh' label for '{item_name}'")

            for obj_pair in [("labels", "Labels"), ("tags", "Tags"), ("alternate_names", "AlternateNames")]:
                obj_key, table_name = obj_pair
                for preview_object in item.get(obj_key, []):
                    created_object = backend.db.individual.AssociatedObject(
                        None, individual=posted_source.id, name=preview_object.get("name"),
                        type=preview_object.get("type"), time=preview_object.get("time"))
                    posted_object = backend.db.individual.upsert_associated_object(
                        created_object, table_name, connection)
                    if posted_object == app_constants.ERROR:
                        raise RuntimeError(f"Failed to create {obj_key} for '{item_name}'")

            # --- Step E: Save image file to disk ---
            if item_data_url:
                file_path = _image_path(posted_media.filename, filename_only=True)
                saved_files.append(file_path)
                save_media(handler, path, data_url=item_data_url, filename=posted_media.filename)

            results.append({
                "media_id": posted_media.id,
                "individual_id": posted_individual.id,
                "source_id": posted_source.id,
                "name": item_name,
            })

        # --- Phase 3: Commit ---
        connection.commit()
        connection.close()

        # Log actions after successful commit (logging uses its own connections).
        for i, item in enumerate(items):
            try:
                media_obj = backend.db.media.get_medias(ids=[results[i]["media_id"]])[0]
                frontend.action.log_action(user_id, "create", "media", None, media_obj)
                log_individual = backend.db.individual.get_individuals(
                    ids=[results[i]["individual_id"]], get_full=True, get_source=True)[0]
                frontend.action.log_action(user_id, "create", "individual", None, log_individual)
            except Exception as log_ex:
                print(f"Warning: Failed to log action for '{item.get('name')}': {log_ex}")

        return {"results": results, "message": f"Successfully uploaded {len(results)} image(s)."}, http.HTTPStatus.OK, None

    except Exception as ex:
        # --- Rollback: undo all DB changes and remove saved files ---
        traceback.print_exc()
        print(f"Upload batch failed, rolling back: {ex}")
        try:
            connection.rollback()
        except Exception as rb_ex:
            print(f"Warning: Rollback failed: {rb_ex}")
        try:
            connection.close()
        except Exception:
            pass

        # Clean up any saved files.
        for file_path in saved_files:
            try:
                os.remove(file_path)
                print(f"Rollback: removed saved file {file_path}")
            except OSError:
                print(f"Warning: could not remove file during rollback: {file_path}")

        error_message = 'The upload could not be completed'
        error_response = {
            "errors": [{"filename": "batch", "reason": error_message}],
            "message": f"Upload failed: {error_message}. No images were uploaded."
        }
        return json.dumps(error_response), http.HTTPStatus.INTERNAL_SERVER_ERROR, None


def _image_path(filename, filename_only=False):
    """Apply the shared containment boundary to application image operations."""
    try:
        return app_utils.resolve_served_path(app_constants.IMAGES_DIR, filename, filename_only)
    except (ValueError, OSError) as ex:
        raise frontend.common.APIError('Invalid image filename.') from ex


def _data_url_to_bytes(data_url):
    """Convert a data URL string to raw image bytes and file extension."""
    extension = ""
    if not isinstance(data_url, str):
        raise frontend.common.APIError('Invalid image data.')
    if len(data_url) > app_constants.MAX_IMAGE_BYTES * 4 // 3 + 256:
        raise frontend.common.APIError('Image exceeds the configured size limit.', 413)
    if data_url.startswith("data:"):
        if ';' not in data_url:
            raise frontend.common.APIError('Invalid image data.')
        data_info, data_url = data_url.split(";", 1)
        mime = data_info.split(":")[-1]
        guess_extension = mimetypes.guess_extension(mime)
        if guess_extension is not None:
            extension = guess_extension

    if data_url.startswith("base64,"):
        data_url = data_url[7:]

    try:
        image_bytes = base64.b64decode(data_url, validate=True)
        if len(image_bytes) > app_constants.MAX_IMAGE_BYTES:
            raise frontend.common.APIError('Image exceeds the configured size limit.', 413)
        with PIL.Image.open(io.BytesIO(image_bytes)) as image:
            if image.width * image.height > app_constants.MAX_IMAGE_PIXELS:
                raise frontend.common.APIError('Image exceeds the configured pixel limit.', 413)
            image.verify()
    except frontend.common.APIError:
        raise
    except PIL.Image.DecompressionBombError as ex:
        raise frontend.common.APIError('Image exceeds the configured pixel limit.', 413) from ex
    except (ValueError, OSError) as ex:
        raise frontend.common.APIError('Invalid image data.') from ex
    return image_bytes, extension


def _get_exif_timestamp(time):
    """Format a datetime object into a standard EXIF timestamp string."""
    # Set timezone to utc.
    time = time.astimezone(datetime.timezone.utc)

    # Format.
    return time.strftime(app_constants.EXIF_DATETIME_FORMAT)


def _set_exif_data(image, metadata, now):
    """Embed metadata and creation timestamps into image EXIF data."""
    exif = image.getexif()

    # This seems correct, but some tools will complain about the encoding:
    # https://github.com/python-pillow/Pillow/issues/5254
    exif[PIL.ExifTags.Base.UserComment] = json.dumps(metadata)

    # Set original (taken) time.
    if "time" in metadata["image"]:
        create_time = datetime.datetime.fromtimestamp(metadata["image"]["time"], tz = datetime.timezone.utc)
        exif[PIL.ExifTags.Base.DateTimeOriginal] = _get_exif_timestamp(create_time)
        exif[PIL.ExifTags.Base.DateTimeDigitized] = _get_exif_timestamp(create_time)

    # Set creation time (now).
    exif[PIL.ExifTags.Base.DateTime] = _get_exif_timestamp(now)

    return exif
