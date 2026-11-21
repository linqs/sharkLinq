"""API handlers for individual and associated object operations."""

import http
from copy import deepcopy

from core import app_constants

import backend.db.action
import backend.db.individual
import backend.db.media
from core import app_utils
import frontend.action
import frontend.common
import frontend.inference_client
import demo.runtime

# Mapping from frontend table names to backend DB table names.
ASSOCIATED_OBJECT_TABLE_MAP = {
    "alternate_name": "AlternateNames",
    "tag": "Tags",
    "label": "Labels",
}

# Individual filter fields, shared by the search, match and export handlers.
FILTER_LIST_FIELDS = ("medias", "locations", "qualities", "max_qualities", "alternate_names",
                      "tags", "labels", "ids", "names", "keywords")
FILTER_SCALAR_FIELDS = ("min_length", "max_length", "min_time", "max_time", "min_associated",
                        "max_associated", "sex", "allow_unknown")
PAGE_FIELDS = ("paginate", "page", "page_size", "attribute", "order")


# pylint: disable=unused-argument
@app_utils.accepts("id", "name", "length", "sex", "source", "reference", "media")
def post_individual(handler, path, **kwargs):
    """Create or update an individual's direct properties."""
    print("Post Individual function called")

    scalar_params = ["id", "name", "length", "sex", "source", "media", "reference", "user_id"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    if not app_utils.valid_name(kwargs["name"]):
        return "Invalid name.", http.HTTPStatus.BAD_REQUEST, None

    # If updating an existing individual, fetch original and do a single upsert (atomic by itself).
    if kwargs["id"] is not None:
        original_individuals = backend.db.individual.get_individuals(ids=[kwargs["id"]], get_full=True, get_source=True)
        if not original_individuals:
            return "Individual not found.", http.HTTPStatus.NOT_FOUND, None
        original_individual = original_individuals[0]

        updated_individual = backend.db.individual.Individual(length=kwargs["length"], sex=kwargs["sex"],
                                               id=kwargs["id"], name=kwargs["name"],
                                               source=kwargs["source"], media=kwargs["media"],
                                               reference=kwargs["reference"])
        updated_individual = backend.db.individual.upsert_individual(updated_individual)
        if updated_individual == app_constants.ERROR:
            return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

        # Get the full media individual for logging.
        log_individual = backend.db.individual.get_individuals(ids=[updated_individual.id], get_full=True,
                                                get_source=False)[0]
        action_status = frontend.action.log_action(kwargs["user_id"], "update",
                                                   "individual", original_individual, log_individual)
        return {"action_status": action_status.to_dict()}, http.HTTPStatus.OK, None

    # Atomically create the media individual, source individual, and "fresh" label.
    new_individual_obj = backend.db.individual.Individual(length=kwargs["length"], sex=kwargs["sex"],
                                           id=None, name=kwargs["name"],
                                           source=kwargs["source"], media=kwargs["media"],
                                           reference=kwargs["reference"])
    connection = app_utils.db_connect()
    try:
        # 1. Create media individual.
        posted_individual = backend.db.individual.upsert_individual(new_individual_obj, connection)
        if posted_individual == app_constants.ERROR:
            raise RuntimeError("Failed to create media individual.")

        # 2. Create source individual and update both individuals' source pointers.
        new_source_individual, updated_media_individual = _create_source_individual(posted_individual, connection)

        # 3. Add "fresh" label to source individual.
        created_label = backend.db.individual.AssociatedObject(individual=new_source_individual.id, name="fresh")
        if backend.db.individual.upsert_associated_object(created_label, "Labels", connection) == app_constants.ERROR:
            raise RuntimeError("Failed to add 'fresh' label.")

        connection.commit()

    except Exception as ex:
        print(f"Post individual (create) failed, rolling back: {ex}")
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
            ids=[updated_media_individual.id], get_full=True, get_source=False)[0]
        action_status = frontend.action.log_action(
            kwargs["user_id"], "create", "individual", None, log_individual).to_dict()
    except Exception as log_ex:
        print(f"Warning: logging failed during post individual: {log_ex}")
        action_status = {}

    return {"action_status": action_status}, http.HTTPStatus.OK, None


@app_utils.timeit
# pylint: disable=unused-argument
@app_utils.accepts("ids", "names", "keywords", "get_full", "get_source")
def get_individuals(handler, path, **kwargs):
    """Get individuals by ID, name, or keywords with optional details."""
    print("Get individuals function called")

    # Reformat parameters via utils.
    list_params = ["ids", "names", "keywords"]
    scalar_params = ["get_full", "get_source"]
    app_utils.normalize_list_params(kwargs, list_params)
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    ids = kwargs["ids"]

    # If names/keywords are present, ensure ids is a list to append to
    if kwargs["names"] or kwargs["keywords"]:
        if ids is None:
            ids = []

    connection = app_utils.db_connect()
    try:
        # If names are given, use those to find ids to get.
        if kwargs["names"]:
            name_individual_ids = backend.db.individual.get_matching_individual_ids(
                names=kwargs["names"], source_only=False, connection=connection)
            ids.extend(name_individual_ids)

        # If keywords are given, use those to find ids to get.
        if kwargs["keywords"]:
            keywords = kwargs["keywords"]
            keyword_individual_ids = _get_keyword_individual_ids(keywords, connection=connection)
            ids.extend(keyword_individual_ids)

        # Get individuals.
        individual_list = backend.db.individual.get_individuals(
            ids=ids, get_full=kwargs["get_full"], get_source=kwargs["get_source"],
            connection=connection)
    finally:
        connection.close()

    # All individual objects to dicts for frontend.
    return_individual_list = []
    for individual in individual_list:
        if isinstance(individual, backend.db.individual.Individual):
            return_individual_list.append(individual.to_dict())
        else:
            print("Warning, a requested individual returned None.")

    return return_individual_list, http.HTTPStatus.OK, None


@app_utils.timeit
# pylint: disable=unused-argument, too-many-locals, too-many-branches, too-many-statements
@app_utils.accepts(*FILTER_LIST_FIELDS, *FILTER_SCALAR_FIELDS, *PAGE_FIELDS)
def filter_individuals(handler, path, connection=None, **kwargs):
    """Filter individuals based on provided query parameters and return matching source records."""
    print("Filter Individual function called")

    # Pop the ids-only short-circuit before normalize_*; it isn't a real filter.
    return_ids_only = kwargs.pop("return_ids_only", False)

    # Robust parameter normalization via utils.
    app_utils.normalize_list_params(kwargs, list(FILTER_LIST_FIELDS))
    app_utils.normalize_scalar_params(kwargs, list(FILTER_SCALAR_FIELDS + PAGE_FIELDS))

    # Open one shared connection for the entire filter pipeline so that every
    # sequential read query reuses the same SQLite connection.
    _local_connection = connection is None
    if _local_connection:
        connection = app_utils.db_connect()

    try:
        # Initialize a set of individual IDs that meets filter constraints.
        # Initialize to None to represent "Everything" until a filter restricts it.
        candidate_individual_ids = None

        # If a list of ids is given, filter this list.
        if kwargs["ids"] is not None:
            candidate_individual_ids = set(backend.db.individual.individual_ids_get_source_individual_ids(
                kwargs["ids"], connection=connection))

        # If keywords are given, use those to find ids to get.
        if kwargs["keywords"] is not None:
            keyword_individual_ids = _get_keyword_individual_ids(kwargs["keywords"], connection=connection)
            keyword_individual_ids = set(backend.db.individual.individual_ids_get_source_individual_ids(
                keyword_individual_ids, connection=connection))

            if candidate_individual_ids is None:
                candidate_individual_ids = keyword_individual_ids
            else:
                candidate_individual_ids = candidate_individual_ids.intersection(keyword_individual_ids)


        # If names are given, filter by names. TODO check if this is really needed
        if kwargs["names"] is not None:
            print("Filtering for names", kwargs["names"])
            name_individual_ids = backend.db.individual.get_matching_individual_ids(
                names=kwargs["names"], source_only=False, connection=connection)
            name_individual_ids = set(backend.db.individual.individual_ids_get_source_individual_ids(
                name_individual_ids, connection=connection))

            if candidate_individual_ids is None:
                candidate_individual_ids = name_individual_ids
            else:
                candidate_individual_ids = candidate_individual_ids.intersection(name_individual_ids)

        # Filter by associated objects.
        for filter_obj in ((kwargs["alternate_names"], "AlternateNames"), (kwargs["tags"], "Tags"), (kwargs["labels"], "Labels")):
            if filter_obj[0] is not None:
                # Filtering for any associated object.
                if "*" in filter_obj[0]:
                    obj_ids = backend.db.individual.get_matching_associated_object_individual_ids(
                            table=filter_obj[1], any=True, connection=connection)
                # Keep only individual ids that pass all object filters.
                else:
                    obj_ids = backend.db.individual.get_matching_associated_object_individual_ids(
                            names=filter_obj[0], table=filter_obj[1], connection=connection)

                if candidate_individual_ids is None:
                    candidate_individual_ids = set(obj_ids)
                else:
                    candidate_individual_ids = candidate_individual_ids.intersection(obj_ids)


        if ((kwargs["min_time"] is not None) or (kwargs["max_time"] is not None)
                or (kwargs["locations"] is not None) or (kwargs["qualities"] is not None)
                or (kwargs.get("max_qualities") is not None)):
            candidate_media_ids = backend.db.media.get_matching_media_ids(
                min_time=kwargs["min_time"], max_time=kwargs["max_time"],
                locations=kwargs["locations"], qualities=kwargs["qualities"],
                max_qualities=kwargs.get("max_qualities"),
                allow_unknown=kwargs["allow_unknown"], connection=connection)

            # Keep only individual ids that pass all applied filters.
            candidate_media_individual_ids = backend.db.media.get_individual_ids_by_media_ids(
                list(candidate_media_ids), connection=connection)
            candidate_media_individual_ids = backend.db.individual.individual_ids_get_source_individual_ids(
                candidate_media_individual_ids, connection=connection)

            if candidate_individual_ids is None:
                candidate_individual_ids = set(candidate_media_individual_ids)
            else:
                 candidate_individual_ids = candidate_individual_ids.intersection(candidate_media_individual_ids)

        # Only resolve to source IDs if a filter has narrowed the candidate set.
        if candidate_individual_ids is not None:
            candidate_individual_ids = set(
                backend.db.individual.individual_ids_get_source_individual_ids(
                    list(candidate_individual_ids), connection=connection))

        # Filter by (source) individual properties.
        if (kwargs["min_length"] is not None or kwargs["max_length"] is not None or
            kwargs["sex"] is not None):
            individual_property_candidate_ids = backend.db.individual.get_matching_individual_ids(
                min_length=kwargs["min_length"], max_length=kwargs["max_length"], sex=kwargs["sex"],
                allow_unknown=kwargs["allow_unknown"], source_only=True, connection=connection)

            if candidate_individual_ids is None:
                 candidate_individual_ids = set(individual_property_candidate_ids)
            else:
                 candidate_individual_ids = candidate_individual_ids.intersection(individual_property_candidate_ids)

        # Filter by association count.
        if kwargs["min_associated"] is not None or kwargs["max_associated"] is not None:
            assoc_count_ids = backend.db.individual.get_source_individual_ids_by_association_count(
                min_associated=kwargs["min_associated"], max_associated=kwargs["max_associated"],
                connection=connection)
            if candidate_individual_ids is None:
                 candidate_individual_ids = set(assoc_count_ids)
            else:
                 candidate_individual_ids = candidate_individual_ids.intersection(assoc_count_ids)

        # Return early when the caller only needs matching IDs.
        if return_ids_only:
            return candidate_individual_ids, http.HTTPStatus.OK, None

        # Sort candidate IDs in SQL, slice to the requested page, then load full records for that page.
        paginate = kwargs.get("paginate") in (True, "true", "True", 1, "1")
        if paginate:
            page = max(1, int(kwargs.get("page") or 1))
            page_size = max(1, int(kwargs.get("page_size") or 50))
            if page_size > app_constants.MAX_PAGE_SIZE:
                raise frontend.common.APIError('Page size exceeds the configured limit.')
            attribute = kwargs.get("attribute") or "name"
            order = kwargs.get("order") or "ascending"
            offset = (page - 1) * page_size

            sorted_ids = backend.db.individual.get_sorted_source_individual_ids(
                ids=candidate_individual_ids, attribute=attribute, order=order,
                connection=connection)
            total_count = len(sorted_ids)
            page_ids = sorted_ids[offset: offset + page_size]

            page_individuals = backend.db.individual.get_individuals(
                ids=page_ids, get_full=True, get_source=True, connection=connection)

            # Preserve the sort order returned by the helper; get_individuals
            # returns rows in SQL order, not input order.
            ind_by_id = {ind.id: ind for ind in page_individuals
                         if isinstance(ind, backend.db.individual.Individual)}
            ordered_page = [ind_by_id[pid] for pid in page_ids if pid in ind_by_id]

            return {
                "results": [ind.to_dict() for ind in ordered_page],
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "has_more": (offset + page_size) < total_count,
            }, http.HTTPStatus.OK, None

        # Fetch full source individuals for candidate IDs, or all individuals if no filters were applied.
        ids_to_fetch = None
        if candidate_individual_ids is not None:
            ids_to_fetch = list(candidate_individual_ids)

        # Fetch full individuals as expected by the frontend.
        individual_list = backend.db.individual.get_individuals(ids=ids_to_fetch,
                                                    get_full=True, get_source=True,
                                                    connection=connection)

    finally:
        if _local_connection:
            connection.close()

    # All individual objects to dicts for frontend.
    if kwargs.get("return_objects") is True:
        return individual_list, http.HTTPStatus.OK, None

    return_individual_list = []
    for individual in individual_list:
        if isinstance(individual, backend.db.individual.Individual):
            return_individual_list.append(individual.to_dict())
        else:
            print("Warning, a requested individual returned None.")

    return return_individual_list, http.HTTPStatus.OK, None


def _has_source_individual(individual_id):
    """Return whether an individual exists together with its source individual."""
    return bool(backend.db.individual.get_individuals(ids=[individual_id], get_full=True, get_source=False)
                and backend.db.individual.get_individuals(ids=[individual_id], get_full=True, get_source=True))


# pylint: disable=unused-argument, too-many-branches
@app_utils.accepts("id")
def delete_individual(handler, path, **kwargs):
    """Delete an individual and clean up orphaned source records atomically."""
    print("Delete Individual function called")

    scalar_params = ["id", "user_id"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    # Pre-fetch needed data before transaction.
    original_individuals = backend.db.individual.get_individuals(ids=[kwargs["id"]], get_full=True,
                                                  get_source=False)
    if not original_individuals:
        return app_constants.ERROR, http.HTTPStatus.OK, None
    original_individual = original_individuals[0]

    if not isinstance(original_individual, backend.db.individual.Individual):
        return app_constants.ERROR, http.HTTPStatus.OK, None

    if not _has_source_individual(kwargs["id"]):
        return "Individual not found.", http.HTTPStatus.NOT_FOUND, None
    source_individual = backend.db.individual.get_individuals(ids=[kwargs["id"]], get_full=True,
                                                   get_source=True)[0]

    fetch_individual = None
    connection = app_utils.db_connect()
    try:
        # 1. If this is a reference individual, update the source's reference pointer.
        if backend.db.individual.is_reference_individual(original_individual):
            if backend.db.individual.set_reference_individual(source_individual, None,
                                                              connection=connection) == app_constants.ERROR:
                raise RuntimeError("Failed to update source individual reference pointer.")

        # 2. Delete media individual's associated objects.
        for assoc_object in original_individual.alternate_names:
            if backend.db.individual.delete_associated_object(assoc_object.id, "AlternateNames", connection) == app_constants.ERROR:
                raise RuntimeError("Failed to delete alternate name.")
        for assoc_object in original_individual.tags:
            if backend.db.individual.delete_associated_object(assoc_object.id, "Tags", connection) == app_constants.ERROR:
                raise RuntimeError("Failed to delete tag.")
        for assoc_object in original_individual.labels:
            if backend.db.individual.delete_associated_object(assoc_object.id, "Labels", connection) == app_constants.ERROR:
                raise RuntimeError("Failed to delete label.")

        # 3. Delete the media individual.
        if backend.db.individual.delete_individual(original_individual.id, connection) == app_constants.ERROR:
            raise RuntimeError("Failed to delete media individual.")

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
            for table in ["AlternateNames", "Tags", "Labels"]:
                source_objects = backend.db.individual.get_individual_associated_objects(
                    source_individual.id, table, connection=connection)
                if source_objects and source_objects != app_constants.ERROR:
                    for assoc_object in source_objects:
                        if backend.db.individual.delete_associated_object(assoc_object.id, table, connection) == app_constants.ERROR:
                            raise RuntimeError(f"Failed to delete source {table}.")

            # Delete the source individual itself (only if distinct from the already-deleted individual).
            if source_individual.id != original_individual.id:
                if backend.db.individual.delete_individual(source_individual.id, connection) == app_constants.ERROR:
                    raise RuntimeError("Failed to delete source individual.")

        # 5. Commit transaction.
        connection.commit()

    except Exception as ex:
        print(f"Delete individual failed, rolling back: {ex}")
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
        action_status = frontend.action.log_action(
            kwargs["user_id"], "delete", "individual", original_individual, None).to_dict()
    except Exception as log_ex:
        print(f"Warning: logging failed during delete individual: {log_ex}")
        action_status = {}

    return {"action_status": action_status, "fetch_individual": fetch_individual}, http.HTTPStatus.OK, None


# pylint: disable=unused-argument, too-many-locals, too-many-branches
@app_utils.accepts("major_id", "minor_id", "preview")
def merge_individuals(handler, path, **kwargs):
    """Merge a minor individual into a major reference individual."""
    print("Merge Individual function called")

    frontend.common.require_fields(kwargs, "major_id", "minor_id")

    # Get original individuals.
    connection = app_utils.db_connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        original_individuals = backend.db.individual.get_individuals(ids=[kwargs["major_id"], kwargs["minor_id"]],
                                                      get_full=True, get_source=True, connection=connection)
        # Bulk reads are ordered by the database, not the caller's major/minor order.
        by_member = {member.id: source for source in original_individuals
                     for member in source.associated_individuals}
        major_source_individual = by_member.get(int(kwargs["major_id"]))
        minor_source_individual = by_member.get(int(kwargs["minor_id"]))

        # If the individuals are not found, return an error.
        if not isinstance(major_source_individual, backend.db.individual.Individual) or not isinstance(
                minor_source_individual, backend.db.individual.Individual):
            raise RuntimeError("Major or minor individual not found.")
        if major_source_individual.id == minor_source_individual.id:
            raise ValueError("Cannot merge a shark with itself.")
        source_ids = [major_source_individual.id, minor_source_individual.id]
        preview_individual = kwargs.get("preview")

        # Adjust all minor individual associated media individuals to point to the major individual.
        for associated_individual in minor_source_individual.associated_individuals:
            if backend.db.individual.is_media_individual(associated_individual):
                if backend.db.individual.set_source_individual(associated_individual, major_source_individual, connection=connection) == app_constants.ERROR:
                     raise RuntimeError("Failed to set associated media individual pointer.")

        # Delete the major and minor individual associated source objects, they will be added back by reading the preview.
        for source_individual in [major_source_individual, minor_source_individual]:
            for assoc_object in source_individual.alternate_names:
                if assoc_object.individual in source_ids:
                    if backend.db.individual.delete_associated_object(assoc_object.id, "AlternateNames", connection=connection) == app_constants.ERROR:
                        raise RuntimeError("Failed to delete alternate name.")
            for assoc_object in source_individual.tags:
                if assoc_object.individual in source_ids:
                    if backend.db.individual.delete_associated_object(assoc_object.id, "Tags", connection=connection) == app_constants.ERROR:
                        raise RuntimeError("Failed to delete tag.")
            for assoc_object in source_individual.labels:
                if assoc_object.individual in source_ids:
                    if backend.db.individual.delete_associated_object(assoc_object.id, "Labels", connection=connection) == app_constants.ERROR:
                        raise RuntimeError("Failed to delete label.")

        # Delete the minor source individual.
        delete_status = backend.db.individual.delete_individual(minor_source_individual.id, connection=connection)
        if delete_status != app_constants.SUCCESS:
            raise RuntimeError("Failed to delete minor source individual.")

        # Add new associated objects and remove deleted ones to match the posted individual with the preview.
        for object_pair in [("alternate_names", "AlternateNames"), ("tags", "Tags"),
                            ("labels", "Labels")]:
            for preview_object in preview_individual[object_pair[0]]:
                if preview_object.get("time") is None:
                    preview_object["time"] = None
                created_object = backend.db.individual.AssociatedObject(
                    None, individual=major_source_individual.id, name=preview_object["name"],
                    type=preview_object["type"], time=preview_object["time"])
                posted_object = backend.db.individual.upsert_associated_object(created_object,
                                                                          object_pair[1], connection=connection)
                if posted_object == app_constants.ERROR:
                    raise RuntimeError("Failed to upsert new associated object.")

        # Get the merged individual for the property updates.
        merged_major_source_individual = backend.db.individual.get_individuals(ids=[major_source_individual.id],
                                                                get_full=True, get_source=True, connection=connection)[0]

        # Update the reference individual if the preview individual specifies a different reference individual than currently set.
        if preview_individual["reference"] != merged_major_source_individual.reference:
            if backend.db.individual.set_reference_individual(merged_major_source_individual,
                                                 preview_individual["reference"], connection=connection) == app_constants.ERROR:
                raise RuntimeError("Failed to update reference individual.")

        # Update the major individual to reflect any property or reference changes requested with the merge.
        merged_major_source_individual.sex = preview_individual["sex"]
        merged_major_source_individual.length = preview_individual["length"]
        updated_major_individual = backend.db.individual.upsert_individual(merged_major_source_individual, connection=connection)
        if updated_major_individual == app_constants.ERROR:
            raise RuntimeError("Failed to update major individual properties.")

        demo_results = []
        if demo.runtime.current() is not None:
            demo_results = demo.runtime.current().grade_merge(updated_major_individual.id, connection)
        connection.commit()

    except Exception as ex:
        print(f"Merge individuals failed, rolling back: {ex}")
        try:
            connection.rollback()
        except Exception as rb_ex:
            print(f"Warning: Rollback failed: {rb_ex}")
        finally:
            connection.close()

        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    connection.close()

    # DB Successful - fetch updated individual and log action
    try:
        # Get the fully updated major individual for logging.
        log_individual = backend.db.individual.get_individuals(ids=[updated_major_individual.id],
                                                get_full=True, get_source=True)[0]

        # Get the reference individual to fetch.
        fetch_individual = backend.db.individual.individual_get_reference_individual(log_individual).to_dict()

        # Log action.
        action_status_minor = frontend.action.log_action(kwargs["user_id"], "merge", "individual",
                                                         minor_source_individual, log_individual).to_dict()
        action_status_major = frontend.action.log_action(kwargs["user_id"], "merge", "individual",
                                                         major_source_individual, log_individual).to_dict()
    except Exception as log_ex:
        print(f"Warning: logging failed during merge: {log_ex}")
        action_status_minor = {}
        action_status_major = {}
        fetch_individual = None

    response = {"action_status_major": action_status_major,
                "action_status_minor": action_status_minor,
                "fetch_individual": fetch_individual}
    if demo.runtime.current() is not None:
        response["demo_results"] = demo_results
    return response, http.HTTPStatus.OK, None


def detach_individual(individual_id, connection, labels=("unmatched",)):
    """Detach an observation using the caller's transaction, also used by demo rounds."""
    original = backend.db.individual.get_individuals(
        ids=[individual_id], get_full=True, get_source=False, connection=connection)[0]
    source = backend.db.individual.get_individuals(
        ids=[individual_id], get_full=True, get_source=True, connection=connection)[0]
    if not isinstance(original, backend.db.individual.Individual) or not isinstance(
            source, backend.db.individual.Individual):
        raise ValueError("Individual not found.")
    if backend.db.individual.is_reference_individual(original, connection=connection):
        reference_id = backend.db.individual.set_reference_individual(source, None, connection)
        if reference_id == app_constants.ERROR:
            raise RuntimeError("Failed to update source reference.")
        reference = backend.db.individual.get_individuals(
            ids=[reference_id], get_full=False, get_source=False, connection=connection)[0]
    else:
        reference = backend.db.individual.individual_get_reference_individual(source, connection)
    fetch_individual = reference.to_dict() if reference is not None else None
    # Keep the original object for ordinary action logging.
    new_source, updated = _create_source_individual(deepcopy(original), connection)
    if app_constants.ERROR in (new_source, updated):
        raise RuntimeError("Failed to create unmatched source.")
    for name in labels:
        label = backend.db.individual.AssociatedObject(individual=new_source.id, name=name)
        if backend.db.individual.upsert_associated_object(label, "Labels", connection) == app_constants.ERROR:
            raise RuntimeError("Failed to label unmatched source.")
    if demo.runtime.current() is not None:
        demo.runtime.current().grade_merge(new_source.id, connection)
    return original, updated, fetch_individual


# pylint: disable=unused-argument
@app_utils.accepts("id")
def unmatch_individual(handler, path, **kwargs):
    """Unmatch a media individual into a new standalone source individual atomically."""
    print("Unmatch Individual function called")

    frontend.common.require_fields(kwargs, "id")
    if not _has_source_individual(kwargs["id"]):
        return "Individual not found.", http.HTTPStatus.NOT_FOUND, None

    fetch_individual = None
    connection = app_utils.db_connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        original_individual, updated_media_individual, fetch_individual = detach_individual(
            kwargs["id"], connection)
        connection.commit()

    except Exception as ex:
        print(f"Unmatch individual failed, rolling back: {ex}")
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
            ids=[updated_media_individual.id], get_full=True, get_source=False)[0]
        action_status = frontend.action.log_action(kwargs["user_id"], "unmatch", "individual",
                                                   original_individual, log_individual).to_dict()
    except Exception as log_ex:
        print(f"Warning: logging failed during unmatch: {log_ex}")
        action_status = {}

    return {"action_status": action_status,
            "fetch_individual": fetch_individual}, http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts("id")
def update_reference(handler, path, **kwargs):
    """Update a source individual's reference pointer to the specified pointer individual."""
    print("Individual update reference function called")

    frontend.common.require_fields(kwargs, "id")

    # Get the source individual that we're updating.
    original_source_individuals = backend.db.individual.get_individuals(ids=[kwargs["id"]], get_full=True, get_source=True)
    if not original_source_individuals:
        return "Individual not found.", http.HTTPStatus.NOT_FOUND, None
    original_source_individual = original_source_individuals[0]

    # If this individual is not found, return an error.
    if not isinstance(original_source_individual, backend.db.individual.Individual):
        return app_constants.ERROR, http.HTTPStatus.OK, None

    # Update the reference, this is the individual to display now.
    new_reference_individual_id = backend.db.individual.set_reference_individual(original_source_individual,
                                                                  kwargs["id"])
    fetch_individual = backend.db.individual.get_individuals(ids=[new_reference_individual_id], get_full=False,
                                              get_source=False)[0].to_dict()

    # Get the full individual for logging.
    log_individual = backend.db.individual.get_individuals(ids=[kwargs["id"]], get_full=True,
                                            get_source=True)[0]

    # Log action.
    action_status = frontend.action.log_action(kwargs["user_id"], "update", "individual",
                                               original_source_individual, log_individual)

    return {"action_status": action_status.to_dict(),
            "fetch_individual": fetch_individual}, http.HTTPStatus.OK, None


# Object functions.
# TODO change to get objects plural.
# pylint: disable=unused-argument
@app_utils.accepts("id", "individual", "name", "table", "time", "type")
def post_object(handler, path, **kwargs):
    """Create or update an associated object in the specified table."""
    print("Post object function called")

    frontend.common.require_fields(kwargs, "table", "name", "individual")
    table_name = ASSOCIATED_OBJECT_TABLE_MAP.get(kwargs["table"])
    if not table_name:
        return "Invalid table specified.", http.HTTPStatus.BAD_REQUEST, None

    if not app_utils.valid_name(kwargs["name"]):
        return "Invalid name.", http.HTTPStatus.BAD_REQUEST, None

    # Update object in database.
    updated_object = backend.db.individual.AssociatedObject(
        id=kwargs.get("id"), individual=kwargs["individual"], name=kwargs["name"],
        type=kwargs.get("type"), time=kwargs.get("time"))
    posted_object = backend.db.individual.upsert_associated_object(updated_object, table_name)
    if posted_object == app_constants.ERROR:
        return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    # Log action.
    action_status = frontend.action.log_action(kwargs["user_id"], "create", kwargs["table"],
                                               None, posted_object)

    return {"action_status": action_status.to_dict()}, http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts("ids", "table")
def get_objects(handler, path, **kwargs):
    """Get associated objects by IDs from the specified table."""
    frontend.common.require_fields(kwargs, "table")
    table_name = ASSOCIATED_OBJECT_TABLE_MAP.get(kwargs["table"])
    if not table_name:
        return "Invalid table specified.", http.HTTPStatus.BAD_REQUEST, None

    # Robust parameter normalization via utils.
    list_params = ["ids"]
    app_utils.normalize_list_params(kwargs, list_params)

    objects = backend.db.individual.get_associated_objects(ids=kwargs["ids"], table=table_name)
    if not objects:
        return "Objects not found.", http.HTTPStatus.NOT_FOUND, None
    object_list = [object_obj.to_dict() for object_obj in objects]

    return object_list, http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts("individual", "table")
def filter_objects(handler, path, **kwargs):
    """Get associated objects of a specific table for an individual."""
    frontend.common.require_fields(kwargs, "table", "individual")
    table_name = ASSOCIATED_OBJECT_TABLE_MAP.get(kwargs["table"])
    if not table_name:
        return "Invalid table specified.", http.HTTPStatus.BAD_REQUEST, None

    objects = backend.db.individual.get_individual_associated_objects(individual_id=kwargs["individual"], table=table_name)
    if not objects:
        return "Objects not found.", http.HTTPStatus.NOT_FOUND, None
    object_list = [object_obj.to_dict() for object_obj in objects]

    return object_list, http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts("id", "table")
def delete_object(handler, path, **kwargs):
    """Delete an associated object from the specified table."""
    print("Delete object function called")

    frontend.common.require_fields(kwargs, "table", "id")
    table_name = ASSOCIATED_OBJECT_TABLE_MAP.get(kwargs["table"])
    if not table_name:
        return "Invalid table specified.", http.HTTPStatus.BAD_REQUEST, None

    original_objects = backend.db.individual.get_associated_objects(ids=[kwargs["id"]], table=table_name)
    if not original_objects:
        return "Object to delete not found.", http.HTTPStatus.NOT_FOUND, None
    original_object = original_objects[0]

    delete_status = backend.db.individual.delete_associated_object(kwargs["id"], table_name)
    if delete_status != app_constants.SUCCESS:
        return "Failed to delete object.", http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    # Log action.
    action_status = frontend.action.log_action(kwargs["user_id"], "delete", kwargs["table"],
                                               original_object, None)

    return {"action_status": action_status.to_dict()}, http.HTTPStatus.OK, None


@app_utils.timeit
def _get_individual_with_distances(query_id, connection=None):
    """Retrieve an individual along with feature distance metrics to associated individuals."""
    print("Get individual distances function called")
    query_individual = backend.db.individual.get_individuals(ids=[query_id], get_full=False, get_source=False, connection=connection)[0]

    # Get the source individual to get all associated individuals.
    source_individual = backend.db.individual.get_individuals(ids=[query_id], get_full=True, get_source=True, connection=connection)[0]
    if not isinstance(query_individual, backend.db.individual.Individual) or not isinstance(source_individual, backend.db.individual.Individual):
        print("Warning, individual for matching individual not found in database.")
        return app_constants.ERROR

    # Get neighbor images for the query individual's image via the inference service.
    query_neighbors = frontend.inference_client.get_matches(
        image_id=query_individual.media, retrieval_method=0)
    neighbor_ids = query_neighbors["ids"]
    neighbor_distances = query_neighbors["distances"]
    neighbor_distance_dict = dict(zip(neighbor_ids, neighbor_distances))

    # Each associated individual that is a media individual gets a distance value.
    for associated_individual in source_individual.associated_individuals:
        if backend.db.individual.is_media_individual(associated_individual):
            associated_individual_distance = neighbor_distance_dict.get(associated_individual.media)
            if associated_individual_distance is None:
                associated_individual.distance = 1000
            else:
                associated_individual.distance = associated_individual_distance

    return {"query_individual": query_individual, "source_individual": source_individual}


@app_utils.timeit
def _get_keyword_individual_ids(keywords, connection=None):
    """Return all individual IDs matching the provided keywords across metadata tables."""
    print("Get keyword individuals function called")

    kw_in, values = app_utils.build_sql_placeholders(keywords, "kw")

    sql = f"""
        SELECT individual FROM AlternateNames
            WHERE individual >= 0 AND name COLLATE NOCASE IN {kw_in}
        UNION
        SELECT individual FROM Tags
            WHERE individual >= 0 AND name COLLATE NOCASE IN {kw_in}
        UNION
        SELECT individual FROM Labels
            WHERE individual >= 0 AND name COLLATE NOCASE IN {kw_in}
        UNION
        SELECT id FROM Individuals
            WHERE id >= 0 AND name COLLATE NOCASE IN {kw_in}
        UNION
        SELECT ind.id FROM Individuals ind
            JOIN Media m ON ind.media = m.id
            WHERE ind.id >= 0 AND m.name COLLATE NOCASE IN {kw_in}
        UNION
        SELECT id FROM Individuals
            WHERE id >= 0 AND id COLLATE NOCASE IN {kw_in};
    """

    result = app_utils.execute_sql(sql, values, connection=connection)
    if result == app_constants.ERROR:
        return []
    return [row[0] for row in result]


def _create_source_individual(media_individual, connection=None):
    """Create a new source individual container for a standalone media individual."""
    # New source individual is created referencing the media individual.
    new_source_individual = backend.db.individual.Individual(id=None, name=None, length=media_individual.length,
                                              sex=media_individual.sex, source=None,
                                              reference=media_individual.id)
    new_source_individual = backend.db.individual.upsert_individual(new_source_individual, connection)
    if new_source_individual == app_constants.ERROR:
        print("Warning: error creating new source individual.")

    # Update the media and source individual's pointer to the newly created source individual.
    media_individual.source = new_source_individual.id
    posted_media_individual = backend.db.individual.upsert_individual(media_individual, connection)
    new_source_individual.source = new_source_individual.id
    posted_source_individual = backend.db.individual.upsert_individual(new_source_individual, connection)
    if app_constants.ERROR in (posted_media_individual, posted_source_individual):
        print("Warning: error updating new media individual after creating source individual.")

    return posted_source_individual, posted_media_individual
