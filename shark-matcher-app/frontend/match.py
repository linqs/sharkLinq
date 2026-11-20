# pylint: disable=protected-access
"""API handlers for candidate matching and match filtering."""

import http

from core import app_constants
from core import app_utils
import backend.db.individual
import backend.db.media
import frontend.action
import frontend.common
import frontend.individual
import frontend.inference_client


@app_utils.timeit
@app_utils.accepts("name", "image_name", "method", "model", "allow_unknown")
def get_matches(handler, path, **kwargs):
    """Get ranked candidate matching individuals for a query individual or image."""
    print("Get Matches function called")

    scalar_params = ["allow_unknown"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    connection = app_utils.db_connect()
    try:
        # Get neighbors, calling this helper.
        try:
            (neighbor_ids, _, _,
             neighbor_distance_dict, query_individual) = _get_neighbors(handler, path, connection=connection, **kwargs)
        except frontend.common.APIError:
            raise
        except Exception:  # pylint: disable=broad-exception-caught
            if kwargs.get("image_name") is not None:
                raise
            print("Warning, got no neighbors for unknown individual.")
            return ({"matches": False, "match_suggestions": [], "query_individual": None},
                    http.HTTPStatus.OK, None)

        # If no neighbor images are found, mark matches as false while returning candidates.
        if len(neighbor_ids) == 0:
            matches = False
        else:
            matches = True

        # All individuals are fetched and then ordered by distance.
        neighbor_individual_list = backend.db.individual.get_individuals(
            ids=None, get_full=True, get_source=True, connection=connection)
    finally:
        connection.close()

    # Sort neighbor individual list by distance
    neighbor_individual_list = _process_individual_distances(neighbor_individual_list, neighbor_distance_dict)

    # All individual objects to dicts for frontend.
    return_individual_list = []
    for individual in neighbor_individual_list:
        if isinstance(individual, backend.db.individual.Individual):
            return_individual_list.append(individual.to_dict())
        else:
            print("Warning, a requested individual returned None.")

    return ({"matches": matches, "match_suggestions": return_individual_list,
             "query_individual": query_individual}, http.HTTPStatus.OK, None)


@app_utils.timeit
@app_utils.accepts("name", "image_name", "method", "model",
                   *frontend.individual.FILTER_LIST_FIELDS, *frontend.individual.FILTER_SCALAR_FIELDS)
def filter_matches(handler, path, **kwargs):
    """Get matching individuals that satisfy the specified metadata filter criteria."""
    print("Filter Matches function called")

    # Robust parameter normalization via utils.
    list_params = ["medias", "locations", "qualities", "max_qualities", "alternate_names", "tags", "labels", "ids", "names", "keywords"]
    scalar_params = ["min_length", "max_length", "min_time", "max_time", "min_associated", "max_associated", "sex", "allow_unknown"]

    app_utils.normalize_list_params(kwargs, list_params)
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    # Check if any filter parameters are actually set, if not, skip filtering.
    filter_keys = list_params + scalar_params

    has_filters = False
    for key in filter_keys:
        val = kwargs.get(key)
        if val is not None:
            if key == "allow_unknown":
                if val is False:
                    has_filters = True
            else:
                has_filters = True

    if not has_filters:
        return get_matches(handler, path, **kwargs)

    connection = app_utils.db_connect()
    try:
        # If there are filters, get the neighbors first to have the distance dict.
        try:
            (_, _, _,
             neighbor_distance_dict, query_individual) = _get_neighbors(handler, path, connection=connection, **kwargs)
        except frontend.common.APIError:
            raise
        except Exception:  # pylint: disable=broad-exception-caught
            if kwargs.get("image_name") is not None:
                raise
            print("Warning, got no neighbors for unknown individual.")
            return ({"matches": False, "match_suggestions": [], "query_individual": None},
                    http.HTTPStatus.OK, None)

        # If no neighbors were found, there are no embeddings for the query.
        has_matches = len(neighbor_distance_dict) > 0

        # Filter individuals based on the text/metadata filters, sharing the connection.
        filtered_individual_list = frontend.individual.filter_individuals(
            handler, path, connection=connection, return_objects=True, **kwargs)
    finally:
        connection.close()

    # Add distances to the filtered individuals and sort them.
    # Since filter_individuals returns (individual_list, status, error), we take index 0.
    final_individual_list = _process_individual_distances(filtered_individual_list[0], neighbor_distance_dict)

    # All individual objects to dicts for frontend.
    return_individual_list = []
    for individual in final_individual_list:
        if isinstance(individual, backend.db.individual.Individual):
            return_individual_list.append(individual.to_dict())

    return ({"matches": has_matches,
             "match_suggestions": return_individual_list,
             "query_individual": query_individual},
            http.HTTPStatus.OK, None)


# TODO, future function to score compatibility of two individuals based on metadata, a form of symbolic ranking in addition to the neural network based ranking.
def score_compatibility(query_individual, result_individual):
    """Assign a compatibility score to a pair of individuals based on metadata compatibility."""
    print(query_individual)
    print(result_individual)
    sex_compatibility = True
    # Unknown sexes are compatible.
    # Unknown sexes are compatible.
    if query_individual["sex"] != "U" and result_individual["sex"] != "U":
        # Known opposite sexes are incompatible.
        if query_individual["sex"] != result_individual["sex"]:
            sex_compatibility = False

    # Length compatilbility.

    # Location compatibility.

    # Associated object compatibility.

    return sex_compatibility


@app_utils.timeit
def _get_neighbors(handler, _path, connection=None, **kwargs):
    """Retrieve neighbor media IDs and distances for a query individual from the inference service."""
    print("Get neighbors helper function called")
    print(f"Model: {kwargs.get('model')}")

    allow_unknown = kwargs.get("allow_unknown")
    frontend.common.require_fields(kwargs, "method")
    if kwargs.get("image_name") is None:
        frontend.common.require_fields(kwargs, "name")

    # Determine the query individual, checking whether name corresponds to media or individual.
    if kwargs.get('image_name') is not None:
        image_ids = backend.db.media.get_matching_media_ids(
            names=[kwargs['image_name']], connection=connection)
        if not image_ids:
            raise frontend.common.APIError('Query image not found.', http.HTTPStatus.NOT_FOUND)
        if len(image_ids) != 1:
            raise frontend.common.APIError('Query image name is ambiguous.', http.HTTPStatus.CONFLICT)
        individual_ids = backend.db.media.get_individual_ids_by_media_ids(
            ids=image_ids, connection=connection)
        if len(individual_ids) != 1:
            raise frontend.common.APIError('Query image has no unique individual.', http.HTTPStatus.CONFLICT)
        name_individual_id = individual_ids[0]
    elif len(kwargs["name"]) >= 10:
        try:
            name_media_ids = backend.db.media.get_matching_media_ids(names=[kwargs["name"]], allow_unknown=allow_unknown, connection=connection)
            name_individual_id = backend.db.media.get_individual_ids_by_media_ids(ids=name_media_ids, connection=connection)[0]
        except Exception:  # pylint: disable=broad-exception-caught
            print("Warning, match for media name attempted, query media not found.")
            return app_constants.ERROR
    # If no valid image name is given, assume you're searching for a individual name.
    else:
        try:
            name_individual_id = backend.db.individual.get_matching_individual_ids(names=[kwargs["name"]], source_only=False, allow_unknown=allow_unknown, connection=connection)[0]
        except Exception:  # pylint: disable=broad-exception-caught
            print("Warning, match for individual name attempted, query media not found.")
            return app_constants.ERROR

    # Get the query individual object, both the media and source individual.
    queries = frontend.individual._get_individual_with_distances(query_id=name_individual_id, connection=connection)
    if not isinstance(queries["query_individual"], backend.db.individual.Individual) or \
            not isinstance(queries["source_individual"], backend.db.individual.Individual):
        print("Warning, individual for matching individual not found in database.")
        return app_constants.ERROR

    # Look up the retrieval method integer and call the inference service.
    retrieval_method_int = app_constants.MATCH_METHOD_DICT[kwargs["method"]]
    neighbors = frontend.inference_client.get_matches(
        image_id=queries["query_individual"].media,
        retrieval_method=retrieval_method_int,
        model=kwargs.get("model"),
    )

    neighbor_ids = neighbors["ids"]
    neighbor_distances = neighbors["distances"]
    neighbor_labels = neighbors.get("labels", [])
    neighbor_distance_dict = dict(zip(neighbor_ids, neighbor_distances))
    return (neighbor_ids, neighbor_distances, neighbor_labels,
            neighbor_distance_dict, queries["source_individual"].to_dict())


def _process_individual_distances(individual_list, neighbor_distance_dict):
    """Assign minimum distances to candidate individuals and sort by distance."""
    for individual in individual_list:
        # Find the minimum distance or distance of the closest image from each individual.
        individual_min_distance = 100000
        if isinstance(individual, backend.db.individual.Individual):
            # Add the distance to the query image for each of the media individuals of this source individual.
            for associated_individual in individual.associated_individuals[1:]:
                if not backend.db.individual.is_media_individual(associated_individual):
                    continue

                neighbor_distance = neighbor_distance_dict.get(associated_individual.media)
                if neighbor_distance is None:
                    neighbor_distance = 1000

                associated_individual.distance = neighbor_distance
                individual_min_distance = min(individual_min_distance, associated_individual.distance)
        individual.distance = individual_min_distance

    # Sort neighbor individual list by name first (so ties in distance are ordered by name)
    def sort_key_name(ind):
        try:
            return (0, int(ind.name))
        except (ValueError, TypeError):
            return (1, str(ind.name))

    individual_list.sort(key=sort_key_name)

    # Sort neighbor individual list by distance
    individual_list.sort(key=lambda individual: individual.distance)

    return individual_list
