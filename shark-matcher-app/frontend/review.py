"""Review API handlers surfacing unmatch and merge suggestions from the inference service."""

import http

import backend.db.individual
import frontend.individual
import frontend.inference_client

# Filter parameters forwarded to filter_individuals, synchronized with webui/review.js.
_FILTER_KEYS = [
    "min_length", "max_length", "min_time", "max_time",
    "min_associated", "max_associated", "sex", "locations",
    "qualities", "max_qualities", "alternate_names", "labels",
    "tags", "allow_unknown", "keywords",
]

_SUPPORTED_REVIEW_TYPES = ("unmatch_suggestions", "merge_suggestions")


def get_review(handler, path, **kwargs):
    """Retrieve a page of unmatch or merge suggestions from the inference service cache."""
    print("Get review function called")

    page = max(1, int(kwargs.get("page") or 1))
    page_size = max(1, int(kwargs.get("page_size") or 50))
    model = kwargs.get("model") or None
    review_type = kwargs.get("review_type") or "unmatch_suggestions"
    if review_type not in _SUPPORTED_REVIEW_TYPES:
        review_type = "unmatch_suggestions"

    raw_suggestions = frontend.inference_client.get_suggestions(
        review_type=review_type, model=model)
    suggestions_available = bool(raw_suggestions)

    # Filter candidate IDs using search-style filters, skipping intersection if no filter was applied.
    filter_kwargs = {key: kwargs.get(key) for key in _FILTER_KEYS}
    candidate_ids, _, _ = frontend.individual.filter_individuals(
        None, None, return_ids_only=True, **filter_kwargs)
    candidate_set = set(candidate_ids) if candidate_ids is not None else None

    if review_type == "merge_suggestions":
        return _build_merge_response(
            raw_suggestions, candidate_set, page, page_size,
            suggestions_available)
    return _build_unmatch_response(
        raw_suggestions, candidate_set, page, page_size,
        suggestions_available)


def _build_unmatch_response(raw_suggestions, candidate_set, page, page_size,
                            suggestions_available):
    """Build a paginated response of unmatch suggestions grouped by source individual."""
    # Group by label (source individual id), preserving worst-first order.
    grouped_order = []
    grouped = {}
    for image_id, k_value, label in raw_suggestions:
        if label not in grouped:
            grouped[label] = {
                "flagged_image_id": image_id,
                "k_value": k_value,
                "flagged_image_k_map": {image_id: k_value},
            }
            grouped_order.append(label)
        else:
            entry = grouped[label]
            entry["flagged_image_k_map"][image_id] = k_value
            if k_value > entry["k_value"]:
                entry["k_value"] = k_value
                entry["flagged_image_id"] = image_id

    if candidate_set is not None:
        grouped_order = [label for label in grouped_order if label in candidate_set]

    start = (page - 1) * page_size
    end = start + page_size
    page_labels = grouped_order[start:end]
    has_more = end < len(grouped_order)

    if not page_labels:
        return ({
            "has_more": False,
            "page": page,
            "suggestions": [],
            "suggestions_available": suggestions_available,
        }, http.HTTPStatus.OK, None)

    individuals = backend.db.individual.get_individuals(
        ids=page_labels, get_full=True, get_source=True)
    individual_by_id = {
        ind.id: ind for ind in individuals
        if isinstance(ind, backend.db.individual.Individual)
    }

    # Drop stale suggestions: the retriever's annotation snapshot can reference
    # individuals or media that have since been deleted or reassigned.
    suggestions = []
    for label in page_labels:
        individual = individual_by_id.get(label)
        if individual is None:
            print(f"Skipping stale suggestion: individual {label} not in database.")
            continue

        entry = grouped[label]
        valid_media_ids = {m.id for m in (individual.associated_medias or [])}
        valid_k_map = {
            mid: k for mid, k in entry["flagged_image_k_map"].items()
            if mid in valid_media_ids
        }
        if not valid_k_map:
            print(f"Skipping stale suggestion: individual {label} no longer "
                  f"associated with any flagged image.")
            continue

        flagged_image_id, k_value = max(valid_k_map.items(), key=lambda kv: kv[1])

        suggestions.append({
            "individual": individual.to_dict(),
            "flagged_image_id": flagged_image_id,
            "k_value": k_value,
            "flagged_image_k_map": valid_k_map,
        })

    return ({
        "has_more": has_more,
        "page": page,
        "suggestions": suggestions,
        "suggestions_available": suggestions_available,
    }, http.HTTPStatus.OK, None)


def _build_merge_response(raw_suggestions, candidate_set, page, page_size,
                          suggestions_available):
    """Build a paginated response of candidate individual merge suggestion pairs."""
    # Filter raw merge suggestion pairs, keeping only those where both individuals pass the filter.
    pairs = []
    for label_a, label_b, distance in raw_suggestions:
        if candidate_set is not None and (
                label_a not in candidate_set or label_b not in candidate_set):
            continue
        pairs.append((int(label_a), int(label_b), float(distance)))

    start = (page - 1) * page_size
    end = start + page_size
    page_pairs = pairs[start:end]
    has_more = end < len(pairs)

    if not page_pairs:
        return ({
            "has_more": False,
            "page": page,
            "suggestions": [],
            "suggestions_available": suggestions_available,
        }, http.HTTPStatus.OK, None)

    # Hydrate all unique individuals on this page in a single query.
    unique_ids = list({label for pair in page_pairs for label in pair[:2]})
    individuals = backend.db.individual.get_individuals(
        ids=unique_ids, get_full=True, get_source=True)
    individual_by_id = {
        ind.id: ind for ind in individuals
        if isinstance(ind, backend.db.individual.Individual)
    }

    suggestions = []
    for label_a, label_b, distance in page_pairs:
        ind_a = individual_by_id.get(label_a)
        ind_b = individual_by_id.get(label_b)
        if ind_a is None or ind_b is None:
            print(f"Skipping stale merge suggestion: pair ({label_a}, "
                  f"{label_b}) has an individual missing from the database.")
            continue
        suggestions.append({
            "individual_a": ind_a.to_dict(),
            "individual_b": ind_b.to_dict(),
            "distance": distance,
        })

    return ({
        "has_more": has_more,
        "page": page,
        "suggestions": suggestions,
        "suggestions_available": suggestions_available,
    }, http.HTTPStatus.OK, None)
