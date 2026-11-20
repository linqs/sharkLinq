"""API handlers for proposing, reviewing, and deleting database changes."""

import datetime
import http
import json

from core import app_constants
from core import app_utils
import backend.db.action
import backend.db.media
import backend.db.proposal
import frontend.action
import frontend.common
import frontend.user


# pylint: disable=unused-argument
@app_utils.accepts("id", "original", "final", "action", "object", "status")
def post_proposal(handler, path, **kwargs):
    """Create a new proposal or update the status of an existing proposal."""
    print("Post Proposal function called")

    scalar_params = ["id", "original", "final", "action", "object", "status", "user_id"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    # Check for new creation, if no id is given.
    if kwargs["id"] is None:
        proposal_id = None
    else:
        # If updating a proposal, use existing.
        proposal_id = kwargs["id"]

    proposal_time = datetime.datetime.now().timestamp()

    # If creating a new proposal (id is None), validate original/final objects based on action.
    if proposal_id is None:
        original_object = kwargs["original"]
        final_object = kwargs["final"]

        # No original object must mean a create or upload.
        if original_object is None:
            if kwargs["action"] != "create" and kwargs["action"] != "upload":
                return "Invalid proposal.", http.HTTPStatus.BAD_REQUEST, None
            original_object = frontend.action.EmptyObject().to_dict()

        # No final object must mean a delete.
        if final_object is None:
            if kwargs["action"] != "delete":
                return "Invalid proposal.", http.HTTPStatus.BAD_REQUEST, None
            final_object = frontend.action.EmptyObject().to_dict()

        for field, proposed_object in (("original", original_object), ("final", final_object)):
            if "id" not in proposed_object:
                raise frontend.common.APIError(f"Invalid value for field: {field}.")

        # Build log elements.
        action_time = datetime.datetime.now().timestamp()
        action_notes = json.dumps({"original": _to_compact(original_object),
                                   "final": _to_compact(final_object)})

        # Build proposal object and update log in database.
        proposal = backend.db.proposal.Proposal(id=None, original=original_object["id"],
                                                final=final_object["id"], action=kwargs["action"],
                                                object=kwargs["object"], status=kwargs["status"],
                                                author=kwargs["user_id"], time=action_time, notes=action_notes)

        # Post the proposal to the database.
        final_proposal = backend.db.proposal.upsert_proposal(proposal)
        if final_proposal == app_constants.ERROR:
            return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    # Updates only modify proposal status, preserving original content.
    else:
        # If updating, get original proposal and modify.
        requester = frontend.user.get_authenticated_user(handler)
        if requester is not None and frontend.user.role_dict.get(requester.role, -1) < 2:
            return 'Only admins can update proposals.', http.HTTPStatus.FORBIDDEN, None
        original_proposals = backend.db.proposal.get_proposals(ids=[kwargs["id"]])
        if not original_proposals:
            return "Proposal not found.", http.HTTPStatus.NOT_FOUND, None
        original_proposal = original_proposals[0]
        original_proposal.status = kwargs["status"]
        final_proposal = backend.db.proposal.upsert_proposal(original_proposal)
        if final_proposal == app_constants.ERROR:
            return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    return None, http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts("ids")
def get_proposals(handler, path, **kwargs):
    """Get a list of proposals matching the requested IDs."""
    print("Get proposals function called")

    # Robust parameter normalization via utils.
    list_params = ["ids"]
    app_utils.normalize_list_params(kwargs, list_params)

    # Get the proposals.
    proposal_list = backend.db.proposal.get_proposals(ids=kwargs["ids"])
    proposal_list = [proposal_obj.to_dict() for proposal_obj in proposal_list]

    return proposal_list, http.HTTPStatus.OK, None


# pylint: disable=unused-argument, too-many-locals, too-many-branches, too-many-nested-blocks
@app_utils.accepts("min_time", "max_time", "object", "action", "original", "final", "status",
                   "author", "ids", "keywords", "paginate", "page", "page_size", "attribute",
                   "order")
def filter_proposals(handler, path, **kwargs):
    """Filter proposals based on query parameters and pagination settings."""
    print("Filter proposals function called")

    # Robust parameter normalization via utils.
    list_params = ["ids", "keywords"]
    scalar_params = ["original", "final", "author", "min_time", "max_time", "action", "object",
                     "status", "paginate", "page", "page_size", "attribute", "order"]

    app_utils.normalize_list_params(kwargs, list_params, splitter=" ")
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    # Initialize a set of IDs that meet the filter constraints.
    # If a list of ids is given, filter this list, otherwise filter starting with all possible.
    if kwargs["ids"] is not None:
        candidate_ids = set(kwargs["ids"])
    else:
        candidate_ids = None

    # Check if any database filters are active.
    db_filters = ["original", "final", "author", "min_time", "max_time", "action", "object", "status"]
    has_db_filters = any(kwargs.get(key) is not None for key in db_filters)

    # ── Paginated path ────────────────────────────────────────────────────────
    paginate = kwargs.get("paginate") in (True, "true", "True", 1, "1")
    if paginate:
        page = max(1, int(kwargs.get("page") or 1))
        page_size = max(1, int(kwargs.get("page_size") or 50))
        attribute = kwargs.get("attribute") or "time"
        order = kwargs.get("order") or "descending"
        offset = (page - 1) * page_size

        filter_args = dict(
            ids=candidate_ids,
            original=kwargs["original"], final=kwargs["final"],
            author=kwargs["author"], min_time=kwargs["min_time"], max_time=kwargs["max_time"],
            action=kwargs["action"], object=kwargs["object"], status=kwargs["status"],
        )

        if kwargs["keywords"] is not None:
            # Keyword filter must run in Python; fetch all sorted candidates then filter.
            all_ids = backend.db.proposal.get_matching_proposal_ids(
                **filter_args, attribute=attribute, order=order
            )
            all_proposals = backend.db.proposal.get_proposals(all_ids)
            all_proposals = _apply_keyword_filter_proposals(all_proposals, kwargs["keywords"])
            total_count = len(all_proposals)
            page_proposals = all_proposals[offset: offset + page_size]
        else:
            total_count = backend.db.proposal.get_matching_proposal_ids(
                **filter_args, count_only=True
            )
            page_ids = backend.db.proposal.get_matching_proposal_ids(
                **filter_args, attribute=attribute, order=order,
                offset=offset, limit=page_size
            )
            page_proposals = backend.db.proposal.get_proposals(page_ids)

        return {
            "results": [p.to_dict() for p in page_proposals],
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "has_more": (offset + page_size) < total_count,
        }, http.HTTPStatus.OK, None

    # ── Unpaginated path (original behavior) ─────────────────────────────────
    if has_db_filters:
        candidate_ids = backend.db.proposal.get_matching_proposal_ids(
            ids=candidate_ids, original=kwargs["original"], final=kwargs["final"],
            author=kwargs["author"], min_time=kwargs["min_time"], max_time=kwargs["max_time"],
            action=kwargs["action"], object=kwargs["object"], status=kwargs["status"],
            keywords=kwargs["keywords"]
        )

    # For all candidate ids, get the proposals.
    # Note: If candidate_ids is None, get_proposals can handle it (fetch all or limit).
    if candidate_ids is not None:
        proposal_list = backend.db.proposal.get_proposals(list(candidate_ids))
    else:
        proposal_list = backend.db.proposal.get_proposals(None)

    # If a set of keywords is given, filter the list to only include those. Must do this here
    # not in SQL due to injection threats.
    if kwargs["keywords"] is not None:
        proposal_list = _apply_keyword_filter_proposals(proposal_list, kwargs["keywords"])

    # Convert to dictionaries for the frontend.
    proposal_list = [proposal_obj.to_dict() for proposal_obj in proposal_list]

    return proposal_list, http.HTTPStatus.OK, None


# pylint: disable=unused-argument
@app_utils.accepts("id")
def delete_proposal(handler, path, **kwargs):
    """Delete an existing proposal by ID."""
    print("Delete proposal function called")

    scalar_params = ["id"]
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    requester = frontend.user.get_authenticated_user(handler)
    if requester is not None:
        proposals = backend.db.proposal.get_proposals(ids=[kwargs['id']])
        if not proposals:
            return 'Proposal not found.', http.HTTPStatus.NOT_FOUND, None
        if proposals[0].author != requester.id:
            return 'Can only delete your own proposals.', http.HTTPStatus.FORBIDDEN, None
    delete_status = backend.db.proposal.delete_proposal(kwargs["id"])
    if delete_status != app_constants.SUCCESS:
        return "Failed to delete proposal.", http.HTTPStatus.INTERNAL_SERVER_ERROR, None

    return delete_status, http.HTTPStatus.OK, None


def _apply_keyword_filter_proposals(proposal_list, keywords):
    """Return proposals whose notes contain at least one of the keywords."""
    filtered = []
    skip_keys = ["hash"]
    for proposal_obj in proposal_list:
        object_string = ""
        notes = json.loads(proposal_obj.notes)
        for note_dict in notes.values():
            for key, value in note_dict.items():
                if key not in skip_keys:
                    if isinstance(value, list):
                        for item in value:
                            object_string += "," + str(item).lower()
                    else:
                        object_string += "," + str(value).lower()
        for keyword in keywords:
            if keyword.lower() in object_string:
                filtered.append(proposal_obj)
                break
    return filtered


def _to_compact(obj):
    """Convert a proposal object or dictionary to a compact serializable representation."""
    if obj is None:
        return None

    # Keys to skip when converting to compact dictionary.
    skip_keys = ["source", "media"]
    name_keys = ["associated_medias", "alternate_names", "labels", "tags"]
    id_keys = ["associated_individuals", "associated_sharks"]

    # If it's already a dictionary, process it
    if isinstance(obj, dict):
        final_dict = {}
        for key, value in obj.items():
            # Only include specific keys that we want to keep
            if key not in skip_keys:
                if key in name_keys:
                    object_names = []
                    for item in value:
                        if isinstance(item, dict):
                            if "name" in item.keys():
                                name = item["name"]
                                if name is not None and name != "":
                                    object_names.append(name)
                        else:
                            # Item is already a primitive (frontend already compacted it)
                            if item is not None and item != "":
                                object_names.append(item)

                    final_dict[key] = object_names
                elif key in id_keys:
                    object_ids = []
                    for item in value:
                        if isinstance(item, dict):
                            if "id" in item.keys():
                                obj_id = item["id"]
                                if obj_id is not None and obj_id != "":
                                    object_ids.append(obj_id)
                        else:
                            # Item is already a primitive (frontend already compacted it)
                            if item is not None and item != "":
                                object_ids.append(item)

                    final_dict[key] = object_ids
                else:
                    final_dict[key] = value
        return final_dict

    # If it's a list, convert each item
    if isinstance(obj, list):
        print("Warning, object is a list")
        return [_to_compact(item) for item in obj]

    # If it's an object with a to_dict method, use it
    if hasattr(obj, 'to_dict'):
        print("Warning, object has a to_dict method")
        return _to_compact(obj.to_dict())

    # For any other type, return as-is
    print("Warning, object is a primitive")
    return obj
