"""API handlers and helpers to store and read audit actions in the log."""

import datetime
import http
import json

from core import app_utils
import backend.db.action
import demo.runtime


class EmptyObject:
    """Represent an empty audit action entity."""

    def __init__(self):
        """Initialize an empty object with a sentinel identifier."""
        self.id = -1

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False):  # pylint: disable=unused-argument
        """Represent the object as a dictionary."""
        item_dict = {"id": self.id}
        return item_dict

    def print(self):
        """Print the object representation as a dictionary."""
        print(self.to_dict())


# pylint: disable=unused-argument
def get_actions(handler, path, **kwargs):
    """Get a list of actions matching request parameters."""
    print("Get action function called")

    # Robust parameter normalization via utils.
    list_params = ["ids"]
    app_utils.normalize_list_params(kwargs, list_params)

    # Get the actions.
    action_list = backend.db.action.get_actions(ids=kwargs["ids"])
    action_list = [action_obj.to_dict() for action_obj in action_list]

    return action_list, http.HTTPStatus.OK, None


# pylint: disable=unused-argument, too-many-locals, too-many-branches, too-many-nested-blocks
@app_utils.accepts("min_time", "max_time", "object", "action", "original", "final", "author",
                   "ids", "keywords", "paginate", "page", "page_size", "attribute", "order")
def filter_actions(handler, path, **kwargs):
    """Filter actions based on provided parameters and pagination settings."""
    print("Filter action function called")

    # Robust parameter normalization via utils.
    list_params = ["ids", "keywords"]
    scalar_params = ["original", "final", "author", "min_time", "max_time", "action", "object",
                     "paginate", "page", "page_size", "attribute", "order"]

    app_utils.normalize_list_params(kwargs, list_params, splitter=" ")
    app_utils.normalize_scalar_params(kwargs, scalar_params)

    # Initialize a set of IDs that meets filter constraints.
    # If a list of ids is given, filter this list, otherwise filter starting with all possible.
    if kwargs["ids"] is not None:
        candidate_ids = set(kwargs["ids"])
    else:
        candidate_ids = None

    # Check if any database filters are active.
    db_filters = ["original", "final", "author", "min_time", "max_time", "action", "object"]
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
            action=kwargs["action"], object=kwargs["object"],
        )

        if kwargs["keywords"] is not None:
            # Keyword filter must run in Python; fetch all sorted candidates then filter.
            all_ids = backend.db.action.get_matching_action_ids(
                **filter_args, attribute=attribute, order=order
            )
            all_actions = backend.db.action.get_actions(all_ids)
            all_actions = _apply_keyword_filter_actions(all_actions, kwargs["keywords"])
            total_count = len(all_actions)
            page_actions = all_actions[offset: offset + page_size]
        else:
            total_count = backend.db.action.get_matching_action_ids(
                **filter_args, count_only=True
            )
            page_ids = backend.db.action.get_matching_action_ids(
                **filter_args, attribute=attribute, order=order,
                offset=offset, limit=page_size
            )
            page_actions = backend.db.action.get_actions(page_ids)

        return {
            "results": [a.to_dict() for a in page_actions],
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "has_more": (offset + page_size) < total_count,
        }, http.HTTPStatus.OK, None

    # ── Unpaginated path (original behavior) ─────────────────────────────────
    # Only call the database filter if necessary.
    if has_db_filters:
        candidate_ids = backend.db.action.get_matching_action_ids(
            ids=candidate_ids, original=kwargs["original"], final=kwargs["final"],
            author=kwargs["author"], min_time=kwargs["min_time"], max_time=kwargs["max_time"],
            action=kwargs["action"], object=kwargs["object"], keywords=kwargs["keywords"]
        )

    # For all candidate ids (or all actions if None), get the action objects.
    # Note: If candidate_ids is None, get_actions gets all actions.
    if candidate_ids is not None:
        action_list = backend.db.action.get_actions(list(candidate_ids))
    else:
        action_list = backend.db.action.get_actions(None)

    # If a set of keywords is given, filter the list to only include those. Must do this here
    # not in SQL due to injection threats.
    if kwargs["keywords"] is not None:
        action_list = _apply_keyword_filter_actions(action_list, kwargs["keywords"])

    action_list = [action_obj.to_dict() for action_obj in action_list]

    return action_list, http.HTTPStatus.OK, None


def log_action(user_id, action_type, object_type, original_object, final_object, connection=None):
    """Log an audit action in the database actions table."""
    if demo.runtime.current() is not None:
        return EmptyObject()

    # Handle empty objects.
    if original_object is None:
        original_object = EmptyObject()
    if final_object is None:
        final_object = EmptyObject()

    # Build log elements.
    action_time = datetime.datetime.now().timestamp()
    action_notes = json.dumps({"original": original_object.to_dict(compact=True),
                               "final": final_object.to_dict(compact=True)})

    # Build action object and update log in database.
    action = backend.db.action.Action(id=None, original=original_object["id"],
                                      final=final_object["id"], action=action_type,
                                      object=object_type, author=user_id, time=action_time,
                                      notes=action_notes)

    return backend.db.action.upsert_action(action, connection=connection)


def _apply_keyword_filter_actions(action_list, keywords):
    """Return actions whose notes contain at least one of the keywords."""
    filtered = []
    skip_keys = ["hash"]
    for action_obj in action_list:
        note_string = ""
        notes = json.loads(action_obj.notes)
        for note_dict in notes.values():
            for key, value in note_dict.items():
                if key not in skip_keys:
                    if isinstance(value, list):
                        for item in value:
                            note_string += "," + str(item).lower()
                    else:
                        note_string += "," + str(value).lower()
        for keyword in keywords:
            if keyword.lower() in note_string:
                filtered.append(action_obj)
                break
    return filtered
