"""
Action class and database CRUD functions for action objects.
"""

from core import app_utils
from core import app_constants

class Action: # pylint: disable=too-many-instance-attributes
    """
    Action class.
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, redefined-builtin
    def __init__(self, id=None, original=None, final=None, action=None, object=None,
                 author=None, time=None, notes=None):
        """Initialize an Action record with the given attributes."""
        self.id = id
        # Note for original and final, filtering on "" gets enpty originals/finals.
        self.original = original
        self.final = final
        self.action = action
        self.object = object
        self.author = author
        self.time = time
        self.notes = notes

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False): # pylint: disable=unused-argument
        """Convert the Action object into a dictionary representation."""
        item_dict = {"id": self.id,
                "original": self.original,
                "final": self.final,
                "action": self.action,
                "object": self.object,
                "author": self.author,
                "time": self.time,
                "notes": self.notes}

        return item_dict

    def print(self):
        """Print the dictionary representation of the Action object."""
        print(self.to_dict())

def db_to_action(db_object):
    """Convert a database row tuple into an Action object."""
    if db_object is None:
        return app_constants.ERROR

    action_id = db_object[0]
    original = db_object[1]
    final = db_object[2]
    action = db_object[3]
    target_object = db_object[4]
    author = db_object[5]
    time = db_object[6]
    notes = db_object[7]

    return_object = Action(id=action_id, original=original, final=final, action=action,
                           object=target_object, author=author, time=time, notes=notes)

    return return_object

# Create/update functions.
def upsert_action(action, connection=None):
    """Insert or update an action record in the database."""
    data = ({"id": action.id,
            "original": action.original,
            "final": action.final,
            "action": action.action,
            "object": action.object,
            "author": action.author,
            "time": action.time,
            "notes": action.notes})
    sql = """INSERT INTO Actions (id, original, final, action, object, author, time, notes)
    VALUES (:id, :original, :final, :action, :object, :author, :time, :notes)
    ON CONFLICT(id) DO UPDATE SET original = :original, final = :final, action = :action,
    object = :object, author = :author, time = :time, notes = :notes RETURNING *;
    """

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR
    return db_to_action(result[0])

# @app_utils.timeit
def get_actions(ids=None, limit=None, connection=None):
    """Retrieve action records from the database by ID list or query limit."""
    # 1. Handle when id list is provided.
    if ids is not None:
        if not isinstance(ids, list):
             ids = [ids]
        
        ids_to_fetch = ids[:limit] if limit is not None else ids
        if not ids_to_fetch:
            return []

        # Fetch action objects by IDs.
        placeholders, values = app_utils.build_sql_placeholders(ids_to_fetch, "id")
        sql = f"""SELECT id, original, final, action, object, author, time, notes FROM Actions
                WHERE id IN {placeholders};"""
        action_rows = app_utils.execute_sql(sql, values, connection=connection)
        if action_rows == app_constants.ERROR:
            return []

        actions_dict = {row[0]: db_to_action(row) for row in action_rows}

        # Return the final list of action objects, preserving original order.
        return [actions_dict[action_id] for action_id in ids_to_fetch if action_id in actions_dict]
    
    # If no IDs provided, fetch all with limit.
    limit_string = ""
    if limit is not None:
        limit_string = f" LIMIT {limit}"
    
    sql = f"""SELECT id, original, final, action, object, author, time, notes FROM Actions{limit_string};"""
    action_rows = app_utils.execute_sql(sql, {}, connection=connection)

    if action_rows == app_constants.ERROR:
        return []
    
    return [db_to_action(row) for row in action_rows]

# Columns allowed in ORDER BY (whitelist to prevent SQL injection via column names).
_ACTION_SORT_COLUMNS = frozenset({'id', 'time', 'author', 'action', 'object'})

# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
# pylint: disable=redefined-builtin, unused-argument
def get_matching_action_ids(ids=None, original=None, object=None, final=None, author=None,
                            min_time=None, max_time=None, action=None, keywords=None,
                            allow_unknown=False,
                            attribute=None, order=None, offset=0, limit=None,
                            count_only=False, connection=None):
    """Retrieve action IDs matching the specified filter, sorting, and pagination criteria."""
    # Build WHERE clauses and query data.
    where_clauses = []
    keyword_clause = ""
    data = {}

    # Value filters.
    for item in [["action", action, "==", "action"], ["author", author, "==", "author"],
                   ["original", original, "==", "original"], ["object", object, "==", "object"],
                   ["final", final, "==", "final"], ["time", min_time, ">=", "min_time"],
                   ["time", max_time, "<=", "max_time"]]:
        if item[1] is not None and item[1] != "":
            # Allow unknown.
            prefix = ""
            if allow_unknown is True:
                prefix = f"{item[0]} IS NULL OR "
            else:
                prefix = f"{item[0]} IS NOT NULL AND "

            sql_ids, sql_values = app_utils.build_sql_placeholders(
                [item[1]], basename=f"x_{item[3]}")
            where_clauses.append(f"({prefix}{item[0]} {item[2]} {sql_ids})")
            data = data | sql_values

    # List filters.
    for item in [["id", ids, "id"]]:
        if item[1] is not None and item[1] != "":
            sql_ids, sql_values = app_utils.build_sql_placeholders(
                item[1], basename=f"x_{item[2]}")
            where_clauses.append(f"({item[0]} COLLATE NOCASE in {sql_ids})")
            data = data | sql_values

    # Build where clauses string.
    where_string = ""
    if len(where_clauses) == 0 and keyword_clause != "":
        where_string = " WHERE " + keyword_clause[:-4]
    elif len(where_clauses) != 0:
        where_string = " WHERE " + keyword_clause + app_utils.list_to_string(
            where_clauses, " AND ")

    # COUNT-only mode: return total matches without fetching IDs.
    if count_only:
        sql = f"SELECT COUNT(*) FROM Actions{where_string};"
        result = app_utils.execute_sql(sql, data, connection=connection)
        if result == app_constants.ERROR:
            return 0
        return result[0][0] if result else 0

    # ORDER BY (column name is whitelist-validated, not parameterised).
    sort_col = attribute if attribute in _ACTION_SORT_COLUMNS else 'time'
    sort_dir = 'DESC' if order == 'descending' else 'ASC'
    tiebreaker = '' if sort_col == 'id' else f', id {sort_dir}'
    order_string = f" ORDER BY {sort_col} {sort_dir}{tiebreaker}"

    # LIMIT / OFFSET for server-side pagination.
    pagination_string = ""
    if limit is not None:
        pagination_string = " LIMIT :pg_limit OFFSET :pg_offset"
        data = data | {"pg_limit": int(limit), "pg_offset": int(offset) if offset else 0}

    # Build SQL statement.
    sql = f"SELECT id FROM Actions{where_string}{order_string}{pagination_string};"

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR
    return [item[0] for item in result]

# Delete functions.
def delete_action(action_id, connection=None):
    """Delete an action record from the database by its ID."""
    if app_utils.id_exists(action_id, "Actions", connection=connection) is False:
        return app_constants.ERROR

    data = {"id": action_id}
    sql = """DELETE FROM Actions where id = :id;"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR
    return app_constants.SUCCESS
