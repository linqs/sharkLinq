"""
Proposal class and database CRUD functions for proposal objects.
"""

from core import app_constants
from core import app_utils

class Proposal: # pylint: disable=too-many-instance-attributes
    """
    Proposal class.
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, redefined-builtin
    def __init__(self, id=None, original=None, final=None, action=None, object=None, status=None, author=None, time=None, notes=None):
        """Initialize a Proposal record with the provided attributes."""
        self.id = id
        self.original = original
        self.final = final
        self.action = action
        self.object = object
        self.status = status
        self.author = author
        self.time = time
        self.notes = notes

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False): # pylint: disable=unused-argument
        """Convert the Proposal object into a dictionary representation."""
        item_dict = {"id": self.id,
                     "original": self.original,
                     "final": self.final,
                     "action": self.action,
                     "object": self.object,
                     "status": self.status,
                     "author": self.author,
                     "time": self.time,
                     "notes": self.notes}

        return item_dict

    def print(self):
        """Print the dictionary representation of the Proposal object."""
        print(self.to_dict())

def db_to_proposal(db_object):
    """Convert a database row tuple into a Proposal object."""
    if db_object is None:
        return app_constants.ERROR

    proposal_id = db_object[0]
    original = db_object[1]
    final = db_object[2]
    action = db_object[3]
    target_object = db_object[4]
    status = db_object[5]
    author = db_object[6]
    time = db_object[7]
    notes = db_object[8]

    return_object = Proposal(id=proposal_id, original=original, final=final, action=action,
                             object=target_object, status=status, author=author, time=time,
                             notes=notes)

    return return_object

# Create/update functions.
def upsert_proposal(proposal, connection=None):
    """Insert or update a proposal record in the database."""
    data = ({"id": proposal.id,
             "original": proposal.original,
             "final": proposal.final,
             "action": proposal.action,
             "object": proposal.object,
             "status": proposal.status,
             "author": proposal.author,
             "time": proposal.time,
             "notes": proposal.notes})
    sql = """INSERT INTO Proposals (id, original, final, action, object, status, author, time, notes)
    VALUES (:id, :original, :final, :action, :object, :status, :author, :time, :notes)
    ON CONFLICT(id) DO UPDATE SET original = :original, final = :final, action = :action,
    object = :object, status = :status, author = :author, time = :time, notes = :notes RETURNING *;
    """

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR
    return db_to_proposal(result[0])

# Get/filter fucntions.
# @app_utils.timeit
def get_proposals(ids=None, limit=None, connection=None):
    """Retrieve proposal records from the database by ID list or query limit."""
    # 1. Handle when id list is provided.
    if ids is not None:
        if not isinstance(ids, list):
             ids = [ids]
        
        ids_to_fetch = ids[:limit] if limit is not None else ids
        if not ids_to_fetch:
            return []

        # Fetch proposal objects by IDs.
        placeholders, values = app_utils.build_sql_placeholders(ids_to_fetch, "id")
        sql = f"""SELECT id, original, final, action, object, status, author, time, notes FROM Proposals
                WHERE id IN {placeholders};"""
        proposal_rows = app_utils.execute_sql(sql, values, connection=connection)
        if proposal_rows == app_constants.ERROR:
            return []

        proposals_dict = {row[0]: db_to_proposal(row) for row in proposal_rows}

        # Return the final list of proposal objects, preserving original order.
        return [proposals_dict[proposal_id] for proposal_id in ids_to_fetch if proposal_id in proposals_dict]

    # If no IDs provided, fetch all with limit.
    limit_string = ""
    if limit is not None:
        limit_string = f" LIMIT {limit}"
    
    sql = f"""SELECT id, original, final, action, object, status, author, time, notes FROM Proposals{limit_string};"""
    proposal_rows = app_utils.execute_sql(sql, {}, connection=connection)

    if proposal_rows == app_constants.ERROR:
        return []
    
    return [db_to_proposal(row) for row in proposal_rows]

# Columns allowed in ORDER BY (whitelist to prevent SQL injection via column names).
_PROPOSAL_SORT_COLUMNS = frozenset({'id', 'time', 'author', 'action', 'object', 'status'})

# TODO keywords search.
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
# pylint: disable=redefined-builtin, unused-argument
def get_matching_proposal_ids(ids=None, original=None, object=None, final=None, author=None,
                              min_time=None, max_time=None, action=None, status=None,
                              keywords=None, allow_unknown=False,
                              attribute=None, order=None, offset=0, limit=None,
                              count_only=False, connection=None):
    """Retrieve proposal IDs matching the specified filter, sorting, and pagination criteria."""
    # Build WHERE clauses and query data.
    where_clauses = []
    keyword_clause = ""
    data = {}

    # Value filters.
    for item in [["action", action, "==", "action"], ["status", status, "==", "status"], ["author", author, "==", "author"],
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
        sql = f"SELECT COUNT(*) FROM Proposals{where_string};"
        result = app_utils.execute_sql(sql, data, connection=connection)
        if result == app_constants.ERROR:
            return 0
        return result[0][0] if result else 0

    # ORDER BY (column name is whitelist-validated, not parameterised).
    sort_col = attribute if attribute in _PROPOSAL_SORT_COLUMNS else 'time'
    sort_dir = 'DESC' if order == 'descending' else 'ASC'
    tiebreaker = '' if sort_col == 'id' else f', id {sort_dir}'
    order_string = f" ORDER BY {sort_col} {sort_dir}{tiebreaker}"

    # LIMIT / OFFSET for server-side pagination.
    pagination_string = ""
    if limit is not None:
        pagination_string = " LIMIT :pg_limit OFFSET :pg_offset"
        data = data | {"pg_limit": int(limit), "pg_offset": int(offset) if offset else 0}

    # Build SQL statement.
    sql = f"SELECT id FROM Proposals{where_string}{order_string}{pagination_string};"

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR
    return [item[0] for item in result]

# Delete functions.
def delete_proposal(proposal_id, connection=None):
    """Delete a proposal record from the database by its ID."""
    if app_utils.id_exists(proposal_id, "Proposals", connection=connection) is False:
        return app_constants.ERROR

    data = {"id": proposal_id}
    sql = """DELETE FROM Proposals where id = :id;"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR
    return app_constants.SUCCESS
