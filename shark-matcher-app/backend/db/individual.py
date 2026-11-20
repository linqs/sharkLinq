"""
Individual class and database CRUD functions for individual objects.
Individual adjacent classes and CRUD functions for those AssociatedObjects.
"""

import sys
from core import app_constants
import backend.db
import backend.db.media
from core import app_utils

class Individual: # pylint: disable=too-many-instance-attributes
    """
    Individual class.
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, redefined-builtin
    def __init__(self, length=None, sex=None, id=None, name=None, source=None, reference=None, media=None,
                 associated_individuals=None, associated_medias=None, alternate_names=None,
                 labels=None, tags=None):
        """Initialize an individual from its stored attributes."""
        self.id = id
        self.name = name
        self.length = length
        self.sex = sex
        self.media = media
        self.source = source
        self.reference = reference
        self.associated_individuals = associated_individuals if associated_individuals is not None else []
        self.associated_medias = associated_medias if associated_medias is not None else []
        self.alternate_names = alternate_names if alternate_names is not None else []
        self.labels = labels if labels is not None else []
        self.tags = tags if tags is not None else []

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False):
        """Return the individual as a dictionary; compact=True lists associated objects by ID only."""
        # All associated objects to dictionaries.
        associated_medias = [associated_media.to_dict() for associated_media
                             in self.associated_medias]
        associated_individuals = [associated_individual.to_dict() for associated_individual
                             in self.associated_individuals]
        alternate_names = [alternate_name.to_dict() for alternate_name in self.alternate_names]
        labels = [label.to_dict() for label in self.labels]
        tags = [tag.to_dict() for tag in self.tags]

        # If returning compact dict, only return ids of associated objects.
        if compact is True:
            associated_medias = [associated_media["name"] for associated_media in associated_medias]
            associated_individuals = [associated_individual["id"] for associated_individual in associated_individuals]
            alternate_names = [alternate_name["name"] for alternate_name in alternate_names]
            labels = [label["name"] for label in labels]
            tags = [tag["name"] for tag in tags]

        item_dict = {"id": self.id,
            "name": self.name,
            "length": self.length,
            "sex": self.sex,
            "source": self.source,
            "reference": self.reference,
            "media": self.media,
            "associated_individuals": associated_individuals,
            "associated_medias": associated_medias,
            "alternate_names": alternate_names,
            "labels": labels,
            "tags": tags}

        if hasattr(self, "distance"):
            item_dict["distance"] = self.distance

        return item_dict

    def print(self, compact=False):
        """Print the individual's dictionary representation."""
        print(self.to_dict(compact))

# Source/media/reference/full individual functions.
def is_full_individual(individual):
    """Return whether an individual has populated associated media and associated individuals."""
    # Full individuals have populated associated_medias and associated_individuals lists.
    return len(individual.associated_individuals) != 0 and len(individual.associated_medias) != 0

def is_media_individual(individual):
    """Return whether an individual is a media individual: it has a media record and a source other than itself."""
    # Media individuals always have an associated media.
    return (individual.source != individual.id) and (individual.media is not None)

def is_reference_individual(individual, connection=None):
    """Return whether an individual is its own reference individual."""
    reference_individual = individual_get_reference_individual(individual, connection=connection)
    return reference_individual.id == individual.id

def is_source_individual(individual):
    """Determine whether an individual is a source individual (utility individual or reference without media)."""
    # The utility individual is a source individual
    if individual.id == -1:
        return True
    # Matching individuals always have an associated reference, never a media.
    return (individual.source == individual.id) and (individual.media is None) and \
           (individual.reference is not None)

def is_archive_individual(individual):
    """Determine whether an individual is an archive individual having neither associated media nor reference."""
    # Archive individuals never have an associated reference or media.
    return (individual.media is None) and (individual.reference is None)

def individual_get_reference_individual(individual, connection=None):
    """Retrieve the full reference individual associated with the given individual."""
    # If a full source individual is provided, use that.
    if is_full_individual(individual) is True and is_source_individual(individual) is True:
        source_individual = individual
    # Otherwise, get the source individual.
    else:
        source_individual = get_individuals(ids=[individual.source], get_full=True, get_source=True, connection=connection)[0]
        if not isinstance(source_individual, Individual):
            print("Could not find individual with ID", individual.source)
            return None

    # By now you need to have a source individual.
    assert is_source_individual(source_individual) is True

    reference_individual_id = source_individual.reference

    for associated_individual in source_individual.associated_individuals:
        if associated_individual.id == reference_individual_id:
            return associated_individual

    print("Warning, could not find reference individual for individual ID", individual.id)
    return None

def individual_get_associated_individual_ids(individual, connection=None):
    """Retrieve all associated individual IDs for a given individual, starting with the source ID."""
    # If this is not a full individual get the corresponding source individual.
    if is_full_individual(individual) is True:
        full_individual = individual
    else:
        full_individual = get_individuals(ids=[individual.id], get_full=True, get_source=False, connection=connection)[0]

    result_list = []
    for assoc_individual in full_individual.associated_individuals:
        result_list.append(assoc_individual.id)

    return result_list

def individual_ids_get_source_individual_ids(ids, connection=None):
    """Return the source individual IDs for the given individual IDs."""
    data = {}
    where_clause = ""

    # List filters.
    for item in [["id", ids, "id"]]:
        if item[1] is not None and item[1] != "":
            sql_ids, sql_values = app_utils.build_sql_placeholders(
                item[1], basename=f"x_{item[2]}")
            where_clause = f"({item[0]} COLLATE NOCASE in {sql_ids})"
            data = data | sql_values

    if where_clause == "":
        return []

    sql = f"""SELECT DISTINCT source FROM Individuals WHERE {where_clause};"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR

    return [item[0] for item in result]

class AssociatedObject:
    """
    AssociatedObject class.
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, redefined-builtin
    def __init__(self, id=None, individual=None, name=None, type=None, time=None):
        """Initialize an associated object from its stored attributes."""
        self.id = id
        self.individual = individual
        self.name = name
        self.type = type
        self.time = time

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False): # pylint: disable=unused-argument
        """Return the associated object as a dictionary; compact is accepted for parity with Individual.to_dict."""
        item_dict = {"id": self.id,
                        "individual": self.individual,
                        "name": self.name,
                        "type": self.type}

        if self.time is not None and self.time != "":
            item_dict["time"] = self.time

        return item_dict


    def print(self):
        """Print the associated object's dictionary representation."""
        print(self.to_dict())

def db_to_individual(db_object):
    """Convert a database row into an Individual, or return ERROR for a missing row."""
    if db_object is None:
        return app_constants.ERROR

    individual_id = db_object[0]
    name = db_object[1]
    sex = db_object[2]
    length = db_object[3]
    media = db_object[4]
    source = db_object[5]
    reference = db_object[6]

    return_object = Individual(length=length, sex=sex, id=individual_id, name=name, source=source,
                          reference=reference, media=media)

    return return_object

def db_to_associated_object(db_object):
    """Convert a database row into an AssociatedObject, or return ERROR for a missing row."""
    if db_object is None:
        return app_constants.ERROR

    assoc_id = db_object[0]
    individual = db_object[1]
    name = db_object[2]
    assoc_type = db_object[3]
    time = db_object[4]

    return_object = AssociatedObject(id=assoc_id, individual=individual, name=name, type=assoc_type,
                                     time=time)

    return return_object

# Counting functions.
def count_individuals(connection=None):
    """Count total individuals broken down into total, media, source, and archive categories."""
    return_dict = {}
    data = {}

    # Count the total number of individuals.
    sql = """SELECT COUNT(*) FROM Individuals WHERE id >= 0"""
    result = app_utils.execute_sql(sql, data, connection=connection)
    return_dict["total"] = result[0][0]

    # Count the total number of media individuals.
    # A media individual is defined as a individual that has a media object (media is not null)
    # and is not a source individual (source != id).
    sql = """SELECT COUNT(*) FROM Individuals WHERE id >= 0 AND media IS NOT NULL AND source != id"""
    result = app_utils.execute_sql(sql, data, connection=connection)
    return_dict["media"] = result[0][0]

    # Count the total number of source individuals.
    sql = """SELECT COUNT(*) FROM Individuals WHERE id >= 0 AND source = id AND
             reference IS NOT NULL"""
    result = app_utils.execute_sql(sql, data, connection=connection)
    return_dict["source"] = result[0][0]

    # Count the total number of archive individuals.
    sql = """SELECT COUNT(*) FROM Individuals WHERE id >= 0 AND source = id AND
             reference IS NULL"""
    result = app_utils.execute_sql(sql, data, connection=connection)
    return_dict["archive"] = result[0][0]

    # Error check.
    for value in return_dict.values():
        if value == app_constants.ERROR:
            return app_constants.ERROR

    return return_dict

def count_associated_objects(table=None, connection=None):
    """Return the number of rows in an associated object table."""
    # Input validation.
    if table not in (app_utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")

    data = {}
    sql = f"""SELECT COUNT(*) FROM {table}"""
    result = app_utils.execute_sql(sql, data, connection=connection)

    if result == app_constants.ERROR:
        return app_constants.ERROR

    return result[0][0]

# Create/update functions.
def upsert_individual(individual, connection=None):
    """Insert or update an individual, inside the caller's transaction when a connection is given."""
    data = {"id": individual.id,
            "name": individual.name,
            "length": app_utils.convert_unknown_to_none(individual.length),
            "sex": app_utils.normalize_sex(individual.sex),
            "media": individual.media,
            "source": individual.source,
            "reference": individual.reference}
    sql = """INSERT INTO Individuals (id, name, length, sex, media, source, reference)
    VALUES (:id, :name, :length, :sex, :media, :source, :reference)
    ON CONFLICT(id) DO UPDATE SET name = :name, length = :length, sex = :sex,
    media = :media, source = :source, reference = :reference RETURNING *;
    """
    result = app_utils.execute_sql(sql, data, connection=connection)
    if not result or result == app_constants.ERROR:
        return app_constants.ERROR

    return db_to_individual(result[0])

def upsert_associated_object(assoc_object, table, connection=None):
    """Insert or update an associated object in table, inside the caller's transaction when a connection is given."""
    # Input validation.
    if table not in (app_utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")
    if app_utils.id_exists(assoc_object.individual, "Individuals", connection=connection) is False:
        return app_constants.ERROR

    data = {"id": assoc_object.id,
            "individual": assoc_object.individual,
            "name": assoc_object.name,
            "type": assoc_object.type,
            "time": assoc_object.time}
    sql = f"""INSERT INTO {table} (id, individual, name, type, time) VALUES (:id, :individual, :name, :type, :time)
    ON CONFLICT(id) DO UPDATE SET individual = :individual, name = :name, type = :type, time = :time RETURNING *;"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if not result or result == app_constants.ERROR:
        return app_constants.ERROR

    return db_to_associated_object(result[0])

# Get/filter fucntions.
# @app_utils.timeit
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def get_individuals(ids=None, get_full=True, get_source=False, limit=None, source_only=True, connection=None):
    """Retrieve individuals matching provided IDs with optional associated data and source resolution."""
    if ids is None:
        # Fetch all matching individuals directly in a single query when ids is None.
        if source_only or get_source:
            where = "id >= 0 AND source = id AND reference IS NOT NULL"
        else:
            where = "id >= 0"
        limit_string = f" LIMIT {limit}" if limit is not None else ""
        sql = f"""SELECT id, name, sex, length, media, source, reference
                  FROM Individuals WHERE {where}{limit_string};"""
        main_individual_rows = app_utils.execute_sql(sql, {}, connection=connection)
        if main_individual_rows == app_constants.ERROR or not main_individual_rows:
            return []
        individuals_dict = {row[0]: db_to_individual(row) for row in main_individual_rows}
        # ids_to_fetch is used by the source_only=False return path below.
        ids_to_fetch = [row[0] for row in main_individual_rows]
    else:
        # Explicit ID list supplied (or a non-list value — preserved for backward
        # compatibility with callers that pass a single int or similar).
        if not isinstance(ids, list):
            ids = app_utils.get_all_ids("Individuals", source=source_only, connection=connection)

        # Handle get_source: resolve provided IDs to their source individuals.
        if get_source is True:
            source_ids = individual_ids_get_source_individual_ids(ids, connection=connection)
            ids_to_fetch = list(set(source_ids))
        else:
            ids_to_fetch = ids

        if limit is not None:
            ids_to_fetch = ids_to_fetch[:limit]

        if ids_to_fetch == []:
            return []

        # Fetch all primary individuals in one go.
        placeholders, values = app_utils.build_sql_placeholders(ids_to_fetch, "id")
        sql = f"""SELECT id, name, sex, length, media, source, reference
                  FROM Individuals WHERE id IN {placeholders};"""
        main_individual_rows = app_utils.execute_sql(sql, values, connection=connection)
        if main_individual_rows == app_constants.ERROR:
            return []

        # Key by id, not source. An explicit id list may contain several individuals that
        # share a source (every media individual of one shark), and keying by source would
        # silently collapse them down to whichever row the database returned last.
        individuals_dict = {row[0]: db_to_individual(row) for row in main_individual_rows}

    # Store an id to source dict so associated objects can be matched with the individual.
    id_to_source_dict = {}

    if get_full is False:
        return list(individuals_dict.values())

    # Handle get_full - fetch all associated data in bulk.
    # Several primary individuals can share one source, so map each source to every primary
    # individual that belongs to it.
    source_to_primary_individuals = {}
    for individual in individuals_dict.values():
        if individual.source is None:
            continue
        source_to_primary_individuals.setdefault(individual.source, []).append(individual)
    source_ids_to_query = list(source_to_primary_individuals.keys())

    all_media_ids = set()
    associated_individual_rows = None

    # Fetch all associated individuals (media individuals + source individuals themselves).
    # Archive individuals have no source, so there is nothing to look up for them.
    if source_ids_to_query:
        placeholders, values = app_utils.build_sql_placeholders(source_ids_to_query, "source")
        sql = f"""SELECT id, name, sex, length, media, source, reference
                  FROM Individuals WHERE source IN {placeholders};"""
        associated_individual_rows = app_utils.execute_sql(sql, values, connection=connection)

    # Match associated individuals to the primary individuals that share their source.
    if associated_individual_rows != app_constants.ERROR and associated_individual_rows:
        for row in associated_individual_rows:
            individual_obj = db_to_individual(row)
            individual_id = individual_obj.id
            source_id = individual_obj.source

            id_to_source_dict[individual_id] = source_id

            primary_individuals = source_to_primary_individuals.get(source_id)
            if primary_individuals:
                for primary_individual in primary_individuals:
                    primary_individual.associated_individuals.append(individual_obj)
                if individual_obj.media is not None:
                    all_media_ids.add(individual_obj.media)

    # Ensure source individual is first in the associated_individuals list for each individual.
    for individual in individuals_dict.values():
        # Sorts with the source individual (where id == source) at the beginning.
        individual.associated_individuals.sort(key=lambda s: s.id != s.source)

    # Get all individual ids to query for associated objects.
    # We want to fetch objects for ALL loaded individuals (source and associated).
    all_individuals_map = {individual.id: individual for individual in individuals_dict.values()}
    for individual in individuals_dict.values():
        for assoc_individual in individual.associated_individuals:
            if assoc_individual.id not in all_individuals_map:
                all_individuals_map[assoc_individual.id] = assoc_individual

    individual_ids_to_query = list(all_individuals_map.keys())

    # Fetch all associated objects (AlternateNames, Tags, Labels).
    for table in ["AlternateNames", "Tags", "Labels"]:
        placeholders, values = app_utils.build_sql_placeholders(individual_ids_to_query, "individual")
        sql = f"""SELECT id, individual, name, type, time FROM {table} WHERE individual IN {placeholders};"""
        object_rows = app_utils.execute_sql(sql, values, connection=connection)

        if object_rows != app_constants.ERROR and object_rows:
            for row in object_rows:
                obj = db_to_associated_object(row)
                
                # 1. Attach to the specific individual that owns this object.
                if obj.individual in all_individuals_map:
                    target_individual = all_individuals_map[obj.individual]
                    attr_name = "alternate_names" if table == "AlternateNames" else table.lower()
                    getattr(target_individual, attr_name).append(obj)

                    # Aggregate associated objects onto the parent source individual when target is a media individual.
                    source_id = target_individual.source
                    if source_id in all_individuals_map:
                        source_individual = all_individuals_map[source_id]
                        # Avoid adding twice if the target is already the source individual.
                        if source_individual.id != target_individual.id:
                             getattr(source_individual, attr_name).append(obj)

    # Fetch all associated media.
    if all_media_ids:
        # Use the optimized get_medias function to fetch all media objects at once.
        media_list = backend.db.media.get_medias(ids=list(all_media_ids), get_full=False, connection=connection)

        media_dict = {}
        if media_list:
            media_dict = {media.id: media for media in media_list}

        # Attach media objects to the source individuals
        for individual in individuals_dict.values():
            for associated_individual in individual.associated_individuals:
                if associated_individual.media in media_dict:
                    individual.associated_medias.append(media_dict[associated_individual.media])

    # Return the final list of individuals.
    if source_only is True:
        # Return only source individuals
        return list(individuals_dict.values())
    # Return the actual requested individuals, not just source individuals
    # We need to find which individuals were originally requested
    requested_individuals = []
    seen_ids = set()
    for individual in individuals_dict.values():
        # Check source individual itself
        if individual.id in ids_to_fetch and individual.id not in seen_ids:
             requested_individuals.append(individual)
             seen_ids.add(individual.id)
        # Check associated individuals
        for associated_individual in individual.associated_individuals:
            if associated_individual.id in ids_to_fetch and associated_individual.id not in seen_ids:
                requested_individuals.append(associated_individual)
                seen_ids.add(associated_individual.id)
    
    return requested_individuals

# @app_utils.timeit
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
# pylint: disable=too-many-branches
def get_matching_individual_ids(ids=None, names=None, min_length=None, max_length=None, sex=None,
                           medias=None, sources=None, keywords=None, allow_unknown=False,
                           limit=None, source_only=True, connection=None):
    """Return IDs of individuals matching every given filter; allow_unknown also admits unknown values and source_only keeps only source individuals."""
    # "Unknown" selects individuals whose sex is not recorded.
    sex = app_utils.normalize_sex(sex, unknown="Unknown")

    # Build WHERE clauses and query data.
    where_clauses = []
    keyword_clause = ""
    data = {}

    # Value filters.
    for item in [["length", min_length, ">=", "min_length"],
                 ["length", max_length, "<=", "max_length"], ["sex", sex, "==", "sex"]]:
        if item[1] is not None and item[1] != "":
            # Allow unknown.
            prefix = ""
            if allow_unknown is True:
                prefix = f"{item[0]} IS NULL OR "
            else:
                prefix = f"{item[0]} IS NOT NULL AND "

            # Cast to real for numeric comparison.
            if item[0] == "length":
                item[1] = float(item[1])

            sql_ids, sql_values = app_utils.build_sql_placeholders(
                [item[1]], basename=f"x_{item[3]}")
            where_clauses.append(f"({prefix}{item[0]} {item[2]} {sql_ids})")
            data = data | sql_values

    # List filters.
    for item in [["id", ids, "id"], ["name", names, "name"], ["media", medias, "media"],
                   ["source", sources, "source"], ["name", keywords, "keyword"]]:
        if item[1] is not None and item[1] != "":
            sql_ids, sql_values = app_utils.build_sql_placeholders(item[1], basename=f"x_{item[2]}")
            if item[2] == "keyword":
                keyword_clause = f"{item[0]} COLLATE NOCASE in {sql_ids} OR "
            else:
                where_clauses.append(f"({item[0]} COLLATE NOCASE in {sql_ids})")
            data = data | sql_values

    # Build where clauses string.
    # Always start by filtering out utility objects with negative ids and meeting the source_only condion.
    where_clause = " WHERE id >= 0"
    if source_only is True:
        where_clause = where_clause + " AND source = id"

    if len(where_clauses) == 0 and keyword_clause != "":
        where_clause = where_clause + " AND " + keyword_clause[:-4]
    elif len(where_clauses) != 0:
        where_clause = where_clause + " AND " + keyword_clause + \
            app_utils.list_to_string(where_clauses, " AND ")

    # Build the limit string.
    limit_string = ""
    if limit is not None:
        limit_string = f" LIMIT ${limit}"
        data = data | {"limit": limit}

    # Build SQL statement.
    sql = f"""SELECT id FROM Individuals{where_clause}{limit_string};"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR

    return [item[0] for item in result]

# Sort attributes recognized by get_sorted_source_individual_ids. Anything else
# falls back to 'name' (the default the frontend uses for the search page).
_SOURCE_SORT_ATTRIBUTES = frozenset({'id', 'name', 'associated_medias', 'associated images'})

def get_sorted_source_individual_ids(ids=None, attribute=None, order=None, connection=None):
    """Return source individual IDs sorted by the requested attribute for paginated search."""
    data = {}
    # Restrict to source individuals (excludes utility rows and media individuals).
    where = "src.id >= 0 AND src.source = src.id AND src.reference IS NOT NULL"

    if ids is not None:
        ids_list = list(ids)
        if not ids_list:
            return []
        placeholders, values = app_utils.build_sql_placeholders(ids_list, basename="flt_id")
        where += f" AND src.id IN {placeholders}"
        data = data | values

    sort_dir = 'DESC' if order == 'descending' else 'ASC'

    if attribute in ('associated_medias', 'associated images'):
        # Count media individuals under each source. LEFT JOIN so sources with
        # zero media still appear (at the low end).
        sql = f"""
            SELECT src.id
            FROM Individuals src
            LEFT JOIN Individuals mi
              ON mi.source = src.id AND mi.media IS NOT NULL
            WHERE {where}
            GROUP BY src.id
            ORDER BY COUNT(mi.id) {sort_dir}, src.id {sort_dir};
        """
    elif attribute == 'id':
        sql = f"""
            SELECT src.id
            FROM Individuals src
            WHERE {where}
            ORDER BY src.id {sort_dir};
        """
    else:
        # Default: sort by reference individual's name. Numeric compare first
        # (matches the frontend's parseInt behavior) then lexical; NULL names
        # sort last regardless of direction.
        sql = f"""
            SELECT src.id
            FROM Individuals src
            LEFT JOIN Individuals ref ON ref.id = src.reference
            WHERE {where}
            ORDER BY ref.name IS NULL,
                     CAST(ref.name AS INTEGER) {sort_dir},
                     ref.name {sort_dir},
                     src.id {sort_dir};
        """

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return []
    return [row[0] for row in result]

def get_source_individual_ids_by_association_count(min_associated=None, max_associated=None, connection=None):
    """Retrieve source individual IDs filtered by their count of associated media individuals."""
    having_clauses = []
    data = {}
    
    # Build having clauses string based on min_associated and max_associated.
    if min_associated is not None:
        having_clauses.append(f"COUNT(*) >= :min_count")
        data["min_count"] = int(min_associated) + 1
        
    if max_associated is not None:
        having_clauses.append(f"COUNT(*) <= :max_count")
        data["max_count"] = int(max_associated) + 1
        
    having_string = ""
    if having_clauses:
        having_string = " HAVING " + " AND ".join(having_clauses)
    
    # We only care about source individuals, so we return the source.
    sql = f"""SELECT source FROM Individuals GROUP BY source{having_string};"""
    
    result = app_utils.execute_sql(sql, data, connection=connection)
    
    if result == app_constants.ERROR:
        return []
        
    return [row[0] for row in result]

def get_associated_objects(ids=None, names=None, limit=None, table=None, connection=None):
    """Return associated objects from table that match the given IDs or names."""
    # Input validation.
    if table not in (app_utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")

    # Build where clauses.
    where_clauses = []
    data = {}

    if ids is not None:
        if not isinstance(ids, list):
            ids = [ids]
        sql_ids, sql_values = app_utils.build_sql_placeholders(ids, basename="id")
        where_clauses.append(f"(id IN {sql_ids})")
        data.update(sql_values)

    if names is not None:
        if not isinstance(names, list):
            names = [names]
        sql_names, sql_values = app_utils.build_sql_placeholders(names, basename="name")
        where_clauses.append(f"(name IN {sql_names})")
        data.update(sql_values)

    # If no filters provided, get all.
    if not where_clauses:
        ids = app_utils.get_all_ids(table, connection=connection)
        if not ids:
            return []
        # Fallback to get all behavior if needed.

    where_string = ""
    if where_clauses:
        where_string = " WHERE " + " OR ".join(where_clauses)

    limit_string = ""
    if limit is not None:
        limit_string = f" LIMIT {limit}"

    sql = f"""SELECT id, individual, name, type, time FROM {table}{where_string}{limit_string};"""

    object_rows = app_utils.execute_sql(sql, data, connection=connection)
    if object_rows == app_constants.ERROR or not object_rows:
        return []

    return [db_to_associated_object(row) for row in object_rows]

# TODO, non exact search for names.
# @app_utils.timeit
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
# pylint: disable=redefined-builtin, unused-argument
def get_matching_associated_object_individual_ids(ids=None, names=None, individuals=None, types=None,
                                             keywords=None, allow_unknown=False, table=None,
                                             limit=None, any=False, connection=None):
    """Return IDs of individuals linked to matching associated objects; any=True ignores every filter except ids."""
    # Input validation.
    if table not in (app_utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")

    # If the any parameter is passed, return individual ids that have any of this associated object.
    if any is True:
        sql = f"""SELECT DISTINCT individual FROM {table} WHERE individual >= 0;"""
        result = app_utils.execute_sql(sql, {}, connection=connection)
        if result == app_constants.ERROR:
            return app_constants.ERROR
        return [item[0] for item in result]

    # Build WHERE clauses and query data.
    where_clauses = []
    keyword_clause = ""
    data = {}

    # List filters.
    for item in [("id", ids, "id"), ("name", names, "name"), ("individual", individuals, "individual"),
                 ("type", types, "type"), ("name", keywords, "keyword")]:
        if item[1] is not None and item[1] != "":
            sql_ids, sql_values = app_utils.build_sql_placeholders(
                item[1], basename=f"x_{item[2]}")
            if item[2] == "keyword":
                keyword_clause = f"{item[0]} COLLATE NOCASE in {sql_ids} OR "
            else:
                where_clauses.append(f"({item[0]} COLLATE NOCASE in {sql_ids})")
            data = data | sql_values

    # Build where clauses string.
    # Always start by filtering out utility objects with negative ids.
    where_clause = " WHERE individual >= 0"
    if len(where_clauses) == 0 and keyword_clause != "":
        where_clause = where_clause + " AND " + keyword_clause[:-4]
    elif len(where_clauses) != 0:
        where_clause = where_clause + " AND " + keyword_clause + \
            app_utils.list_to_string(where_clauses, " AND ")

    # Build the limit string.
    limit_string = ""
    if limit is not None:
        limit_string = f" LIMIT ${limit}"
        data = data | {"limit": limit}

    # Build SQL statement.
    sql = f"""SELECT individual FROM {table}{where_clause}{limit_string};"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR

    return [item[0] for item in result]

# Delete functions.
def delete_individual(individual_id, connection=None):
    """Delete an individual by ID, inside the caller's transaction when a connection is given."""
    # Check if individual id exists
    if app_utils.id_exists(individual_id, "Individuals", connection=connection) is False:
        return app_constants.ERROR

    data = {"id": individual_id}
    sql = """DELETE FROM Individuals where id = :id;"""

    try:
        result = app_utils.execute_sql(sql, data, connection=connection)
        if result == app_constants.ERROR:
            return app_constants.ERROR
        return app_constants.SUCCESS
    except Exception as e:
        print(f"Error deleting individual: {e}")
        return app_constants.ERROR

def delete_associated_object(assoc_id, table, connection=None):
    """Delete an associated object from table by ID, inside the caller's transaction when a connection is given."""
    if table not in (app_utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")
    if app_utils.id_exists(assoc_id, table, connection=connection) is False:
        return app_constants.ERROR

    data = {"id": assoc_id}
    sql = f"""DELETE FROM {table} where id = :id;"""

    try:
        result = app_utils.execute_sql(sql, data, connection=connection)
        if result == app_constants.ERROR:
            return app_constants.ERROR
        return app_constants.SUCCESS
    except Exception as e:
        print(f"Error deleting associated object: {e}")
        return app_constants.ERROR

# Reference and source individual modification.
def set_reference_individual(source_individual, reference_id=None, connection=None):
    """Set the reference individual pointer for a given source individual."""
    # Get the full source individual, check that it is indeed a source individual.
    if is_full_individual(source_individual) is False:
        source_individual = get_individuals(ids=[source_individual.id], get_full=True, get_source=True, connection=connection)[0]
    if is_source_individual(source_individual) is False:
        print("Error, invalid source individual.")
        return app_constants.ERROR

    # Check that the reference individual is matched to this source individual.
    if reference_id is not None:
        associated_individual_ids = individual_get_associated_individual_ids(source_individual, connection=connection)
        if reference_id not in associated_individual_ids:
            print("Error, invalid new reference individual requested.", reference_id)
            return app_constants.ERROR

    # If no reference id is given, default to the oldest associated individual.
    if reference_id is None:
        associated_individuals = sort_individuals_except_source(source_individual.associated_individuals, "media_time", connection=connection)
        reference_id = associated_individuals[0].id
        # If picking the oldest associated individual doesn't change the reference id,
        # choose the next oldest associated individual or set to None it if there is only one associated individual (and one source individual).
        if source_individual.reference == reference_id:
            if len(associated_individuals) > 2:
                reference_id = associated_individuals[1].id
            else:
                print(f"Warning, source individual with id {source_individual.id} has no more associated "
                      f"individuals, setting reference to None")
                reference_id = None
                # TODO, When setting a source individual's reference ID to "", you are turning it
                # into an archive individual.
                # This may be the time to delete that individual.

    # Update the source's reference individual pointer.
    source_individual.reference = reference_id
    upsert_individual(source_individual, connection=connection)

    return reference_id

def set_source_individual(pointer_individual, source_individual, connection=None):
    """Update the source individual pointer for a given media individual."""
    # Check that the source is a source individual.
    if is_source_individual(source_individual) is False:
        print("Error, invalid source individual.")
        return app_constants.ERROR

    # Check that the pointer is a media individual.
    if is_media_individual(pointer_individual) is False:
        print("Error, invalid media individual.")
        return app_constants.ERROR

    # Update the pointer individual.
    pointer_individual.source = source_individual.id
    upsert_individual(pointer_individual, connection=connection)

    return app_constants.SUCCESS

# @app_utils.timeit
def sort_individuals_except_source(individual_list, parameter="id", connection=None):
    """Sort associated individuals ascending by parameter, placing the media-less source individual last."""
    # Sort on individual attributes
    if parameter == "id":
        individual_list.sort(key=lambda x : x.id)
    elif parameter == "length":
        individual_list.sort(key=lambda x : x.length)

    # Sort on media attributes
    elif parameter == "media_time":
        # Get media data to sort on.
        # Filter None values to avoid unnecessary queries or issues
        media_ids = [individual.media for individual in individual_list if individual.media is not None]
        medias = backend.db.media.get_medias(ids=media_ids, connection=connection)

        media_map = {m.id: m for m in medias}

        individual_media_list = []
        for individual in individual_list:
            # Individuals with malformed or missing media go to the end of the list.
            if individual.media is not None and individual.media in media_map:
                media_item = media_map[individual.media]
                if isinstance(media_item.time, int):
                    individual_media_list.append((individual, media_item.time))
                else:
                    individual_media_list.append((individual, sys.maxsize))
            else:
                individual_media_list.append((individual, sys.maxsize))

        # Sort by media, rebuild individual_list.
        individual_media_list.sort(key=lambda x : x[1])
        individual_list = [item[0] for item in individual_media_list]

    return individual_list

# Cross table functions.
# @app_utils.timeit
def get_individual_associated_objects(individual_id, table, connection=None):
    """
    Get associated objects for a individual id from a specific table.
    """
    # Input validation.
    if table not in (app_utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")

    data = {"id": individual_id}
    sql = f"""SELECT id, individual, name, type, time FROM {table} WHERE individual = :id;"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR

    return [db_to_associated_object(item) for item in result]

# Helper functions.
def _get_next_individual_name(connection=None):
    """Return the next monotonically increasing integer individual name from the database."""
    data = {}
    sql = """SELECT name FROM Individuals"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR

    # Add all names to a list, casted as integers.
    name_list = []
    for item in result:
        if item[0] is not None and item[0] != "":
            try:
                name_list.append(int(item[0]))
            except ValueError:
                pass

    # Sort the list by name, return one more than the highest name.
    name_list.sort()
    if len(name_list) == 0:
        return 0
    return name_list[-1] + 1
