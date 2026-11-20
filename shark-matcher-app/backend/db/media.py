"""
Media class and database CRUD functions for media objects.
"""

import backend.db.individual
from core import app_constants
from core import app_utils

class Media: # pylint: disable=too-many-instance-attributes
    """
    Media class.
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, redefined-builtin
    def __init__(self, id=None, hash=None, name=None, filename=None, source=None, individual=None,
                 metadata=None, location=None, time=None, type=None, quality=None):
        """Initialize a Media record with the provided attributes."""
        self.id = id
        self.hash = hash
        self.name = name
        self.filename = filename
        self.source = source
        self.individual = individual
        self.metadata = metadata
        self.location = location
        self.time = time
        self.type = type
        self.quality = quality

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False): # pylint: disable=unused-argument
        """Convert the Media object into a dictionary representation."""
        if isinstance(self.individual, backend.db.individual.Individual):
            self.individual = self.individual.to_dict()

        item_dict = {"id": self.id,
                     "hash": self.hash,
                     "name": self.name,
                     "filename": self.filename,
                     "source": self.source,
                     "individual": self.individual,
                     "metadata": self.metadata,
                     "location": self.location,
                     "time": self.time,
                     "type": self.type,
                     "quality": self.quality}

        return item_dict

    def print(self):
        """Print the dictionary representation of the Media object."""
        print(self.to_dict())

def db_to_media(db_object):
    """Convert a database row tuple into a Media object."""
    if db_object is None:
        return app_constants.ERROR

    media_id = db_object[0]
    media_hash = db_object[1]
    name = db_object[2]
    filename = db_object[3]
    source = db_object[4]
    metadata = db_object[5]
    location = db_object[6]
    time = db_object[7]
    media_type = db_object[8]
    quality = db_object[9]

    return_object = Media(id=media_id, hash=media_hash, name=name, filename=filename, source=source,
                          individual=None, metadata=metadata, location=location, time=time,
                          type=media_type, quality=quality)

    return return_object

# Create/update.
def upsert_media(media, connection=None):
    """Insert or update a media record in the database."""
    data = {"id": media.id,
            "hash": media.hash,
            "name": media.name,
            "filename": media.filename,
            "source": media.source,
            "individual": media.individual,
            "metadata": media.metadata,
            "location": app_utils.convert_unknown_to_none(media.location),
            "time": app_utils.convert_unknown_to_none(media.time),
            "type": app_utils.convert_unknown_to_none(media.type),
            "quality": app_utils.convert_unknown_to_none(media.quality)}
    sql = """INSERT INTO Media (id, hash, name, filename, source, metadata, location, time, type, quality)
    VALUES (:id, :hash, :name, :filename, :source, :metadata, :location, :time, :type, :quality)
    ON CONFLICT(id) DO UPDATE SET hash = :hash, name = :name, filename = :filename, source = :source,
    metadata = :metadata, location = :location, time = :time, type = :type,
    quality = :quality RETURNING *;"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if not result or result == app_constants.ERROR:
        return app_constants.ERROR

    return db_to_media(result[0])

# Get/filter fucntions.
# @app_utils.timeit
# pylint: disable=unused-argument
def get_medias(ids=None, names=None, get_full=False, get_source=False, limit=None, connection=None):
    """Retrieve media objects from the database matching the provided IDs or names."""
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

    ids_to_fetch = []

    limit_string = ""
    if limit is not None:
        limit_string = f" LIMIT {limit}"

    # If no filters provided, get keys for all.
    if not where_clauses:
        sql = f"""SELECT id, hash, name, filename, source, metadata, location, time, type,
                  quality FROM Media{limit_string};"""
        media_rows = app_utils.execute_sql(sql, {}, connection=connection)

        if media_rows != app_constants.ERROR:
            ids_to_fetch = [row[0] for row in media_rows]
    else:
        # Filtered fetch.
        where_string = " WHERE " + " OR ".join(where_clauses)

        sql = f"""SELECT id, hash, name, filename, source, metadata, location, time, type,
                  quality FROM Media{where_string}{limit_string};"""
        media_rows = app_utils.execute_sql(sql, data, connection=connection)

    if media_rows == app_constants.ERROR:
        return []

    media_dict = {row[0]: db_to_media(row) for row in media_rows}

    # If we filtered, we need to establish the retrieval order or just use the keys.
    if where_clauses:
        ids_to_fetch = list(media_dict.keys())

    # 3. Handle get_full - fetch associated individuals.
    if get_full:
        media_ids = list(media_dict.keys())
        individual_media_map = get_individual_ids_by_media_ids(media_ids, dict_format=True, connection=connection)
        all_individual_ids = list(individual_media_map.values())

        if all_individual_ids:
            # Use the optimized get_individuals to fetch all individuals at once.
            individuals = backend.db.individual.get_individuals(ids=all_individual_ids, get_full=True, get_source=False, connection=connection)
            # TODO Can add source_only = False here to not get the source individuals
            # but the media's exact individuals.
            individuals_dict = {individual.id: individual for individual in individuals}

            # Attach individual objects to their corresponding media objects.
            for media_id, media_obj in media_dict.items():
                if media_id in individual_media_map and individual_media_map[media_id] in individuals_dict:
                    media_obj.individual = individuals_dict[individual_media_map[media_id]]

    # 4. Return the final list of media objects, preserving original order from ids_to_fetch.
    return [media_dict[media_id] for media_id in ids_to_fetch if media_id in media_dict]

# @app_utils.timeit
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
# pylint: disable=too-many-branches, unused-argument
def get_matching_media_ids(ids=None, names=None, sources=None, types=None,
                           locations=None, min_time=None, max_time=None,
                           qualities=None, max_qualities=None, keywords=None, allow_unknown=False,
                           limit=None, connection=None):
    """Return IDs of media matching every given filter, with qualities and max_qualities bounding quality; allow_unknown also admits unknown values."""
    # Make qualities inclusive, i.e, Medium allows Medium and above, so Good too.
    if qualities is not None:
        for i, quality in enumerate(qualities):
            qualities[i] = quality.lower()
        if "unmatchable" in qualities:
            qualities.extend(["poor", "medium", "good"])
        if "poor" in qualities:
            qualities.extend(["medium", "good"])
        if "medium" in qualities:
            qualities.extend(["good"])
        qualities = list(set(qualities))

    # Make max_qualities inclusive, i.e, Medium allows Medium and below, so Poor and Unmatchable too.
    if max_qualities is not None:
        for i, quality in enumerate(max_qualities):
            max_qualities[i] = quality.lower()
        if "good" in max_qualities:
            max_qualities.extend(["medium", "poor", "unmatchable"])
        if "medium" in max_qualities:
            max_qualities.extend(["poor", "unmatchable"])
        if "poor" in max_qualities:
            max_qualities.extend(["unmatchable"])
        max_qualities = list(set(max_qualities))

    # Build WHERE clauses and query data.
    where_clauses = []
    keyword_clause = ""
    data = {}

    # Value filters.
    for item in [["time", min_time, ">=", "min_time"], ["time", max_time, "<=", "max_time"]]:
        if item[1] is not None and item[1] != "":
            # Allow unknown.
            prefix = ""
            if allow_unknown is True:
                prefix = f"{item[0]} IS NULL OR "
            else:
                prefix = f"{item[0]} IS NOT NULL AND "

            print(item)
            # Cast to real for numeric comparison.
            if item[0] == "time":
                item[1] = float(item[1])

            print(item)

            sql_ids, sql_values = app_utils.build_sql_placeholders([item[1]], basename=f"x_{item[3]}")
            where_clauses.append(f"({prefix}{item[0]} {item[2]} {sql_ids})")
            data = data | sql_values

    # Configure list filter fields that permit NULL values when allow_unknown is set.
    fields_allow_unknown = ["quality", "location", "type"]

    for item in [("id", ids, "id"), ("name", names, "name"), ("source", sources, "source"),
                 ("type", types, "type"), ("quality", qualities, "quality"),
                 ("quality", max_qualities, "max_quality"),
                 ("location", locations, "location"), ("name", keywords, "keyword")]:
        if item[1] is not None and item[1] != "":
            sql_ids, sql_values = app_utils.build_sql_placeholders(
                item[1], basename=f"x_{item[2]}")
            
            clause = ""
            if item[2] == "keyword":
                clause = f"{item[0]} COLLATE NOCASE in {sql_ids}"
                # Append keyword disjunction clause.
                keyword_clause = f"{clause} OR "
            else:
                clause = f"{item[0]} COLLATE NOCASE in {sql_ids}"
                
                # If this field allows unknown and allow_unknown flag is set, include NULLs.
                if allow_unknown and item[0] in fields_allow_unknown:
                     clause = f"({clause} OR {item[0]} IS NULL)"
                
                where_clauses.append(f"({clause})")
                
            data = data | sql_values

    # Build where clauses string.
    where_string = ""
    if len(where_clauses) == 0 and keyword_clause != "":
        where_string = " WHERE " + keyword_clause[:-4]
    elif len(where_clauses) != 0:
        where_string = " WHERE " + keyword_clause + app_utils.list_to_string(where_clauses, " AND ")

    # Build the limit string.
    limit_string = ""
    if limit is not None:
        limit_string = f" LIMIT ${limit}"
        data = data | {"limit": limit}

    # Build SQL statement.
    sql = f"""SELECT id FROM Media{where_string}{limit_string};"""

    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR

    return [item[0] for item in result]

# Delete functions.
def delete_media(media_id, connection=None):
    """Delete a media record from the database by its ID."""
    if app_utils.id_exists(media_id, "Media", connection=connection) is False:
        return app_constants.ERROR

    data = {"id": media_id}
    sql = """DELETE FROM Media where id = :id;"""

    try:
        result = app_utils.execute_sql(sql, data, connection=connection)
        if result == app_constants.ERROR:
            return app_constants.ERROR
        return app_constants.SUCCESS
    except Exception as e:
        print(f"Error deleting media: {e}")
        return app_constants.ERROR

# Cross table functions.
# @app_utils.timeit
def get_individual_ids_by_media_ids(ids, dict_format=False, connection=None):
    """Retrieve individual IDs associated with a list of media IDs."""
    # Build WHERE clauses and query data.
    where_clauses = []
    data = {}

    # List filters.
    for item in [["media", ids]]:
        if item[1] is not None and item[1] != "":
            sql_ids, sql_values = app_utils.build_sql_placeholders(item[1], basename=f"x_{item[0]}")
            where_clauses.append(f"({item[0]} COLLATE NOCASE in {sql_ids})")
            data = data | sql_values

    # Build where clauses string.
    where_string = ""
    if len(where_clauses) != 0:
        where_string = " WHERE " + app_utils.list_to_string(where_clauses, " AND ")

    # Build SQL statement.
    sql = f"""SELECT id, media FROM Individuals{where_string};"""

    # DB execution.
    result = app_utils.execute_sql(sql, data, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR

    # Dict format returns the individual ids and media ids in a dictionary indexed by the media ids.
    if dict_format is True:
        result_dict = {}
        for item in result:
            result_dict[item[1]] = item[0]
        return result_dict

    # Otherwise just return a list of media ids.
    result = [(item[0]) for item in result]
    return list(set(result))
