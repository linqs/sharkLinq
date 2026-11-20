# Shark Matcher App — Performance Audit

> **Status:** Historical as-found audit with a current disposition summary.

- The Implementation Status table records what was completed for V1 and what
  remains partial or pending for the planned V2 large-query work.
- The detailed query counts, line references, code samples, and proposed
  solutions below are investigation evidence, not maintained source documentation.
- When they differ, the status table and current source take precedence.

## Implementation Status

| ID | Description | Status | Notes |
|---|---|---|---|
| B1 | No database indexes | **Done** | 14 indexes in `db/sql/add_indexes.sql`, applied by `init` and to existing `.db` files |
| B2 | `get_individuals(ids=None)` double round trip | **Done** | `backend/db/individual.py` — `ids=None` now issues one direct `SELECT` instead of `get_all_ids` + source-resolution + `WHERE id IN` |
| B6 | Unconditional `individual_ids_get_source_individual_ids(None)` | **Done** | Folded into B9 fix; call is now guarded by `if candidate_individual_ids is not None` |
| B9 | One SQLite connection per query in multi-query paths | **Done** | Shared connection plumbed through `filter_individuals`, `get_individuals` handler, `_get_keyword_individual_ids`, `_get_individual_with_distances`, `get_matches`, `filter_matches`, `_get_neighbors`, and `filter_media` |
| B3 | `get_matches()` loads entire Individuals table | Pending | |
| B4 | Keyword search fires 5 sequential queries | **Done** | `frontend/individual.py:_get_keyword_individual_ids` — 7 sequential queries + Python set.union() replaced with a single 6-branch UNION query |
| B5 | `filter_individuals` ~18 sequential round trips | Partial | B9 eliminates per-query connection overhead; set-intersection query count unchanged |
| B7 | `get_matching_associated_object_individual_ids(any=True)` double query | **Done** | `backend/db/individual.py:845` — replaced `get_all_ids` + `WHERE id IN` with single `SELECT DISTINCT individual WHERE individual >= 0` |
| B8 | `get_source_individual_ids_by_association_count` no-args redundancy | **Done** | `backend/db/individual.py:748` — removed `get_all_ids` early return; `having_string` is now conditionally built, GROUP BY query handles all cases |
| B10 | No pagination / LIMIT on list endpoints | Partial | Opt-in `paginate`/`page`/`page_size` on individual, action and proposal filters; matching is unpaginated |

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Flow for Large Queries](#2-data-flow-for-large-queries)
3. [Identified Bottlenecks](#3-identified-bottlenecks)
   - [B1 — No Database Indexes](#b1--no-database-indexes)
   - [B2 — `get_individuals(ids=None)` Makes Two SQL Round Trips](#b2--get_individualsidsnone-makes-two-sql-round-trips)
   - [B3 — `get_matches()` Fetches the Entire Individual Table](#b3--get_matches-fetches-the-entire-individual-table)
   - [B4 — Keyword Search Fires Five Sequential Queries](#b4--keyword-search-fires-five-sequential-queries)
   - [B5 — `filter_individuals()` Makes ~15–20 Sequential DB Round Trips](#b5--filter_individuals-makes-1520-sequential-db-round-trips)
   - [B6 — Unconditional `individual_ids_get_source_individual_ids(None)` Call](#b6--unconditional-individual_ids_get_source_individual_idsnone-call)
   - [B7 — `get_matching_associated_object_individual_ids(any=True)` Fetches All IDs Then Immediately Discards Them](#b7--get_matching_associated_object_individual_idsanytrue-fetches-all-ids-then-immediately-discards-them)
   - [B8 — `get_source_individual_ids_by_association_count()` With No Args Runs Two Queries](#b8--get_source_individual_ids_by_association_count-with-no-args-runs-two-queries)
   - [B9 — One SQLite Connection Per Query in Multi-Query Paths](#b9--one-sqlite-connection-per-query-in-multi-query-paths)
   - [B10 — No Pagination / LIMIT on Any List Endpoint](#b10--no-pagination--limit-on-any-list-endpoint)
4. [Summary Table](#4-summary-table)
5. [Optimized Solutions](#5-optimized-solutions)
   - [O1 — Add Database Indexes](#o1--add-database-indexes)
   - [O2 — Remove the Double Round Trip in `get_individuals`](#o2--remove-the-double-round-trip-in-get_individuals)
   - [O3 — Only Fetch Relevant Individuals in `get_matches`](#o3--only-fetch-relevant-individuals-in-get_matches)
   - [O4 — Collapse Keyword Search into a Single Query](#o4--collapse-keyword-search-into-a-single-query)
   - [O5 — Reduce `filter_individuals` Round Trips With a Shared Connection](#o5--reduce-filter_individuals-round-trips-with-a-shared-connection)
   - [O6 — Remove the Redundant `individual_ids_get_source_individual_ids(None)` Call](#o6--remove-the-redundant-individual_ids_get_source_individual_idsnone-call)
   - [O7 — Fix the `any=True` Path in `get_matching_associated_object_individual_ids`](#o7--fix-the-anytrue-path-in-get_matching_associated_object_individual_ids)
   - [O8 — Fix the No-Args Path in `get_source_individual_ids_by_association_count`](#o8--fix-the-no-args-path-in-get_source_individual_ids_by_association_count)
   - [O9 — Reuse a Single Connection Across `get_individuals` Sub-Queries](#o9--reuse-a-single-connection-across-get_individuals-sub-queries)
   - [O10 — Add Pagination to List Endpoints](#o10--add-pagination-to-list-endpoints)
6. [Implementation Priority](#6-implementation-priority)

---

## 1. System Overview

The shark-matcher app uses a SQLite database accessed directly from Python. There is no ORM; all queries are hand-built SQL executed via `app_utils.execute_sql`. The key hot paths from the UI perspective are:

| UI Action | Backend Call Chain |
|---|---|
| View all sharks | `GET /api/individual/filter` -> `filter_individuals()` -> `get_individuals(ids=None, get_full=True)` |
| Search / filter sharks | `GET /api/individual/filter` -> `filter_individuals()` (multi-step set intersection) -> `get_individuals(ids=..., get_full=True)` |
| Get match results | `GET /api/match/filter` -> `filter_matches()` -> `_get_neighbors()` (inference service) + `filter_individuals()` -> `_process_individual_distances()` |
| View all sharks (match mode, no filter) | `GET /api/match/get` -> `get_matches()` -> `get_individuals(ids=None, get_full=True, get_source=True)` |

---

## 2. Data Flow for Large Queries

### `filter_individuals()` — full unfiltered call

```
filter_individuals(no filters)
  ├─ candidate_individual_ids = None   # no work done
  ├─ individual_ids_get_source_individual_ids(None)  <- WASTED QUERY always runs
  │    └─ ids=None -> returns []
  └─ get_individuals(ids=None, get_full=True, get_source=True)
       ├─ get_all_ids("Individuals", source=True)          [SQL #1]  <- extra round trip
       ├─ individual_ids_get_source_individual_ids(all_ids)[SQL #2]  <- redundant (ids already source IDs)
       ├─ SELECT * FROM Individuals WHERE id IN (all_ids)  [SQL #3]
       ├─ SELECT * FROM Individuals WHERE source IN (...)  [SQL #4]
       ├─ SELECT * FROM AlternateNames WHERE individual IN [SQL #5]
       ├─ SELECT * FROM Tags WHERE individual IN (...)     [SQL #6]
       ├─ SELECT * FROM Labels WHERE individual IN (...)   [SQL #7]
       └─ SELECT * FROM Media WHERE id IN (...)            [SQL #8]
  Total: 8 SQL queries, each opening and closing its own SQLite connection
```

### `filter_individuals()` — with filters applied

```
filter_individuals(keywords, locations, qualities, min_time, sex, min_associated)
  ├─ _get_keyword_individual_ids(keywords)         <- 5 queries (B4)
  │    ├─ get_matching_associated_object_individual_ids(AlternateNames) [SQL #1]
  │    ├─ get_matching_associated_object_individual_ids(Tags)           [SQL #2]
  │    ├─ get_matching_associated_object_individual_ids(Labels)         [SQL #3]
  │    ├─ get_matching_media_ids(keywords=...)                          [SQL #4]
  │    └─ get_matching_individual_ids(keywords=...)                     [SQL #5]
  ├─ get_matching_media_ids(locations, qualities, min_time)             [SQL #6]
  ├─ get_individual_ids_by_media_ids(...)                               [SQL #7]
  ├─ individual_ids_get_source_individual_ids(media_individual_ids)     [SQL #8]
  ├─ individual_ids_get_source_individual_ids(candidate_ids) <- ALWAYS  [SQL #9] (B6)
  ├─ get_matching_individual_ids(sex=...)                               [SQL #10]
  ├─ get_source_individual_ids_by_association_count(min_associated=...) [SQL #11]
  └─ get_individuals(ids=filtered_ids, get_full=True, get_source=True)
       ├─ individual_ids_get_source_individual_ids(filtered_ids)        [SQL #12]
       ├─ SELECT * FROM Individuals WHERE id IN (filtered_ids)          [SQL #13]
       ├─ SELECT * FROM Individuals WHERE source IN (...)               [SQL #14]
       ├─ SELECT * FROM AlternateNames WHERE individual IN (...)        [SQL #15]
       ├─ SELECT * FROM Tags WHERE individual IN (...)                  [SQL #16]
       ├─ SELECT * FROM Labels WHERE individual IN (...)                [SQL #17]
       └─ SELECT * FROM Media WHERE id IN (...)                         [SQL #18]
  Total: ~18 SQL queries, each with its own SQLite connection open/close
```

### `get_matches()` — match endpoint, no filters

```
get_matches(name="123")
  ├─ _get_neighbors()
  │    ├─ get_matching_individual_ids(names=["123"])           [SQL #1]
  │    ├─ get_individuals(ids=[id], get_full=True, ...)        [SQL #2–8]
  │    └─ inference_client.get_matches(image_id=...)           [HTTP]
  ├─ get_individuals(ids=None, get_full=True, get_source=True) <- ALL INDIVIDUALS (B3)
  │    └─ 8 queries as shown above                            [SQL #9–16]
  └─ _process_individual_distances(all_individuals, distances) <- Python-side sort
  Total: ~16 SQL queries + 1 HTTP call; loads entire DB into memory
```

---

## 3. Identified Bottlenecks

### B1 — No Database Indexes

- **File:** `db/sql/create_database.sql`

The schema defines only `PRIMARY KEY` constraints (which SQLite implicitly indexes). Every other column used in `WHERE`, `JOIN`, or `ORDER BY` clauses is unindexed.

- **Impact:** Every filter, lookup, and join is a full table scan. Columns hit without indexes:

| Table | Column(s) | Used By |
|---|---|---|
| `Individuals` | `source` | All `WHERE source IN (...)` queries — fetching associated individuals |
| `Individuals` | `media` | `get_individual_ids_by_media_ids` — reverse media->individual lookup |
| `Individuals` | `name` | Name filtering, keyword search |
| `Individuals` | `reference` | `get_all_ids(source=True)` — `WHERE reference IS NOT NULL` |
| `Media` | `quality` | Quality filtering |
| `Media` | `location` | Location filtering |
| `Media` | `time` | Time-range filtering |
| `Media` | `name` | Keyword/name search |
| `AlternateNames` | `individual` | All associated-object bulk fetches |
| `AlternateNames` | `name` | Keyword search on alternate names |
| `Tags` | `individual` | All associated-object bulk fetches |
| `Tags` | `name` | Keyword search on tags |
| `Labels` | `individual` | All associated-object bulk fetches |
| `Labels` | `name` | Keyword search on labels |

- **Severity:** Critical. This is the single highest-leverage change available.

---

### B2 — `get_individuals(ids=None)` Makes Two SQL Round Trips

- **File:** `backend/db/individual.py:460–484`

```python
def get_individuals(ids=None, get_full=True, get_source=False, ...):
    if not isinstance(ids, list) or ids is None:
        ids = app_utils.get_all_ids("Individuals", source=source_only, ...)  # SQL #1
    if get_source is True:
        source_ids = individual_ids_get_source_individual_ids(ids, ...)       # SQL #2
        ids_to_fetch = list(set(source_ids))
    ...
    sql = f"SELECT ... FROM Individuals WHERE id IN {placeholders};"          # SQL #3
```

When `ids=None`, the function first calls `get_all_ids()` which runs `SELECT id FROM Individuals WHERE id >= 0`, yielding a list of all IDs. It then immediately uses those IDs to build `WHERE id IN (id_0, id_1, ...)` — which is functionally equivalent to `WHERE id >= 0`. This is two queries doing the work of one.

When `get_source=True` is also set (the common case), `individual_ids_get_source_individual_ids(all_ids)` runs a third query (`SELECT DISTINCT source FROM Individuals WHERE id IN (all_ids)`) — but since all source IDs already satisfy `source = id`, the result is identical to the input set. This is a no-op query.

- **Severity:** High. Every call to `filter_individuals()` with no ID filter hits this path.

---

### B3 — `get_matches()` Fetches the Entire Individual Table

- **File:** `frontend/match.py:60`

```python
neighbor_individual_list = backend.db.individual.get_individuals(
    ids=None, get_full=True, get_source=True
)
```

This unconditionally loads every source individual in the database — with all their associated individuals, media, tags, labels, and alternate names — into Python memory. The `_process_individual_distances()` function then iterates this full list in Python to assign distances and sort.

The inference service returns a `neighbor_distance_dict` keyed by media ID:

- An individual whose media is not in `neighbor_distance_dict` gets `distance=100000` (a sentinel) and appears last in results.
- The final sorted list is the matched individuals (by distance) followed by all unmatched individuals.

**The database is loaded entirely to display a ranked list where only a small fraction of individuals actually have computed distances.**

- **Severity:** High. This is the match endpoint's dominant cost and scales linearly with database size.

---

### B4 — Keyword Search Fires Five Sequential Queries

- **File:** `frontend/individual.py:845–891`

```python
def _get_keyword_individual_ids(keywords):
    # 3 queries: one per associated-object table
    for assoc_object in (("alternate_names", "AlternateNames"), ("tags", "Tags"), ("labels", "Labels")):
        candidate_individual_ids = candidate_individual_ids.union(
            backend.db.individual.get_matching_associated_object_individual_ids(
                keywords=keywords, table=assoc_object[1]))           # SQL #1, #2, #3

    # 2 queries: media names, then individual names
    candidate_media_ids = backend.db.media.get_matching_media_ids(keywords=keywords)   # SQL #4
    candidate_individual_ids = candidate_individual_ids.union(
        backend.db.media.get_individual_ids_by_media_ids(...))
    candidate_individual_ids = candidate_individual_ids.union(
        backend.db.individual.get_matching_individual_ids(keywords=keywords, ...))     # SQL #5
```

Five queries are fired sequentially against five different tables, each opening and closing its own connection. The results are then unioned in Python. This could be one UNION query.

- **Severity:** Medium-High. Keyword search is a common UI interaction.

---

### B5 — `filter_individuals()` Makes ~15–20 Sequential DB Round Trips

- **File:** `frontend/individual.py:83–232`

The filter logic is a Python-side set-intersection loop that fires one DB query per active filter type, then a final bulk fetch. With many filters active, this becomes:

1. Keyword search: 5 queries (B4)
2. Name filter: 1 query
3. AlternateNames / Tags / Labels: up to 3 queries
4. Media-based filters (time, location, quality): 3 queries
5. Individual property filter (sex, length): 1 query
6. Association count: 1 query
7. Unconditional `individual_ids_get_source_individual_ids`: 1 query (B6)
8. Final `get_individuals(get_full=True)`: 6 queries

Each query opens and closes a separate SQLite connection (B9). Total: up to ~21 DB round trips per filter request.

- **Severity:** Medium. The Python set-intersection approach is logically sound but the sequential execution adds latency. With indexes (B1), each individual query becomes fast, but the connection overhead and sequential nature remain.

---

### B6 — Unconditional `individual_ids_get_source_individual_ids(None)` Call

- **File:** `frontend/individual.py:180–186`

```python
# This runs regardless of whether any filter has been applied:
candidate_media_individual_ids = backend.db.individual.individual_ids_get_source_individual_ids(
    candidate_individual_ids   # May be None
)

if candidate_individual_ids is not None:
    candidate_individual_ids = set(candidate_media_individual_ids)
```

When `candidate_individual_ids` is `None` (no filters applied), this call passes `None` to `individual_ids_get_source_individual_ids`, which detects the empty input and returns `[]` immediately — but only after opening a SQLite connection. The returned `[]` is never used because the `if candidate_individual_ids is not None:` guard prevents the assignment. This is a pure waste.

When `candidate_individual_ids` is not `None`, the intent is to resolve all IDs to their source IDs. This is correct but could also be skipped if all IDs in `candidate_individual_ids` are already known to be source IDs (which is the case after the media-filter step at line 172–178 already calls this function).

- **Severity:** Low-Medium. One wasted DB call per request when no filters are applied.

---

### B7 — `get_matching_associated_object_individual_ids(any=True)` Fetches All IDs Then Immediately Discards Them

- **File:** `backend/db/individual.py:817–829`

```python
if any is True:
    ids = app_utils.get_all_ids(table, connection=connection)   # SQL #1: SELECT id FROM {table}
    sql_ids, sql_values = app_utils.build_sql_placeholders(ids)
    sql = f"""SELECT individual FROM {table} WHERE id IN {sql_ids};"""  # SQL #2: uses all IDs
    result = app_utils.execute_sql(sql, data, connection=connection)
```

When the user filters by `*` (any tag/label/alternate name), this code fetches all row IDs from the table, then immediately uses them all in a `WHERE id IN (...)` clause. This is logically equivalent to `SELECT DISTINCT individual FROM {table} WHERE individual >= 0` — one query instead of two.

- **Severity:** Low-Medium. Only triggered when `*` filter is used.

---

### B8 — `get_source_individual_ids_by_association_count()` With No Args Runs Two Queries

- **File:** `backend/db/individual.py:721–722`

```python
if not having_clauses:
    return app_utils.get_all_ids("Individuals", source=True, connection=connection)
```

When neither `min_associated` nor `max_associated` is set, this is a no-op path that returns all source individual IDs. However, this only runs when `filter_individuals` explicitly passes non-None values for these params, so it cannot be reached with no args by normal code paths. No action needed, but the code path is misleading — callers should check before calling.

- **Severity:** Low.

---

### B9 — One SQLite Connection Per Query in Multi-Query Paths

- **File:** `app_utils.py:414–449`

```python
def execute_sql(sql, data=None, commit=False, connection=None):
    is_local_connection = (connection is None)
    conn = connection or db_connect()   # opens new connection if none provided
    try:
        ...
    finally:
        if is_local_connection:
            conn.close()              # closes connection after every single query
```

`get_individuals(get_full=True)` makes 6 sequential `execute_sql` calls. When no outer `connection` is passed (the common case in `get_matches` and `filter_individuals`), each call opens and closes a distinct `sqlite3.connect()` connection. SQLite connection setup is lightweight but not free, and sequential serial I/O over many small connections adds measurable latency.

The `connection` parameter is already propagated correctly through all DB functions, meaning the fix is simply to obtain one connection at the call site and pass it through.

- **Severity:** Low-Medium. Each connection open/close is ~0.1–0.5ms, but with 15–20 per request this adds 5–10ms of pure overhead.

---

### B10 — No Pagination / LIMIT on Any List Endpoint

- **Files:** `frontend/individual.py:218`, `frontend/match.py:60`, `backend/db/individual.py:460`

None of the list endpoints (`/api/individual/filter`, `/api/match/get`, `/api/match/filter`) apply a LIMIT to the final result. The `limit` parameter exists in the DB layer signatures but is never threaded through from frontend handler to final query. Returning 10,000 individuals when the UI can only display 50 at a time serializes, transmits, and deserializes far more data than needed.

- **Severity:** Medium. Impact grows proportionally with database size. For small datasets (<500 sharks) this is acceptable; for thousands it becomes the dominant cost.

---

## 4. Summary Table

| ID | Location | Description | Severity | Correctness Risk of Fix | Status |
|---|---|---|---|---|---|
| B1 | `db/sql/create_database.sql` | No indexes on any filtered/joined column | Critical | None — indexes are transparent | Done |
| B2 | `backend/db/individual.py:460` | Double round trip when `ids=None` | High | Low — logic change, well-testable | Done |
| B3 | `frontend/match.py:60` | Full table load in `get_matches` | High | Medium — changes what is returned | Pending |
| B4 | `frontend/individual.py:845` | 5 sequential queries for keyword search | Medium-High | Low — results are equivalent | Done |
| B5 | `frontend/individual.py:83` | ~18 sequential DB round trips per filter | Medium | Low — same set-intersection logic | Partial |
| B6 | `frontend/individual.py:180` | Wasted `individual_ids_get_source_individual_ids(None)` | Low-Medium | None — pure dead code removal | Done |
| B7 | `backend/db/individual.py:817` | Double query for `any=True` path | Low-Medium | None — logically equivalent | Done |
| B8 | `backend/db/individual.py:721` | Redundant no-op path for association count | Low | None | Done |
| B9 | `app_utils.py:414` + call sites | One connection per query in multi-query paths | Low-Medium | None — connection reuse is safe | Done |
| B10 | Multiple endpoints | No pagination / LIMIT | Medium | Medium — API contract change | Partial |

---

## 5. Optimized Solutions

### O1 — Add Database Indexes

- **Addresses:** B1

Added to `db/sql/add_indexes.sql`:

```sql
-- Speed up all "get associated individuals" queries (WHERE source IN (...))
CREATE INDEX IF NOT EXISTS idx_individuals_source ON Individuals(source);

-- Speed up reverse media->individual lookups (WHERE media IN (...))
CREATE INDEX IF NOT EXISTS idx_individuals_media ON Individuals(media);

-- Speed up source individual lookups (WHERE reference IS NOT NULL)
CREATE INDEX IF NOT EXISTS idx_individuals_reference ON Individuals(reference);

-- Speed up name/keyword search on individuals
CREATE INDEX IF NOT EXISTS idx_individuals_name ON Individuals(name COLLATE NOCASE);

-- Speed up quality filtering (most common media filter)
CREATE INDEX IF NOT EXISTS idx_media_quality ON Media(quality COLLATE NOCASE);

-- Speed up location filtering
CREATE INDEX IF NOT EXISTS idx_media_location ON Media(location COLLATE NOCASE);

-- Speed up time-range filtering
CREATE INDEX IF NOT EXISTS idx_media_time ON Media(time);

-- Speed up media name/keyword search
CREATE INDEX IF NOT EXISTS idx_media_name ON Media(name COLLATE NOCASE);

-- Speed up all associated-object bulk fetches (the hottest path in get_individuals)
CREATE INDEX IF NOT EXISTS idx_alternatenames_individual ON AlternateNames(individual);
CREATE INDEX IF NOT EXISTS idx_tags_individual ON Tags(individual);
CREATE INDEX IF NOT EXISTS idx_labels_individual ON Labels(individual);

-- Speed up keyword search through associated-object names
CREATE INDEX IF NOT EXISTS idx_alternatenames_name ON AlternateNames(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_tags_name ON Tags(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_labels_name ON Labels(name COLLATE NOCASE);
```

For existing databases (without dropping and recreating), these same statements can be run as a migration script. SQLite supports `CREATE INDEX IF NOT EXISTS`, so they are safe to run multiple times.

- **Expected impact:** The three `associated-object bulk fetch` queries inside `get_individuals` — currently the dominant cost for any full-load operation — drop from full table scans to index lookups. The `WHERE source IN (...)` query for associated individuals similarly becomes an index range scan.
- **Test compatibility:** All existing tests pass unchanged. Indexes are transparent to query results.

---

### O2 — Remove the Double Round Trip in `get_individuals`

- **Addresses:** B2
- **File:** `backend/db/individual.py:460–484`

Current logic:
```python
if not isinstance(ids, list) or ids is None:
    ids = app_utils.get_all_ids("Individuals", source=source_only, ...)  # Query 1
# ...
placeholders, values = app_utils.build_sql_placeholders(ids_to_fetch, "id")
sql = f"SELECT ... FROM Individuals WHERE id IN {placeholders};"         # Query 2
```

Optimized logic — skip `get_all_ids` and query directly:
```python
if ids is None or not isinstance(ids, list):
    # Fetch all individuals directly without a pre-query for IDs
    where = "WHERE id >= 0"
    if source_only:
        where += " AND source = id AND reference IS NOT NULL"
    sql = f"SELECT id, name, sex, length, media, source, reference FROM Individuals {where};"
    main_individual_rows = app_utils.execute_sql(sql, {}, connection=connection)
    if main_individual_rows == app_constants.ERROR:
        return []
    individuals_dict = {row[5]: db_to_individual(row) for row in main_individual_rows}
else:
    # Existing path for explicit ID list
    if get_source is True:
        source_ids = individual_ids_get_source_individual_ids(ids, connection=connection)
        ids_to_fetch = list(set(source_ids))
    else:
        ids_to_fetch = ids
    placeholders, values = app_utils.build_sql_placeholders(ids_to_fetch, "id")
    sql = f"SELECT ... FROM Individuals WHERE id IN {placeholders};"
    main_individual_rows = app_utils.execute_sql(sql, values, connection=connection)
    individuals_dict = {row[5]: db_to_individual(row) for row in main_individual_rows}
```

Also, when `get_source=True` and `ids` is already a list of source IDs (the case after `filter_individuals` resolves everything to source IDs), calling `individual_ids_get_source_individual_ids` is still a round trip but returns the same set. The caller should pass `get_source=False` in that case, or the function should detect that all IDs already have `source = id`.

- **Expected impact:** Eliminates 1–2 SQL queries from every full-table call.
- **Test compatibility:** Full — behavior is identical, just fewer round trips.

---

### O3 — Only Fetch Relevant Individuals in `get_matches`

- **Addresses:** B3
- **File:** `frontend/match.py:59–76`

Current:
```python
# Fetches ALL individuals unconditionally
neighbor_individual_list = backend.db.individual.get_individuals(
    ids=None, get_full=True, get_source=True
)
neighbor_individual_list = _process_individual_distances(neighbor_individual_list, neighbor_distance_dict)
```

The match endpoint result is: matched individuals (sorted by distance, ascending) followed by all unmatched individuals (sorted by name). The UI uses this to show the closest matches first with all others still accessible below.

An optimized approach that preserves this exact semantics:

```python
# Step 1: fetch ONLY the individuals that appear in the neighbor list (fast, small set)
neighbor_media_ids = list(neighbor_distance_dict.keys())
if neighbor_media_ids:
    matched_individual_ids = backend.db.media.get_individual_ids_by_media_ids(neighbor_media_ids)
    matched_source_ids = backend.db.individual.individual_ids_get_source_individual_ids(
        matched_individual_ids)
    matched_individuals = backend.db.individual.get_individuals(
        ids=list(set(matched_source_ids)), get_full=True, get_source=False)
else:
    matched_individuals = []

# Step 2: for the "show all" tail, fetch source IDs only (not full data) to build the complete list
all_source_ids = set(app_utils.get_all_ids("Individuals", source=True))
matched_source_id_set = {ind.id for ind in matched_individuals}
unmatched_source_ids = list(all_source_ids - matched_source_id_set)
unmatched_individuals = backend.db.individual.get_individuals(
    ids=unmatched_source_ids, get_full=True, get_source=False)

# Step 3: assign distances and sort (same as before)
neighbor_individual_list = matched_individuals + unmatched_individuals
neighbor_individual_list = _process_individual_distances(
    neighbor_individual_list, neighbor_distance_dict)
```

- **Expected impact:** The "matched" set (individuals the inference model actually ranked) is typically small (tens to low hundreds). Fetching only those individuals with full data instead of all of them dramatically reduces the dominant query cost. The unmatched tail still requires a full fetch but can be lazily paginated by the frontend once pagination is added (O10).
- **Test compatibility:** Medium — `test_match.py` needs to verify the ordering of the result is preserved. The return structure (`matches`, `match_suggestions`, `query_individual`) is unchanged.

---

### O4 — Collapse Keyword Search into a Single Query

- **Addresses:** B4
- **File:** `frontend/individual.py:845–891`

Current: 5 sequential queries with Python-side `set.union`.

Optimized: one UNION query returning all matching individual IDs:

```python
def _get_keyword_individual_ids(keywords, connection=None):
    placeholders, values = app_utils.build_sql_placeholders(keywords, "kw")
    kw_in = placeholders  # e.g. (:kw_0, :kw_1)

    sql = f"""
        SELECT DISTINCT i.id
        FROM Individuals i
        WHERE i.id >= 0
          AND i.name COLLATE NOCASE IN {kw_in}

        UNION

        SELECT DISTINCT an.individual
        FROM AlternateNames an
        WHERE an.individual >= 0
          AND an.name COLLATE NOCASE IN {kw_in}

        UNION

        SELECT DISTINCT t.individual
        FROM Tags t
        WHERE t.individual >= 0
          AND t.name COLLATE NOCASE IN {kw_in}

        UNION

        SELECT DISTINCT l.individual
        FROM Labels l
        WHERE l.individual >= 0
          AND l.name COLLATE NOCASE IN {kw_in}

        UNION

        SELECT DISTINCT ind.id
        FROM Individuals ind
        JOIN Media m ON ind.media = m.id
        WHERE ind.id >= 0
          AND m.name COLLATE NOCASE IN {kw_in};
    """

    result = app_utils.execute_sql(sql, values, connection=connection)
    if result == app_constants.ERROR:
        return []
    return [row[0] for row in result]
```

Note: the `values` dict is the same for all UNION branches since SQLite named placeholders are shared within a statement. SQLite will reuse `:kw_0`, `:kw_1` etc. across all UNION branches — this is valid SQLite behavior.

- **Expected impact:** 5 connections -> 1 connection; 5 round trips -> 1 round trip. Also benefits from indexes on `name` columns (O1).
- **Test compatibility:** Full — semantics are identical (union of all matches across tables). `test_individual.py` keyword tests should pass unchanged.

---

### O5 — Reduce `filter_individuals` Round Trips With a Shared Connection

- **Addresses:** B5, B9
- **File:** `frontend/individual.py:83–232`, `backend/db/individual.py:460`

All DB functions already accept an optional `connection` parameter. The fix is to obtain one connection at the start of `filter_individuals` and pass it through to every DB call:

```python
def filter_individuals(handler, path, **kwargs):
    # ... parameter normalization ...

    connection = app_utils.db_connect()
    try:
        candidate_individual_ids = None

        if kwargs["keywords"] is not None:
            keyword_individual_ids = _get_keyword_individual_ids(
                kwargs["keywords"], connection=connection)   # <- pass connection
            ...

        if kwargs["names"] is not None:
            name_individual_ids = backend.db.individual.get_matching_individual_ids(
                names=kwargs["names"], source_only=False,
                connection=connection)                       # <- pass connection
            ...

        # ... (all subsequent DB calls get connection=connection) ...

        individual_list = backend.db.individual.get_individuals(
            ids=ids_to_fetch, get_full=True, get_source=True,
            connection=connection)                           # <- pass connection
    finally:
        connection.close()
```

The same pattern applies to `get_matches()` and `filter_matches()`.

- **Expected impact:** Reduces SQLite connection open/close overhead from ~18 per request to 1. On a local SQLite file, each connection open takes approximately 0.1–0.5ms, so this saves ~5–10ms of pure overhead per filter request. More importantly, it opens the door to read isolation and consistent snapshots.
- **Test compatibility:** Full — connection passing is already supported throughout. All tests pass unchanged.

---

### O6 — Remove the Redundant `individual_ids_get_source_individual_ids(None)` Call

- **Addresses:** B6
- **File:** `frontend/individual.py:180–186`

#### Current Code

Always runs, even with no filters:

```python
candidate_media_individual_ids = backend.db.individual.individual_ids_get_source_individual_ids(
    candidate_individual_ids  # May be None -> returns [] immediately after opening a connection
)

if candidate_individual_ids is not None:
    candidate_individual_ids = set(candidate_media_individual_ids)
```

#### Problem

- The guard `if candidate_individual_ids is not None` means the assignment only happens when `candidate_individual_ids` is already set. The function call itself always runs, even when it is `None`.
- When `candidate_individual_ids` is not None, the purpose is to resolve all IDs (which may include media individual IDs from the keyword/name search paths) to their source IDs.
- After the media-filter block at lines 172–178, `candidate_media_individual_ids` was already set to source IDs by the call at line 172.
- Calling it again at line 180 with the combined set is correct but may include IDs that are already source IDs.

#### Fix

```python
# Only resolve to source IDs if candidate set exists and wasn't already resolved
if candidate_individual_ids is not None:
    candidate_individual_ids = set(
        backend.db.individual.individual_ids_get_source_individual_ids(
            list(candidate_individual_ids), connection=connection
        )
    )
```

- **Test compatibility:** Full — behavior is identical when `candidate_individual_ids is not None`, and the wasted call when it is `None` is eliminated.

---

### O7 — Fix the `any=True` Path in `get_matching_associated_object_individual_ids`

- **Addresses:** B7
- **File:** `backend/db/individual.py:817–829`

Current:
```python
if any is True:
    ids = app_utils.get_all_ids(table, connection=connection)   # SQL #1
    sql_ids, sql_values = app_utils.build_sql_placeholders(ids)
    sql = f"""SELECT individual FROM {table} WHERE id IN {sql_ids};"""  # SQL #2
```

Optimized:
```python
if any is True:
    sql = f"""SELECT DISTINCT individual FROM {table} WHERE individual >= 0;"""
    result = app_utils.execute_sql(sql, {}, connection=connection)
    if result == app_constants.ERROR:
        return app_constants.ERROR
    return [item[0] for item in result]
```

- **Test compatibility:** Full — returns identical set of individual IDs.

---

### O8 — Fix the No-Args Path in `get_source_individual_ids_by_association_count`

- **Addresses:** B8
- **File:** `backend/db/individual.py:721–722`

#### Current Code

The no-args fallback:

```python
if not having_clauses:
    return app_utils.get_all_ids("Individuals", source=True, connection=connection)
```

#### Problem

- This path is only reachable if `filter_individuals` passes non-None values for `min_associated`/`max_associated` (otherwise the outer `if` in `filter_individuals` prevents calling this function).
- It is internally inconsistent: if neither bound is specified, all source individuals qualify, but calling `get_all_ids` with `source=True` uses `WHERE reference IS NOT NULL` which excludes archive individuals.

#### Fix

The GROUP BY query itself would be the correct implementation for all paths:

```python
if not having_clauses:
    # No count constraints — all source individuals qualify
    sql = """SELECT DISTINCT source FROM Individuals WHERE id >= 0 AND source = id AND reference IS NOT NULL;"""
    result = app_utils.execute_sql(sql, {}, connection=connection)
    if result == app_constants.ERROR:
        return []
    return [row[0] for row in result]
```

This is consistent with what `get_all_ids("Individuals", source=True)` returns and avoids the slightly different semantics of that utility.

- **Test compatibility:** Full.

---

### O9 — Reuse a Single Connection Across `get_individuals` Sub-Queries

- **Addresses:** B9
- **File:** `backend/db/individual.py:460–606`

`get_individuals` already accepts a `connection` parameter and passes it to most sub-calls. The gap is at the call site.

- As described in O5, passing a connection from `filter_individuals`, `get_matches`, and `filter_matches` eliminates the repeated open/close cycle.
- Within `get_individuals` itself, the call to `individual_ids_get_source_individual_ids` at line 478 and the call to `backend.db.media.get_medias` at line 575 both receive the `connection` parameter correctly. They benefit automatically once a connection is passed from the outer call site.
- No code changes are needed within `get_individuals` itself; changes are at the call sites (O5 covers this).

---

### O10 — Add Pagination to List Endpoints

- **Addresses:** B10
- **Files:** `frontend/individual.py`, `frontend/match.py`, `backend/db/individual.py`

Add `page` (default `0`) and `page_size` (default `None`, meaning no limit) parameters to `filter_individuals` and `get_matches`. Apply `LIMIT` and `OFFSET` in the final `get_individuals` call:

```python
# In filter_individuals, before the final get_individuals call:
page = kwargs.get("page", 0)
page_size = kwargs.get("page_size")  # None = no pagination (current behavior)

if ids_to_fetch is not None and page_size is not None:
    start = page * page_size
    ids_to_fetch = ids_to_fetch[start : start + page_size]
```

At the DB layer, the `limit` parameter already exists in `get_individuals`. The frontend layer just needs to thread `page` and `page_size` through, and return the total count alongside the page:

```python
return {
    "individuals": return_individual_list,
    "total": len(all_candidate_ids),  # count before pagination
    "page": page,
    "page_size": page_size,
}, http.HTTPStatus.OK, None
```

This is a breaking API change and requires frontend JS updates to consume `total` and issue paginated requests. Implement after the non-breaking fixes.

- **Test compatibility:** Medium — tests that assert on full result sets must be updated to use `page_size=None` or a sufficiently large value. New tests should cover paginated behavior.

---

## 6. Implementation Priority

| Priority | Fix | Expected Latency Reduction | Risk | Status |
|---|---|---|---|---|
| 1 | **O1** — Add indexes | 60–90% on filtered queries | None | Done |
| 2 | **O6** — Remove wasted connection call | Eliminates one useless round trip | None | Done |
| 3 | **O2** — Remove double round trip in `get_individuals(ids=None)` | Saves 1–2 queries per full fetch | Low | Done |
| 4 | **O5 + O9** — Share connection across all multi-query paths | Saves per-query open/close overhead (~5–10ms per request) | None | Done |
| 5 | **O7** — Fix `any=True` double query | Saves one query per `*` filter use | None | Done |
| 6 | **O4** — Collapse keyword search to single UNION query | Saves 4 queries per keyword search | Low | Done |
| 7 | **O3** — Limit `get_matches` to relevant individuals | Largest absolute reduction for match endpoint | Medium | Pending |
| 8 | **O10** — Pagination | Scales the app to large datasets | Medium (API change) | Pending |

Items 1–6 are implemented. Item 7 is pending and would not change return values or API contracts; item 8 is a breaking API change requiring frontend updates.
