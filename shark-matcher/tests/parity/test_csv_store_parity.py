"""Local validation parity tests comparing CSV store query outputs against the live SQLite database."""

import pytest

from src.csv_store._loader import load_individuals
import src.csv_store.individual as csv_individual
import src.csv_store.media as csv_media


class TestGetMedias:

    def test_count_sync(self, db):
        _assert_media_counts_match(db)

    def test_all_ids_match(self, db):
        _assert_media_counts_match(db)
        db_ids = sorted(_db_column(db, "SELECT id FROM Media"))
        csv_ids = sorted(m.id for m in csv_media.get_medias())
        assert db_ids == csv_ids

    def test_core_attributes(self, db):
        _assert_media_counts_match(db)
        db_rows = {
            row["id"]: dict(row)
            for row in db.execute(
                "SELECT id, name, filename, location, time, type, quality FROM Media"
            )
        }
        csv_objs = {m.id: m for m in csv_media.get_medias()}

        for mid, db_row in db_rows.items():
            assert mid in csv_objs, f"Media id {mid} missing from CSV store"
            m = csv_objs[mid]
            assert m.name == db_row["name"],         f"name mismatch for media {mid}"
            assert m.filename == db_row["filename"],  f"filename mismatch for media {mid}"
            assert m.location == db_row["location"],  f"location mismatch for media {mid}"
            assert m.time == db_row["time"],          f"time mismatch for media {mid}"
            assert m.type == db_row["type"],          f"type mismatch for media {mid}"
            assert m.quality == db_row["quality"],    f"quality mismatch for media {mid}"

    def test_filter_by_ids(self, db):
        _assert_media_counts_match(db)
        all_ids = _db_column(db, "SELECT id FROM Media LIMIT 10")
        sample = all_ids[:5]

        db_result = sorted(_db_column(
            db, f"SELECT id FROM Media WHERE id IN ({','.join('?'*len(sample))})", sample))
        csv_result = sorted(m.id for m in csv_media.get_medias(ids=sample))
        assert db_result == csv_result

    def test_get_full_attaches_individual(self, db):
        """When get_full=True, every Media that has a linked Individual should have
        the individual attribute populated."""
        _assert_media_counts_match(db)

        # Find media that have an individual in the DB.
        db_linked = {
            row["media"]: row["id"]
            for row in db.execute(
                "SELECT id, media FROM Individuals WHERE media IS NOT NULL AND id >= 0"
            )
        }

        sample_media_ids = list(db_linked.keys())[:20]
        csv_objs = {m.id: m for m in csv_media.get_medias(ids=sample_media_ids, get_full=True)}

        for media_id, ind_id in db_linked.items():
            if media_id not in csv_objs:
                continue
            m = csv_objs[media_id]
            assert m.individual is not None, \
                f"get_full=True: individual not attached to media {media_id}"
            assert m.individual.id == ind_id, \
                f"Wrong individual on media {media_id}: expected {ind_id}, got {m.individual.id}"


class TestGetMatchingMediaIds:

    def test_no_filter_returns_all(self, db):
        _assert_media_counts_match(db)
        db_ids = sorted(_db_column(db, "SELECT id FROM Media"))
        csv_ids = sorted(csv_media.get_matching_media_ids())
        assert db_ids == csv_ids

    def test_filter_by_ids(self, db):
        _assert_media_counts_match(db)
        sample = _db_column(db, "SELECT id FROM Media LIMIT 8")
        db_ids = sorted(_db_column(
            db, f"SELECT id FROM Media WHERE id IN ({','.join('?'*len(sample))})", sample))
        csv_ids = sorted(csv_media.get_matching_media_ids(ids=sample))
        assert db_ids == csv_ids

    def test_filter_by_type(self, db):
        _assert_media_counts_match(db)
        types = _db_column(db, "SELECT DISTINCT type FROM Media WHERE type IS NOT NULL LIMIT 2")
        if not types:
            pytest.skip("No typed media in dataset")
        for t in types:
            db_ids = sorted(_db_column(
                db, "SELECT id FROM Media WHERE type = ? COLLATE NOCASE", (t,)))
            csv_ids = sorted(csv_media.get_matching_media_ids(types=[t]))
            assert db_ids == csv_ids, f"type filter mismatch for type={t!r}"

    def test_filter_by_location(self, db):
        _assert_media_counts_match(db)
        locs = _db_column(
            db, "SELECT DISTINCT location FROM Media WHERE location IS NOT NULL LIMIT 2")
        if not locs:
            pytest.skip("No media with location in dataset")
        for loc in locs:
            db_ids = sorted(_db_column(
                db, "SELECT id FROM Media WHERE location = ? COLLATE NOCASE", (loc,)))
            csv_ids = sorted(csv_media.get_matching_media_ids(locations=[loc]))
            assert db_ids == csv_ids, f"location filter mismatch for location={loc!r}"

    def test_filter_by_min_quality(self, db):
        _assert_media_counts_match(db)
        quality_order = ['unmatchable', 'poor', 'medium', 'good']
        for threshold in quality_order:
            # Build the expected set: threshold and anything above it.
            allowed = quality_order[quality_order.index(threshold):]
            placeholders = ','.join('?' * len(allowed))
            db_ids = sorted(_db_column(
                db,
                f"SELECT id FROM Media WHERE quality IS NOT NULL "
                f"AND lower(quality) IN ({placeholders})",
                allowed,
            ))
            csv_ids = sorted(csv_media.get_matching_media_ids(qualities=[threshold]))
            assert db_ids == csv_ids, f"min quality mismatch for threshold={threshold!r}"

    def test_filter_by_min_quality_allow_unknown(self, db):
        _assert_media_counts_match(db)
        quality_order = ['unmatchable', 'poor', 'medium', 'good']
        threshold = quality_order[0]  # 'unmatchable' -> all qualities + NULLs
        allowed = quality_order
        placeholders = ','.join('?' * len(allowed))
        db_ids = sorted(_db_column(
            db,
            f"SELECT id FROM Media WHERE quality IS NULL "
            f"OR lower(quality) IN ({placeholders})",
            allowed,
        ))
        csv_ids = sorted(csv_media.get_matching_media_ids(
            qualities=[threshold], allow_unknown=True))
        assert db_ids == csv_ids

    def test_filter_by_time_range(self, db):
        _assert_media_counts_match(db)
        times = _db_column(
            db, "SELECT time FROM Media WHERE time IS NOT NULL ORDER BY time")
        if len(times) < 2:
            pytest.skip("Not enough timed media in dataset")
        min_t = times[len(times) // 4]
        max_t = times[3 * len(times) // 4]

        db_ids = sorted(_db_column(
            db,
            "SELECT id FROM Media WHERE time IS NOT NULL AND time >= ? AND time <= ?",
            (min_t, max_t),
        ))
        csv_ids = sorted(csv_media.get_matching_media_ids(min_time=min_t, max_time=max_t))
        assert db_ids == csv_ids

    def test_combined_filters(self, db):
        _assert_media_counts_match(db)
        # Type + location together.
        pairs = db.execute(
            "SELECT DISTINCT type, location FROM Media "
            "WHERE type IS NOT NULL AND location IS NOT NULL LIMIT 1"
        ).fetchall()
        if not pairs:
            pytest.skip("No media with both type and location set")
        t, loc = pairs[0]["type"], pairs[0]["location"]
        db_ids = sorted(_db_column(
            db,
            "SELECT id FROM Media WHERE type = ? COLLATE NOCASE AND location = ? COLLATE NOCASE",
            (t, loc),
        ))
        csv_ids = sorted(csv_media.get_matching_media_ids(types=[t], locations=[loc]))
        assert db_ids == csv_ids


class TestGetIndividualIdsByMediaIds:

    def test_list_format(self, db):
        _assert_media_counts_match(db)
        _assert_individual_counts_match(db)

        all_media_ids = _db_column(db, "SELECT id FROM Media LIMIT 20")
        if not all_media_ids:
            pytest.skip("No media in dataset")

        # DB: find individuals whose media column is in the sample.
        placeholders = ','.join('?' * len(all_media_ids))
        db_ind_ids = sorted(set(_db_column(
            db,
            f"SELECT id FROM Individuals WHERE media IN ({placeholders}) AND id >= 0",
            all_media_ids,
        )))
        csv_ind_ids = sorted(csv_media.get_individual_ids_by_media_ids(all_media_ids))
        assert db_ind_ids == csv_ind_ids

    def test_dict_format(self, db):
        _assert_media_counts_match(db)
        _assert_individual_counts_match(db)

        sample_media = _db_column(db, "SELECT id FROM Media LIMIT 15")
        if not sample_media:
            pytest.skip("No media in dataset")

        placeholders = ','.join('?' * len(sample_media))
        db_rows = db.execute(
            f"SELECT id, media FROM Individuals "
            f"WHERE media IN ({placeholders}) AND id >= 0",
            sample_media,
        ).fetchall()
        db_dict = {row["media"]: row["id"] for row in db_rows}
        csv_dict = csv_media.get_individual_ids_by_media_ids(sample_media, dict_format=True)
        assert db_dict == csv_dict

    def test_unknown_media_ids_are_ignored(self, db):
        csv_result = csv_media.get_individual_ids_by_media_ids([999999999])
        assert csv_result == []


class TestIsMediaIndividual:

    def test_all_individuals_classified_correctly(self, db):
        _assert_individual_counts_match(db)

        db_media_ids = set(_db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND media IS NOT NULL AND source != id",
        ))
        db_non_media_ids = set(_db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND (media IS NULL OR source = id)",
        ))

        for ind in load_individuals():
            if ind.id < 0:
                continue
            result = csv_individual.is_media_individual(ind)
            if ind.id in db_media_ids:
                assert result is True, \
                    f"Expected is_media_individual=True for ind {ind.id}"
            elif ind.id in db_non_media_ids:
                assert result is False, \
                    f"Expected is_media_individual=False for ind {ind.id}"


class TestGetIndividuals:

    def test_count_sync(self, db):
        _assert_individual_counts_match(db)

    def test_source_only_ids_match(self, db):
        _assert_individual_counts_match(db)
        db_ids = sorted(_db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL",
        ))
        csv_ids = _sorted_ids(csv_individual.get_individuals(
            ids=None, get_full=False, source_only=True))
        assert db_ids == csv_ids

    def test_all_ids_match(self, db):
        _assert_individual_counts_match(db)
        db_ids = sorted(_db_column(
            db, "SELECT id FROM Individuals WHERE id >= 0"))
        csv_ids = sorted(csv_individual.get_matching_individual_ids(
            source_only=False, allow_unknown=True))
        assert db_ids == csv_ids

    def test_core_attributes(self, db):
        _assert_individual_counts_match(db)
        db_rows = {
            row["id"]: dict(row)
            for row in db.execute(
                "SELECT id, name, sex, length, media, source, reference "
                "FROM Individuals WHERE id >= 0"
            )
        }
        csv_objs = {
            i.id: i
            for i in csv_individual.get_individuals(
                ids=None, get_full=False, source_only=False, allow_unknown=True)
        }

        for ind_id, db_row in db_rows.items():
            assert ind_id in csv_objs, f"Individual id {ind_id} missing from CSV store"
            i = csv_objs[ind_id]
            assert i.name == db_row["name"],         f"name mismatch for individual {ind_id}"
            assert i.sex == db_row["sex"],           f"sex mismatch for individual {ind_id}"
            assert i.length == db_row["length"],     f"length mismatch for individual {ind_id}"
            assert i.media == db_row["media"],       f"media mismatch for individual {ind_id}"
            assert i.source == db_row["source"],     f"source mismatch for individual {ind_id}"
            assert i.reference == db_row["reference"], \
                f"reference mismatch for individual {ind_id}"

    def test_filter_by_ids(self, db):
        _assert_individual_counts_match(db)
        sample = _db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL LIMIT 5",
        )
        if not sample:
            pytest.skip("No source individuals in dataset")
        db_ids = sorted(sample)
        csv_ids = _sorted_ids(
            csv_individual.get_individuals(ids=sample, get_full=False))
        assert db_ids == csv_ids

    def test_get_full_associated_individual_count(self, db):
        """Each source individual's associated_individuals count must match the DB."""
        _assert_individual_counts_match(db)
        source_ids = _db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL",
        )
        if not source_ids:
            pytest.skip("No source individuals in dataset")

        # DB: count of associated individuals per source.
        db_counts = {}
        for sid in source_ids:
            count = _db_scalar(
                db, "SELECT COUNT(*) FROM Individuals WHERE source = ? AND id >= 0", (sid,))
            db_counts[sid] = count

        csv_objs = {
            i.id: i
            for i in csv_individual.get_individuals(
                ids=source_ids, get_full=True, source_only=True)
        }

        for sid, expected_count in db_counts.items():
            assert sid in csv_objs, f"Source individual {sid} missing"
            actual = len(csv_objs[sid].associated_individuals)
            assert actual == expected_count, (
                f"associated_individuals count mismatch for source {sid}: "
                f"DB={expected_count}, CSV={actual}"
            )

    def test_get_full_associated_media_ids(self, db):
        """Each source individual's associated_medias must contain exactly the media
        records linked to its media individuals."""
        _assert_individual_counts_match(db)
        _assert_media_counts_match(db)
        source_ids = _db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL",
        )
        if not source_ids:
            pytest.skip("No source individuals in dataset")

        csv_objs = {
            i.id: i
            for i in csv_individual.get_individuals(
                ids=source_ids, get_full=True, source_only=True)
        }

        for sid in source_ids:
            # DB: collect media ids belonging to this source individual's media individuals.
            db_media_ids = sorted(_db_column(
                db,
                "SELECT media FROM Individuals "
                "WHERE source = ? AND id >= 0 AND media IS NOT NULL",
                (sid,),
            ))
            csv_media_ids = sorted(
                m.id for m in csv_objs[sid].associated_medias
                if m is not None
            )
            assert db_media_ids == csv_media_ids, (
                f"associated_medias mismatch for source {sid}: "
                f"DB={db_media_ids}, CSV={csv_media_ids}"
            )

    def test_get_full_associated_individual_ids(self, db):
        """Each source individual's associated_individuals IDs must match the DB."""
        _assert_individual_counts_match(db)
        source_ids = _db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL",
        )
        if not source_ids:
            pytest.skip("No source individuals in dataset")

        csv_objs = {
            i.id: i
            for i in csv_individual.get_individuals(
                ids=source_ids, get_full=True, source_only=True)
        }

        for sid in source_ids:
            db_assoc_ids = sorted(_db_column(
                db,
                "SELECT id FROM Individuals WHERE source = ? AND id >= 0",
                (sid,),
            ))
            csv_assoc_ids = sorted(a.id for a in csv_objs[sid].associated_individuals)
            assert db_assoc_ids == csv_assoc_ids, (
                f"associated_individuals IDs mismatch for source {sid}: "
                f"DB={db_assoc_ids}, CSV={csv_assoc_ids}"
            )

    def test_get_source_traverses_to_source_individual(self, db):
        """Passing a media individual id with get_source=True should return the
        source individual, not the media individual itself."""
        _assert_individual_counts_match(db)
        rows = db.execute(
            "SELECT id, source FROM Individuals "
            "WHERE id >= 0 AND source != id AND media IS NOT NULL LIMIT 5"
        ).fetchall()
        if not rows:
            pytest.skip("No media individuals in dataset")

        for row in rows:
            media_ind_id = row["id"]
            expected_source_id = row["source"]
            result = csv_individual.get_individuals(
                ids=[media_ind_id], get_full=False, get_source=True)
            assert len(result) == 1
            assert result[0].id == expected_source_id, (
                f"get_source=True: expected source {expected_source_id} "
                f"for media individual {media_ind_id}, got {result[0].id}"
            )


class TestGetMatchingIndividualIds:

    def test_no_filter_source_only(self, db):
        _assert_individual_counts_match(db)
        db_ids = sorted(_db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL",
        ))
        csv_ids = sorted(csv_individual.get_matching_individual_ids(source_only=True))
        assert db_ids == csv_ids

    def test_filter_by_ids(self, db):
        _assert_individual_counts_match(db)
        sample = _db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL LIMIT 5",
        )
        if not sample:
            pytest.skip("No source individuals")
        db_ids = sorted(sample)
        csv_ids = sorted(csv_individual.get_matching_individual_ids(ids=sample))
        assert db_ids == csv_ids

    def test_filter_by_sex(self, db):
        _assert_individual_counts_match(db)
        sexes = _db_column(
            db,
            "SELECT DISTINCT sex FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL AND sex IS NOT NULL",
        )
        if not sexes:
            pytest.skip("No sex data in dataset")
        for s in sexes:
            db_ids = sorted(_db_column(
                db,
                "SELECT id FROM Individuals "
                "WHERE id >= 0 AND source = id AND reference IS NOT NULL AND sex = ?",
                (s,),
            ))
            # CSV store accepts sex as a list.
            csv_ids = sorted(csv_individual.get_matching_individual_ids(sex=[s]))
            assert db_ids == csv_ids, f"sex filter mismatch for sex={s!r}"

    def test_filter_by_sex_allow_unknown(self, db):
        _assert_individual_counts_match(db)
        # With allow_unknown=True, individuals with sex IS NULL are also included.
        sexes = _db_column(
            db,
            "SELECT DISTINCT sex FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL AND sex IS NOT NULL LIMIT 1",
        )
        if not sexes:
            pytest.skip("No sex data in dataset")
        s = sexes[0]
        db_ids = sorted(_db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL "
            "AND (sex = ? OR sex IS NULL)",
            (s,),
        ))
        csv_ids = sorted(csv_individual.get_matching_individual_ids(
            sex=[s], allow_unknown=True))
        assert db_ids == csv_ids

    def test_filter_by_min_length(self, db):
        _assert_individual_counts_match(db)
        lengths = _db_column(
            db,
            "SELECT DISTINCT length FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL AND length IS NOT NULL "
            "ORDER BY length",
        )
        if len(lengths) < 2:
            pytest.skip("Not enough length data in dataset")
        threshold = lengths[len(lengths) // 2]
        db_ids = sorted(_db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL "
            "AND length IS NOT NULL AND length >= ?",
            (threshold,),
        ))
        csv_ids = sorted(csv_individual.get_matching_individual_ids(min_length=threshold))
        assert db_ids == csv_ids

    def test_filter_by_max_length(self, db):
        _assert_individual_counts_match(db)
        lengths = _db_column(
            db,
            "SELECT DISTINCT length FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL AND length IS NOT NULL "
            "ORDER BY length",
        )
        if len(lengths) < 2:
            pytest.skip("Not enough length data in dataset")
        threshold = lengths[len(lengths) // 2]
        db_ids = sorted(_db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL "
            "AND length IS NOT NULL AND length <= ?",
            (threshold,),
        ))
        csv_ids = sorted(csv_individual.get_matching_individual_ids(max_length=threshold))
        assert db_ids == csv_ids

    def test_filter_by_length_range(self, db):
        _assert_individual_counts_match(db)
        lengths = _db_column(
            db,
            "SELECT DISTINCT length FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL AND length IS NOT NULL "
            "ORDER BY length",
        )
        if len(lengths) < 3:
            pytest.skip("Not enough length data in dataset")
        lo = lengths[len(lengths) // 4]
        hi = lengths[3 * len(lengths) // 4]
        db_ids = sorted(_db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL "
            "AND length IS NOT NULL AND length >= ? AND length <= ?",
            (lo, hi),
        ))
        csv_ids = sorted(csv_individual.get_matching_individual_ids(
            min_length=lo, max_length=hi))
        assert db_ids == csv_ids

    def test_combined_sex_and_length(self, db):
        _assert_individual_counts_match(db)
        row = db.execute(
            "SELECT sex, length FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL "
            "AND sex IS NOT NULL AND length IS NOT NULL LIMIT 1"
        ).fetchone()
        if not row:
            pytest.skip("No individuals with both sex and length")
        s, lo = row["sex"], row["length"]
        db_ids = sorted(_db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL "
            "AND sex = ? AND length IS NOT NULL AND length >= ?",
            (s, lo),
        ))
        csv_ids = sorted(csv_individual.get_matching_individual_ids(
            sex=[s], min_length=lo))
        assert db_ids == csv_ids

    def test_allow_unknown_length(self, db):
        _assert_individual_counts_match(db)
        lengths = _db_column(
            db,
            "SELECT DISTINCT length FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL AND length IS NOT NULL "
            "ORDER BY length LIMIT 1",
        )
        if not lengths:
            pytest.skip("No length data in dataset")
        threshold = lengths[0]
        db_ids = sorted(_db_column(
            db,
            "SELECT id FROM Individuals "
            "WHERE id >= 0 AND source = id AND reference IS NOT NULL "
            "AND (length IS NULL OR length >= ?)",
            (threshold,),
        ))
        csv_ids = sorted(csv_individual.get_matching_individual_ids(
            min_length=threshold, allow_unknown=True))
        assert db_ids == csv_ids


def _db_scalar(db, sql, params=()):
    """Execute query and return the first column of the first row."""
    return db.execute(sql, params).fetchone()[0]


def _db_column(db, sql, params=()):
    """Execute query and return all values from the first column."""
    return [row[0] for row in db.execute(sql, params).fetchall()]


def _sorted_ids(objs):
    """Return sorted IDs from an iterable of objects."""
    return sorted(o.id for o in objs)


def _obj_attrs(obj, *attrs):
    """Extract a tuple of attributes from an object."""
    return tuple(getattr(obj, a) for a in attrs)


def _assert_individual_counts_match(db):
    """Assert that database and CSV store individual counts match."""
    db_total = _db_scalar(db, "SELECT COUNT(*) FROM Individuals WHERE id >= 0")
    csv_total = len([i for i in csv_individual.get_matching_individual_ids(
        source_only=False, allow_unknown=True)])
    assert db_total == csv_total, (
        f"Individual count mismatch: DB has {db_total}, CSV has {csv_total}. "
        "Ensure PARITY_CSV_DIR matches the data used to populate PARITY_DB_PATH."
    )


def _assert_media_counts_match(db):
    """Assert that database and CSV store media counts match."""
    db_total = _db_scalar(db, "SELECT COUNT(*) FROM Media")
    csv_total = len(csv_media.get_matching_media_ids())
    assert db_total == csv_total, (
        f"Media count mismatch: DB has {db_total}, CSV has {csv_total}. "
        "Ensure PARITY_CSV_DIR matches the data used to populate PARITY_DB_PATH."
    )
