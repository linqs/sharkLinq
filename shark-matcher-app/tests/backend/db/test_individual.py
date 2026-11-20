import unittest
from backend.db.individual import (
    Individual, AssociatedObject, upsert_individual, get_individuals, count_individuals, get_matching_individual_ids,
    is_full_individual, is_media_individual, is_reference_individual, is_source_individual, is_archive_individual,
    individual_get_reference_individual, individual_get_associated_individual_ids,
    individual_ids_get_source_individual_ids,
    upsert_associated_object, get_associated_objects, count_associated_objects,
    get_matching_associated_object_individual_ids,
    delete_individual, delete_associated_object, set_reference_individual, set_source_individual,
    sort_individuals_except_source, get_individual_associated_objects, _get_next_individual_name,
    get_source_individual_ids_by_association_count
)
from backend.db.media import Media, upsert_media, delete_media
from core.app_constants import ERROR, SUCCESS
from core import app_constants

class TestIndividual(unittest.TestCase):
    # --- Individual Class ---
    def test_individual_initialization_and_dict(self):
        """Test Individual object init and to_dict."""
        individual = Individual(id=10001, name="S1", length=10.5, sex="F")

        self.assertEqual(individual.name, "S1")
        d = individual.to_dict()
        self.assertEqual(d["name"], "S1")
        self.assertEqual(d["sex"], "F")

    def test_upsert_individual_create(self):
        """Test Creating a Individual."""
        i_id = 10001
        try:
            i = Individual(id=i_id, name="Original", length=1.0, sex="U")
            res = upsert_individual(i)
            self.assertEqual(res.name, "Original")
            
            fetched = get_individuals(ids=[i_id])[0]
            self.assertEqual(fetched.name, "Original")
        finally:
            delete_individual(i_id)
    
    def test_upsert_individual_update(self):
        """Test Updating a Individual."""
        i_id = 10003
        try:
            i = Individual(id=i_id, name="Original", length=1.0, sex="U")
            upsert_individual(i)
            
            i.name = "Updated"
            res2 = upsert_individual(i)
            self.assertEqual(res2.name, "Updated")
        finally:
            delete_individual(i_id)
    
    def test_get_individuals_single(self):
        """Test Reading a single Individual."""
        # Seeded ID 21
        fetched = get_individuals(ids=[21])
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0].id, 21)

    def test_get_individuals_batch(self):
        """Test Reading multiple Individuals."""
        ids = [21, 22]
        fetched = get_individuals(ids=ids)
        self.assertEqual(len(fetched), 2)
        res_ids = sorted([i.id for i in fetched])
        self.assertEqual(res_ids, ids)

    def test_get_individuals_options(self):
        """Test get_individuals with various options using seeded data."""
        # In seeded data, individual 21 is a source individual while individual 0 is an associated media individual.
        
        src_id = 21
        media_individual_id = 0
        
        # Source Only.
        # Fetching media individual ID with source_only=True should return its source (21).
        res_src = get_individuals(ids=[media_individual_id], get_source=True, source_only=True)
        self.assertEqual(len(res_src), 1)
        self.assertEqual(res_src[0].id, src_id)
        
        # Get Full.
        # Fetching source individual 21 with get_full=True should return it with associated individuals.
        res_full = get_individuals(ids=[src_id], get_full=True)
        self.assertEqual(len(res_full), 1)
        full_individual = res_full[0]
        self.assertEqual(full_individual.id, src_id)
        
        # Verify associated individuals include the media individual 0.
        assoc_ids = [i.id for i in full_individual.associated_individuals]
        self.assertIn(media_individual_id, assoc_ids)
        # Also Individual 1 should be there (which is the reference).
        self.assertIn(1, assoc_ids)

    def test_get_individuals_non_existent(self):
        """Test get_individuals with non-existent ID."""
        res = get_individuals(ids=[-9999])
        self.assertEqual(res, [])

    def test_get_matching_individual_ids(self):
        """Test get_matching_individual_ids filters with seeded data."""
        # Seeded records include individual 0 (male) and individuals 1 and 21 (female).
        
        # Name Match.
        ids_found = get_matching_individual_ids(names=["0"], source_only=False)
        self.assertIn(0, ids_found)
        
        # Sex Match.
        # "F" should include Individual 21 and Individual 1.
        ids_sex = get_matching_individual_ids(sex="F", source_only=False)
        self.assertIn(21, ids_sex)
        self.assertIn(1, ids_sex)
        
        # Verify length filtering includes individuals 0 and 22 while excluding individual 1.
        ids_len = get_matching_individual_ids(min_length=14.0, source_only=False)
        self.assertIn(0, ids_len)
        self.assertIn(22, ids_len) # Source 22 (14.0).
        self.assertNotIn(1, ids_len) # 11.5.

        # No match.
        ids_none = get_matching_individual_ids(names=["GhostIndividual"], source_only=False)
        self.assertEqual(len(ids_none), 0)

    def test_get_matching_individual_ids_combined(self):
        """Test get_matching_individual_ids with combined criteria using seeded data."""
        # Filtering by male sex and minimum length 14.5 matches individuals 0 and 23 while excluding 21 and 1.
        ids = get_matching_individual_ids(sex="M", min_length=14.5, source_only=False)
        self.assertIn(0, ids)
        self.assertIn(23, ids)
        self.assertNotIn(21, ids) # F.
        self.assertNotIn(1, ids) # F.

    def test_get_source_individual_ids_by_association_count(self):
        """Test get_source_individual_ids_by_association_count with seeded data."""
        # In seeded data, source individuals 21, 22, and 27 have 2, 4, and 1 associated media individuals respectively.
        
        # Parameter min_associated means X media individuals (total count >= X+1).
        
        # Min 2 associated (Count >= 3) -> Should include 21 and 22. Exclude 27.
        ids_min = get_source_individual_ids_by_association_count(min_associated=2)
        self.assertIn(21, ids_min)
        self.assertIn(22, ids_min)
        self.assertNotIn(27, ids_min)
        
        # Max 2 associated (Count <= 3) -> Should include 21 (2) and 27 (1). Exclude 22 (4).
        ids_max = get_source_individual_ids_by_association_count(max_associated=2)
        self.assertIn(21, ids_max)
        self.assertIn(27, ids_max)
        self.assertNotIn(22, ids_max)
        
        # Matching 1 to 3 associated individuals includes 21 and 27 while excluding 22.
        ids_range = get_source_individual_ids_by_association_count(min_associated=1, max_associated=3)
        self.assertIn(21, ids_range)
        self.assertIn(27, ids_range)
        self.assertNotIn(22, ids_range)

    def test_delete_individual(self):
        """Test delete_individual."""
        i_id = 10050

        try:
            i = Individual(id=i_id, name="ToDelete")
            upsert_individual(i)
            
            res = delete_individual(i_id)
            self.assertEqual(res, SUCCESS)
            
            fetched = get_individuals(ids=[i_id])
            self.assertEqual(len(fetched), 0)
        finally:
            delete_individual(i_id)

    def test_delete_individual_non_existent(self):
        """Test delete_individual with non-existent ID."""
        res = delete_individual(-9999)
        self.assertEqual(res, app_constants.ERROR)

    def test_count_individuals(self):
        """Test count_individuals breakdown using seeded data."""
        # Total in CSV is 27 rows (IDs 0-20, 21-27).        
        counts = count_individuals()
        
        # We assume the seed data is present.
        self.assertGreaterEqual(counts["total"], 27)
        
        # Check specific types based on seed analysis
        # Source individuals: 21, 22, 23, 24, 25, 27 (from CSV) = 6.
        # Utility individual -1 is excluded from count_individuals query.
        self.assertGreaterEqual(counts["source"], 6)
        self.assertGreaterEqual(counts["media"], 20)

    def test_get_next_individual_name(self):
        """Test _get_next_individual_name with seeded data."""
        # Because the highest seeded individual name is 21, the next name must be at least 22.
        next_name = _get_next_individual_name()
        self.assertGreaterEqual(int(next_name), 22)

    def test_is_full_individual(self):
        """Test is_full_individual with seeded data."""
        # Fetching source individual 21 with get_full=True populates associated lists.
        i = get_individuals(ids=[21], get_full=True)[0]
        self.assertTrue(is_full_individual(i))
        
        # If we fetch with get_full=False, lists are empty, so is_full_individual is False.
        i2 = get_individuals(ids=[21], get_full=False)[0]
        self.assertFalse(is_full_individual(i2))

    def test_is_media_individual(self):
        """Test is_media_individual with seeded data."""
        # Individual 0 is a Media Individual (Source=21 != ID=0, Media=0).
        i = get_individuals(ids=[0])[0]
        self.assertTrue(is_media_individual(i))
        
        # Individual 21 is Source (Source=21 == ID=21). Not Media Individual.
        i_not = get_individuals(ids=[21])[0]
        self.assertFalse(is_media_individual(i_not))

    def test_is_reference_individual(self):
        """Test is_reference_individual with seeded data."""
        # Individual 1 is the Reference for Individual 21.
        ref = get_individuals(ids=[1])[0]
        self.assertTrue(is_reference_individual(ref)) 
        
        # Individual 0 is not a reference individual.
        not_ref = get_individuals(ids=[0])[0]
        self.assertFalse(is_reference_individual(not_ref))

    def test_is_source_individual(self):
        """Test is_source_individual with seeded data."""
        # Individual 21 is a Source Individual.
        i1 = get_individuals(ids=[21])[0]
        self.assertTrue(is_source_individual(i1))
        
        # Utility individual -1 is also a source individual.
        i_util = get_individuals(ids=[-1])[0]
        self.assertTrue(is_source_individual(i_util))
        
        # Individual 0 (Media Individual) is NOT a source individual.
        i3 = get_individuals(ids=[0])[0]
        self.assertFalse(is_source_individual(i3))

    def test_is_archive_individual(self):
        """Test is_archive_individual."""
        # Seeded data might not have archive individuals (Source=ID, No Media, No Ref).
        # We'll stick to creating one transiently for this test.
        i = Individual(id=10020, media=None, reference=None)
        self.assertTrue(is_archive_individual(i))
        i2 = Individual(id=10021, media="m")
        self.assertFalse(is_archive_individual(i2))

    def test_individual_get_reference_individual(self):
        """Test individual_get_reference_individual with seeded data."""
        # Individual 0 (Media Individual) -> Source 21 -> Ref 1.
        src = get_individuals(ids=[0])[0]
        res = individual_get_reference_individual(src)
        self.assertEqual(res.id, 1)
        
        # Individual 1 (Reference Individual) -> Source 21 -> Ref 1.
        ref = get_individuals(ids=[1])[0]
        res2 = individual_get_reference_individual(ref)
        self.assertEqual(res2.id, 1)

    def test_individual_get_associated_individual_ids(self):
        """Test individual_get_associated_individual_ids with seeded data."""
        # Individual 0 (Media Individual) -> belongs to Source 21. 
        # Source 21 has associated individuals: 21, 0, 1, etc.
        
        src_individual_ids = individual_get_associated_individual_ids(get_individuals(ids=[0])[0])
        self.assertIn(21, src_individual_ids)
        self.assertIn(0, src_individual_ids)
        self.assertIn(1, src_individual_ids)

    def test_individual_ids_get_source_individual_ids(self):
        """Test individual_ids_get_source_individual_ids with seeded data."""
        # ID 0 -> Source 21.
        src_ids = individual_ids_get_source_individual_ids([0])
        self.assertEqual(src_ids, [21])
        
        # ID 21 -> Source 21.
        src_ids_2 = individual_ids_get_source_individual_ids([21])
        self.assertEqual(src_ids_2, [21])
        
        # ID 22 -> Source 22.
        src_ids_3 = individual_ids_get_source_individual_ids([22])
        self.assertEqual(src_ids_3, [22])

    def test_set_reference_individual(self):
        """Test set_reference_individual logic."""
        m_ids = [10010, 10020]
        i_ids = [11000, 11001, 11002]
        try:
            upsert_media(Media(id=m_ids[0], time=10, hash="hA", name="nA", filename="fA", source=None, metadata="{}", location="l", type="img", quality="g"))
            upsert_media(Media(id=m_ids[1], time=20, hash="hB", name="nB", filename="fB", source=None, metadata="{}", location="l", type="img", quality="g"))
            
            src = Individual(id=i_ids[0], source=11000, reference=11001)
            upsert_individual(src)
            a = Individual(id=i_ids[1], source=11000, media=10010)
            upsert_individual(a)
            b = Individual(id=i_ids[2], source=11000, media=10020)
            upsert_individual(b)
            
            full_src = get_individuals(ids=[i_ids[0]], get_full=True)[0]
            
            # Set explicit.
            set_reference_individual(full_src, reference_id=i_ids[2])
            updated = get_individuals(ids=[i_ids[0]])[0]
            self.assertEqual(updated.reference, i_ids[2])
            
            # Set Default (oldest -> m_A (time=10) -> 1001).
            full_src = get_individuals(ids=[i_ids[0]], get_full=True)[0]
            set_reference_individual(full_src, reference_id=None)
            updated_2 = get_individuals(ids=[i_ids[0]])[0]
            self.assertEqual(updated_2.reference, i_ids[1])
        finally:
            for i in i_ids:
                delete_individual(i)
            for i in m_ids:
                delete_media(i)

    def test_set_source_individual(self):
        """Test set_source_individual."""
        ids = [11100, 11200]
        try:
            ptr = Individual(id=ids[0], source=11099, media=555) 
            upsert_individual(ptr)
            
            src = Individual(id=ids[1], source=11200, reference=11300) # Valid source..

            upsert_individual(src)
            
            fetched_ptr = get_individuals(ids=[ids[0]])[0]
            
            res = set_source_individual(fetched_ptr, src)
            self.assertEqual(res, SUCCESS)
            
            fetched = get_individuals(ids=[ids[0]])[0]
            self.assertEqual(fetched.source, ids[1])
        finally:
            for i in ids:
                delete_individual(i)

    def test_sort_individuals_except_source(self):
        """Test sort_individuals_except_source."""
        m_ids = [100, 50]
        try:
            upsert_media(Media(id=m_ids[0], time=100, hash="h1", name="n1", filename="f1", source=None, metadata="{}", location="l", type="img", quality="g"))
            upsert_media(Media(id=m_ids[1], time=50, hash="h2", name="n2", filename="f2", source=None, metadata="{}", location="l", type="img", quality="g"))
            
            i1 = Individual(id=1, media=100)
            i2 = Individual(id=2, media=50)
            src = Individual(id=0, source=0) 
            
            individuals = [i1, i2, src] 
            
            # Sort by ID.
            sorted_id = sort_individuals_except_source(list(individuals), "id")
            self.assertEqual(sorted_id[0].id, 0)
            self.assertEqual(sorted_id[1].id, 1)
            self.assertEqual(sorted_id[2].id, 2)
            
            # Sort by Media Time (Low to High: m_2(50), m_1(100)).
            individuals_ordered = [src, i1, i2]
            sorted_time = sort_individuals_except_source(list(individuals_ordered), "media_time")
            
            self.assertEqual(sorted_time[0].id, 2) # lowest time (50).
            self.assertEqual(sorted_time[1].id, 1) # next time (100).
            self.assertEqual(sorted_time[2].id, 0) # source (no media/time -> end).
        finally:
            for i in m_ids:
                delete_media(i)


    # --- Associated Object Class ---
    def test_upsert_associated_object_create(self):
        """Test Creating an Associated Object."""
        i_id = 11700
        obj_id = 10001
        try:
            individual = Individual(id=i_id, name="Host")
            upsert_individual(individual)
            
            obj = AssociatedObject(id=obj_id, individual=i_id, name="Tag1", type="tag")
            res = upsert_associated_object(obj, "Tags")
            self.assertEqual(res.name, "Tag1")
            
            fetched = get_associated_objects(ids=[obj_id], table="Tags")[0]
            self.assertEqual(fetched.name, "Tag1")
            
            # Test get by name.
            fetched_name = get_associated_objects(names=["Tag1"], table="Tags")[0]
            self.assertEqual(fetched_name.id, obj_id)

            count = count_associated_objects(table="Tags")
            self.assertGreaterEqual(count, 1)
        finally:
            delete_associated_object(obj_id, "Tags")
            delete_individual(i_id)

    def test_upsert_associated_object_update(self):
        """Test Updating an Associated Object."""
        i_id = 11701
        obj_id = 10002
        try:
            individual = Individual(id=i_id, name="Host2")
            upsert_individual(individual)
            
            obj = AssociatedObject(id=obj_id, individual=i_id, name="TagOrig", type="tag")
            upsert_associated_object(obj, "Tags")
            
            obj.name = "TagUpdated"
            res = upsert_associated_object(obj, "Tags")
            self.assertEqual(res.name, "TagUpdated")
            
            fetched = get_associated_objects(ids=[obj_id], table="Tags")[0]
            self.assertEqual(fetched.name, "TagUpdated")
        finally:
            delete_associated_object(obj_id, "Tags")
            delete_individual(i_id)

    def test_get_associated_objects_single(self):
        """Test Reading single Associated Objects."""
        # Seeded AlternateNames: ID 1 -> Individual 22, Name "merged boy".
        objs = get_associated_objects(ids=[1], table="AlternateNames")
        self.assertEqual(len(objs), 1)
        self.assertEqual(objs[0].name, "merged boy")
        self.assertEqual(objs[0].individual, 22)
        
        # Seeded Tags: ID 1 -> Individual 22, Name "test-tag"
        tags = get_associated_objects(ids=[1], table="Tags")
        self.assertEqual(len(tags), 1)
        self.assertEqual(tags[0].name, "test-tag")

    def test_get_individual_associated_objects(self):
        """Test get_individual_associated_objects with seeded data."""
        # Seeded individual 21 has label 'fresh' and alternate names 'james' and 'dean'.
        
        # Labels..
        labels = get_individual_associated_objects(21, "Labels")
        # In CSV: 3,21,fresh,,..
        label_names = [l.name for l in labels]
        self.assertIn("fresh", label_names)
        self.assertEqual(len(labels), 1)
        
        # AlternateNames..
        alt_names = get_individual_associated_objects(21, "AlternateNames")
        names = sorted([a.name for a in alt_names])
        self.assertEqual(names, ["dean", "james"])

    def test_get_associated_objects_non_existent(self):
        """Test get_associated_objects with non-existent ID."""
        res = get_associated_objects(ids=[-9999], table="Tags")
        self.assertEqual(res, [])

    def test_get_matching_associated_object_individual_ids(self):
        """Test get_matching_associated_object_individual_ids using seeded data."""
        # Seeded records map label 'fresh' to 21 and 27, alternate name 'james' to 21, and tag 'test-tag' to 22.
        
        # Filter by Name (Labels)..
        # Note: Individual -1 is a utility individual, explicitly excluded by implementation (individual >= 0).
        ids_fresh = get_matching_associated_object_individual_ids(names=["fresh"], table="Labels")
        self.assertIn(21, ids_fresh)
        self.assertIn(27, ids_fresh)
        self.assertNotIn(-1, ids_fresh)
        
        # Filter by Name (AlternateNames)..
        ids_james = get_matching_associated_object_individual_ids(names=["james"], table="AlternateNames")
        self.assertIn(21, ids_james)
        self.assertEqual(len(ids_james), 1)
        
        # Filter by Name (Tags)..
        ids_sat = get_matching_associated_object_individual_ids(names=["test-tag"], table="Tags")
        self.assertIn(22, ids_sat)
        self.assertEqual(len(ids_sat), 1)
        
    def test_get_matching_associated_object_individual_ids_combined(self):
        """Test get_matching_associated_object_individual_ids with combined criteria."""
        # Restrict label filter to individual 21.
        ids_individual_restricted = get_matching_associated_object_individual_ids(names=["fresh"], individuals=[21], table="Labels")
        self.assertEqual(ids_individual_restricted, [21])
        
        # Restricting to [21] excludes individual 27 despite having the same label.
        ids_fresh_all = get_matching_associated_object_individual_ids(names=["fresh"], table="Labels")
        self.assertIn(27, ids_fresh_all)

        # No match..
        ids_none = get_matching_associated_object_individual_ids(names=["NonExistentTag"], table="Tags")
        self.assertEqual(len(ids_none), 0)

    
    def test_delete_associated_object(self):
        """Test delete_associated_object."""
        i_id = 11850
        obj_id = 10010

        try:
            individual = Individual(id=i_id)
            upsert_individual(individual)
            upsert_associated_object(AssociatedObject(id=obj_id, individual=i_id, name="DelTag", type="tag"), "Tags")
            
            # Verify existence..
            self.assertIsNotNone(get_associated_objects(ids=[obj_id], table="Tags"))
            
            # Delete..
            res = delete_associated_object(obj_id, "Tags")
            self.assertEqual(res, SUCCESS)
            
            # Verify deletion..
            fetched = get_associated_objects(ids=[obj_id], table="Tags")
            self.assertEqual(len(fetched), 0)
        finally:
            delete_associated_object(obj_id, "Tags")
            delete_individual(i_id)


    def test_delete_associated_object_non_existent(self):
        """Test delete_associated_object with non-existent ID."""
        res = delete_associated_object(-9999, "Tags")
        self.assertEqual(res, app_constants.ERROR)