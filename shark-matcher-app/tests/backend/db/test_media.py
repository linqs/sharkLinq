import unittest

from core.app_constants import ERROR, SUCCESS
from backend.db.media import Media, upsert_media, get_medias, delete_media, get_matching_media_ids, get_individual_ids_by_media_ids

class TestMedia(unittest.TestCase):
    def setUp(self):
        # Use seeded media data without clearing.
        pass

    def test_upsert_media_create(self):
        """Test Creating a Media."""
        m_id = 12001
        try:
            media = Media(id=m_id, hash="hash_2001", name="media_2001", filename="file_2001.jpg", 
                          source=None, individual=None, metadata="meta", location="locA", time=1000, 
                          type="image", quality="good")
            res = upsert_media(media)
            self.assertNotEqual(res, ERROR)
            self.assertEqual(res.name, "media_2001")
            self.assertEqual(res.id, m_id)
        finally:
            delete_media(m_id)

    def test_upsert_media_update(self):
        """Test Updating a Media."""
        m_id = 12002
        try:
            media = Media(id=m_id, hash="hash_2002", name="media_2002", filename="file_2002.jpg", 
                          source=None, individual=None, metadata="meta", location="locA", time=1000, 
                          type="image", quality="good")
            upsert_media(media)

            media.metadata = "updated_meta"
            res_update = upsert_media(media)
            self.assertEqual(res_update.metadata, "updated_meta")
        finally:
            delete_media(m_id)

    def test_get_medias_single(self):
        """Test Reading a single Media."""
        # Seeded ID 1.
        fetched = get_medias(ids=[1])
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0].id, 1)
        self.assertEqual(fetched[0].name, "AN11101602")

    def test_get_medias_batch(self):
        """Test Reading multiple Medias."""
        # Seeded IDs 1, 2.
        fetched = get_medias(ids=[1, 2])
        self.assertEqual(len(fetched), 2)
        # Verify both present.
        ids = sorted([m.id for m in fetched])
        self.assertEqual(ids, [1, 2])

    def test_get_medias_by_name(self):
        """Test get_medias by Names."""
        fetched_name = get_medias(names=["AN16101002"])
        self.assertEqual(len(fetched_name), 1)
        self.assertEqual(fetched_name[0].id, 2)

    def test_get_medias_options(self):
        """Test get_medias options (get_full)."""
        # Get Full (requires Individual).
        # Seeded Media 1 is linked to Individual 1.
        fetched_full = get_medias(ids=[1], get_full=True)
        self.assertEqual(len(fetched_full), 1)
        self.assertIsNotNone(fetched_full[0].individual)
        self.assertEqual(fetched_full[0].individual.id, 1)
        self.assertEqual(fetched_full[0].individual.sex, "F")
        self.assertEqual(fetched_full[0].individual.length, 11.5)

    def test_get_medias_non_existent(self):
        """Test get_medias with non-existent ID."""
        res = get_medias(ids=[-9999])
        self.assertEqual(res, [])

    def test_get_medias_by_name_non_existent(self):
        """Test get_medias with non-existent Name."""
        res = get_medias(names=["non_existent_media_name"])
        self.assertEqual(res, [])

    def test_get_matching_media_ids(self):
        """Test get_matching_media_ids with seeded data."""
        # Seeded media rows provide distinct locations, types, and quality values for filtering.
        
        # 1. IDs.
        res = get_matching_media_ids(ids=[1])
        self.assertEqual(res, [1])

        # 2. Names.
        res = get_matching_media_ids(names=["AN11101602"])
        self.assertEqual(res, [1])

        # 3. Locations.
        res = get_matching_media_ids(locations=["FAR"])
        self.assertIn(4, res) # FAR11102905.
        self.assertIn(5, res)
        self.assertNotIn(1, res) # 1 is AN.

        # 4. Types.
        res = get_matching_media_ids(types=["topshot"])
        self.assertIn(1, res)
        self.assertNotIn(4, res) # 4 is underwater.

        # Filter by minimum and maximum timestamps across seeded media.
        res = get_matching_media_ids(min_time=1400000000)
        self.assertIn(2, res)
        self.assertNotIn(1, res)

        res = get_matching_media_ids(max_time=1350000000)
        self.assertIn(1, res)
        self.assertNotIn(2, res)

        # Filtering by 'good' quality matches media 1 and 21 while excluding null quality media.
        res = get_matching_media_ids(qualities=["good"], allow_unknown=False)
        self.assertIn(1, res)
        self.assertIn(21, res)
        self.assertNotIn(4, res) # Media 4 has null quality in seeded data.

        # 9. No match.
        res = get_matching_media_ids(locations=["NonExistentLoc"])
        self.assertEqual(len(res), 0)

    def test_get_matching_media_ids_combined(self):
        """Test filtering media with combined criteria."""
        # Combining location 'AN' and type 'topshot' matches only media 1.
        res = get_matching_media_ids(locations=["AN"], types=["topshot"])
        self.assertIn(1, res)
        
        # Location FAR AND Type Underwater -> ID 4, 5.
        res = get_matching_media_ids(locations=["FAR"], types=["underwater"])
        self.assertIn(4, res)
        self.assertIn(5, res)

    def test_delete_media(self):
        """Test delete_media."""
        m_id = 12004
        try:
            media = Media(id=m_id, hash="hash_2004", name="media_2004", filename="file_2004.jpg")
            upsert_media(media)

            # Verify exists.
            fetched = get_medias(ids=[m_id])
            self.assertEqual(len(fetched), 1)

            # Delete.
            res = delete_media(m_id)
            self.assertEqual(res, SUCCESS)

            # Verify gone.
            fetched_after = get_medias(ids=[m_id])
            self.assertEqual(len(fetched_after), 0)
        finally:
            delete_media(m_id)

    def test_delete_media_non_existent(self):
        """Test delete_media with non-existent ID."""
        res = delete_media(-9999)
        self.assertEqual(res, ERROR)

if __name__ == '__main__':
    unittest.main()
