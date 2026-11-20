"""Unit tests for frontend individual CRUD, filtering, and merge operations."""

import http
import unittest

from core import app_constants
import backend.db.action
import backend.db.individual
import backend.db.user
import frontend.individual


class MockHandler:
    def __init__(self):
        self._models = []

class TestIndividualFrontend(unittest.TestCase):
    def setUp(self):
        self.handler = MockHandler()
        # Do not clear DB (seeded data)..
        
        # Create user (transient).
        self.username = "individual_tester"
        u = backend.db.user.User(id=892, username=self.username, name="Individual Tester", email="individual@test.com", role="admin", password="pass", salt="salt")
        backend.db.user.upsert_user(u)

    def tearDown(self):
        # Cleanup transient user.
        try:
             backend.db.user.delete_user(892)
        except:
             pass

    def test_get_individuals_success(self):
        """Test getting individuals with seeded data (success)."""
        # Basic get (Seeded Individual 0).
        individuals, code, _ = frontend.individual.get_individuals(self.handler, "", ids=0, names=None, keywords=None)
        self.assertEqual(code, http.HTTPStatus.OK)
        # Note: Individual 0 name is "0" in seed.
        self.assertEqual(len(individuals), 1)
        self.assertEqual(individuals[0]['name'], "0")
        
        # Keyword search for 'fresh' matches individuals 21 and 27 by label.
        individuals, code, _ = frontend.individual.get_individuals(self.handler, "", ids=None, names=None, keywords="fresh")
        self.assertGreaterEqual(len(individuals), 1)
        ids = [s['id'] for s in individuals]
        self.assertIn(21, ids)

    def test_get_individuals_non_existent(self):
        """Test getting non-existent individual."""
        individuals, code, _ = frontend.individual.get_individuals(self.handler, "", ids=9999)
        self.assertEqual(len(individuals), 0)

    def test_filter_individuals(self):
        """Test filtering individuals with seeded data."""
        # Seeded source individuals include 21 (F, 12.0), 22 (U, 14.0), and 23 (M, 15.0).
        
        # Standard Filter (Sex="F").
        kwargs = {
            "ids": None, "keywords": None, 
            "medias": "", "locations": "", "qualities": "", "allow_unknown": False,
            "min_time": None, "max_time": None,
            "min_length": None, "max_length": None,
            "sex": "F",
            "alternate_names": "", "tags": "", "labels": "",
            "min_associated": None, "max_associated": None
        }
        individuals, _, _ = frontend.individual.filter_individuals(self.handler, "", **kwargs)
        self.assertGreaterEqual(len(individuals), 1)
        ids = [s['id'] for s in individuals]
        self.assertIn(21, ids)
        self.assertNotIn(0, ids) # M.

        # Filtering by min_length=14.0 returns source individuals 22 and 23, excluding media individual 0 and source 21.
        kwargs["sex"] = ""
        kwargs["min_length"] = 14.0
        individuals, _, _ = frontend.individual.filter_individuals(self.handler, "", **kwargs)
        ids = [s['id'] for s in individuals]
        self.assertIn(22, ids)
        self.assertIn(23, ids)
        self.assertNotIn(0, ids)
        self.assertNotIn(21, ids) # 12.0.

        # Attribute Filter (Label "fresh").
        # Should include 21, 27.
        kwargs["min_length"] = None
        kwargs["labels"] = "fresh"
        individuals, _, _ = frontend.individual.filter_individuals(self.handler, "", **kwargs)
        ids = [s['id'] for s in individuals]
        self.assertIn(21, ids)
        self.assertIn(27, ids)

        # Attribute Filter (Tag "test-tag").
        # Should include 22.
        kwargs["labels"] = ""
        kwargs["tags"] = "test-tag"
        individuals, _, _ = frontend.individual.filter_individuals(self.handler, "", **kwargs)
        ids = [s['id'] for s in individuals]
        self.assertIn(22, ids)
        self.assertNotIn(21, ids)

        # Combining male sex and min_length 14.5 matches only source individual 23.
        kwargs["tags"] = ""
        kwargs["sex"] = "M"
        kwargs["min_length"] = 14.5
        individuals, _, _ = frontend.individual.filter_individuals(self.handler, "", **kwargs)
        ids = [s['id'] for s in individuals]
        self.assertIn(23, ids)
        self.assertNotIn(0, ids)
        self.assertNotIn(22, ids)
        self.assertNotIn(21, ids)

        # 6. No Match.
        kwargs["sex"] = ""
        kwargs["min_length"] = ""
        kwargs["labels"] = "NonExistentLabel"
        individuals, _, _ = frontend.individual.filter_individuals(self.handler, "", **kwargs)
        self.assertEqual(len(individuals), 0)

    def test_sex_filter_accepts_any_letter_case_and_unknown_markers(self):
        """Lowercase letters match like uppercase, and U filters for unrecorded sex instead of failing."""
        kwargs = {"ids": None, "keywords": None, "medias": "", "locations": "", "qualities": "",
                  "allow_unknown": False, "min_time": None, "max_time": None,
                  "min_length": None, "max_length": None, "alternate_names": "", "tags": "",
                  "labels": "", "min_associated": None, "max_associated": None}
        upper, code, _ = frontend.individual.filter_individuals(self.handler, "", sex="F", **kwargs)
        lower, _, _ = frontend.individual.filter_individuals(self.handler, "", sex="f", **kwargs)
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertEqual([s['id'] for s in lower], [s['id'] for s in upper])
        unknown, code, _ = frontend.individual.filter_individuals(
            self.handler, "", sex="U", **{**kwargs, "allow_unknown": True})
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertIn(22, [s['id'] for s in unknown])

    def test_filter_individuals_date_keyword_intersection(self):
        """Test that date filters correctly intersect with keyword matches based on source individuals."""
        # Create a source individual linked to media with an explicit timestamp.
        source_id = 9001
        media_id = 9002
        s_source = backend.db.individual.Individual(id=source_id, name="TestDateSource", source=source_id, reference=source_id)
        backend.db.individual.upsert_individual(s_source)
        
        media_hash = "hash_date_9002"
        mock_media = backend.db.media.Media(id=media_id+1, name="SpecialImageName23", time=1673784000, hash=media_hash, filename="fake.jpg", source=None, metadata="{}", location="l", type="img", quality="g")
        backend.db.media.upsert_media(mock_media)
        
        s_media = backend.db.individual.Individual(id=media_id, name="MediaInd23", source=source_id, media=media_id+1)
        backend.db.individual.upsert_individual(s_media)

        # Search by keyword alone returns the source individual.
        kwargs = {
            "ids": None, "keywords": ["SpecialImageName23"], 
            "names": None, "medias": "", "locations": "", "qualities": "", "allow_unknown": False,
            "min_time": None, "max_time": None, "min_length": None, "max_length": None, "sex": "",
            "alternate_names": "", "tags": "", "labels": "", "min_associated": None, "max_associated": None
        }
        inds, _, _ = frontend.individual.filter_individuals(self.handler, "", **kwargs)
        self.assertEqual(len(inds), 1)
        self.assertEqual(inds[0]['id'], source_id)

        # Search by keyword with a date range matching the media timestamp returns the source individual.
        kwargs["min_time"] = 1673697600
        kwargs["max_time"] = 1673870400
        inds, _, _ = frontend.individual.filter_individuals(self.handler, "", **kwargs)
        self.assertEqual(len(inds), 1)
        self.assertEqual(inds[0]['id'], source_id)

        # Search by keyword with a non-overlapping date range returns no results.
        kwargs["min_time"] = 1673870400
        kwargs["max_time"] = 1673956800
        inds, _, _ = frontend.individual.filter_individuals(self.handler, "", **kwargs)
        self.assertEqual(len(inds), 0)

    def test_post_individual_create(self):
        """Test creating an individual."""
        kwargs = {
            "id": None,
            "name": None,
            "length": 150,
            "sex": "F",
            "source": None,
            "media": "media_hash_create",
            "reference": None,
            "user_id": 1,
            "type": None
        }
        resp, code, _ = frontend.individual.post_individual(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.OK)
        action_status = resp["action_status"]
        self.assertEqual(action_status["action"], "create")
        
        # Verify Action Log (Create Individual).
        actions = backend.db.action.get_matching_action_ids(author=1, action="create", object="individual")
        self.assertTrue(len(actions) > 0, "No create individual action found")

    def test_post_individual_update(self):
        """Test updating an individual."""
        # Setup.
        s = backend.db.individual.Individual(id=1005, name="UpdIndividual", source=1005)
        backend.db.individual.upsert_individual(s)
        
        kwargs_update = {
            "id": 1005,
            "name": "UpdatedName",
            "length": 160,
            "sex": "M",
            "source": 1005,
            "media": None,
            "reference": None,
            "user_id": 1,
            "type": None
        }
        resp, code, _ = frontend.individual.post_individual(self.handler, "", **kwargs_update)
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertEqual(resp["action_status"]["action"], "update")
        
        # Verify.
        s_upd = backend.db.individual.get_individuals(ids=[1005])[0]
        self.assertEqual(s_upd.name, "UpdatedName")
        
        # Verify Action Log (Update Individual).
        actions = backend.db.action.get_matching_action_ids(author=1, action="update", object="individual")
        self.assertTrue(len(actions) > 0, "No update individual action found")

    def test_post_individual_rejects_injection_name(self):
        """A name that would execute as HTML is refused, and nothing is written."""
        s = backend.db.individual.Individual(id=1006, name="SafeIndividual", source=1006)
        backend.db.individual.upsert_individual(s)

        kwargs = {
            "id": 1006,
            "name": "<img src=x onerror=alert(1)>",
            "length": 160,
            "sex": "M",
            "source": 1006,
            "media": None,
            "reference": None,
            "user_id": 1,
            "type": None
        }
        _, code, _ = frontend.individual.post_individual(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.BAD_REQUEST)

        # The original name must survive the rejected write.
        unchanged = backend.db.individual.get_individuals(ids=[1006])[0]
        self.assertEqual(unchanged.name, "SafeIndividual")

    def test_post_object_rejects_injection_name(self):
        """Test that post_object rejects script injection in object names."""
        kwargs = {
            "id": None,
            "individual": 1,
            "name": "<script>alert(1)</script>",
            "type": None,
            "time": None,
            "table": "tag",
            "user_id": 1,
            "user_username": self.username,
        }
        _, code, _ = frontend.individual.post_object(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.BAD_REQUEST)

    def test_post_object_accepts_apostrophe_name(self):
        """Test that post_object accepts object names containing apostrophes."""
        kwargs = {
            "id": None,
            "individual": 1,
            "name": "O'Brien",
            "type": None,
            "time": None,
            "table": "tag",
            "user_id": 1,
            "user_username": self.username,
        }
        _, code, _ = frontend.individual.post_object(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.OK)

    def test_delete_individual(self):
        """Test deleting an individual."""
        s = backend.db.individual.Individual(id=5001, name="DeleteMe", source=5001, reference=5001)
        backend.db.individual.upsert_individual(s)
        
        resp, code, _ = frontend.individual.delete_individual(self.handler, "", id=5001, user_id=1)
        self.assertEqual(code, http.HTTPStatus.OK)
        
        gone = backend.db.individual.get_individuals(ids=[5001])
        self.assertEqual(len(gone), 0)
        
        # Verify Action Log (Delete Individual).
        actions = backend.db.action.get_matching_action_ids(author=1, action="delete", object="individual")
        self.assertTrue(len(actions) > 0, "No delete individual action found")

    def test_delete_individual_non_existent(self):
        """Test deleting non-existent individual."""
        resp, code, _ = frontend.individual.delete_individual(self.handler, "", id=9999, user_id=1)
        # ERROR gracefully.
        self.assertEqual(resp, app_constants.ERROR)
        self.assertEqual(code, http.HTTPStatus.OK)

    def test_unmatch_and_delete_of_an_individual_without_a_source_are_404(self):
        """A record with no source individual is reported as not found, not a server error."""
        orphan = backend.db.individual.Individual(id=5002, name="Orphan")
        backend.db.individual.upsert_individual(orphan)
        try:
            _, unmatch_code, _ = frontend.individual.unmatch_individual(
                self.handler, "", id=5002, user_id=1)
            _, delete_code, _ = frontend.individual.delete_individual(
                self.handler, "", id=5002, user_id=1)
        finally:
            backend.db.individual.delete_individual(5002)
        self.assertEqual(unmatch_code, http.HTTPStatus.NOT_FOUND)
        self.assertEqual(delete_code, http.HTTPStatus.NOT_FOUND)

    def test_merge_individuals(self):
        """Test merging two individuals."""
        s_major = backend.db.individual.Individual(id=2001, name="MajorIndividual", source=2001, reference=2001)
        backend.db.individual.upsert_individual(s_major)
        s_minor = backend.db.individual.Individual(id=2002, name="MinorIndividual", source=2002, reference=2002)
        backend.db.individual.upsert_individual(s_minor)
        m_minor = backend.db.individual.Individual(id=2003, name="MinorMedia", source=2002, media="some_media_hash")
        backend.db.individual.upsert_individual(m_minor)
        
        preview = {
            "sex": "F", "length": 200, "reference": 2001,
            "alternate_names": [], "tags": [], "labels": []
        }
        
        resp, code, _ = frontend.individual.merge_individuals(self.handler, "", major_id=2001, minor_id=2002, preview=preview, user_id=1)
        self.assertEqual(code, http.HTTPStatus.OK)
        
        # Verify logic included in original test (deleted minor, moved media).
        self.assertEqual(len(backend.db.individual.get_individuals(ids=[2002])), 0)
        
        # Verify Action Logs (Merge).
        actions = backend.db.action.get_matching_action_ids(author=1, action="merge", object="individual")
        # Should be 2 logs (one for major, one for minor).
        self.assertTrue(len(actions) >= 2, "Insufficient merge actions found")

    def test_unmatch_individual(self):
        """Test unmatching an individual."""
        s_source = backend.db.individual.Individual(id=3001, name="SourceIndividual", source=3001, reference=3001)
        backend.db.individual.upsert_individual(s_source)
        s_media = backend.db.individual.Individual(id=3002, name="MediaIndividual", source=3001, media="hash1")
        backend.db.individual.upsert_individual(s_media)
        
        resp, code, _ = frontend.individual.unmatch_individual(self.handler, "", id=3002, user_id=1)
        self.assertEqual(code, http.HTTPStatus.OK)
        
        # Verify Action Log (Unmatch).
        actions = backend.db.action.get_matching_action_ids(author=1, action="unmatch", object="individual")
        self.assertTrue(len(actions) > 0, "No unmatch individual action found")

    def test_associated_objects(self):
        """Test creating, getting, filter, delete associated objects + logs."""
        s = backend.db.individual.Individual(id=1004, name="ObjectIndividual", source=1004)
        backend.db.individual.upsert_individual(s)
        
        # Post (Create) Tag.
        kwargs = {
            "id": None, "individual": 1004, "name": "BittenByDog",
            "type": "behavior", "time": None, "table": "tag",
            "user_id": 1
        }
        frontend.individual.post_object(self.handler, "", **kwargs)
        
        # Verify Tag Create Log.
        actions = backend.db.action.get_matching_action_ids(author=1, action="create", object="tag")
        self.assertTrue(len(actions) > 0, "No create tag action found")
        
        # Filter & Get logic (existing test covered this well, keeping brief).
        objects, _, _ = frontend.individual.filter_objects(self.handler, "", individual=1004, table="tag")
        self.assertEqual(len(objects), 1)
        obj_id = objects[0]['id']
        
        # Delete Tag.
        frontend.individual.delete_object(self.handler, "", id=obj_id, table="tag", user_id=1)
        
        # Verify Tag Delete Log.
        actions_del = backend.db.action.get_matching_action_ids(author=1, action="delete", object="tag")
        self.assertTrue(len(actions_del) > 0, "No delete tag action found")


if __name__ == '__main__':
    unittest.main()
