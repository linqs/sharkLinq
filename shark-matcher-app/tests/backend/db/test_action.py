import unittest

from core.app_constants import ERROR, SUCCESS
from backend.db.action import Action, upsert_action, get_actions, delete_action, get_matching_action_ids

class TestAction(unittest.TestCase):
    def setUp(self):
        # Do not clear actions. Use seeded data.
        pass

    def test_upsert_action_create(self):
        """Test Creating an Action."""
        id = 10001
        try:
            action = Action(id=id, original="A", final="B", action="update", object="Individual", 
                            author=1, time=123456, notes="test note")
            res = upsert_action(action)
            self.assertEqual(res.notes, "test note")
        finally:
            delete_action(id)

    def test_upsert_action_update(self):
        """Test Updating an Action."""
        id = 10002
        try:
            action = Action(id=id, original="A", final="B", action="update", object="Individual", 
                            author=1, time=123456, notes="test note")
            upsert_action(action)
            
            action.notes = "updated note"
            res = upsert_action(action)
            self.assertEqual(res.notes, "updated note")
        finally:
            delete_action(id)

    def test_get_actions_single(self):
        """Test Reading an Action with seeded data."""
        id = 1
        # Seeded Action 1: update individual, author testowner.
        fetched = get_actions(ids=[id])
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0].object, "individual")
        self.assertEqual(fetched[0].author, 0)

    def test_get_actions_batch(self):
        """Test Reading multiple Actions with seeded data."""
        ids = [1, 2]
        # Action 1: update individual, Action 2: update media.
        fetched = get_actions(ids=ids)
        self.assertEqual(len(fetched), 2)
        # Order usually preserved.
        self.assertEqual(fetched[0].id, 1)
        self.assertEqual(fetched[1].id, 2)

    def test_get_actions_non_existent(self):
        """Test get_actions with non-existent ID."""
        res = get_actions(ids=[-9999])
        self.assertEqual(res, [])

    def test_get_matching_action_ids_filters(self):
        """Test filtering actions with seeded data."""
        # Seeded actions cover updates, merges, and tags created by author 0.
        
        # 1. Filter by Author.
        res_ids = get_matching_action_ids(author=0)
        self.assertGreaterEqual(len(res_ids), 4)
        self.assertIn(1, res_ids)

        # 2. Filter by Action Type.
        res_ids = get_matching_action_ids(action="update")
        # ID 1, 2, 3, 4 are updates.
        self.assertIn(1, res_ids)
        self.assertIn(2, res_ids)
        
        res_merge = get_matching_action_ids(action="merge")
        self.assertIn(6, res_merge)

        # Filter by time range where ID 6 matches and ID 1 is excluded.
        res_ids = get_matching_action_ids(min_time=1770004000)
        self.assertIn(6, res_ids)
        self.assertNotIn(1, res_ids)

        # 4. Filter by Object.
        res_ids = get_matching_action_ids(object="media")
        self.assertIn(2, res_ids)
        # ID 20, 23 also media.
        self.assertIn(20, res_ids)

        # 6. No matches.
        res_ids = get_matching_action_ids(author=999)
        self.assertEqual(len(res_ids), 0)

    def test_get_matching_action_ids_combined(self):
        """Test filtering actions with combined criteria."""
        # 5. Combined Filter (Author + Action).
        res_ids = get_matching_action_ids(author=0, action="merge")
        self.assertIn(6, res_ids)

    def test_delete_action(self):
        """Test Deleting an Action."""
        id = 10003
        try:
            action = Action(id=id, original="A", final="B", action="update", object="Individual", 
                            author=1, time=123456, notes="test note")
            upsert_action(action)

            status = delete_action(id)
            self.assertEqual(status, SUCCESS)
            
            fetched = get_actions(ids=[id])
            self.assertEqual(len(fetched), 0)
        finally:
            delete_action(id)

    def test_delete_action_non_existent(self):
        """Test delete_action with non-existent ID."""
        res = delete_action(-9999)
        self.assertEqual(res, ERROR)
