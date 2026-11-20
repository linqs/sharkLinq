import unittest
import http
import json
import frontend.action
import backend.db.action
import backend.db.user
from backend.db.action import Action, upsert_action, delete_action, get_actions
from core import app_constants

class MockHandler:
    pass

class TestActionFrontend(unittest.TestCase):
    def setUp(self):
        self.handler = MockHandler()
        self.u1_name = 0
        
        # Create a dedicated user for frontend action tests.
        self.username = "action_tester"
        u = backend.db.user.User(id=890, username=self.username, name="Action Tester", email="action@test.com", role="admin", password="pass", salt="salt")
        backend.db.user.upsert_user(u)

    def tearDown(self):
        # Cleanup transient user.
        try:
            backend.db.user.delete_user(890)
        except:
            pass

    def test_get_actions(self):
        """Test get_actions function with seeded data."""
        # Test getting existing seeded actions.
        kwargs = {"ids": [1, 2]}
        res, code, _ = frontend.action.get_actions(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["id"], 1)

    def test_get_actions_non_existent(self):
        """Test get_actions with non-existent IDs."""
        kwargs = {"ids": [-9999]}
        res, code, _ = frontend.action.get_actions(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertEqual(res, [])

    def test_filter_actions(self):
        """Test filter_actions function with seeded data."""
        # Filter by Author "testowner".
        kwargs = {
            "ids": "", "keywords": "", "original": "", "final": "",
            "author": self.u1_name, "min_time": "", "max_time": "", "action": "", "object": ""
        }
        res, code, _ = frontend.action.filter_actions(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.OK)
        # All seeded actions seem to be by "testowner".
        self.assertGreaterEqual(len(res), 10)
        ids = [a["id"] for a in res]
        self.assertIn(1, ids)

        # Filter by Object "media".
        # Action ID 2 (update media), Action ID 20 (create media), Action ID 23 (delete media).
        kwargs["author"] = ""
        kwargs["object"] = "media"
        res, code, _ = frontend.action.filter_actions(self.handler, "", **kwargs)
        self.assertGreaterEqual(len(res), 1)
        ids = [a["id"] for a in res]
        self.assertIn(2, ids)
        self.assertIn(20, ids)

        # Filter by Action "merge".
        # Action 6 and 7 are merges.
        kwargs["object"] = ""
        kwargs["action"] = "merge"
        res, code, _ = frontend.action.filter_actions(self.handler, "", **kwargs)
        self.assertGreaterEqual(len(res), 2)
        ids = [a["id"] for a in res]
        self.assertIn(6, ids)
        self.assertIn(7, ids)

        # Combined Filter (Author "testowner" + Object "tag").
        # Action 8 is create tag by testowner.
        kwargs["action"] = ""
        kwargs["author"] = self.u1_name
        kwargs["object"] = "tag"
        res, code, _ = frontend.action.filter_actions(self.handler, "", **kwargs)
        self.assertGreaterEqual(len(res), 1)
        self.assertIn(8, [a["id"] for a in res])

        # Filtering by min_time > 1770004000 includes action 6 and excludes action 1.
        kwargs["object"] = ""
        kwargs["min_time"] = 1770004000
        res, code, _ = frontend.action.filter_actions(self.handler, "", **kwargs)
        ids = [a["id"] for a in res]
        self.assertNotIn(1, ids)
        self.assertIn(6, ids)

        # No Match.
        kwargs["min_time"] = ""
        kwargs["author"] = "NonExistentUser"
        res, code, _ = frontend.action.filter_actions(self.handler, "", **kwargs)
        self.assertEqual(len(res), 0)

        # Keyword Filter.
        # "fresh" appears in Label creation actions (ID 12, 14) and potentially individual creation (ID 21).
        kwargs["author"] = ""
        kwargs["keywords"] = "fresh"
        res, code, _ = frontend.action.filter_actions(self.handler, "", **kwargs)
        self.assertGreaterEqual(len(res), 1)

    def test_log_action(self):
        """Test log_action function."""
        # Log a new action.
        self.created_ids = []
        
        class MockObj:
            def __init__(self, id):
                self.id = id
            def to_dict(self, compact=False):
                return {"id": self.id}
            def __getitem__(self, item):
                return getattr(self, item)

        orig = MockObj(10)
        final = MockObj(20)
        
        res = frontend.action.log_action(self.u1_name, "update", "testing", orig, final)
        
        self.assertIsNotNone(res)
        self.assertNotEqual(res, app_constants.ERROR)
        self.created_ids.append(res.id)
        
        self.assertEqual(res.author, self.u1_name)
        self.assertEqual(res.object, "testing")
        self.assertEqual(res.action, "update")
        
        # Verify stored in DB.
        fetched = backend.db.action.get_actions(ids=[res.id])
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0].author, self.u1_name)

if __name__ == '__main__':
    unittest.main()
