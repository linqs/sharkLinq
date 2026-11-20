"""Unit tests for frontend proposal retrieval, filtering, and deletion."""

import http
import unittest

import backend.db.proposal
from backend.db.proposal import Proposal, upsert_proposal
import backend.db.user
import frontend.proposal


class MockHandler:
    pass

class TestProposalFrontend(unittest.TestCase):
    def setUp(self):
        self.handler = MockHandler()
        # Seeded test data remains persistent.
        
        # Create user (transient).
        self.username = "proposer"
        u = backend.db.user.User(id=888, username=self.username, name="Proposer Name", email="prop@test.com", role="collaborator", password="pass", salt="salt")
        backend.db.user.upsert_user(u)

    def tearDown(self):
        # Cleanup transient user.
        try:
            backend.db.user.delete_user(888)
        except:
            pass

    def test_get_proposals(self):
        """Test getting proposals."""
        p = Proposal(id=20, action="delete", author=self.username, status="open", object="individual", time=12345.6, notes="{}")
        upsert_proposal(p)
        
        # Existing.
        resp, code, _ = frontend.proposal.get_proposals(self.handler, "", ids=[20])
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertEqual(len(resp), 1)
        self.assertEqual(resp[0]["id"], 20)

    def test_get_proposals_non_existent(self):
        """Test getting non-existent proposals."""
        resp_none, code_none, _ = frontend.proposal.get_proposals(self.handler, "", ids=[-9999])
        self.assertEqual(code_none, http.HTTPStatus.OK)
        self.assertEqual(resp_none, [])

    def test_filter_proposals(self):
        """Test filtering proposals with seeded data."""
        # Seeded proposals cover actions, statuses, and target objects for author 0.

        # Standard Filter (Author).
        kwargs = {
            "ids": "", "keywords": "", "original": "", "final": "",
            "author": 0, "min_time": "", "max_time": "", "action": "", "object": "", "status": ""
        }
        res, code, _ = frontend.proposal.filter_proposals(self.handler, "", **kwargs)
        self.assertEqual(code, http.HTTPStatus.OK)
        # Should be at least 5 seeded proposals for testowner.
        self.assertGreaterEqual(len(res), 5)
        ids = [r["id"] for r in res]
        self.assertIn(11, ids)
        self.assertIn(12, ids)
        self.assertIn(15, ids)

        # Filter by Action "delete".
        # IDs 11, 15.
        kwargs["author"] = ""
        kwargs["action"] = "delete"
        res, code, _ = frontend.proposal.filter_proposals(self.handler, "", **kwargs)
        self.assertGreaterEqual(len(res), 2)
        ids = [r["id"] for r in res]
        self.assertIn(11, ids)
        self.assertIn(15, ids)
        
        # Filter by Status "archive".
        # IDs 11, 13.
        kwargs["action"] = ""
        kwargs["status"] = "archive"
        res, code, _ = frontend.proposal.filter_proposals(self.handler, "", **kwargs)
        ids = [r["id"] for r in res]
        self.assertIn(11, ids)
        self.assertIn(13, ids)
        self.assertNotIn(12, ids) # open.

        # Filter by Object "media".
        # IDs 12, 15.
        kwargs["status"] = ""
        kwargs["object"] = "media"
        res, code, _ = frontend.proposal.filter_proposals(self.handler, "", **kwargs)
        ids = [r["id"] for r in res]
        self.assertIn(12, ids)
        self.assertIn(15, ids)

        # Combined (Status "open" + Action "merge").
        # ID 14.
        kwargs["object"] = ""
        kwargs["status"] = "open"
        kwargs["action"] = "merge"
        res, code, _ = frontend.proposal.filter_proposals(self.handler, "", **kwargs)
        ids = [r["id"] for r in res]
        self.assertIn(14, ids)
        self.assertNotIn(11, ids) # delete/archive.

        # No Match.
        kwargs["status"] = ""
        kwargs["action"] = ""
        kwargs["author"] = "NonExistentUser"
        res, code, _ = frontend.proposal.filter_proposals(self.handler, "", **kwargs)
        self.assertEqual(len(res), 0)

    def test_delete_proposal(self):
        """Test deleting proposals."""
        p = Proposal(id=30, action="delete", author=self.username, status="open", object="individual", time=12345.6, notes="{}")
        upsert_proposal(p)

        resp, code, _ = frontend.proposal.delete_proposal(self.handler, "", id=30)
        self.assertEqual(code, http.HTTPStatus.OK)
        
        left = backend.db.proposal.get_proposals(ids=[30])
        self.assertEqual(len(left), 0)
    
    def test_delete_proposal_non_existent(self):
        """Test deleting non-existent proposal."""
        # Should return ERROR gracefully.
        resp, code, _ = frontend.proposal.delete_proposal(self.handler, "", id=-9999)
        self.assertEqual(resp, "Failed to delete proposal.")
        self.assertEqual(code, http.HTTPStatus.INTERNAL_SERVER_ERROR)


if __name__ == '__main__':
    unittest.main()
