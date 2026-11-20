import unittest

from backend.db.proposal import Proposal, upsert_proposal, get_proposals, delete_proposal, get_matching_proposal_ids
from core.app_constants import ERROR, SUCCESS

class TestProposal(unittest.TestCase):
    def setUp(self):
        pass

    def test_upsert_proposal_create(self):
        """Test Creating a Proposal."""
        p_id = 1001
        try:
            proposal = Proposal(id=p_id, original=1, final=2, action="merge", object="Individual1", 
                                status="open", author=1, time=1000, notes="note")
            res = upsert_proposal(proposal)
            self.assertNotEqual(res, ERROR)
            self.assertEqual(res.status, "open")
            self.assertEqual(res.id, p_id)
        finally:
            delete_proposal(p_id)

    def test_upsert_proposal_update(self):
        """Test Updating a Proposal."""
        p_id = 1002
        try:
            proposal = Proposal(id=p_id, original=1, final=2, action="merge", object="Individual1", 
                                status="open", author=1, time=1000, notes="note")
            upsert_proposal(proposal)

            proposal.status = "archive"
            res_update = upsert_proposal(proposal)
            self.assertEqual(res_update.status, "archive")
        finally:
            delete_proposal(p_id)

    def test_get_proposals_single(self):
        """Test Reading a single Proposal."""
        p_ids = [12, 13]
        fetched = get_proposals(ids=[p_ids[0]])
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0].id, p_ids[0])

    def test_get_proposals_batch(self):
        """Test Reading multiple Proposals."""
        p_ids = [12, 13]
        fetched_multi = get_proposals(ids=p_ids)
        self.assertEqual(len(fetched_multi), 2)
        
    def test_get_proposals_non_existent(self):
        """Test get_proposals with non-existent ID."""
        res = get_proposals(ids=[-9999])
        self.assertEqual(res, [])

    def test_get_matching_proposal_ids(self):
        """Test get_matching_proposal_ids with seeded data."""
        # Seeded proposals cover actions, statuses, and target objects across author 0.

        # Filter by merge action.
        res = get_matching_proposal_ids(action="merge")
        self.assertIn(14, res)
        # Filter by delete action.
        res_del = get_matching_proposal_ids(action="delete")
        self.assertIn(11, res_del)
        self.assertIn(15, res_del)
        
        # Filter by open status.
        res_open = get_matching_proposal_ids(status="open")
        self.assertIn(12, res_open)
        self.assertIn(14, res_open)
        self.assertIn(15, res_open)
        
        # Filter by archive status.
        res_arch = get_matching_proposal_ids(status="archive")
        self.assertIn(11, res_arch)
        self.assertIn(13, res_arch)
        
        # Filter by author 0.
        res_auth = get_matching_proposal_ids(author=0)
        self.assertGreaterEqual(len(res_auth), 5)
        self.assertIn(12, res_auth)

        # Filter by target object type.
        res_med = get_matching_proposal_ids(object="media")
        self.assertIn(12, res_med)
        self.assertIn(15, res_med)
        
        # 6. No matches.
        res_none = get_matching_proposal_ids(author=999) # NonExistentUser.
        self.assertEqual(len(res_none), 0)

    def test_get_matching_proposal_ids_combined(self):
        """Test get_matching_proposal_ids with combined criteria."""
        # Combine action and status filters to target open deletions.
        res_comb = get_matching_proposal_ids(action="delete", status="open")
        self.assertIn(15, res_comb)
        self.assertNotIn(11, res_comb) # 11 is archive.


    def test_delete_proposal(self):
        """Test delete_proposal."""
        p_id = 1004
        try:
            proposal = Proposal(id=p_id, original=1, final=2, action="merge", object="Individual1", status="open", author=0, time=1000)
            upsert_proposal(proposal)

            # Verify exists.
            fetched = get_proposals(ids=[p_id])
            self.assertEqual(len(fetched), 1)

            # Delete.
            res = delete_proposal(p_id)
            self.assertEqual(res, SUCCESS)

            # Verify gone.
            fetched_after = get_proposals(ids=[p_id])
            self.assertEqual(len(fetched_after), 0)
        finally:
            # Ensure cleanup if delete failed or logic broke.
            delete_proposal(p_id)

    def test_delete_proposal_non_existent(self):
        """Test delete_proposal with non-existent ID."""
        res = delete_proposal(-9999)
        self.assertEqual(res, ERROR)

if __name__ == '__main__':
    unittest.main()
