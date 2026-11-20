"""Unit tests for frontend match generation, filtering, and scoring logic."""

import http
import unittest
from unittest.mock import patch

import backend.db.individual
import backend.db.user
import frontend.match

# Mock match data keyed by model name.
_MOCK_MATCH_DATA = {
    "ViT":  {"ids": [0, 1], "distances": [0.1,  0.2],  "labels": ["a", "b"]},
    "Miew": {"ids": [0, 1], "distances": [0.05, 0.15], "labels": ["a", "b"]},
}
_MOCK_MATCH_DEFAULT = _MOCK_MATCH_DATA["ViT"]

class MockHandler:
    def __init__(self):
        self._models = ["ViT"]


class TestMatchFrontend(unittest.TestCase):
    def setUp(self):
        self.handler = MockHandler()

        # Patch inference client so tests don't make real HTTP requests.
        self._patch_matches = patch(
            'frontend.inference_client.get_matches',
            side_effect=_mock_get_matches,
        )
        self._patch_matches.start()

        # Create user (transient). ID 893 for test_match.
        self.username = "matcher"
        u = backend.db.user.User(id=893, username=self.username, name="Matcher Name", email="match@test.com", role="collaborator", password="pass", salt="salt")
        backend.db.user.upsert_user(u)

    def tearDown(self):
        self._patch_matches.stop()
        try:
            backend.db.user.delete_user(893)
        except:
            pass

    def test_get_neighbors_individual_name(self):
        """Test _get_neighbors identifying individual by name with seeded data."""
        ids, dists, labels, dist_dict, source_individual_dict = frontend.match._get_neighbors(
            self.handler, "", name="0", method="nearest neighbor", model="ViT"
        )

        # Verify.
        self.assertEqual(ids, [0, 1])
        self.assertEqual(dists, [0.1, 0.2])
        # Individual 0 (Media) source is Individual 21.
        self.assertEqual(source_individual_dict["id"], 21)

    def test_get_matches(self):
        """Test get_matches success flow and sorting with seeded data."""
        # Result: Source 21 with min dist 0.1.

        resp, code, _ = frontend.match.get_matches(self.handler, "", name="0", method="nearest neighbor", model="ViT")

        # Verify.
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertTrue(resp["matches"])

        suggestions = resp["match_suggestions"]
        # Expected:
        # Shark 21: Dist 0.1 (min of 0.1 and 0.2.

        # Check first match.
        self.assertEqual(suggestions[0]["id"], 21)
        self.assertEqual(suggestions[0]["distance"], 0.1)

    def test_get_matches_multi_model(self):
        """Test that different models return different results based on model param."""
        # Test ViT (distances 0.1, 0.2).
        resp_vit, code_vit, _ = frontend.match.get_matches(self.handler, "", name="0", method="nearest neighbor", model="ViT")
        self.assertEqual(code_vit, http.HTTPStatus.OK)

        # Test Miew (distances 0.05, 0.15).
        resp_miew, code_miew, _ = frontend.match.get_matches(self.handler, "", name="0", method="nearest neighbor", model="Miew")
        self.assertEqual(code_miew, http.HTTPStatus.OK)

        suggestions_vit = resp_vit["match_suggestions"]
        suggestions_miew = resp_miew["match_suggestions"]

        # Confirm both return Shark 21 (as IDs 0,1 map to it).
        self.assertEqual(suggestions_vit[0]["id"], 21)
        self.assertEqual(suggestions_miew[0]["id"], 21)

        # Confirm distances are different.
        self.assertAlmostEqual(suggestions_vit[0]["distance"], 0.1)
        self.assertAlmostEqual(suggestions_miew[0]["distance"], 0.05)
        self.assertNotEqual(suggestions_vit[0]["distance"], suggestions_miew[0]["distance"])

    def test_filter_matches(self):
        """Test filter_matches logic with seeded data."""
        # Inference client mock returns [0, 1].
        # Seeded Data:
        # Shark 21: Sex F. (Matches neighbors 0 and 1).

        # Call.
        resp, code, _ = frontend.match.filter_matches(self.handler, "", name="0", method="nearest neighbor", model="ViT", sex="F")

        # Verify.
        self.assertEqual(code, http.HTTPStatus.OK)

        suggestions = resp["match_suggestions"]
        # Should contain Shark 21 at top (Dist 0.1).
        # Others (Dist 100000) follow.
        self.assertGreaterEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["id"], 21)
        self.assertEqual(suggestions[0]["distance"], 0.1)

        # Verify others are high distance.
        if len(suggestions) > 1:
             self.assertGreater(suggestions[1]["distance"], 100)

    def test_filter_matches_combined(self):
        """Test filter_matches with combined criteria."""
        # Seeded Data:
        # Shark 21: Sex="F", Label="fresh".

        # Successful Combination (Sex="F" + Label="fresh").
        resp, code, _ = frontend.match.filter_matches(
            self.handler, "", name="0", method="nearest neighbor", model="ViT",
            sex="F", labels="fresh"
        )
        self.assertEqual(code, http.HTTPStatus.OK)
        suggestions = resp["match_suggestions"]
        self.assertGreaterEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["id"], 21)

        # Failed Combination (Sex="M" + Label="fresh").
        # Shark 21 is F, so this should exclude it.
        resp_fail, code_fail, _ = frontend.match.filter_matches(
            self.handler, "", name="0", method="nearest neighbor", model="ViT",
            sex="M", labels="fresh"
        )
        self.assertEqual(code_fail, http.HTTPStatus.OK)
        suggestions_fail = resp_fail["match_suggestions"]
        # Should NOT contain 21. If it returns anything, it's non-matches.
        ids = [s["id"] for s in suggestions_fail]
        self.assertNotIn(21, ids)

    def test_get_matches_non_existant(self):
        """Test get_matches with non-existent individual/media."""
        # Logic is in _get_neighbors which resolves the name via DB.
        # DB will fail to find an individual ID for this name.

        resp, code, _ = frontend.match.get_matches(self.handler, "", name="GhostShark", method="nearest neighbor", model="ViT")

        # Logic returns success but empty matches.
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertFalse(resp["matches"])
        self.assertEqual(resp["match_suggestions"], [])

    def test_score_compatibility(self):
        """Test compatibility scoring logic."""
        # Case 1: Same sex -> True.
        self.assertTrue(frontend.match.score_compatibility({"sex": "M"}, {"sex": "M"}))
        # Case 2: Opposite sex -> False.
        self.assertFalse(frontend.match.score_compatibility({"sex": "M"}, {"sex": "F"}))
        # Case 3: Unknown sex -> True.
        self.assertTrue(frontend.match.score_compatibility({"sex": "U"}, {"sex": "M"}))
        self.assertTrue(frontend.match.score_compatibility({"sex": "F"}, {"sex": "U"}))

def _mock_get_matches(image_id, retrieval_method=0, model=None):
    """Return mock match results for the requested model."""
    return _MOCK_MATCH_DATA.get(model, _MOCK_MATCH_DEFAULT)

if __name__ == "__main__":
    unittest.main()
