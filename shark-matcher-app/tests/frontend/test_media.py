"""Unit tests for frontend media retrieval, filtering, creation, update, and deletion."""

import http
import unittest
from unittest.mock import MagicMock, patch

from core import app_constants
from core import app_utils
import backend.db.action
import backend.db.individual
import backend.db.media
import backend.db.user
import frontend.media
from tests.frontend.test_upload_batch import _make_tiny_jpeg_data_url


class MockHandler:
    pass

class TestMediaFrontend(unittest.TestCase):
    def setUp(self):
        self.handler = MockHandler()
        # Do not clear DB. Seeded data is persistent..
        
        # Create user (transient).
        self.username = "media_tester"
        u = backend.db.user.User(id=891, username=self.username, name="Media Tester", email="media@test.com", role="admin", password="pass", salt="salt")
        backend.db.user.upsert_user(u)

    def tearDown(self):
        # Cleanup transient user.
        try:
            backend.db.user.delete_user(891)
        except:
            pass

    def test_get_medias(self):
        """Test getting medias with seeded data (success)."""
        # Fetch seeded media record 1.
        medias, code, _ = frontend.media.get_medias(self.handler, "", ids=1, get_full=False)
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertEqual(len(medias), 1)
        self.assertEqual(medias[0]['name'], "AN11101602")

    def test_get_medias_non_existent(self):
        """Test getting medias with non-existent ID."""
        medias_none, code_none, _ = frontend.media.get_medias(self.handler, "", ids=[-9999], get_full=False)
        self.assertEqual(code_none, http.HTTPStatus.OK)
        self.assertEqual(medias_none, [])

    def test_filter_medias(self):
        """Test filter_medias with seeded data."""
        # Seeded media rows provide distinct locations and types for filtering.
        
        # Filter by location FAR.
        data, code, _ = frontend.media.filter_medias(self.handler, "", locations=["FAR"])
        self.assertEqual(code, http.HTTPStatus.OK)
        ids = [m['id'] for m in data]
        self.assertIn(4, ids)
        self.assertIn(5, ids)
        self.assertNotIn(1, ids)
        
        # Filter by topshot type.
        data_ts, code_ts, _ = frontend.media.filter_medias(self.handler, "", types=["topshot"])
        self.assertEqual(code_ts, http.HTTPStatus.OK)
        ids_ts = [m['id'] for m in data_ts]
        self.assertIn(1, ids_ts)
        self.assertNotIn(4, ids_ts) # underwater.

        # Filtering by good quality matches media 1 and 21 while excluding null quality media 4.
        data_q, code_q, _ = frontend.media.filter_medias(self.handler, "", qualities=["good"])
        self.assertEqual(code_q, http.HTTPStatus.OK)
        ids_q = [m['id'] for m in data_q]
        self.assertIn(1, ids_q)
        self.assertIn(21, ids_q)
        self.assertNotIn(4, ids_q)
        
        # Filtering by min_time includes media 2 and excludes earlier media 1.
        data_time, code_time, _ = frontend.media.filter_medias(self.handler, "", min_time=1400000000)
        self.assertEqual(code_time, http.HTTPStatus.OK)
        ids_time = [m['id'] for m in data_time]
        self.assertIn(2, ids_time)
        self.assertNotIn(1, ids_time)
        
        # Combining location and type filters matches only media 1.
        data_comb, code_comb, _ = frontend.media.filter_medias(self.handler, "", locations=["AN"], types=["topshot"])
        self.assertEqual(code_comb, http.HTTPStatus.OK)
        ids_comb = [m['id'] for m in data_comb]
        self.assertIn(1, ids_comb)
        self.assertNotIn(4, ids_comb) # FAR.
        self.assertNotIn(21, ids_comb) # Underwater.

        # 6. No Match.
        data_none, code_none, _ = frontend.media.filter_medias(self.handler, "", locations=["NonExistentLoc"])
        self.assertEqual(code_none, http.HTTPStatus.OK)
        self.assertEqual(len(data_none), 0)

    @patch('os.rename')
    @patch('frontend.individual.post_individual')
    @patch('backend.db.individual._get_next_individual_name', return_value="NewIndividualForMedia")
    @patch('backend.db.media.get_individual_ids_by_media_ids')
    def test_post_media_create(self, mock_get_individual_ids, mock_get_next_name, mock_post_individual, mock_rename):
        """Test creating new media."""
        # Transient IDs.
        s_id = 15000
        delete_media_id = None
        
        # Mocking individual creation internal call.
        mock_post_individual.return_value = ({"action_status": {}}, http.HTTPStatus.OK, None)
        mock_get_individual_ids.return_value = [s_id] # Match the created individual ID.
        
        try:
            s = backend.db.individual.Individual(id=s_id, name="LinkedIndividual", source=s_id)
            res_s = backend.db.individual.upsert_individual(s)
            self.assertNotEqual(res_s, app_constants.ERROR, "Individual upsert failed")
            
            # Verify individual exists.
            check_s = backend.db.individual.get_individuals(ids=[s_id])
            self.assertEqual(len(check_s), 1, "Individual 5000 not found after upsert")
    
            # Define the arguments for testing
            test_kwargs = {
                "id": None, "hash": "newhash", "name": "NewMedia", "source": None,
                "individual": None, "metadata": None, "location": None, "time": None,
                "type": "image", "quality": None, "length": 250, "sex": "F", "user_id": 891
            }
            kwargs = {
                "id": "",
                "hash": "hash_new_media_fe",
                "name": "NewMediaFrontend",
                "source": None,
                "individual": None,
                "metadata": None,
                "location": None,
                "time": None,
                "type": "image",
                "quality": None,
                "length": None,
                "sex": None,
                "user_id": 891,
                "alternate_names": [],
                "tags": [],
                "labels": []
            }
    
            resp, code, _ = frontend.media.post_media(self.handler, "", **kwargs)
            
            self.assertEqual(code, http.HTTPStatus.OK)
            found_list = backend.db.media.get_medias(names=["NewMediaFrontend"])
            self.assertTrue(len(found_list) > 0)
            found = found_list[0]
            delete_media_id = found.id
            self.assertEqual(found.hash, "hash_new_media_fe")
    
            # Verify Action Log (Create Media).
            # Expect a "create" action for object "media" by self.username.
            actions = backend.db.action.get_matching_action_ids(author=891, action="create", object="media")
            self.assertTrue(len(actions) > 0, "No create media action found")
        finally:
            if delete_media_id:
                backend.db.media.delete_media(delete_media_id)
            backend.db.individual.delete_individual(s_id)

    @patch('os.rename')
    def test_post_media_update(self, mock_rename):
        """Test updating existing media and renaming file."""
        # Create initial media and individual.
        m_id = 13001
        m = backend.db.media.Media(id=m_id, name="OriginalMediaFE", filename="OriginalMediaFE.jpeg", hash="hash3001", individual=m_id)
        backend.db.media.upsert_media(m)
        
        s = backend.db.individual.Individual(id=m_id, name="IndividualFE", source=m_id, media=m_id, length=100, sex="U")
        backend.db.individual.upsert_individual(s)
        
        try:
            # Update details.
            kwargs = {
                "id": m_id,
                "hash": "hash3001",
                "name": "RenamedMediaFE", # Name change triggers rename.
                "source": None,
                "individual": m_id,
                "metadata": None,
                "location": None,
                "time": None,
                "type": "image",
                "quality": None,
                "length": 150, # Individual prop change.
                "sex": "F", # Individual prop change.
                "user_id": 891
            }
            
            resp, code, _ = frontend.media.post_media(self.handler, "", **kwargs)
            
            self.assertEqual(code, http.HTTPStatus.OK)
            
            # Verify media updated.
            m_upd = backend.db.media.get_medias(ids=[m_id])[0]
            self.assertEqual(m_upd.name, "RenamedMediaFE")
            self.assertEqual(m_upd.filename, "RenamedMediaFE.jpeg")
            
            # Verify individual updated.
            s_upd = backend.db.individual.get_individuals(ids=[m_id])[0]
            self.assertEqual(s_upd.length, 150)
            self.assertEqual(s_upd.sex, "F")
    
            # Verify Action Log (Update Media).
            actions = backend.db.action.get_matching_action_ids(author=891, action="update", object="media")
            self.assertTrue(len(actions) > 0, "No update media action found")
        finally:
            backend.db.media.delete_media(m_id)
            backend.db.individual.delete_individual(m_id)

    @patch('os.remove')
    def test_delete_media(self, mock_remove):
        """Test deleting media (existing)."""
        # Setup media linked to individual.
        m_id = 14000
        m = backend.db.media.Media(id=m_id, name="DeleteMediaFE", filename="del.jpg", hash="hashDel", individual=m_id)
        backend.db.media.upsert_media(m)
        
        # Setup source individual
        s_source = backend.db.individual.Individual(id=m_id+1, source=m_id+1, reference=m_id)
        backend.db.individual.upsert_individual(s_source)
        
        # Setup media individual pointing to source
        s = backend.db.individual.Individual(id=m_id, media=m_id, source=m_id+1)
        backend.db.individual.upsert_individual(s)
    
        try:
            resp, code, _ = frontend.media.delete_media(self.handler, "", id=m_id, user_id=891)
    
            self.assertEqual(code, http.HTTPStatus.OK)
            self.assertIn("action_status", resp)
            
            # Verify Action Log (Delete Media).
            actions = backend.db.action.get_matching_action_ids(author=891, action="delete", object="media")
            self.assertTrue(len(actions) > 0, "No delete media action found")
        except Exception as e:
            self.fail(f"delete_media raised exception: {e}")
        finally:
            # Ensure cleanup if delete failed.
            app_utils.execute_sql("DELETE FROM Media WHERE id = :id", {"id": m_id})
            app_utils.execute_sql("DELETE FROM Individuals WHERE id IN (:id1, :id2)", {"id1": m_id, "id2": m_id+1})

    def test_delete_media_non_existent(self):
        """Test deleting non-existent media."""
        resp, code, _ = frontend.media.delete_media(self.handler, "", id=-9999, user_id=891)
        # Should return (ERROR, OK, None).
        self.assertEqual(resp, app_constants.ERROR)
        self.assertEqual(code, http.HTTPStatus.OK)

    def test_save_media(self):
        """Test save_media (file write mock)."""
        data_url = _make_tiny_jpeg_data_url()
        with patch('PIL.Image.Image.save') as save:
            
            resp, code, _ = frontend.media.save_media(self.handler, "", data_url=data_url, filename="test.jpg")
            
            self.assertEqual(code, http.HTTPStatus.OK)
            save.assert_called_once()


if __name__ == '__main__':
    unittest.main()
