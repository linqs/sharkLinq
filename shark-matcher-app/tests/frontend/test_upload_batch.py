"""
Tests for the atomic batch upload endpoint (frontend.media.upload_batch).
"""
import unittest
import http
import os
import json
import io
import base64
from unittest.mock import patch, MagicMock

import backend.db.media
import backend.db.individual
import backend.db.user
from core import app_constants
import frontend.common
import frontend.media


class MockHandler:
    pass


# Pre-generate a valid tiny JPEG data URL at module load time (before any patches).
def _make_tiny_jpeg_data_url():
    """Create a valid minimal JPEG as a data URL for testing."""
    import PIL.Image
    img = PIL.Image.new("RGB", (2, 2), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# Cache it to avoid re-generation.
_CACHED_DATA_URL = _make_tiny_jpeg_data_url()


def _make_batch_item(name, hash_val=None, data_url=None, **extra):
    """Helper to build a single batch item dict."""
    item = {
        "name": name,
        "hash": hash_val or f"hash_{name}",
        "metadata": None,
        "location": None,
        "time": None,
        "type": "underwater",
        "quality": None,
        "length": None,
        "sex": None,
        "data_url": data_url or _CACHED_DATA_URL,
        "alternate_names": [],
        "tags": [],
        "labels": [],
    }
    item.update(extra)
    return item


class TestUploadBatch(unittest.TestCase):
    def setUp(self):
        self.handler = MockHandler()
        # Create user (transient).
        self.username = "batch_tester"
        u = backend.db.user.User(
            id=892, username=self.username, name="Batch Tester",
            email="batch@test.com", role="admin", password="pass", salt="salt"
        )
        backend.db.user.upsert_user(u)

    def tearDown(self):
        try:
            backend.db.user.delete_user(892)
        except Exception:
            pass

    def _cleanup_results(self, results):
        """Helper to clean up DB entries created during tests."""
        for res in results:
            try:
                backend.db.individual.delete_individual(res["individual_id"])
            except Exception:
                pass
            try:
                backend.db.individual.delete_individual(res["source_id"])
            except Exception:
                pass
            try:
                backend.db.media.delete_media(res["media_id"])
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Test 1: Happy path — batch of 3 unique images succeeds.
    # -----------------------------------------------------------------------
    @patch("os.remove")
    @patch("frontend.media.PIL.Image.Image.save")
    def test_happy_path_batch_upload(self, mock_img_save, mock_remove):
        """Batch upload of 3 unique images should succeed and create DB entries."""
        items = [
            _make_batch_item("test_batch_a", "hash_unique_a"),
            _make_batch_item("test_batch_b", "hash_unique_b"),
            _make_batch_item("test_batch_c", "hash_unique_c"),
        ]

        result, code, _ = frontend.media.upload_batch(
            self.handler, "", items=items, user_id=892
        )

        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 3)

        # Verify all media are in the DB.
        medias = backend.db.media.get_medias(ids=None, get_full=False)
        names = [m.name for m in medias]
        for item in items:
            self.assertIn(item["name"], names)

        # Cleanup.
        self._cleanup_results(result["results"])

    @patch("os.remove")
    @patch("frontend.media.PIL.Image.Image.save")
    def test_lowercase_sex_is_stored_uppercase(self, mock_img_save, mock_remove):
        """An image uploaded with sex 'f' creates individuals stored as 'F', so sex filters find them."""
        result, code, _ = frontend.media.upload_batch(
            self.handler, "", items=[_make_batch_item("test_batch_sex", "hash_sex", sex="f")],
            user_id=892)
        self.assertEqual(code, http.HTTPStatus.OK)
        try:
            entry = result["results"][0]
            for individual_id in (entry["individual_id"], entry["source_id"]):
                individual = backend.db.individual.get_individuals(ids=[individual_id])[0]
                self.assertEqual(individual.sex, "F")
        finally:
            self._cleanup_results(result["results"])

    def test_invalid_sex_rejects_the_batch(self):
        """A sex value outside F, M and the unknown markers is a 400 naming the field."""
        with self.assertRaises(frontend.common.APIError) as raised:
            frontend.media.upload_batch(
                self.handler, "", items=[_make_batch_item("test_batch_bad", "hash_bad", sex="x")],
                user_id=892)
        self.assertIn("sex", str(raised.exception))

    # -----------------------------------------------------------------------
    # Test 2: Duplicate hash against DB — entire batch rejected.
    # -----------------------------------------------------------------------
    def test_duplicate_hash_against_db_rejects_batch(self):
        """If one image has a hash matching an existing DB entry, the entire batch is rejected."""
        existing_media = backend.db.media.get_medias(ids=[1], get_full=False)
        existing_hash = existing_media[0].hash if existing_media else "seeded_hash"

        items = [
            _make_batch_item("test_dup_hash_a", "hash_fresh_dup_1"),
            _make_batch_item("test_dup_hash_b", existing_hash),  # Duplicate!
            _make_batch_item("test_dup_hash_c", "hash_fresh_dup_3"),
        ]

        result, code, _ = frontend.media.upload_batch(
            self.handler, "", items=items, user_id=892
        )

        self.assertEqual(code, http.HTTPStatus.CONFLICT)
        parsed = json.loads(result)
        self.assertIn("errors", parsed)
        self.assertTrue(len(parsed["errors"]) > 0)
        self.assertIn("Duplicate hash", parsed["errors"][0]["reason"])

    # -----------------------------------------------------------------------
    # Test 3: Duplicate name against DB — entire batch rejected.
    # -----------------------------------------------------------------------
    def test_duplicate_name_against_db_rejects_batch(self):
        """If one image has a name matching an existing DB entry, the entire batch is rejected."""
        existing_media = backend.db.media.get_medias(ids=[1], get_full=False)
        existing_name = existing_media[0].name if existing_media else "AN11101602"

        items = [
            _make_batch_item("test_dup_name_a", "hash_fresh_name_1"),
            _make_batch_item(existing_name, "hash_fresh_name_2"),  # Duplicate name!
            _make_batch_item("test_dup_name_c", "hash_fresh_name_3"),
        ]

        result, code, _ = frontend.media.upload_batch(
            self.handler, "", items=items, user_id=892
        )

        self.assertEqual(code, http.HTTPStatus.CONFLICT)
        parsed = json.loads(result)
        self.assertIn("errors", parsed)
        name_errors = [e for e in parsed["errors"] if "Duplicate name" in e["reason"]]
        self.assertTrue(len(name_errors) > 0)

    # -----------------------------------------------------------------------
    # Test 4: Duplicate within batch — pre-validation catches it.
    # -----------------------------------------------------------------------
    def test_duplicate_within_batch_rejects(self):
        """If two images in the same batch have the same name, pre-validation rejects."""
        items = [
            _make_batch_item("test_within_dup", "hash_within_1"),
            _make_batch_item("test_within_dup", "hash_within_2"),  # Same name!
        ]

        result, code, _ = frontend.media.upload_batch(
            self.handler, "", items=items, user_id=892
        )

        self.assertEqual(code, http.HTTPStatus.CONFLICT)
        parsed = json.loads(result)
        self.assertIn("errors", parsed)
        batch_dup_errors = [e for e in parsed["errors"] if "within this batch" in e["reason"]]
        self.assertTrue(len(batch_dup_errors) > 0)

    # -----------------------------------------------------------------------
    # Test 5: File write failure — DB transaction rolled back.
    # -----------------------------------------------------------------------
    @patch("frontend.media.PIL.Image.Image.save")
    def test_file_write_failure_rolls_back_db(self, mock_img_save):
        """If saving an image file fails, the DB transaction should be rolled back."""
        mock_img_save.side_effect = IOError("Disk full")

        items = [
            _make_batch_item("test_rollback_a", "hash_rollback_1"),
        ]

        result, code, _ = frontend.media.upload_batch(
            self.handler, "", items=items, user_id=892
        )

        self.assertEqual(code, http.HTTPStatus.INTERNAL_SERVER_ERROR)
        parsed = json.loads(result)
        self.assertIn("errors", parsed)

        # Verify the DB was rolled back — no media with this name should exist.
        all_media = backend.db.media.get_medias(ids=None, get_full=False)
        names = [m.name for m in all_media]
        self.assertNotIn("test_rollback_a", names)

    # -----------------------------------------------------------------------
    # Test 6: Empty batch — returns error.
    # -----------------------------------------------------------------------
    def test_empty_batch_returns_error(self):
        """Sending an empty batch returns BAD_REQUEST."""
        result, code, _ = frontend.media.upload_batch(
            self.handler, "", items=[], user_id=892
        )
        self.assertEqual(code, http.HTTPStatus.BAD_REQUEST)

    # -----------------------------------------------------------------------
    # Test 7: Error response format verification.
    # -----------------------------------------------------------------------
    def test_error_response_contains_filename_and_reason(self):
        """Error response should contain 'filename' and 'reason' for each error."""
        items = [
            _make_batch_item("test_format_a", "hash_format_1"),
            _make_batch_item("test_format_a", "hash_format_2"),  # Duplicate name.
        ]

        result, code, _ = frontend.media.upload_batch(
            self.handler, "", items=items, user_id=892
        )

        self.assertEqual(code, http.HTTPStatus.CONFLICT)
        parsed = json.loads(result)
        for error in parsed["errors"]:
            self.assertIn("filename", error)
            self.assertIn("reason", error)
            self.assertIsInstance(error["filename"], str)
            self.assertIsInstance(error["reason"], str)


if __name__ == "__main__":
    unittest.main()
