"""Tests for project frontend endpoints."""

import datetime
import http
import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from core import app_constants
import frontend.common
import frontend.project

_MOCK_INFERENCE_STATS = {
    "filenames": ["test.jpg"],
    "embeddings_metadata": {"start_time": 100, "update_time": 200},
    "loaded_models": ["test_model"],
    "model_statuses": [{
        "display_name": "test_model",
        "embedding_set": "test_model",
        "required": True,
        "state": "ready",
        "error": None,
    }],
    "generation_id": "22222222-2222-4222-8222-222222222222",
    "snapshot_id": "11111111-1111-4111-8111-111111111111",
}

class MockHandlerProject:
    def __init__(self):
        self._models = ["test_model"]

class TestProjectFrontend(unittest.TestCase):
    def setUp(self):
        self.handler = MockHandlerProject()

        # Patch inference client so tests don't make real HTTP requests.
        self._patch_stats = patch(
            'frontend.inference_client.get_stats',
            return_value=_MOCK_INFERENCE_STATS,
        )
        self._patch_stats.start()

        # Stats are memoized process-wide; start each test from a cold cache.
        frontend.project.invalidate_dataset_stats()

    def tearDown(self):
        self._patch_stats.stop()
        frontend.project.invalidate_dataset_stats()

    def test_get_dataset_stats(self):
        """Test getting dataset stats."""
        # It queries DB counts. We rely on DB state (empty or whatever).
        stats, code, _ = frontend.project.get_dataset_stats(self.handler, "")
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertIn("source_individuals", stats)
        self.assertIn("embedded_images", stats)
        # Mocked inference stats returns 1 image neighbor.
        self.assertEqual(stats["embedded_images"], 1)

    def test_get_dataset_stats_is_cached(self):
        """A second call within the TTL is served from the memo, not rebuilt."""
        first, _, _ = frontend.project.get_dataset_stats(self.handler, "")

        with patch('frontend.project._get_dataset_stats') as mock_build:
            second, code, _ = frontend.project.get_dataset_stats(self.handler, "")

        self.assertEqual(code, http.HTTPStatus.OK)
        mock_build.assert_not_called()
        self.assertIs(second, first)

    def test_invalidate_dataset_stats_forces_rebuild(self):
        """Invalidation drops the memo so the next call rebuilds."""
        frontend.project.get_dataset_stats(self.handler, "")
        frontend.project.invalidate_dataset_stats()

        with patch('frontend.project._get_dataset_stats', return_value={}) as mock_build:
            frontend.project.get_dataset_stats(self.handler, "")

        mock_build.assert_called_once()

    def test_stats_say_whether_the_matcher_answered(self):
        """Home can tell an unreachable matcher apart from a matcher with zero embeddings."""
        self.assertTrue(frontend.project._get_dataset_stats()["inference_available"])
        with patch('frontend.inference_client.get_stats', return_value=None):
            self.assertFalse(frontend.project._get_dataset_stats()["inference_available"])

    def test_stats_computed_across_a_write_are_not_cached(self):
        """A write that lands while stats are being built leaves the cache empty, not stale."""
        def build_while_a_write_lands():
            frontend.project.invalidate_dataset_stats()
            return {"embedded_images": 0}

        with patch('frontend.project._get_dataset_stats', side_effect=build_while_a_write_lands):
            frontend.project.get_dataset_stats(self.handler, "")
        with patch('frontend.project._get_dataset_stats', return_value={}) as mock_build:
            frontend.project.get_dataset_stats(self.handler, "")
        mock_build.assert_called_once()

    @patch('shutil.make_archive')
    @patch('frontend.project._export_annotations')
    @patch('builtins.open')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_export_data(self, mock_isdir, mock_exists, mock_open, mock_export_anno, mock_make_archive):
        """Test export data logic."""
        mock_exists.return_value = True
        mock_isdir.return_value = True

        with tempfile.TemporaryDirectory() as export_directory:
            mock_export_anno.return_value = export_directory

            # Mock file read.
            m = MagicMock()
            m.__enter__.return_value.read.return_value = b"fake data"
            mock_open.return_value = m

            resp, code, _ = frontend.project.export_data(
                self.handler,
                "",
                content="annotations",
                format="snapshot",
            )
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertIsInstance(resp, str)

    def test_export_data_rejects_unknown_content(self):
        """An unsupported export content type is a 400 raised before filesystem work."""
        with self.assertRaises(frontend.common.APIError) as raised:
            frontend.project.export_data(
                self.handler,
                "",
                content="unsupported",
                format="snapshot",
            )

        self.assertEqual(raised.exception.status, http.HTTPStatus.BAD_REQUEST)

    @patch('time.sleep')
    @patch('frontend.inference_client.get_embeddings_status')
    @patch('frontend.inference_client.start_embeddings_generation')
    def test_generate_and_monitor_polls_until_succeeded(
            self, mock_start, mock_status, _mock_sleep):
        """Non-blocking generation polls status until the job reports succeeded."""
        mock_start.return_value = {"status": "started", "job_id": "job-1"}
        mock_status.side_effect = [
            {"status": "running", "job_id": "job-1"},
            {"status": "succeeded", "job_id": "job-1"},
        ]

        ok = frontend.project._generate_and_monitor("test_model", "snap-1")
        self.assertTrue(ok)
        self.assertEqual(mock_status.call_count, 2)

    @patch('time.sleep')
    @patch('frontend.inference_client.get_embeddings_status')
    @patch('frontend.inference_client.start_embeddings_generation')
    def test_generate_and_monitor_raises_on_worker_failure(
            self, mock_start, mock_status, _mock_sleep):
        """A failed background worker surfaces its diagnostic error string."""
        mock_start.return_value = {"status": "started", "job_id": "job-1"}
        mock_status.return_value = {
            "status": "failed",
            "job_id": "job-1",
            "error": "Embedding worker process terminated by signal SIGILL (4)",
        }

        with self.assertRaisesRegex(RuntimeError, "SIGILL"):
            frontend.project._generate_and_monitor("test_model", "snap-1")


_SNAPSHOT_ID = "33333333-3333-4333-8333-333333333333"


class TestEmbeddingsUpdateCooldown(unittest.TestCase):
    """Update Embeddings is refused for an hour after an update the app itself started."""

    def setUp(self):
        frontend.project._reset_embeddings_update_job()
        self.addCleanup(frontend.project._reset_embeddings_update_job)
        self._snapshot_root = tempfile.TemporaryDirectory()
        self.addCleanup(self._snapshot_root.cleanup)
        patcher = patch.object(app_constants, "INFERENCE_SNAPSHOT_ROOT", self._snapshot_root.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_snapshot(self, age_seconds):
        created_at = datetime.datetime.fromtimestamp(time.time() - age_seconds, datetime.timezone.utc)
        directory = os.path.join(self._snapshot_root.name, _SNAPSHOT_ID)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as manifest:
            json.dump({"created_at": created_at.isoformat().replace("+00:00", "Z")}, manifest)

    def _call(self, *, age_seconds, role="admin", built_age_seconds=None, **kwargs):
        """Call the update with embeddings from an app snapshot taken age_seconds ago (None: no snapshot)."""
        if age_seconds is None:
            stats = {"snapshot_id": None, "embeddings_metadata": {
                "start_time": time.time() - (built_age_seconds or 0)}}
        else:
            self._write_snapshot(age_seconds)
            stats = {"snapshot_id": _SNAPSHOT_ID, "embeddings_metadata": {
                "start_time": time.time() - (built_age_seconds or age_seconds)}}
        handler = MagicMock(authenticated_user=MagicMock(role=role))
        with patch("frontend.inference_client.get_stats", return_value=stats), \
                patch("threading.Thread") as thread:
            result = frontend.project.update_embeddings(handler, "", **kwargs)
        return result, thread

    def test_command_line_build_does_not_lock(self):
        (_, code, _), thread = self._call(age_seconds=None, built_age_seconds=60)
        self.assertEqual(code, http.HTTPStatus.OK)
        thread.assert_called_once()

    def test_maintenance_rebuild_of_an_old_app_update_does_not_lock(self):
        (_, code, _), _ = self._call(age_seconds=7200, built_age_seconds=60)
        self.assertEqual(code, http.HTTPStatus.OK)

    def test_snapshot_missing_from_disk_does_not_lock(self):
        stats = {"snapshot_id": _SNAPSHOT_ID, "embeddings_metadata": {"start_time": time.time()}}
        handler = MagicMock(authenticated_user=MagicMock(role="admin"))
        with patch("frontend.inference_client.get_stats", return_value=stats), \
                patch("threading.Thread"):
            _, code, _ = frontend.project.update_embeddings(handler, "")
        self.assertEqual(code, http.HTTPStatus.OK)

    def test_stats_give_the_page_the_cooldown_start(self):
        self._write_snapshot(600)
        with patch("frontend.inference_client.get_stats",
                   return_value={"snapshot_id": _SNAPSHOT_ID}):
            stats = frontend.project._get_dataset_stats()
        self.assertAlmostEqual(stats["update_cooldown_started_at"], time.time() - 600, delta=5)

    def test_recent_embeddings_are_refused_with_429_and_retry_after(self):
        (message, code, headers), thread = self._call(age_seconds=600)
        self.assertEqual(code, http.HTTPStatus.TOO_MANY_REQUESTS)
        self.assertIn("last hour", message)
        self.assertAlmostEqual(int(headers["Retry-After"]), 3000, delta=5)
        thread.assert_not_called()

    def test_old_embeddings_may_be_updated(self):
        (result, code, _), thread = self._call(age_seconds=3700)
        self.assertEqual(code, http.HTTPStatus.OK)
        self.assertEqual(result["status"], "in_progress")
        thread.assert_called_once()

    def test_unknown_start_time_does_not_lock(self):
        (_, code, _), _ = self._call(age_seconds=None)
        self.assertEqual(code, http.HTTPStatus.OK)

    def test_owner_may_force_an_update_inside_the_hour(self):
        (_, code, _), thread = self._call(age_seconds=600, role="owner", force=True)
        self.assertEqual(code, http.HTTPStatus.OK)
        thread.assert_called_once()

    def test_owner_without_force_is_still_refused(self):
        (_, code, _), _ = self._call(age_seconds=600, role="owner")
        self.assertEqual(code, http.HTTPStatus.TOO_MANY_REQUESTS)

    def test_admin_cannot_force_the_update(self):
        (_, code, _), _ = self._call(age_seconds=600, role="admin", force=True)
        self.assertEqual(code, http.HTTPStatus.TOO_MANY_REQUESTS)

    def test_in_progress_is_reported_before_the_cooldown(self):
        with patch.dict(frontend.project._embeddings_update_job, {"status": "in_progress"}):
            (_, code, _), _ = self._call(age_seconds=600)
        self.assertEqual(code, http.HTTPStatus.CONFLICT)

    def test_stats_tell_the_page_the_cooldown_length(self):
        with patch("frontend.inference_client.get_stats", return_value={}):
            stats = frontend.project._get_dataset_stats()
        self.assertEqual(stats["update_cooldown_seconds"], 3600)

