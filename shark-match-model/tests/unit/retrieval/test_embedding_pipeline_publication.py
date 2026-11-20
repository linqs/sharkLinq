"""Tests for CLI integration with versioned embedding publication."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, call, patch

from src.core import constants  # pylint: disable=wrong-import-position
from src.retrieval import embeddings  # pylint: disable=wrong-import-position
from src.retrieval import generation_store  # pylint: disable=wrong-import-position


class TestEmbeddingPipelinePublication(unittest.TestCase):
    """Verify orchestration reuse at the CLI publication boundary."""

    def test_retrieval_mode_precedence(self):
        """Pipeline flags select the one artifact layout readers will load."""
        cases = (
            ({}, "baseline"),
            ({"use_psl": True}, "psl"),
            ({"use_multimodal": True}, "multimodal"),
            ({"use_psl": True, "use_multimodal": True}, "multimodal"),
            ({"stacked": True}, "stacked"),
        )
        for flags, expected in cases:
            with self.subTest(flags=flags):
                self.assertEqual(
                    generation_store.resolve_retrieval_mode(
                        use_psl=flags.get("use_psl", False),
                        use_multimodal=flags.get("use_multimodal", False),
                        use_stacked=flags.get("stacked", False),
                    ),
                    expected,
                )

    def test_retrieval_mode_cli_flags_round_trip(self):
        """Every artifact mode maps to CLI flags that resolve back to that mode."""
        flag_kwargs = {
            "--use-psl": "use_psl",
            "--use-multimodal": "use_multimodal",
            "--stacked": "use_stacked",
        }
        for mode in generation_store._ARTIFACT_LAYOUTS:  # pylint: disable=protected-access
            with self.subTest(mode=mode):
                flags = generation_store.retrieval_mode_cli_flags(mode)
                kwargs = {flag_kwargs[flag]: True for flag in flags}
                self.assertEqual(
                    generation_store.resolve_retrieval_mode(**kwargs), mode
                )

    @patch("src.core.utils.resolve_model_dirs")
    @patch("src.core.utils.print_script_banner")
    @patch("src.retrieval.generation_store.rollback_generation")
    def test_rollback_command_bypasses_generation_pipeline(
            self, mock_rollback, _mock_banner, mock_resolve_models):
        """The embedding CLI exposes rollback without resolving a model build."""
        mock_rollback.return_value = SimpleNamespace(generation_id="prior-id")
        argv = [
            "src.retrieval.embeddings",
            "--rollback",
            "--embeddings-name",
            "embedding-set",
        ]

        with patch.object(sys, "argv", argv):
            embeddings.main()

        mock_rollback.assert_called_once_with(
            os.path.join(constants.EMBEDDINGS_DIR, "embedding-set")
        )
        mock_resolve_models.assert_not_called()

    @patch("src.retrieval.generation_store.rollback_generation")
    def test_rollback_command_rejects_generation_modes(self, mock_rollback):
        """Rollback cannot accidentally run alongside a generation operation."""
        argv = [
            "src.retrieval.embeddings",
            "--rollback",
            "--update-embeddings",
            "--embeddings-name",
            "embedding-set",
        ]

        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            embeddings.main()

        mock_rollback.assert_not_called()

    @patch("src.retrieval.embeddings._build_generation_artifacts", return_value=7)
    @patch("src.retrieval.generation_store.publish_generation")
    @patch("src.core.utils.resolve_csv_dir", return_value="/annotations/csv")
    def test_run_pipeline_publishes_full_generation(
            self, mock_resolve_csv, mock_publish, mock_build_artifacts):
        """A full CLI run delegates staging and publication to generation_store."""
        published = object()
        captured = {}

        def publish(**kwargs):
            """Capture the publisher contract and execute its build callback."""
            captured.update(kwargs)
            self.assertEqual(kwargs["build"]("/staging/new", "/previous"), 7)
            return published

        mock_publish.side_effect = publish
        args = SimpleNamespace(
            update_embeddings=False,
            annotations_version="annotations-v1",
            use_psl=False,
            use_multimodal=False,
            stacked=False,
        )

        result = embeddings.run_pipeline(
            "trained-model", "embedding-set", args, 1_000.0
        )

        self.assertIs(result, published)
        mock_resolve_csv.assert_called_once_with(args)
        self.assertEqual(
            captured["embedding_set_root"],
            os.path.join(constants.EMBEDDINGS_DIR, "embedding-set"),
        )
        self.assertEqual(captured["embedding_set"], "embedding-set")
        self.assertEqual(captured["model_name"], "trained-model")
        self.assertEqual(captured["source"], "offline_annotations")
        self.assertIsNone(captured["snapshot_id"])
        self.assertEqual(captured["retrieval_mode"], "baseline")
        self.assertFalse(captured["seed_previous"])
        mock_build_artifacts.assert_called_once_with(
            "trained-model",
            "embedding-set",
            args,
            "/staging/new",
            "/annotations/csv",
            "baseline",
        )

    @patch("src.retrieval.embeddings._build_generation_artifacts", return_value=3)
    @patch("src.retrieval.generation_store.publish_generation")
    @patch("src.core.utils.resolve_csv_dir", return_value="/annotations/csv")
    def test_run_pipeline_seeds_incremental_generation(
            self, mock_resolve_csv, mock_publish, mock_build_artifacts):
        """An incremental CLI run asks the publisher to copy its predecessor."""
        captured = {}

        def publish(**kwargs):
            """Capture the publisher contract and execute its build callback."""
            captured.update(kwargs)
            kwargs["build"]("/staging/update", "/previous")
            return "published"

        mock_publish.side_effect = publish
        args = SimpleNamespace(
            update_embeddings=True,
            annotations_version=None,
            use_psl=True,
            use_multimodal=False,
            stacked=False,
        )

        result = embeddings.run_pipeline(
            "trained-model", "embedding-set", args, 2_000.0
        )

        self.assertEqual(result, "published")
        mock_resolve_csv.assert_called_once_with(args)
        self.assertEqual(captured["retrieval_mode"], "psl")
        self.assertTrue(captured["seed_previous"])
        mock_build_artifacts.assert_called_once_with(
            "trained-model",
            "embedding-set",
            args,
            "/staging/update",
            "/annotations/csv",
            "psl",
        )

    @patch("src.retrieval.embeddings._build_generation_artifacts", return_value=3)
    @patch("src.retrieval.generation_store.publish_generation")
    @patch("src.core.utils.resolve_csv_dir")
    def test_run_pipeline_threads_validated_snapshot_csv_to_existing_builder(
            self, mock_resolve_csv, mock_publish, mock_build_artifacts):
        """A validated snapshot supplies input without another path resolver."""
        captured = {}

        def publish(**kwargs):
            """Exercise the existing build callback against publisher staging."""
            captured.update(kwargs)
            kwargs["build"]("/staging/snapshot", "/previous")
            return "published"

        mock_publish.side_effect = publish
        args = SimpleNamespace(
            update_embeddings=True,
            annotations_version=None,
            use_psl=False,
            use_multimodal=False,
            stacked=False,
        )
        source_snapshot = SimpleNamespace(
            directory="/snapshots/validated",
            snapshot_id="11111111-1111-4111-8111-111111111111",
        )

        result = embeddings.run_pipeline(
            "trained-model",
            "embedding-set",
            args,
            2_500.0,
            source_snapshot=source_snapshot,
        )

        self.assertEqual(result, "published")
        mock_resolve_csv.assert_not_called()
        self.assertTrue(captured["seed_previous"])
        self.assertFalse(captured["seed_neighbors"])
        self.assertEqual(captured["source"], "snapshot")
        self.assertEqual(
            captured["snapshot_id"], source_snapshot.snapshot_id)
        mock_build_artifacts.assert_called_once_with(
            "trained-model",
            "embedding-set",
            args,
            "/staging/snapshot",
            "/snapshots/validated",
            "baseline",
        )

    @patch("src.retrieval.generation_store.publish_generation")
    @patch("src.core.utils.resolve_csv_dir", return_value=None)
    def test_run_pipeline_reports_missing_csv_as_library_error(
            self, _mock_resolve_csv, mock_publish):
        """Shared callers receive a catchable error instead of process exit."""
        args = SimpleNamespace(
            update_embeddings=False,
            annotations_version="annotations-v1",
        )

        with self.assertRaisesRegex(RuntimeError, "No CSV directory"):
            embeddings.run_pipeline(
                "trained-model", "embedding-set", args, 1_000.0
            )

        mock_publish.assert_not_called()

    @patch("src.retrieval.embeddings.regenerate_dataset_only", return_value=4)
    @patch("src.retrieval.generation_store.publish_generation")
    @patch("src.core.utils.resolve_csv_dir", return_value="/annotations/csv")
    @patch("os.path.isdir", return_value=True)
    def test_dataset_maintenance_reuses_pipeline_publication(
            self, _mock_isdir, mock_resolve_csv, mock_publish,
            mock_regenerate_dataset):
        """Dataset-only maintenance updates a copied generation in staging."""
        captured = {}

        def publish(**kwargs):
            """Capture and exercise the maintenance build callback."""
            captured.update(kwargs)
            self.assertEqual(kwargs["build"]("/staging/new", "/active"), 4)
            return "published"

        mock_publish.side_effect = publish
        args = SimpleNamespace(
            update_embeddings=False,
            annotations_version="annotations-v2",
            use_psl=False,
            use_multimodal=False,
            stacked=False,
        )

        result = embeddings.run_pipeline(
            "trained-model",
            "embedding-set",
            args,
            3_000.0,
            maintenance_mode="dataset",
        )

        self.assertEqual(result, "published")
        mock_resolve_csv.assert_called_once_with(args)
        self.assertTrue(captured["seed_previous"])
        self.assertTrue(captured["inherit_previous_identity"])
        mock_regenerate_dataset.assert_called_once_with(
            "trained-model",
            "embedding-set",
            args,
            "/staging/new",
            csv_dir="/annotations/csv",
        )

    @patch("src.retrieval.embeddings.generate_and_store_suggestions")
    @patch("src.retrieval.embeddings._stored_dataset_image_count", return_value=5)
    @patch("src.retrieval.generation_store.publish_generation")
    @patch("src.core.utils.resolve_csv_dir")
    @patch("os.path.isdir", return_value=True)
    def test_suggestions_maintenance_reuses_pipeline_publication(
            self, _mock_isdir, mock_resolve_csv, mock_publish, mock_count,
            mock_suggestions):
        """Suggestions-only maintenance writes through copied staging."""
        captured = {}

        def publish(**kwargs):
            """Capture and exercise the maintenance build callback."""
            captured.update(kwargs)
            self.assertEqual(kwargs["build"]("/staging/new", "/active"), 5)
            return "published"

        mock_publish.side_effect = publish
        args = SimpleNamespace(
            update_embeddings=False,
            annotations_version=None,
            use_psl=False,
            use_multimodal=True,
            stacked=False,
        )

        result = embeddings.run_pipeline(
            "trained-model",
            "embedding-set",
            args,
            4_000.0,
            maintenance_mode="suggestions",
        )

        self.assertEqual(result, "published")
        mock_resolve_csv.assert_not_called()
        self.assertEqual(captured["retrieval_mode"], "multimodal")
        self.assertTrue(captured["seed_previous"])
        self.assertTrue(captured["inherit_previous_identity"])
        mock_count.assert_called_once_with("/staging/new")
        mock_suggestions.assert_called_once_with(
            "embedding-set",
            model_dir="/staging/new",
            retrieval_mode="multimodal",
            raise_on_error=True,
        )

    @patch("src.retrieval.generation_store.publish_generation")
    @patch("os.path.isdir", return_value=False)
    def test_maintenance_requires_existing_embedding_set(
            self, _mock_isdir, mock_publish):
        """A maintenance command cannot create an incomplete embedding set."""
        args = SimpleNamespace(update_embeddings=False)

        with self.assertRaisesRegex(RuntimeError, "directory not found"):
            embeddings.run_pipeline(
                "trained-model",
                "missing-set",
                args,
                5_000.0,
                maintenance_mode="suggestions",
            )

        mock_publish.assert_not_called()

    @patch("src.retrieval.embeddings.run_pipeline")
    @patch("src.core.utils.resolve_model_dirs", return_value=["trained-model"])
    @patch("src.core.utils.print_script_banner")
    def test_cli_maintenance_modes_use_shared_pipeline(
            self, _mock_banner, _mock_resolve_models, mock_run_pipeline):
        """Both standalone CLI modes enter the authoritative coordinator."""
        cases = (
            ("--dataset-only", "dataset"),
            ("--suggestions-only", "suggestions"),
        )
        for flag, expected_mode in cases:
            with self.subTest(flag=flag):
                mock_run_pipeline.reset_mock()
                argv = [
                    "src.retrieval.embeddings",
                    "--model-name",
                    "trained-model",
                    "--embeddings-name",
                    "embedding-set",
                    flag,
                ]
                with patch.object(sys, "argv", argv), patch(
                        "time.time", return_value=6_000.0):
                    embeddings.main()

                mock_run_pipeline.assert_called_once_with(
                    "trained-model",
                    "embedding-set",
                    ANY,
                    6_000.0,
                    maintenance_mode=expected_mode,
                )

    @patch("src.retrieval.suggestions.store_suggestions")
    @patch("src.retrieval.retrieval.Retriever")
    def test_suggestions_use_explicit_staged_generation(
            self, mock_retriever_class, mock_store_suggestions):
        """Suggestion generation loads the explicit staged artifact layout."""
        retriever = mock_retriever_class.return_value

        embeddings.generate_and_store_suggestions(
            "embedding-set",
            model_dir="/generation-root/staged-set",
            retrieval_mode="multimodal",
        )

        mock_retriever_class.assert_called_once_with(
            preload_models=[{
                "display_name": "embedding-set",
                "dir": "staged-set",
                "model": "embedding-set",
                "retrieval_mode": "multimodal",
                "rerank": "none",
            }],
            embeddings_root="/generation-root",
        )
        mock_store_suggestions.assert_called_once_with(
            retriever, "/generation-root/staged-set"
        )

    @patch("src.retrieval.retrieval.Retriever",
           side_effect=ValueError("invalid staged artifacts"))
    def test_required_suggestion_failure_is_catchable(self, _mock_retriever):
        """Standalone suggestion failure aborts before pointer publication."""
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            embeddings.generate_and_store_suggestions(
                "embedding-set",
                model_dir="/generation-root/staged-set",
                raise_on_error=True,
            )

    @patch("src.retrieval.retrieval.Retriever",
           side_effect=ValueError("invalid staged artifacts"))
    def test_optional_suggestion_failure_remains_nonfatal(self, _mock_retriever):
        """Full generation retains the established optional-cache behavior."""
        embeddings.generate_and_store_suggestions(
            "embedding-set",
            model_dir="/generation-root/staged-set",
        )

    @patch("torch.load")
    @patch("os.path.exists", return_value=True)
    @patch("src.individual_dataset.IndividualDataset.create_and_store")
    @patch("src.core.utils.get_model_metadata")
    def test_dataset_maintenance_writes_and_validates_staging(
            self, mock_metadata, mock_create_dataset, _mock_exists, mock_load):
        """The existing dataset operation validates staged embedding alignment."""
        previous_dataset = MagicMock()
        previous_dataset.get_filenames.return_value = ["one.jpg", "two.jpg"]
        staged_dataset = MagicMock()
        staged_dataset.get_filenames.return_value = ["one.jpg", "two.jpg"]
        mock_load.side_effect = [
            previous_dataset, staged_dataset, MagicMock(shape=(2, 8))]
        mock_metadata.return_value = (False, None, (224, 224), {"mean": [0.5]})
        args = SimpleNamespace(annotations_version="annotations-v2")

        count = embeddings.regenerate_dataset_only(
            "trained-model",
            "embedding-set",
            args,
            "/publisher/staging",
            csv_dir="/annotations/csv",
        )

        self.assertEqual(count, 2)
        mock_create_dataset.assert_called_once_with(
            model_name="trained-model",
            channels=3,
            annotations_version="annotations-v2",
            output_dir="/publisher/staging",
            csv_dir="/annotations/csv",
            output_image_size=(224, 224),
            normalization={"mean": [0.5]},
        )

    @patch("torch.load")
    @patch("os.path.exists", return_value=True)
    @patch("src.individual_dataset.IndividualDataset.create_and_store")
    @patch("src.core.utils.get_model_metadata",
           return_value=(False, None, (224, 224), None))
    def test_dataset_maintenance_rejects_row_mismatch(
            self, _mock_metadata, _mock_create_dataset, _mock_exists,
            mock_load):
        """A replacement dataset cannot desynchronize copied embeddings."""
        previous_dataset = MagicMock()
        previous_dataset.get_filenames.return_value = ["one.jpg", "two.jpg"]
        staged_dataset = MagicMock()
        staged_dataset.get_filenames.return_value = ["one.jpg", "two.jpg"]
        mock_load.side_effect = [
            previous_dataset, staged_dataset, MagicMock(shape=(3, 8))]

        with self.assertRaisesRegex(RuntimeError, "indices aligned"):
            embeddings.regenerate_dataset_only(
                "trained-model",
                "embedding-set",
                SimpleNamespace(annotations_version="annotations-v2"),
                "/publisher/staging",
                csv_dir="/annotations/csv",
            )

    @patch("torch.load")
    @patch("os.path.exists", return_value=True)
    @patch("src.individual_dataset.IndividualDataset.create_and_store")
    @patch("src.core.utils.get_model_metadata",
           return_value=(False, None, (224, 224), None))
    def test_dataset_maintenance_rejects_reordered_filenames(
            self, _mock_metadata, _mock_create_dataset, _mock_exists,
            mock_load):
        """Same row count in a different order would pair vectors with the wrong images."""
        previous_dataset = MagicMock()
        previous_dataset.get_filenames.return_value = ["one.jpg", "two.jpg"]
        staged_dataset = MagicMock()
        staged_dataset.get_filenames.return_value = ["two.jpg", "one.jpg"]
        mock_load.side_effect = [
            previous_dataset, staged_dataset, MagicMock(shape=(2, 8))]

        with self.assertRaisesRegex(RuntimeError, "content or order"):
            embeddings.regenerate_dataset_only(
                "trained-model",
                "embedding-set",
                SimpleNamespace(annotations_version="annotations-v2"),
                "/publisher/staging",
                csv_dir="/annotations/csv",
            )

    @patch("src.retrieval.embeddings.generate_and_store_suggestions")
    @patch("torch.load")
    @patch("os.path.exists")
    @patch("src.retrieval.neighbors.store_neighbors")
    @patch("src.retrieval.embeddings.store_centroid_embeddings")
    @patch("src.retrieval.embeddings.store_image_embeddings")
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def test_existing_build_body_targets_publisher_staging(
            self, mock_store_images, mock_store_centroids, mock_store_neighbors,
            mock_exists, mock_load, mock_suggestions):
        """The extracted existing build body writes and reviews only staging."""
        staged_dataset = MagicMock()
        staged_dataset.get_filenames.return_value = ["one.jpg", "two.jpg"]
        mock_load.return_value = staged_dataset
        mock_exists.side_effect = lambda path: path.endswith("stored_dataset.pt")
        args = SimpleNamespace(
            update_embeddings=False,
            annotations_version="annotations-v1",
            use_psl=False,
            use_multimodal=False,
            stacked=False,
        )

        count = embeddings._build_generation_artifacts(  # pylint: disable=protected-access
            "trained-model",
            "embedding-set",
            args,
            "/publisher/staging",
            "/annotations/csv",
            "baseline",
        )

        self.assertEqual(count, 2)
        mock_store_images.assert_called_once_with(
            model_name="trained-model",
            model_dir="/publisher/staging",
            annotations_version="annotations-v1",
            channels=3,
            csv_dir="/annotations/csv",
        )
        mock_store_centroids.assert_called_once_with(
            model_dir="/publisher/staging"
        )
        mock_store_neighbors.assert_not_called()
        mock_suggestions.assert_called_once_with(
            "embedding-set",
            model_dir="/publisher/staging",
            retrieval_mode="baseline",
        )

    @patch("src.retrieval.embeddings.generate_and_store_suggestions")
    @patch("torch.load")
    @patch("os.path.exists")
    @patch("src.retrieval.neighbors.store_neighbors")
    @patch("src.retrieval.embeddings.store_centroid_embeddings")
    @patch("src.retrieval.embeddings.store_image_embeddings")
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def test_existing_build_body_stores_neighbors_when_requested(
            self, _mock_store_images, _mock_store_centroids,
            mock_store_neighbors, mock_exists, mock_load, _mock_suggestions):
        """Passing store_neighbors=True generates image and centroid neighbor dicts."""
        staged_dataset = MagicMock()
        staged_dataset.get_filenames.return_value = ["one.jpg", "two.jpg"]
        mock_load.return_value = staged_dataset
        mock_exists.side_effect = lambda path: path.endswith("stored_dataset.pt")
        args = SimpleNamespace(
            update_embeddings=False,
            annotations_version="annotations-v1",
            use_psl=False,
            use_multimodal=False,
            stacked=False,
            store_neighbors=True,
        )

        embeddings._build_generation_artifacts(  # pylint: disable=protected-access
            "trained-model",
            "embedding-set",
            args,
            "/publisher/staging",
            "/annotations/csv",
            "baseline",
        )

        self.assertEqual(
            mock_store_neighbors.call_args_list,
            [
                call(model_dir="/publisher/staging", centroid=False),
                call(model_dir="/publisher/staging", centroid=True),
            ],
        )

    @patch("src.retrieval.embeddings._stored_dataset_image_count",
           return_value=2)
    @patch("src.retrieval.embeddings.generate_and_store_suggestions")
    @patch("src.retrieval.neighbors.store_neighbors")
    @patch("src.retrieval.embeddings.store_centroid_embeddings")
    @patch("src.retrieval.embeddings.update_embeddings", return_value=None)
    def test_incremental_build_passes_validated_csv_to_existing_updater(
            self, mock_update, _mock_store_centroids, _mock_store_neighbors,
            _mock_suggestions, _mock_count):
        """The shared build body forwards its resolved immutable CSV input."""
        args = SimpleNamespace(
            update_embeddings=True,
            use_psl=False,
            use_multimodal=False,
            stacked=False,
        )

        count = embeddings._build_generation_artifacts(  # pylint: disable=protected-access
            "trained-model",
            "embedding-set",
            args,
            "/publisher/staging",
            "/snapshots/validated",
            "baseline",
        )

        self.assertEqual(count, 2)
        mock_update.assert_called_once_with(
            "trained-model",
            "/publisher/staging",
            csv_dir="/snapshots/validated",
        )


if __name__ == "__main__":
    unittest.main()
