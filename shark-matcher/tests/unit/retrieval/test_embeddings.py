"""
Tests for embeddings generation process.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from src.core import constants
import src.retrieval.rerank
from src.core import utils
from src.retrieval import embeddings
from src.retrieval import neighbors as neighbors_module


class _StubDataset:
    """Pickle-friendly stand-in for IndividualDataset for torch.save/torch.load tests."""
    def __init__(self, filenames, labels, ids, train_ids=None, test_ids=None):
        self.filenames = list(filenames)
        self.labels = torch.tensor(list(labels))
        self.ids = list(ids)
        self.train_ids = list(train_ids) if train_ids is not None else []
        self.test_ids = list(test_ids) if test_ids is not None else []

    def get_filenames(self):
        return self.filenames

    def get_labels(self):
        return self.labels

    def get_ids(self):
        return self.ids

    def get_train_ids(self):
        return self.train_ids

    def get_test_ids(self):
        return self.test_ids

    def __len__(self):
        return len(self.filenames)


class TestEmbeddings(unittest.TestCase):
    """
    Test suite for embeddings generation and update logic.
    """

    # update_embeddings operates directly on model_dir while publication safety is covered by the generation contract tests.

    @patch('src.core.utils.get_model_metadata')
    @patch('src.cli.convert_data.prepare_annotations')
    @patch('src.retrieval.embeddings.store_image_embeddings')
    @patch('os.path.exists')
    def test_update_embeddings_full_generation(
            self, mock_exists, mock_store, mock_prepare, mock_metadata):
        """
        Test that `update_embeddings` resorts to full generation when stored dataset is missing.
        """
        mock_metadata.return_value = (True, "path/to/config.json", (100, 100), None)
        # Neither stored_dataset.pt nor stored_embeddings.pt exists.
        mock_exists.side_effect = lambda path: False

        model_dir = os.path.join(constants.EMBEDDINGS_DIR, "test_model")
        result = embeddings.update_embeddings(
            "test_model", model_dir, csv_dir="validated_csv_dir")

        self.assertIsNone(result)
        prepare_args = mock_prepare.call_args.kwargs
        self.assertEqual(prepare_args["input_dir"], "validated_csv_dir")
        self.assertNotEqual(
            prepare_args["output_directory"], "validated_csv_dir")
        # store_embeddings(centroid=...) was split; the image half now also
        # pins the source CSV directory on the generated dataset.
        mock_store.assert_called_once_with(
            model_name="test_model",
            model_dir=model_dir,
            annotations_version=prepare_args["output_directory"],
            csv_dir="validated_csv_dir",
        )

    @patch('src.core.utils.get_model_metadata',
           return_value=(True, "path/to/config.json", (100, 100), None))
    @patch('src.retrieval.embeddings.store_image_embeddings')
    def test_update_embeddings_keeps_csv_input_read_only(
            self, mock_store, _mock_metadata):
        """Annotation conversion writes only to temporary matcher storage."""
        with tempfile.TemporaryDirectory() as csv_dir, \
                tempfile.TemporaryDirectory() as model_dir:
            individuals_path = os.path.join(csv_dir, "individuals.csv")
            media_path = os.path.join(csv_dir, "media.csv")
            with open(individuals_path, 'w', encoding='utf-8') as file:
                file.write("individual input\n")
            with open(media_path, 'w', encoding='utf-8') as file:
                file.write("media input\n")

            annotation_dirs = []

            def prepare_annotations(**kwargs):
                """Stand in for conversion while recording its write target."""
                self.assertFalse(kwargs["train_test"])
                self.assertEqual(kwargs["input_dir"], csv_dir)
                output_directory = kwargs["output_directory"]
                self.assertNotEqual(output_directory, csv_dir)
                annotation_dirs.append(output_directory)
                with open(
                        os.path.join(
                            output_directory, constants.FULL_ANNOTATIONS),
                        'w', encoding='utf-8') as file:
                    file.write("derived annotations\n")

            with patch(
                    'src.cli.convert_data.prepare_annotations',
                    side_effect=prepare_annotations):
                result = embeddings.update_embeddings(
                    "test_model", model_dir, csv_dir=csv_dir)

            self.assertIsNone(result)
            with open(individuals_path, encoding='utf-8') as file:
                self.assertEqual(file.read(), "individual input\n")
            with open(media_path, encoding='utf-8') as file:
                self.assertEqual(file.read(), "media input\n")
            self.assertFalse(os.path.exists(annotation_dirs[0]))
            mock_store.assert_called_once_with(
                model_name="test_model",
                model_dir=model_dir,
                annotations_version=annotation_dirs[0],
                csv_dir=csv_dir,
            )

    @patch('src.core.utils.get_model_metadata')
    @patch('src.cli.convert_data.prepare_annotations')
    @patch('src.individual_dataset.IndividualDataset.create_and_store')
    @patch('src.core.utils.load_embedding_model')
    @patch('src.retrieval.embeddings._compute_embeddings')
    @patch('torch.save')
    @patch('torch.load')
    @patch('os.path.exists')
    def test_update_embeddings_with_new_images(
            self, mock_exists, mock_load, mock_save, mock_compute,
            _mock_load_model, mock_create, mock_prepare, mock_metadata):
        """
        Test `update_embeddings` when new images are added to the dataset.
        """
        mock_metadata.return_value = (True, "path/to/config.json", (100, 100), None)
        mock_exists.side_effect = lambda path: True

        mock_old_dataset = MagicMock()
        mock_old_dataset.get_filenames.return_value = ["img1.jpg", "img2.jpg"]
        mock_old_dataset.get_labels.return_value = torch.tensor([5, 5])
        mock_new_dataset = MagicMock()
        mock_new_dataset.get_filenames.return_value = ["img1.jpg", "img2.jpg", "img3.jpg"]
        mock_new_dataset.get_labels.return_value = torch.tensor([5, 5, 7])
        mock_new_dataset.__len__.return_value = 3

        mock_old_embeddings = torch.randn(2, 512)
        mock_new_embeddings_subset = torch.randn(1, 512)

        mock_load.side_effect = [mock_old_dataset, mock_old_embeddings, mock_new_dataset]
        mock_compute.return_value = mock_new_embeddings_subset

        result = embeddings.update_embeddings(
            "test_model",
            os.path.join(constants.EMBEDDINGS_DIR, "test_model"),
            csv_dir="validated_csv_dir",
        )

        self.assertEqual(result.new_filenames, ["img3.jpg"])
        # Only label 7 (the added image) is affected. affected_labels is a flat
        # set now that the per-focus dict is gone.
        self.assertEqual(result.affected_labels, {7})
        mock_compute.assert_called_once()
        prepare_args = mock_prepare.call_args.kwargs
        self.assertEqual(prepare_args["input_dir"], "validated_csv_dir")
        mock_create.assert_called_once_with(
            "test_model",
            annotations_version=prepare_args["output_directory"],
            output_image_size=(100, 100),
            normalization=None,
            output_dir=os.path.join(
                constants.EMBEDDINGS_DIR, "test_model"),
            csv_dir="validated_csv_dir",
        )
        # The reassembled embeddings are written straight to model_dir.
        mock_save.assert_called_once()

        saved_tensor = mock_save.call_args[0][0]
        self.assertEqual(saved_tensor.shape[0], 3)

    @patch('src.core.utils.get_model_metadata')
    @patch('src.cli.convert_data.prepare_annotations')
    @patch('src.individual_dataset.IndividualDataset.create_and_store')
    @patch('src.retrieval.embeddings._compute_embeddings')
    @patch('torch.save')
    @patch('torch.load')
    @patch('os.path.exists')
    def test_update_embeddings_with_deletions(
            self, mock_exists, mock_load, mock_save, mock_compute,
            _mock_create, _mock_prepare, mock_metadata):
        """
        Test `update_embeddings` when images are removed or reordered but no new images are added.
        """
        mock_metadata.return_value = (True, "path/to/config.json", (100, 100), None)
        mock_exists.side_effect = lambda path: True

        mock_old_dataset = MagicMock()
        mock_old_dataset.get_filenames.return_value = ["img1.jpg", "img2.jpg", "img3.jpg"]
        mock_old_dataset.get_labels.return_value = torch.tensor([1, 2, 3])
        mock_new_dataset = MagicMock()
        mock_new_dataset.get_filenames.return_value = ["img3.jpg", "img1.jpg"]
        mock_new_dataset.get_labels.return_value = torch.tensor([3, 1])
        mock_new_dataset.__len__.return_value = 2

        emb1 = torch.ones(1, 512)
        emb2 = torch.ones(1, 512) * 2
        emb3 = torch.ones(1, 512) * 3
        mock_old_embeddings = torch.cat([emb1, emb2, emb3])

        mock_load.side_effect = [mock_old_dataset, mock_old_embeddings, mock_new_dataset]

        result = embeddings.update_embeddings(
            "test_model",
            os.path.join(constants.EMBEDDINGS_DIR, "test_model"),
            csv_dir="validated_csv_dir",
        )

        self.assertEqual(result.new_filenames, [])
        # img2.jpg (label 2) was deleted; only label 2 is affected.
        self.assertEqual(result.affected_labels, {2})
        mock_compute.assert_not_called()
        mock_save.assert_called_once()

        saved_tensor = mock_save.call_args[0][0]
        self.assertEqual(saved_tensor.shape[0], 2)
        self.assertTrue(torch.all(saved_tensor[0] == emb3))
        self.assertTrue(torch.all(saved_tensor[1] == emb1))

    # --- Failure recovery tests ---

    @patch('src.core.utils.get_model_metadata')
    @patch('src.cli.convert_data.prepare_annotations')
    @patch('src.individual_dataset.IndividualDataset.create_and_store')
    @patch('src.core.utils.load_embedding_model')
    @patch('torch.save')
    @patch('torch.load')
    @patch('os.path.exists')
    def test_update_embeddings_failure_recovery(
            self, mock_exists, mock_load, mock_save, mock_load_model,
            _mock_create, _mock_prepare, mock_metadata):
        """
        Test that a model-loading failure propagates without writing embeddings.

        The staging/rollback half of the original assertion moved to
        run_pipeline (see the note above); what `update_embeddings` still owes
        its caller is that it does not leave a half-written
        stored_embeddings.pt behind when it fails.
        """
        mock_metadata.return_value = (True, "path/to/config.json", (100, 100), None)
        mock_exists.side_effect = lambda path: True

        mock_old_dataset = MagicMock()
        mock_old_dataset.get_filenames.return_value = ["img1.jpg"]
        mock_new_dataset = MagicMock()
        mock_new_dataset.get_filenames.return_value = ["img1.jpg", "img2.jpg"]
        mock_new_dataset.__len__.return_value = 2
        mock_old_embeddings = torch.randn(1, 512)

        mock_load.side_effect = [mock_old_dataset, mock_old_embeddings, mock_new_dataset]

        # Simulate model loading failure.
        mock_load_model.side_effect = RuntimeError("Failed to load model weights.")

        with self.assertRaises(RuntimeError):
            embeddings.update_embeddings(
                "test_model",
                os.path.join(constants.EMBEDDINGS_DIR, "test_model"),
                csv_dir="validated_csv_dir",
            )

        # No embeddings should be saved.
        mock_save.assert_not_called()

    @patch('src.core.utils.get_model_metadata')
    @patch('src.cli.convert_data.prepare_annotations')
    @patch('src.retrieval.embeddings.store_image_embeddings')
    @patch('torch.save')
    @patch('os.path.exists')
    def test_update_embeddings_full_generation_failure_recovery(
            self, mock_exists, mock_save, mock_store, _mock_prepare,
            mock_metadata):
        """
        A failure during full generation propagates and writes nothing.
        """
        mock_metadata.return_value = (True, "path/to/config.json", (100, 100), None)
        # Neither stored_dataset.pt nor stored_embeddings.pt exists.
        mock_exists.side_effect = lambda path: False

        mock_store.side_effect = RuntimeError("GPU out of memory.")

        with self.assertRaises(RuntimeError):
            embeddings.update_embeddings(
                "test_model",
                os.path.join(constants.EMBEDDINGS_DIR, "test_model"),
                csv_dir="validated_csv_dir",
            )

        mock_save.assert_not_called()

    # --- Unit tests for helper functions ---

    def _make_dataset(self, filenames, labels, ids=None, train_ids=None, test_ids=None):
        """Build a MagicMock dataset that satisfies the IndividualDataset API used by
        `_compute_affected_labels`."""
        ds = MagicMock()
        ds.get_filenames.return_value = list(filenames)
        ds.get_labels.return_value = torch.tensor(list(labels))
        ds.get_ids.return_value = list(ids) if ids is not None else list(filenames)
        ds.get_train_ids.return_value = list(train_ids) if train_ids is not None else []
        ds.get_test_ids.return_value = list(test_ids) if test_ids is not None else []
        return ds

    # _compute_affected_labels identifies labels whose member images changed between datasets.

    def test_compute_affected_labels_no_changes(self):
        """Identical datasets produce an empty affected set."""
        old_ds = self._make_dataset(["a", "b", "c"], [1, 1, 2])
        new_ds = self._make_dataset(["a", "b", "c"], [1, 1, 2])
        affected = embeddings._compute_affected_labels(old_ds, new_ds)
        self.assertEqual(affected, set())

    def test_compute_affected_labels_addition(self):
        """Adding an image affects only its label."""
        old_ds = self._make_dataset(["a", "b"], [1, 1])
        new_ds = self._make_dataset(["a", "b", "c"], [1, 1, 2])
        affected = embeddings._compute_affected_labels(old_ds, new_ds)
        self.assertEqual(affected, {2})

    def test_compute_affected_labels_deletion(self):
        """Deleting an image affects only its label."""
        old_ds = self._make_dataset(["a", "b", "c"], [1, 1, 2])
        new_ds = self._make_dataset(["a", "b"], [1, 1])
        affected = embeddings._compute_affected_labels(old_ds, new_ds)
        self.assertEqual(affected, {2})

    def test_compute_affected_labels_relabel(self):
        """Reassigning an image to a new label affects both old and new labels."""
        old_ds = self._make_dataset(["a", "b"], [1, 2])
        new_ds = self._make_dataset(["a", "b"], [1, 3])
        affected = embeddings._compute_affected_labels(old_ds, new_ds)
        self.assertEqual(affected, {2, 3})

    def test_compute_affected_labels_reorder_no_changes(self):
        """Re-ordering filenames without changing the (filename, label) mapping affects nothing."""
        old_ds = self._make_dataset(["a", "b", "c"], [1, 2, 3])
        new_ds = self._make_dataset(["c", "a", "b"], [3, 1, 2])
        affected = embeddings._compute_affected_labels(old_ds, new_ds)
        self.assertEqual(affected, set())

    def _write_full_centroids(self, model_dir, embeddings_tensor, dataset):
        """Compute and persist full centroid embeddings to model_dir in sorted-label order."""
        emb_np = embeddings_tensor.numpy()
        labels_np = dataset.get_labels().numpy()
        centroid_embs, centroid_lbls = embeddings._compute_centroids(emb_np, labels_np)

        centroid_tensor = torch.tensor(centroid_embs)
        labels_tensor = torch.tensor(centroid_lbls, dtype=torch.long)
        torch.save(centroid_tensor, os.path.join(model_dir, "stored_centroid_embeddings.pt"))
        torch.save(labels_tensor, os.path.join(model_dir, "stored_centroid_labels.pt"))
        return centroid_tensor, labels_tensor

    @patch('src.core.utils.get_torch_device', return_value=torch.device('cpu'))
    def test_update_centroid_embeddings_parity_no_split(self, _mock_device):
        """Incremental centroid update produces a bit-identical result to a full
        recomputation when dataset changes include additions, deletions, and a relabel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.manual_seed(0)

            # Old dataset: 4 images across 2 labels.
            old_filenames = ["a", "b", "c", "d"]
            old_labels_list = [1, 1, 2, 2]
            old_ids = [10, 20, 30, 40]
            old_embs = torch.rand(4, 8, dtype=torch.float32)
            old_ds = _StubDataset(old_filenames, old_labels_list, old_ids)

            # Write old centroid files using the full-path logic.
            self._write_full_centroids(tmpdir, old_embs, old_ds)

            # New dataset: delete 'b', add 'e' with a new label, relabel 'c' from 2 -> 2 (unchanged).
            # Also add another new image to existing label 1 to exercise affected-but-existing.
            new_filenames = ["a", "c", "d", "e", "f"]
            new_labels_list = [1, 2, 2, 3, 1]
            new_ids = [10, 30, 40, 50, 60]
            # Carry over embeddings for a, c, d; generate fresh for e, f.
            new_embs = torch.cat([
                old_embs[0:1],                            # a
                old_embs[2:4],                            # c, d
                torch.rand(2, 8, dtype=torch.float32),    # e, f
            ])
            new_ds = _StubDataset(new_filenames, new_labels_list, new_ids)

            # Write current embeddings + dataset to tmpdir.
            torch.save(new_embs, os.path.join(tmpdir, "stored_embeddings.pt"))
            torch.save(new_ds, os.path.join(tmpdir, "stored_dataset.pt"))

            # Build the UpdateResult the update pipeline would have produced.
            # UpdateResult lost its old_focus_labels field along with the
            # focus concept; it is now (new_filenames, affected_labels).
            affected = embeddings._compute_affected_labels(old_ds, new_ds)
            update_result = embeddings.UpdateResult(
                new_filenames=["e", "f"], affected_labels=affected)

            # Run the incremental centroid update.
            embeddings.update_centroid_embeddings(tmpdir, update_result)

            # Independently compute what a full recomputation would have produced.
            expected_centroids_list, expected_labels_list = embeddings._compute_centroids(
                new_embs.numpy(), new_ds.get_labels().numpy())
            expected_centroids = torch.tensor(expected_centroids_list)
            expected_labels = torch.tensor(expected_labels_list, dtype=torch.long)

            # Load what the incremental path wrote and compare.
            actual_centroids = torch.load(
                os.path.join(tmpdir, "stored_centroid_embeddings.pt"), weights_only=False).cpu()
            actual_labels = torch.load(
                os.path.join(tmpdir, "stored_centroid_labels.pt"), weights_only=False).cpu()

            self.assertTrue(torch.equal(actual_labels, expected_labels),
                            f"Labels differ: {actual_labels} vs {expected_labels}")
            self.assertTrue(torch.equal(actual_centroids, expected_centroids),
                            "Centroids are not bit-identical to full recomputation.")
            # Verify affected set includes updated labels 1 and 3 while leaving unchanged label 2 untouched.
            self.assertIn(1, affected)
            self.assertIn(3, affected)
            self.assertNotIn(2, affected)

    # Centroid updates operate in a single pass with one centroid per individual.

    @patch('src.core.utils.get_torch_device', return_value=torch.device('cpu'))
    def test_update_centroid_embeddings_no_changes(self, _mock_device):
        """When nothing changed, every centroid must be reused verbatim (bit-identical
        to the previously stored file)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.manual_seed(2)
            filenames = ["a", "b", "c"]
            labels_list = [1, 2, 2]
            ids = [10, 20, 30]
            embs = torch.rand(3, 4, dtype=torch.float32)
            ds = _StubDataset(filenames, labels_list, ids)

            prev_centroids, prev_labels = self._write_full_centroids(tmpdir, embs, ds)
            torch.save(embs, os.path.join(tmpdir, "stored_embeddings.pt"))
            torch.save(ds, os.path.join(tmpdir, "stored_dataset.pt"))

            affected = embeddings._compute_affected_labels(ds, ds)
            update_result = embeddings.UpdateResult(
                new_filenames=[], affected_labels=affected)
            # Sanity: nothing affected.
            self.assertEqual(affected, set())

            embeddings.update_centroid_embeddings(tmpdir, update_result)

            actual_centroids = torch.load(
                os.path.join(tmpdir, "stored_centroid_embeddings.pt"), weights_only=False).cpu()
            actual_labels = torch.load(
                os.path.join(tmpdir, "stored_centroid_labels.pt"), weights_only=False).cpu()
            self.assertTrue(torch.equal(actual_labels, prev_labels))
            self.assertTrue(torch.equal(actual_centroids, prev_centroids))

    def _write_centroid_neighbors_via_cdist(self, model_dir, dataset):
        """Compute centroid neighbors via torch.cdist and write them to model_dir."""
        centroid_tensor = torch.load(
            os.path.join(model_dir, "stored_centroid_embeddings.pt"),
            weights_only=False).cpu()
        centroid_labels_np = torch.load(
            os.path.join(model_dir, "stored_centroid_labels.pt"),
            weights_only=False).cpu().numpy()

        ids = dataset.get_ids()
        filenames = dataset.get_filenames()
        labels_np = dataset.get_labels().cpu().detach().numpy()

        centroid_label_order = [int(L) for L in centroid_labels_np]
        n_total = centroid_tensor.shape[0]
        self.assertEqual(n_total, len(set(int(L) for L in labels_np)))

        images_per_label = {}
        for i in range(len(ids)):
            images_per_label.setdefault(int(labels_np[i]), []).append(i)

        if n_total == 0:
            torch.save({}, os.path.join(model_dir, "stored_centroid_neighbors.pt"))
            return {}

        dist_matrix = src.retrieval.rerank._pairwise_l2(
            centroid_tensor.numpy()
        ).astype(np.float32)

        # For each centroid (i.e. each label), the neighbor list over all
        # centroids sorted by distance, expanded to per-image entries.
        expansion_per_label = {}
        for q_idx, q_label in enumerate(centroid_label_order):
            sorted_pairs = sorted(
                ((j, float(dist_matrix[q_idx][j])) for j in range(n_total)),
                key=lambda p: p[1])
            m_ids, m_names, m_labels, m_dists = [], [], [], []
            for neighbor_idx, dist in sorted_pairs:
                n_label = centroid_label_order[neighbor_idx]
                for img_idx in images_per_label.get(n_label, []):
                    m_ids.append(ids[img_idx])
                    m_names.append(filenames[img_idx])
                    m_labels.append(n_label)
                    m_dists.append(dist)
            expansion_per_label[q_label] = (m_ids, m_names, m_labels, m_dists)

        expanded = {}
        for i, img_id in enumerate(ids):
            exp = expansion_per_label.get(int(labels_np[i]))
            if exp is None:
                continue
            m_ids, m_names, m_labels, m_dists = exp
            fn = filenames[i]
            expanded[fn] = {
                "query": img_id,
                "query_name": fn,
                "matches": list(m_ids),
                "match_names": list(m_names),
                "match_distances": list(m_dists),
                "match_labels": list(m_labels),
            }

        torch.save(expanded, os.path.join(model_dir, "stored_centroid_neighbors.pt"))
        return expanded

    def _assert_centroid_neighbors_equal(self, actual, expected):
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        for fn, exp in expected.items():
            act = actual[fn]
            self.assertEqual(act["query"], exp["query"], f"query mismatch for {fn}")
            self.assertEqual(act["query_name"], exp["query_name"])
            self.assertEqual(act["matches"], exp["matches"], f"matches mismatch for {fn}")
            self.assertEqual(act["match_names"], exp["match_names"],
                             f"match_names mismatch for {fn}")
            self.assertEqual(act["match_labels"], exp["match_labels"],
                             f"match_labels mismatch for {fn}")
            self.assertEqual(len(act["match_distances"]), len(exp["match_distances"]))
            for a, e in zip(act["match_distances"], exp["match_distances"]):
                self.assertEqual(a, e, f"distance mismatch for {fn}: {a} vs {e}")

    @patch('src.core.utils.get_torch_device', return_value=torch.device('cpu'))
    def test_update_centroid_neighbors_parity_no_split(self, _mock_device):
        """Incremental centroid-neighbors update is bit-identical to a full
        recompute when additions, deletions, and unchanged labels are mixed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.manual_seed(10)

            old_filenames = ["a", "b", "c", "d"]
            old_labels_list = [1, 1, 2, 2]
            old_ids = [10, 20, 30, 40]
            old_embs = torch.rand(4, 8, dtype=torch.float32)
            old_ds = _StubDataset(old_filenames, old_labels_list, old_ids)

            # Seed old centroid embeddings + labels.
            self._write_full_centroids(tmpdir, old_embs, old_ds)
            # Write old dataset so the helper can compute neighbors from it.
            torch.save(old_ds, os.path.join(tmpdir, "stored_dataset.pt"))
            # Seed old centroid-neighbors dict using torch.cdist math.
            self._write_centroid_neighbors_via_cdist(tmpdir, old_ds)

            # New state: delete 'b', add 'e' (new label 3) and 'f' (existing label 1).
            # Label 2 unchanged in membership.
            new_filenames = ["a", "c", "d", "e", "f"]
            new_labels_list = [1, 2, 2, 3, 1]
            new_ids = [10, 30, 40, 50, 60]
            new_embs = torch.cat([
                old_embs[0:1],                            # a
                old_embs[2:4],                            # c, d
                torch.rand(2, 8, dtype=torch.float32),    # e, f
            ])
            new_ds = _StubDataset(new_filenames, new_labels_list, new_ids)

            # Install new image embeddings + dataset, and produce new centroid files
            # via the incremental centroid path (this is what the pipeline does).
            torch.save(new_embs, os.path.join(tmpdir, "stored_embeddings.pt"))
            torch.save(new_ds, os.path.join(tmpdir, "stored_dataset.pt"))
            affected = embeddings._compute_affected_labels(old_ds, new_ds)
            update_result = embeddings.UpdateResult(
                new_filenames=["e", "f"], affected_labels=affected)
            embeddings.update_centroid_embeddings(tmpdir, update_result)

            # Run the incremental centroid-neighbor update.
            neighbors_module.update_centroid_neighbors(
                tmpdir, update_result, transformed=False)
            actual = torch.load(
                os.path.join(tmpdir, "stored_centroid_neighbors.pt"), weights_only=False)

            # Expected: full-recompute via cdist on the updated centroid file.
            # Temporarily stash and recompute into a second location to avoid clobbering.
            with tempfile.TemporaryDirectory() as expected_dir:
                # Mirror the files needed by the helper.
                for f in ("stored_centroid_embeddings.pt", "stored_centroid_labels.pt",
                          "stored_dataset.pt"):
                    src_p = os.path.join(tmpdir, f)
                    dst_p = os.path.join(expected_dir, f)
                    torch.save(torch.load(src_p, weights_only=False), dst_p)
                expected = self._write_centroid_neighbors_via_cdist(
                    expected_dir, new_ds)

            self._assert_centroid_neighbors_equal(actual, expected)

    @patch('src.core.utils.get_torch_device', return_value=torch.device('cpu'))
    def test_update_centroid_neighbors_no_changes(self, _mock_device):
        """When nothing changed, the incremental path must emit the exact old
        expanded dict (every row reconstructed from the cache)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.manual_seed(12)
            filenames = ["a", "b", "c", "d"]
            labels_list = [1, 2, 2, 3]
            ids = [10, 20, 30, 40]
            embs = torch.rand(4, 5, dtype=torch.float32)
            ds = _StubDataset(filenames, labels_list, ids)

            self._write_full_centroids(tmpdir, embs, ds)
            torch.save(embs, os.path.join(tmpdir, "stored_embeddings.pt"))
            torch.save(ds, os.path.join(tmpdir, "stored_dataset.pt"))
            seeded = self._write_centroid_neighbors_via_cdist(tmpdir, ds)

            affected = embeddings._compute_affected_labels(ds, ds)
            update_result = embeddings.UpdateResult(
                new_filenames=[], affected_labels=affected)
            self.assertEqual(affected, set())

            neighbors_module.update_centroid_neighbors(
                tmpdir, update_result, transformed=False)
            actual = torch.load(
                os.path.join(tmpdir, "stored_centroid_neighbors.pt"), weights_only=False)
            self._assert_centroid_neighbors_equal(actual, seeded)

    def test_compute_centroids(self):
        """Test that _compute_centroids averages embeddings per label and L2-renormalizes onto the unit sphere."""
        emb = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [10.0, 10.0]
        ])
        labels = np.array([1, 1, 2])

        centroids, cent_labels = embeddings._compute_centroids(emb, labels)

        self.assertEqual(len(centroids), 2)
        self.assertEqual(len(cent_labels), 2)
        self.assertEqual(cent_labels, [1, 2])

        mean_1 = np.array([2.0, 3.0])
        self.assertTrue(np.allclose(centroids[0], mean_1 / np.linalg.norm(mean_1)))
        mean_2 = np.array([10.0, 10.0])
        self.assertTrue(np.allclose(centroids[1], mean_2 / np.linalg.norm(mean_2)))

        # Every centroid lands on the unit sphere.
        for centroid in centroids:
            self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=6)

    def test_compute_centroids_zero_vector_is_not_normalized(self):
        """A degenerate all-zero centroid is left alone rather than dividing by 0."""
        emb = np.array([[1.0, 1.0], [-1.0, -1.0]])
        labels = np.array([1, 1])

        centroids, cent_labels = embeddings._compute_centroids(emb, labels)

        self.assertEqual(cent_labels, [1])
        self.assertTrue(np.allclose(centroids[0], [0.0, 0.0]))

    @patch('os.path.exists')
    @patch('src.core.utils.load_json_file')
    def test__get_model_metadata_precanned_miew(self, mock_load_json, mock_exists):
        """
        Test `_get_model_metadata` with precanned model 'miew'.
        """
        mock_exists.return_value = False
        is_finetuned, config_path, img_size, norm = utils.get_model_metadata("miew")
        
        self.assertFalse(is_finetuned)
        self.assertEqual(img_size, (440, 440))
        self.assertEqual(norm["resize_mode"], "squash")
        mock_load_json.assert_not_called()

    @patch('os.path.exists')
    def test__get_model_metadata_precanned_mega(self, mock_exists):
        """
        Test `_get_model_metadata` with precanned model 'mega'.
        """
        mock_exists.return_value = False
        is_finetuned, config_path, img_size, norm = utils.get_model_metadata("mega")
        
        self.assertFalse(is_finetuned)
        self.assertEqual(img_size, (384, 384))
        self.assertEqual(norm["resize_mode"], "squash")

    @patch('os.path.exists')
    @patch('src.core.utils.load_json_file')
    def test__get_model_metadata_finetuned(self, mock_load_json, mock_exists):
        """
        Test `_get_model_metadata` with a fine-tuned model.
        """
        mock_exists.return_value = True
        mock_load_json.return_value = {
            "data": {
                "image_size": [100, 100],
                "normalization": {"mean": [0.5], "std": [0.5]}
            }
        }
        
        is_finetuned, config_path, img_size, norm = utils.get_model_metadata("lora_model")
        
        self.assertTrue(is_finetuned)
        self.assertEqual(img_size, (100, 100))
        self.assertEqual(norm["mean"], [0.5])

    def test_compute_embeddings_squeezing(self):
        """
        Test `_compute_embeddings` correctly handles squeezing and normalization.
        """
        mock_model = MagicMock()
        # Mock model to return [B, 1, D] output that needs squeezing.
        mock_model.return_value = torch.ones(2, 1, 512) 
        
        # Loader yields one batch of shape (2, channels, h, w).
        mock_loader = [torch.randn(2, 3, 100, 100)]
        
        result = embeddings._compute_embeddings(mock_model, mock_loader, torch.device('cpu'))
        
        # Check shape after squeezing [B, 1, D] -> [B, D].
        self.assertEqual(result.shape, (2, 512))
        
        # Check L2 normalization (sum of squares along dim=1 should be 1.0).
        norms = torch.norm(result, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones(2)))

    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoModel.from_config')
    @patch('transformers.AutoModel.from_pretrained')
    def test_load_embedding_model_miew(self, mock_from_pretrained, mock_from_config, mock_config):
        """
        Test `_load_embedding_model` initializes MiewID correctly.
        """
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_from_config.return_value = mock_model
        
        model = utils.load_embedding_model("miew", False, None, torch.device('cpu'))
        
        self.assertEqual(model, mock_model)
        mock_model.eval.assert_called_once()
        mock_config.assert_called_once()
        mock_from_pretrained.assert_called_once()

    @patch('timm.create_model')
    def test_load_embedding_model_mega(self, mock_create_model):
        """
        Test `_load_embedding_model` initializes MegaDescriptor correctly.
        """
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model
        
        model = utils.load_embedding_model("mega", False, None, torch.device('cpu'))
        
        self.assertEqual(model, mock_model)
        mock_model.eval.assert_called_once()
        mock_create_model.assert_called_once_with("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)

    @patch('src.core.utils.load_json_file')
    @patch('src.core.utils.load_model')
    @patch('torch.nn.Sequential')
    def test_load_embedding_model_finetuned(self, mock_seq, mock_load_model, mock_load_json):
        """
        Test `_load_embedding_model` initializes finetuned model correctly.
        """
        mock_load_json.return_value = {"type": "timm"}
        
        mock_head = MagicMock()
        mock_backbone = MagicMock()
        mock_load_model.return_value = (mock_head, mock_backbone)
        
        mock_seq_instance = MagicMock()
        mock_seq_instance.to.return_value = mock_seq_instance
        mock_seq.return_value = mock_seq_instance
        
        model = utils.load_embedding_model("lora_model", True, "config_path", torch.device('cpu'))
        
        mock_head.eval.assert_called_once()
        mock_backbone.eval.assert_called_once()
        mock_seq.assert_called_once_with(mock_backbone, mock_head)
        
        self.assertEqual(model, mock_seq_instance)

    # Storing embeddings is split into store_image_embeddings and store_centroid_embeddings, tested separately below.

    @patch('src.core.utils.get_model_metadata')
    @patch('src.individual_dataset.IndividualDataset.create_and_store')
    @patch('src.core.utils.load_embedding_model')
    @patch('src.retrieval.embeddings._compute_embeddings')
    @patch('torch.save')
    @patch('torch.load')
    @patch('os.makedirs')
    def test_store_image_embeddings(self, mock_makedirs, mock_load, mock_save,
                                    mock_compute_embeddings, mock_load_model,
                                    mock_create, mock_metadata):
        """`store_image_embeddings` builds the dataset, embeds it, and writes one file."""
        mock_metadata.return_value = (True, "config", (100, 100), None)

        mock_dataset = MagicMock()
        mock_dataset.get_ids.return_value = [1]
        mock_dataset.get_labels.return_value = torch.tensor([1])
        mock_load.return_value = mock_dataset

        mock_load_model.return_value = MagicMock()
        mock_compute_embeddings.return_value = torch.ones(1, 512)

        model_dir = os.path.join(constants.EMBEDDINGS_DIR, "test")
        embeddings.store_image_embeddings("test", model_dir)

        mock_create.assert_called_once()
        mock_compute_embeddings.assert_called_once()
        mock_save.assert_called_once()
        self.assertEqual(mock_save.call_args[0][1],
                         os.path.join(model_dir, "stored_embeddings.pt"))

    @patch('src.core.utils.get_torch_device', return_value=torch.device('cpu'))
    @patch('src.retrieval.embeddings._compute_centroids')
    @patch('torch.save')
    @patch('torch.load')
    @patch('os.makedirs')
    def test_store_centroid_embeddings(self, mock_makedirs, mock_load, mock_save,
                                       mock_compute_centroids, _mock_device):
        """`store_centroid_embeddings` writes both the centroids and their labels."""
        mock_dataset = MagicMock()
        mock_dataset.get_labels.return_value = torch.tensor([1, 2])

        mock_load.side_effect = [
            mock_dataset,        # stored_dataset.pt
            torch.ones(2, 512),  # stored_embeddings.pt
        ]
        mock_compute_centroids.return_value = ([[1.0] * 512, [2.0] * 512], [1, 2])

        model_dir = os.path.join(constants.EMBEDDINGS_DIR, "test")
        embeddings.store_centroid_embeddings(model_dir)

        mock_compute_centroids.assert_called_once()
        self.assertEqual(mock_save.call_count, 2)
        saved_paths = [c[0][1] for c in mock_save.call_args_list]
        self.assertEqual(saved_paths, [
            os.path.join(model_dir, "stored_centroid_embeddings.pt"),
            os.path.join(model_dir, "stored_centroid_labels.pt"),
        ])

    @patch('src.core.utils.get_torch_device', return_value=torch.device('cpu'))
    @patch('src.retrieval.embeddings._compute_centroids')
    @patch('torch.save')
    @patch('torch.load')
    @patch('os.makedirs')
    def test_store_centroid_embeddings_uses_shared_suffix(
            self, _mock_makedirs, mock_load, mock_save,
            mock_compute_centroids, _mock_device):
        """A suffix selects matching source and centroid artifact names."""
        mock_dataset = MagicMock()
        mock_dataset.get_labels.return_value = torch.tensor([1, 2])
        mock_load.side_effect = [mock_dataset, torch.ones(2, 512)]
        mock_compute_centroids.return_value = ([[1.0] * 512, [2.0] * 512], [1, 2])

        model_dir = os.path.join(constants.EMBEDDINGS_DIR, "test")
        embeddings.store_centroid_embeddings(model_dir, suffix="_multimodal")

        loaded_paths = [call.args[0] for call in mock_load.call_args_list]
        self.assertEqual(loaded_paths, [
            os.path.join(model_dir, "stored_dataset.pt"),
            os.path.join(model_dir, "stored_embeddings_multimodal.pt"),
        ])
        saved_paths = [call.args[1] for call in mock_save.call_args_list]
        self.assertEqual(saved_paths, [
            os.path.join(model_dir, "stored_centroid_embeddings_multimodal.pt"),
            os.path.join(model_dir, "stored_centroid_labels_multimodal.pt"),
        ])

class TestRequirePslRuntime(unittest.TestCase):
    """The PSL preflight stops early with install guidance instead of failing mid-generation."""

    def test_missing_pslpython_exits_with_guidance(self):
        with patch.dict("sys.modules", {"pslpython": None, "pslpython.runtime": None}):
            with self.assertRaises(SystemExit) as raised:
                embeddings._require_psl_runtime()
        self.assertIn("src/psl/README.md", str(raised.exception))
        self.assertIn("pslpython", str(raised.exception))

    def test_missing_java_exits_with_guidance(self):
        import subprocess
        import jpype
        fake_psl = {"pslpython": MagicMock(), "pslpython.runtime": MagicMock()}
        with patch.dict("sys.modules", fake_psl), patch.object(
                jpype, "getDefaultJVMPath",
                side_effect=subprocess.CalledProcessError(1, "java_home")):
            with self.assertRaises(SystemExit) as raised:
                embeddings._require_psl_runtime()
        self.assertIn("Java", str(raised.exception))

    def test_available_runtime_passes(self):
        import jpype
        fake_psl = {"pslpython": MagicMock(), "pslpython.runtime": MagicMock()}
        with patch.dict("sys.modules", fake_psl), patch.object(
                jpype, "getDefaultJVMPath", return_value="libjvm"):
            embeddings._require_psl_runtime()

    def test_main_checks_runtime_before_any_generation_work(self):
        argv = ["embeddings", "--model-name", "m", "--use-psl"]
        with patch("sys.argv", argv), patch.object(
                embeddings, "_require_psl_runtime", side_effect=SystemExit("no psl")), patch.object(
                embeddings.utils, "print_script_banner"), patch.object(
                embeddings.utils, "resolve_model_dirs") as resolve_dirs:
            with self.assertRaises(SystemExit):
                embeddings.main()
        resolve_dirs.assert_not_called()

    def test_plain_main_skips_the_runtime_check(self):
        argv = ["embeddings", "--model-name", "m"]
        with patch("sys.argv", argv), patch.object(
                embeddings, "_require_psl_runtime") as require_runtime, patch.object(
                embeddings.utils, "print_script_banner"), patch.object(
                embeddings.utils, "resolve_model_dirs", return_value=[]):
            embeddings.main()
        require_runtime.assert_not_called()


if __name__ == '__main__':
    unittest.main()
