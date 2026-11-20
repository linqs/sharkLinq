"""Tests for src.learning.evaluator.evaluators covering embedding computation and hits_at_k metrics."""
import unittest

import torch

import src.learning.evaluator.evaluators as ev


# ---------------------------------------------------------------------------
# Stub datasets
# ---------------------------------------------------------------------------

class _ImageDataset(torch.utils.data.Dataset):
    """``__getitem__`` returns ``(embedding, label)`` — matches IndividualDataset shape."""

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]

    def get_labels(self):
        return self.labels


class _FusionDataset(torch.utils.data.Dataset):
    """``__getitem__`` returns ``(image_embedding, metadata, label)`` — matches FusionDataset shape."""

    def __init__(self, image_embeddings: torch.Tensor, metadata: torch.Tensor,
                 labels: torch.Tensor):
        self.image_embeddings = image_embeddings
        self.metadata = metadata
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.image_embeddings[i], self.metadata[i], self.labels[i]

    def get_labels(self):
        return self.labels


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class _IdentityModel(torch.nn.Module):
    """Returns the input unchanged (embeddings are already pre-computed)."""
    def forward(self, x):
        return x


class _FusionConcatModel(torch.nn.Module):
    """Concatenates the two inputs along dim=1; for fusion smoke testing."""
    def forward(self, image_emb, metadata):
        return torch.cat([image_emb, metadata], dim=1)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _three_cluster_corpus(samples_per_class: int = 2):
    """
    Build deterministic 3-class clusters along the first 3 axes of an
    ``embed_dim``-dim space. Samples of the same class are identical, so
    every same-class neighbor sits at distance 0 and other classes at
    distance sqrt(2) — KNN ordering is unambiguous.
    """
    embed_dim = 8
    centers = torch.eye(3, embed_dim)  # [[1,0,0,...], [0,1,0,...], [0,0,1,...]]

    embeddings = []
    labels = []
    for class_idx in range(3):
        for _ in range(samples_per_class):
            embeddings.append(centers[class_idx])
            labels.append(class_idx)
    return torch.stack(embeddings), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# compute_embeddings
# ---------------------------------------------------------------------------

class TestComputeEmbeddings(unittest.TestCase):

    def test_default_embed_fn_image_only(self):
        """Without an ``embed_fn``, batches are unpacked as ``(images, labels)``."""
        embeddings, labels = _three_cluster_corpus()
        ds = _ImageDataset(embeddings, labels)
        out_emb, out_lbl = ev.compute_embeddings(_IdentityModel(), ds, torch.device("cpu"))
        # Each row is L2-normalized; for orthogonal unit-axis inputs that's a no-op.
        self.assertEqual(out_emb.shape, (6, 8))
        # Labels round-tripped via dataset.get_labels().
        self.assertEqual(list(out_lbl), [0, 0, 1, 1, 2, 2])
        # Norms are unit (within float tolerance).
        norms = (out_emb ** 2).sum(axis=1) ** 0.5
        for n in norms:
            self.assertAlmostEqual(float(n), 1.0, places=5)

    def test_custom_embed_fn_fusion(self):
        """A fusion-shaped ``embed_fn`` that ignores metadata and returns image emb."""
        image_embeddings, labels = _three_cluster_corpus()
        metadata = torch.zeros(len(labels), 4)
        ds = _FusionDataset(image_embeddings, metadata, labels)

        def embed_fn(model, batch, device):
            img, _, _ = batch
            return img.to(device)

        out_emb, _ = ev.compute_embeddings(
            _IdentityModel(), ds, torch.device("cpu"), embed_fn=embed_fn,
        )
        self.assertEqual(out_emb.shape, (6, 8))

    def test_custom_embed_fn_called_with_full_batch(self):
        """Verify ``embed_fn`` receives the model, the raw batch, and the device."""
        image_embeddings, labels = _three_cluster_corpus()
        metadata = torch.randn(len(labels), 4)
        ds = _FusionDataset(image_embeddings, metadata, labels)
        device = torch.device("cpu")

        captured = {}

        def embed_fn(model, batch, dev):
            captured["batch_len"] = len(batch)
            captured["device"] = dev
            return batch[0].to(dev)

        ev.compute_embeddings(_IdentityModel(), ds, device, embed_fn=embed_fn)
        self.assertEqual(captured["batch_len"], 3)
        self.assertEqual(captured["device"], device)


# ---------------------------------------------------------------------------
# hits_at_k
# ---------------------------------------------------------------------------

class TestHitsAtKImageOnly(unittest.TestCase):
    """
    Image-only path through the default ``embed_fn``.

    With two identical samples per class, every train query's nearest
    neighbor in the gallery (after skipping self at index 0) is its
    same-class twin — every hit@k is 1.0.
    """

    def setUp(self):
        self.embeddings, self.labels = _three_cluster_corpus(samples_per_class=2)
        self.train_ds = _ImageDataset(self.embeddings, self.labels)

    def test_returns_validation_hit_at_10_and_full_dict(self):
        valid_emb = torch.tensor([
            [1.0, 0, 0, 0, 0, 0, 0, 0],  # label 0 query.
            [0, 1.0, 0, 0, 0, 0, 0, 0],  # label 1 query.
        ])
        valid_lbl = torch.tensor([0, 1], dtype=torch.long)
        valid_ds = _ImageDataset(valid_emb, valid_lbl)

        metric, results = ev.hits_at_k(
            _IdentityModel(), self.train_ds, valid_ds, torch.device("cpu")
        )
        self.assertAlmostEqual(metric, 1.0)
        for k in (1, 5, 10, 25, 50, 100):
            self.assertIn(f"average_train_hit_at_{k}", results)
            self.assertIn(f"average_valid_hit_at_{k}", results)

    def test_perfect_clusters_give_hit_at_1_one(self):
        valid_emb = torch.tensor([
            [1.0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1.0, 0, 0, 0, 0, 0],
        ])
        valid_lbl = torch.tensor([0, 1, 2], dtype=torch.long)
        valid_ds = _ImageDataset(valid_emb, valid_lbl)

        _, results = ev.hits_at_k(
            _IdentityModel(), self.train_ds, valid_ds, torch.device("cpu")
        )
        self.assertAlmostEqual(results["average_valid_hit_at_1"], 1.0)
        self.assertAlmostEqual(results["average_train_hit_at_1"], 1.0)

    def test_no_valid_dataset_returns_train_only(self):
        metric, results = ev.hits_at_k(
            _IdentityModel(), self.train_ds, None, torch.device("cpu")
        )
        # Validation metric is 0.0 when there's no valid set.
        self.assertEqual(metric, 0.0)
        self.assertAlmostEqual(results["average_train_hit_at_1"], 1.0)
        # Valid keys are still present, defaulted to 0.0.
        self.assertEqual(results["average_valid_hit_at_10"], 0.0)

    def test_empty_valid_dataset_returns_train_only(self):
        valid_ds = _ImageDataset(
            torch.empty(0, 8), torch.empty(0, dtype=torch.long)
        )
        metric, results = ev.hits_at_k(
            _IdentityModel(), self.train_ds, valid_ds, torch.device("cpu")
        )
        self.assertEqual(metric, 0.0)
        self.assertAlmostEqual(results["average_train_hit_at_1"], 1.0)


class TestHitsAtKFusionPath(unittest.TestCase):
    """
    Fusion-shaped datasets via a custom ``embed_fn`` — exercises the same
    code path as ``multimodal_evaluator.multimodal_hits_at_k``.
    """

    def _fusion_embed(self, model, batch, device):
        image_emb, metadata, _ = batch
        return model(image_emb.to(device), metadata.to(device))

    def test_fusion_path_returns_same_shape(self):
        image_embeddings, labels = _three_cluster_corpus(samples_per_class=2)
        # 4-dim metadata that doesn't contribute (zeros) so the cluster
        # structure of image_embeddings dominates.
        metadata = torch.zeros(len(labels), 4)
        train_ds = _FusionDataset(image_embeddings, metadata, labels)

        valid_image = torch.tensor([
            [1.0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0, 0, 0, 0],
        ])
        valid_metadata = torch.zeros(2, 4)
        valid_lbl = torch.tensor([0, 1], dtype=torch.long)
        valid_ds = _FusionDataset(valid_image, valid_metadata, valid_lbl)

        metric, results = ev.hits_at_k(
            _FusionConcatModel(), train_ds, valid_ds, torch.device("cpu"),
            embed_fn=self._fusion_embed,
        )
        # Image emb is 8-dim and metadata 4-dim -> fused 12-dim. After L2
        # normalization the metadata-zero dims don't affect ordering, so
        # hit@1 must still be perfect.
        self.assertAlmostEqual(metric, 1.0)
        for k in (1, 5, 10, 25, 50, 100):
            self.assertIn(f"average_train_hit_at_{k}", results)
            self.assertIn(f"average_valid_hit_at_{k}", results)


class TestHitsAtKEdgeCases(unittest.TestCase):

    def test_singleton_classes_are_skipped(self):
        """
        Train samples whose label only appears once contribute nothing to
        average_train_hit_at_k — the gallery has no possible hit for them.
        """
        # 2 of label 0, 1 of label 1 (singleton), 2 of label 2.
        embeddings = torch.tensor([
            [1.0, 0, 0, 0, 0, 0, 0, 0],
            [1.0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0, 0, 0, 0],   # singleton.
            [0, 0, 1.0, 0, 0, 0, 0, 0],
            [0, 0, 1.0, 0, 0, 0, 0, 0],
        ])
        labels = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
        train_ds = _ImageDataset(embeddings, labels)
        # No valid set — focus on train metric semantics.
        _, results = ev.hits_at_k(
            _IdentityModel(), train_ds, None, torch.device("cpu")
        )
        # 4 contributing train samples (the 2 each of class 0 and 2),
        # all with hit@1 = 1.0 against their own twin in the gallery.
        self.assertAlmostEqual(results["average_train_hit_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
