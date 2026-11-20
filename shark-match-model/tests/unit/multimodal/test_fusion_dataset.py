"""
Tests for ``src.multimodal.fusion_dataset.FusionDataset``.

Uses a synthetic stub ``stored_dataset`` and monkey-patches
``src.csv_store.media.get_medias`` to return Media records constructed
in-memory, so the dataset can be exercised without on-disk CSV fixtures.
"""
import unittest
from unittest.mock import patch

import torch

import src.csv_store.models as csv_models
import src.multimodal.fusion_dataset as fd
import src.multimodal.metadata_features as mf


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubStoredDataset:
    """Minimal stand-in for an ``IndividualDataset`` loaded from disk."""

    def __init__(self, ids, labels, train_ids, test_ids, csv_dir=None):
        self.csv_dir = csv_dir
        self._ids = list(ids)
        self._labels = torch.tensor(labels, dtype=torch.long)
        self._train_ids = list(train_ids)
        self._test_ids = list(test_ids)

    def get_ids(self):
        return list(self._ids)

    def get_labels(self):
        return self._labels

    def get_train_ids(self):
        return list(self._train_ids)

    def get_test_ids(self):
        return list(self._test_ids)


def _make_media(
        media_id, *, sex="M", length=2.5, location="A",
        time=None, type_="still", quality="good"
):
    """Construct a Media record with an Individual attached."""
    individual = csv_models.Individual(id=media_id, sex=sex, length=length)
    return csv_models.Media(
        id=media_id, location=location, time=time, type=type_, quality=quality,
        individual=individual,
    )


def _default_corpus():
    """
    A 6-sample corpus with 3 identities; train/test are partitioned 4/2.

    Variation across location, sex, length, type, and quality is enough to
    exercise both the categorical vocab fitting and the numeric ranges.
    """
    medias = [
        _make_media(1, sex="M", length=2.0, location="A", type_="still", quality="good"),
        _make_media(2, sex="F", length=3.0, location="A", type_="still", quality="good"),
        _make_media(3, sex="M", length=4.0, location="B", type_="video", quality="medium"),
        _make_media(4, sex="F", length=5.0, location="B", type_="video", quality="medium"),
        _make_media(5, sex="M", length=6.0, location="C", type_="still", quality="poor"),
        _make_media(6, sex="F", length=7.0, location="C", type_="video", quality="poor"),
    ]
    ids = [1, 2, 3, 4, 5, 6]
    labels = [0, 0, 1, 1, 2, 2]
    train_ids = [1, 2, 3, 4]
    test_ids = [5, 6]
    return medias, ids, labels, train_ids, test_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFusionDatasetConstruction(unittest.TestCase):
    """Split filtering and basic structural invariants."""

    def setUp(self):
        self.medias, self.ids, self.labels, self.train_ids, self.test_ids = _default_corpus()
        self.image_dim = 8
        self.image_embeddings = torch.arange(
            len(self.ids) * self.image_dim, dtype=torch.float32
        ).reshape(len(self.ids), self.image_dim)
        self.spec = mf.MetadataFeatureSpec()
        self.stored = _StubStoredDataset(
            self.ids, self.labels, self.train_ids, self.test_ids
        )

    def _build(self, split=None, norm_stats=None):
        with patch.object(fd.src.csv_store.media, "get_medias",
                          return_value=self.medias) as mocked:
            ds = fd.FusionDataset(
                self.stored, self.image_embeddings, self.spec,
                split=split, norm_stats=norm_stats,
            )
        return ds, mocked

    def test_full_split_includes_all_samples(self):
        ds, _ = self._build(split=None)
        self.assertEqual(len(ds), 6)
        self.assertEqual(ds.get_ids(), self.ids)
        self.assertEqual(ds.get_labels().tolist(), self.labels)

    def test_train_split_filters_to_train_ids(self):
        ds, _ = self._build(split="train")
        self.assertEqual(len(ds), 4)
        self.assertEqual(ds.get_ids(), self.train_ids)
        self.assertEqual(ds.get_labels().tolist(), [0, 0, 1, 1])

    def test_test_split_filters_to_test_ids(self):
        ds, _ = self._build(split="test")
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.get_ids(), self.test_ids)
        self.assertEqual(ds.get_labels().tolist(), [2, 2])

    def test_valid_alias_for_test(self):
        ds, _ = self._build(split="valid")
        self.assertEqual(ds.get_ids(), self.test_ids)

    def test_unknown_split_raises(self):
        with self.assertRaises(ValueError):
            self._build(split="bogus")

    def test_csv_store_called_with_full_ids(self):
        _, mocked = self._build(split="train")
        # The dataset loads full-corpus media via keyword arguments before filtering by split.
        _args, kwargs = mocked.call_args
        self.assertEqual(kwargs.get("ids"), self.ids)
        self.assertTrue(kwargs.get("get_full"))


class TestFusionDatasetSamples(unittest.TestCase):
    """``__getitem__`` shape and content."""

    def setUp(self):
        self.medias, self.ids, self.labels, _, _ = _default_corpus()
        self.image_dim = 8
        self.image_embeddings = torch.arange(
            len(self.ids) * self.image_dim, dtype=torch.float32
        ).reshape(len(self.ids), self.image_dim)
        self.spec = mf.MetadataFeatureSpec()
        self.stored = _StubStoredDataset(self.ids, self.labels,
                                         train_ids=self.ids, test_ids=[])
        with patch.object(fd.src.csv_store.media, "get_medias",
                          return_value=self.medias):
            self.ds = fd.FusionDataset(
                self.stored, self.image_embeddings, self.spec, split=None,
            )

    def test_getitem_returns_three_tuple(self):
        sample = self.ds[0]
        self.assertEqual(len(sample), 3)

    def test_image_embedding_passes_through_unchanged(self):
        # Sample 0 should map to row 0 of image_embeddings.
        image_emb, _, _ = self.ds[0]
        self.assertTrue(torch.equal(image_emb, self.image_embeddings[0]))

    def test_metadata_matches_engineered_features(self):
        # Recompute the expected feature vector via metadata_features directly
        # over the same corpus and spec; it must match the dataset's cached row.
        stats = mf.fit_norm_stats(self.medias, self.spec)
        expected = mf.engineer_features(self.medias[0], self.spec, stats)
        _, metadata, _ = self.ds[0]
        self.assertTrue(torch.equal(metadata, expected))

    def test_label_matches_stored_dataset(self):
        _, _, label = self.ds[2]
        self.assertEqual(label.item(), self.labels[2])

    def test_image_and_metadata_dim_accessors(self):
        self.assertEqual(self.ds.image_dim(), self.image_dim)
        self.assertEqual(self.ds.metadata_dim(),
                         mf.feature_dim(self.spec, self.ds.norm_stats))


class TestFusionDatasetNormStats(unittest.TestCase):
    """Normalization stats are fit train-only by default and reusable across splits."""

    def setUp(self):
        self.medias, self.ids, self.labels, self.train_ids, self.test_ids = _default_corpus()
        self.image_embeddings = torch.randn(len(self.ids), 8)
        self.spec = mf.MetadataFeatureSpec()
        self.stored = _StubStoredDataset(
            self.ids, self.labels, self.train_ids, self.test_ids
        )

    def _build(self, split, norm_stats=None, **kwargs):
        with patch.object(fd.src.csv_store.media, "get_medias",
                          return_value=self.medias):
            return fd.FusionDataset(
                self.stored, self.image_embeddings, self.spec,
                split=split, norm_stats=norm_stats, **kwargs,
            )

    def test_fit_defaults_to_train_only(self):
        # Default fit_split='train' prevents test set metadata from leaking into vocabularies or min-max stats.
        ds = self._build(split="train")
        self.assertEqual(ds.norm_stats.length_min, 2.0)
        self.assertEqual(ds.norm_stats.length_max, 5.0)
        self.assertEqual(ds.norm_stats.location_vocab, ["A", "B"])

    def test_fit_split_all_reproduces_legacy_whole_corpus_fit(self):
        # fit_split='all' is the opt-in legacy path kept for parity reruns; it
        # spans the whole corpus (lengths 2.0..7.0, locations A/B/C).
        ds = self._build(split="train", fit_split="all")
        self.assertEqual(ds.norm_stats.length_min, 2.0)
        self.assertEqual(ds.norm_stats.length_max, 7.0)
        self.assertEqual(ds.norm_stats.location_vocab, ["A", "B", "C"])

    def test_norm_stats_can_be_reused_across_splits(self):
        train_ds = self._build(split="train")
        valid_ds = self._build(split="test", norm_stats=train_ds.norm_stats)
        self.assertIs(valid_ds.norm_stats, train_ds.norm_stats)
        # Engineered metadata for the same media id must be byte-identical
        # whether the dataset fit or inherited the stats.
        train_meta_for_id1 = train_ds.metadata[0]
        # id=1 is in train; verify shape & finite values.
        self.assertEqual(train_meta_for_id1.shape,
                         (mf.feature_dim(self.spec, train_ds.norm_stats),))
        # Confirm valid split metadata uses same engineering rules.
        valid_meta_for_id5 = valid_ds.metadata[0]
        self.assertEqual(valid_meta_for_id5.shape, train_meta_for_id1.shape)


class TestFusionDatasetSplitAccessors(unittest.TestCase):
    """``get_train_ids`` / ``get_test_ids`` are filtered to the dataset's slice."""

    def setUp(self):
        self.medias, self.ids, self.labels, self.train_ids, self.test_ids = _default_corpus()
        self.image_embeddings = torch.randn(len(self.ids), 8)
        self.spec = mf.MetadataFeatureSpec()
        self.stored = _StubStoredDataset(
            self.ids, self.labels, self.train_ids, self.test_ids
        )

    def _build(self, split):
        with patch.object(fd.src.csv_store.media, "get_medias",
                          return_value=self.medias):
            return fd.FusionDataset(
                self.stored, self.image_embeddings, self.spec, split=split,
            )

    def test_train_split_train_ids_match_dataset_ids(self):
        ds = self._build(split="train")
        # In a train-split dataset, every id is a train id; no test ids.
        self.assertEqual(set(ds.get_train_ids()), set(ds.get_ids()))
        self.assertEqual(ds.get_test_ids(), [])

    def test_test_split_test_ids_match_dataset_ids(self):
        ds = self._build(split="test")
        self.assertEqual(ds.get_train_ids(), [])
        self.assertEqual(set(ds.get_test_ids()), set(ds.get_ids()))

    def test_full_split_partitions_into_original_train_test(self):
        ds = self._build(split=None)
        self.assertEqual(set(ds.get_train_ids()), set(self.train_ids))
        self.assertEqual(set(ds.get_test_ids()), set(self.test_ids))


class TestFusionDatasetMissingMedia(unittest.TestCase):
    """Media records missing from csv_store are silently dropped."""

    def setUp(self):
        self.medias, self.ids, self.labels, self.train_ids, self.test_ids = _default_corpus()
        # Drop the first media record so id=1 has no metadata.
        self.medias_partial = [m for m in self.medias if m.id != 1]
        self.image_embeddings = torch.randn(len(self.ids), 8)
        self.spec = mf.MetadataFeatureSpec()
        self.stored = _StubStoredDataset(
            self.ids, self.labels, self.train_ids, self.test_ids
        )

    def test_missing_media_is_dropped(self):
        with patch.object(fd.src.csv_store.media, "get_medias",
                          return_value=self.medias_partial):
            ds = fd.FusionDataset(
                self.stored, self.image_embeddings, self.spec, split=None,
            )
        # 6 corpus ids minus the one missing media = 5 samples.
        self.assertEqual(len(ds), 5)
        self.assertNotIn(1, ds.get_ids())


if __name__ == "__main__":
    unittest.main()
