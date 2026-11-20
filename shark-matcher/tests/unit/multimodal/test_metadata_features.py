"""
Tests for src.multimodal.metadata_features.

Verifies feature ordering, encoding logic, and normalization stats are
byte-for-byte compatible with the CLI behaviour of the previous
engineer_metadata_features.py script.
"""
import datetime
import math
import unittest

import torch

import src.csv_store.models as csv_models
import src.multimodal.metadata_features as mf


def _make_media(media_id=1, location=None, time=None, type_=None, quality=None,
                sex=None, length=None):
    """Build a csv_store Media record (with Individual attached) for tests."""
    individual = csv_models.Individual(id=media_id, sex=sex, length=length)
    return csv_models.Media(
        id=media_id, location=location, time=time, type=type_, quality=quality,
        individual=individual,
    )


def _unix(year, month=6, day=15):
    """Return a UNIX timestamp for a fixed local-time date."""
    return int(datetime.datetime(year, month, day, 12, 0, 0).timestamp())


class TestSoftEncoders(unittest.TestCase):
    """Verify the probabilistic encoding rules match the original CLI."""

    def test_soft_sex(self):
        self.assertEqual(mf._soft_sex("M"), [1.0, 0.0])
        self.assertEqual(mf._soft_sex("F"), [0.0, 1.0])
        self.assertEqual(mf._soft_sex("U"), [0.5, 0.5])
        self.assertEqual(mf._soft_sex(None), [0.0, 0.0])
        self.assertEqual(mf._soft_sex("other"), [0.0, 0.0])

    def test_soft_type(self):
        self.assertEqual(mf._soft_type("still"), [1.0, 0.0])
        self.assertEqual(mf._soft_type("video"), [0.0, 1.0])
        # Anything else (including None) -> uniform 0.5/0.5.
        self.assertEqual(mf._soft_type(None), [0.5, 0.5])
        self.assertEqual(mf._soft_type("other"), [0.5, 0.5])

    def test_soft_quality(self):
        self.assertEqual(mf._soft_quality("good"), [1.0, 0.0, 0.0])
        self.assertEqual(mf._soft_quality("medium"), [0.0, 1.0, 0.0])
        self.assertEqual(mf._soft_quality("poor"), [0.0, 0.0, 1.0])
        # Out-of-set -> uniform 1/3 per column.
        self.assertEqual(
            mf._soft_quality(None),
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
        )
        self.assertEqual(
            mf._soft_quality("unknown"),
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
        )


class TestNormStatsFitting(unittest.TestCase):
    """Verify vocabularies and min/max ranges are fit correctly."""

    def test_vocab_alphabetical_with_unknown_for_none(self):
        spec = mf.MetadataFeatureSpec(use_soft_encoding=False)
        medias = [
            _make_media(1, location="A", sex="M", quality="good", type_="still"),
            _make_media(2, location="B", sex="F", quality=None, type_="video"),
            _make_media(3, location=None, sex="U", quality="medium", type_=None),
        ]
        stats = mf.fit_norm_stats(medias, spec)
        # None -> 'Unknown'; values sorted alphabetically.
        self.assertEqual(stats.location_vocab, ["A", "B", "Unknown"])
        self.assertEqual(stats.sex_vocab, ["F", "M", "U"])
        self.assertEqual(stats.type_vocab, ["Unknown", "still", "video"])
        self.assertEqual(stats.quality_vocab, ["Unknown", "good", "medium"])

    def test_soft_mode_only_fits_location_vocab(self):
        spec = mf.MetadataFeatureSpec(use_soft_encoding=True)
        medias = [_make_media(1, location="A", sex="M", quality="good", type_="still")]
        stats = mf.fit_norm_stats(medias, spec)
        self.assertEqual(stats.location_vocab, ["A"])
        # sex/type/quality are not vocab-based in soft mode.
        self.assertEqual(stats.sex_vocab, [])
        self.assertEqual(stats.type_vocab, [])
        self.assertEqual(stats.quality_vocab, [])

    def test_length_min_max_skips_none(self):
        spec = mf.MetadataFeatureSpec(use_soft_encoding=False)
        medias = [
            _make_media(1, length=2.0),
            _make_media(2, length=None),
            _make_media(3, length=4.0),
        ]
        stats = mf.fit_norm_stats(medias, spec)
        self.assertEqual(stats.length_min, 2.0)
        self.assertEqual(stats.length_max, 4.0)

    def test_year_min_max_from_unix_timestamps(self):
        spec = mf.MetadataFeatureSpec(use_soft_encoding=False)
        medias = [
            _make_media(1, time=_unix(2020)),
            _make_media(2, time=_unix(2024)),
            _make_media(3, time=None),
        ]
        stats = mf.fit_norm_stats(medias, spec)
        self.assertEqual(stats.year_min, 2020)
        self.assertEqual(stats.year_max, 2024)


class TestOneHotEncoding(unittest.TestCase):
    """Verify one-hot mode output ordering and per-sample encoding."""

    def _basic_corpus(self):
        return [
            _make_media(1, location="A", sex="M", type_="still", quality="good",
                        length=1.0, time=_unix(2020)),
            _make_media(2, location="B", sex="F", type_="video", quality="medium",
                        length=3.0, time=_unix(2024)),
        ]

    def test_one_hot_ordering_sex_loc_type_qual(self):
        spec = mf.MetadataFeatureSpec(
            use_soft_encoding=False,
            include_length=False,
            include_time=False,
        )
        medias = self._basic_corpus()
        stats = mf.fit_norm_stats(medias, spec)
        vec = mf.engineer_features(medias[0], spec, stats)
        # Ordering: [sex_F, sex_M] [loc_A, loc_B] [type_still, type_video]
        # [qual_good, qual_medium]. Sample 0: M, A, still, good.
        self.assertEqual(vec.tolist(), [
            0.0, 1.0,    # sex F, M  -> M
            1.0, 0.0,    # loc A, B  -> A
            1.0, 0.0,    # type still, video -> still
            1.0, 0.0,    # qual good, medium -> good
        ])

    def test_one_hot_out_of_vocab_yields_all_zero(self):
        spec = mf.MetadataFeatureSpec(
            use_soft_encoding=False, include_sex=False, include_type=False,
            include_quality=False, include_length=False, include_time=False,
        )
        medias = [_make_media(1, location="A"), _make_media(2, location="B")]
        stats = mf.fit_norm_stats(medias, spec)
        # Sample with an unseen location.
        unseen = _make_media(3, location="Z")
        vec = mf.engineer_features(unseen, spec, stats)
        self.assertEqual(vec.tolist(), [0.0, 0.0])

    def test_none_categorical_maps_to_unknown_column(self):
        spec = mf.MetadataFeatureSpec(
            use_soft_encoding=False, include_sex=False, include_type=False,
            include_quality=False, include_length=False, include_time=False,
        )
        medias = [
            _make_media(1, location="A"),
            _make_media(2, location=None),
        ]
        stats = mf.fit_norm_stats(medias, spec)
        # Vocab: ['A', 'Unknown'].
        none_sample = _make_media(99, location=None)
        vec = mf.engineer_features(none_sample, spec, stats)
        self.assertEqual(vec.tolist(), [0.0, 1.0])


class TestSoftFeatureOrdering(unittest.TestCase):
    """Soft mode emits sex, type, quality, location (different from one-hot)."""

    def test_soft_ordering(self):
        spec = mf.MetadataFeatureSpec(
            use_soft_encoding=True,
            include_length=False,
            include_time=False,
        )
        medias = [
            _make_media(1, location="A", sex="M", type_="still", quality="good"),
        ]
        stats = mf.fit_norm_stats(medias, spec)
        vec = mf.engineer_features(medias[0], spec, stats)
        # [sex_male, sex_female] [type_still, type_video]
        # [qual_good, qual_medium, qual_poor] [loc_A].
        self.assertEqual(vec.tolist(), [
            1.0, 0.0,    # sex (M)
            1.0, 0.0,    # type (still)
            1.0, 0.0, 0.0,  # quality (good)
            1.0,         # loc A
        ])


class TestLengthAndTime(unittest.TestCase):
    """Length normalization and time encoding match the original."""

    def test_length_normalization(self):
        spec = mf.MetadataFeatureSpec(
            use_soft_encoding=False, include_sex=False, include_type=False,
            include_quality=False, include_location=False, include_time=False,
        )
        medias = [
            _make_media(1, length=2.0),
            _make_media(2, length=4.0),
        ]
        stats = mf.fit_norm_stats(medias, spec)
        # Mid-range value normalizes to 0.5.
        v = mf.engineer_features(_make_media(99, length=3.0), spec, stats)
        self.assertAlmostEqual(v.item(), 0.5)
        # Min and max anchor 0.0/1.0.
        self.assertAlmostEqual(
            mf.engineer_features(_make_media(99, length=2.0), spec, stats).item(), 0.0)
        self.assertAlmostEqual(
            mf.engineer_features(_make_media(99, length=4.0), spec, stats).item(), 1.0)
        # None defaults to 0.5.
        self.assertAlmostEqual(
            mf.engineer_features(_make_media(99, length=None), spec, stats).item(), 0.5)

    def test_length_degenerate_stats_default(self):
        spec = mf.MetadataFeatureSpec(
            use_soft_encoding=False, include_sex=False, include_type=False,
            include_quality=False, include_location=False, include_time=False,
        )
        # All None -> no min/max -> every value 0.5.
        medias = [_make_media(1, length=None)]
        stats = mf.fit_norm_stats(medias, spec)
        v = mf.engineer_features(_make_media(99, length=10.0), spec, stats)
        self.assertAlmostEqual(v.item(), 0.5)
        # Single value -> min == max -> degenerate -> 0.5.
        medias = [_make_media(1, length=3.0), _make_media(2, length=3.0)]
        stats = mf.fit_norm_stats(medias, spec)
        v = mf.engineer_features(_make_media(99, length=3.0), spec, stats)
        self.assertAlmostEqual(v.item(), 0.5)

    def test_time_cyclical_and_year(self):
        spec = mf.MetadataFeatureSpec(
            use_soft_encoding=False, include_sex=False, include_type=False,
            include_quality=False, include_location=False, include_length=False,
        )
        medias = [
            _make_media(1, time=_unix(2020, 1, 1)),
            _make_media(2, time=_unix(2024, 1, 1)),
        ]
        stats = mf.fit_norm_stats(medias, spec)
        # Day 1 of 2020 -> angle = 2π/366 -> sin(small), cos(near 1).
        # Year 2020 sits at the min of [2020, 2024] -> year_norm = 0.0.
        v = mf.engineer_features(medias[0], spec, stats)
        expected_angle = 2.0 * math.pi / 366.0
        self.assertAlmostEqual(v[0].item(), math.sin(expected_angle), places=6)
        self.assertAlmostEqual(v[1].item(), math.cos(expected_angle), places=6)
        self.assertAlmostEqual(v[2].item(), 0.0, places=6)

    def test_time_none_yields_zeros_and_year_default(self):
        spec = mf.MetadataFeatureSpec(
            use_soft_encoding=False, include_sex=False, include_type=False,
            include_quality=False, include_location=False, include_length=False,
        )
        medias = [
            _make_media(1, time=_unix(2020)),
            _make_media(2, time=_unix(2024)),
        ]
        stats = mf.fit_norm_stats(medias, spec)
        v = mf.engineer_features(_make_media(99, time=None), spec, stats)
        self.assertAlmostEqual(v[0].item(), 0.0)
        self.assertAlmostEqual(v[1].item(), 0.0)
        # Missing year -> 0.5 default per the original code.
        self.assertAlmostEqual(v[2].item(), 0.5)


class TestFeatureDim(unittest.TestCase):
    """feature_dim() must match the actual length of engineer_features output."""

    def test_dim_one_hot_full(self):
        spec = mf.MetadataFeatureSpec(use_soft_encoding=False)
        medias = [
            _make_media(1, location="A", sex="M", type_="still", quality="good",
                        length=1.0, time=_unix(2020)),
            _make_media(2, location="B", sex="F", type_="video", quality="medium",
                        length=2.0, time=_unix(2024)),
        ]
        stats = mf.fit_norm_stats(medias, spec)
        vec = mf.engineer_features(medias[0], spec, stats)
        self.assertEqual(vec.numel(), mf.feature_dim(spec, stats))

    def test_dim_soft_full(self):
        spec = mf.MetadataFeatureSpec(use_soft_encoding=True)
        medias = [
            _make_media(1, location="A", sex="M", type_="still", quality="good",
                        length=1.0, time=_unix(2020)),
            _make_media(2, location="B", sex="F", type_="video", quality="medium",
                        length=2.0, time=_unix(2024)),
        ]
        stats = mf.fit_norm_stats(medias, spec)
        vec = mf.engineer_features(medias[0], spec, stats)
        self.assertEqual(vec.numel(), mf.feature_dim(spec, stats))

    def test_excluded_features_drop_columns(self):
        spec = mf.MetadataFeatureSpec(
            use_soft_encoding=False, include_sex=False, include_location=False,
            include_type=False, include_quality=False, include_length=True,
            include_time=False,
        )
        medias = [_make_media(1, length=1.0), _make_media(2, length=3.0)]
        stats = mf.fit_norm_stats(medias, spec)
        vec = mf.engineer_features(medias[0], spec, stats)
        self.assertEqual(vec.numel(), 1)


class TestSpecRoundTrip(unittest.TestCase):
    """from_dict / to_dict are stable for config persistence."""

    def test_round_trip(self):
        original = mf.MetadataFeatureSpec(use_soft_encoding=True, include_sex=False)
        restored = mf.MetadataFeatureSpec.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_from_dict_ignores_unknown_keys(self):
        spec = mf.MetadataFeatureSpec.from_dict({
            "use_soft_encoding": True,
            "made_up_key": "ignored",
        })
        self.assertTrue(spec.use_soft_encoding)

    def test_from_dict_none_yields_defaults(self):
        spec = mf.MetadataFeatureSpec.from_dict(None)
        self.assertEqual(spec, mf.MetadataFeatureSpec())


if __name__ == "__main__":
    unittest.main()
