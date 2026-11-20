"""Sex values are stored as F, M or None whatever case or unknown marker the caller sends."""
import unittest

from core import app_utils


class TestNormalizeSex(unittest.TestCase):
    def test_known_letters_are_uppercased(self):
        for value, expected in (("f", "F"), ("F", "F"), (" m ", "M"), ("M", "M")):
            with self.subTest(value=value):
                self.assertEqual(app_utils.normalize_sex(value), expected)

    def test_unknown_markers_and_blanks_become_none(self):
        for value in ("U", "u", "Unknown", "unknown", "", "  ", None):
            with self.subTest(value=value):
                self.assertIsNone(app_utils.normalize_sex(value))

    def test_filters_can_keep_unknown_as_a_selection(self):
        self.assertEqual(app_utils.normalize_sex("u", unknown="Unknown"), "Unknown")
        self.assertIsNone(app_utils.normalize_sex("", unknown="Unknown"))

    def test_other_values_are_rejected(self):
        for value in ("x", "female", 1, ["F"]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    app_utils.normalize_sex(value)
