"""
Tests for date parsing utility functions in app_utils.

Covers parse_time_from_name, including overflow months (month 15 is month 3 of
the following year) and edge-case years.
"""
import datetime
import unittest

from core import app_utils


class TestParseTimeFromName(unittest.TestCase):
    """Tests for parse_time_from_name returning unix timestamps."""

    def _assert_date(self, name, expected_month, expected_day, expected_year):
        """Helper to verify a name parses to the expected UTC date."""
        timestamp = app_utils.parse_time_from_name(name)
        self.assertIsNotNone(timestamp, f"parse_time_from_name returned None for '{name}'")
        dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        self.assertEqual(dt.year, expected_year, f"Year mismatch for '{name}'")
        self.assertEqual(dt.month, expected_month, f"Month mismatch for '{name}'")
        self.assertEqual(dt.day, expected_day, f"Day mismatch for '{name}'")

    def test_normal_month(self):
        """Standard month (11) should parse normally."""
        self._assert_date("PR19111803", 11, 18, 2019)

    def test_month_12(self):
        """Month 12 (December) should not overflow."""
        self._assert_date("PR19122103", 12, 21, 2019)

    def test_month_13_overflow(self):
        """Month 13 should wrap to January of the following year."""
        self._assert_date("PR19132103", 1, 21, 2020)

    def test_month_15_overflow(self):
        """Month 15 should wrap to March of the following year."""
        self._assert_date("PR19152103", 3, 21, 2020)

    def test_month_24_overflow(self):
        """Month 24 should wrap to December of the following year."""
        self._assert_date("PR19242103", 12, 21, 2020)

    def test_month_25_overflow(self):
        """Month 25 should wrap to January two years later."""
        self._assert_date("PR19252103", 1, 21, 2021)

    def test_year_above_50(self):
        """Year > 50 should be interpreted as 19XX."""
        self._assert_date("PR90061503", 6, 15, 1990)

    def test_year_zero(self):
        """Year 00 should be interpreted as 2000."""
        self._assert_date("PR00051503", 5, 15, 2000)

    def test_no_digits(self):
        """A name with no 8-digit sequence should return None."""
        result = app_utils.parse_time_from_name("NODIGITS")
        self.assertIsNone(result)

    def test_short_digits(self):
        """A name with fewer than 8 digits should return None."""
        result = app_utils.parse_time_from_name("PR1911180")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
