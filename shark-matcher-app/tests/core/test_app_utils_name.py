"""Tests for server-side name validation in app_utils.valid_name."""
import unittest

from core import app_utils


class TestValidName(unittest.TestCase):
    """Tests for the characters valid_name accepts and rejects."""

    def test_accepts_plain_names(self):
        """Ordinary names pass."""
        for name in ["Notch", "fin_notch-3", "Shark 12", "A1", "ALLCAPS"]:
            self.assertTrue(app_utils.valid_name(name), name)

    def test_accepts_punctuation_present_in_real_data(self):
        """Punctuation present in database records is accepted."""
        for name in ["I'M JUST KEN", "O'Brien", "Shark (juvenile)", "N/A",
                     "Jan. 5, 2020", "left-fin, notched"]:
            self.assertTrue(app_utils.valid_name(name), name)

    def test_accepts_accented_and_non_latin_letters(self):
        """Letters from any script are allowed; the rule is not ASCII-only."""
        for name in ["Café", "Muñoz", "Zoë", "Ærø", "サメ", "Акула"]:
            self.assertTrue(app_utils.valid_name(name), name)

    def test_rejects_html_injection_payloads(self):
        """The characters that make stored XSS work are refused outright."""
        for name in ["<img src=x onerror=alert(1)>",
                     "<script>alert(1)</script>",
                     "Notch<script>",
                     '" onmouseover="alert(1)',
                     "javascript:alert(1)"]:
            self.assertFalse(app_utils.valid_name(name), name)

    def test_rejects_individual_dangerous_characters(self):
        """Each excluded character is rejected on its own, not just in payloads."""
        for char in ['<', '>', '"', '&', ';', '=', '`', '$', '{', '}', '\\',
                     '\n', '\r', '\t', '\x00']:
            self.assertFalse(app_utils.valid_name(f"Notch{char}"), repr(char))

    def test_absent_name_is_valid(self):
        """None and empty strings pass because name is optional on some writes."""
        self.assertTrue(app_utils.valid_name(None))
        self.assertTrue(app_utils.valid_name(""))

    def test_rejects_non_strings(self):
        """A non-string name is never valid."""
        for value in [123, 1.5, [], {}, True]:
            self.assertFalse(app_utils.valid_name(value), repr(value))

    def test_enforces_length_limit(self):
        """A name at the cap passes; one character over does not."""
        self.assertTrue(app_utils.valid_name("a" * app_utils.MAX_NAME_LENGTH))
        self.assertFalse(app_utils.valid_name("a" * (app_utils.MAX_NAME_LENGTH + 1)))


if __name__ == '__main__':
    unittest.main()
