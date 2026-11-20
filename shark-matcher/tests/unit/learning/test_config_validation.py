"""Schema-validation tests verifying training config registries conform to FullConfig specifications."""
import os
import unittest
from unittest.mock import patch

import src.config.config
from src.core import utils


class TestConfigSchemaValidation(unittest.TestCase):

    def test_every_config_validates(self):
        """Every entry in CONFIGS must pass FullConfig validation."""
        self.assertGreater(len(src.config.config.CONFIGS), 0,
                           "CONFIGS is empty — nothing to validate.")
        for key, cfg in src.config.config.CONFIGS.items():
            with self.subTest(config=key):
                utils.ensure_config_schema(cfg)

    def test_registry_names_the_production_config_and_holds_no_parked_entries(self):
        """The MiewID ArcFace config is registered as miewid_arcface and nothing else is parked."""
        self.assertIn("miewid_arcface", src.config.config.CONFIGS)
        self.assertNotIn("supposed_best", src.config.config.CONFIGS)
        self.assertFalse(hasattr(src.config.config, "PAUSED_EXPERIMENTS"))


class TestWorkerCount(unittest.TestCase):

    def _read(self, value):
        with patch.dict(os.environ, {} if value is None else {"SHARK_NUM_WORKERS": value}):
            if value is None:
                os.environ.pop("SHARK_NUM_WORKERS", None)
            return src.config.config._num_workers_from_env()  # pylint: disable=protected-access

    def test_default_is_four_workers(self):
        self.assertEqual(self._read(None), 4)

    def test_environment_overrides_the_count(self):
        self.assertEqual(self._read("2"), 2)
        self.assertEqual(self._read("0"), 0)

    def test_invalid_counts_are_rejected_with_the_variable_name(self):
        for bad in ("-1", "many", ""):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "SHARK_NUM_WORKERS"):
                    self._read(bad)


class TestNotificationRecipients(unittest.TestCase):

    def test_recipients_default_to_none(self):
        """Without SHARK_NOTIFICATION_EMAILS no training notifications are sent."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SHARK_NOTIFICATION_EMAILS", None)
            self.assertEqual(src.config.config._notification_emails_from_env(), [])  # pylint: disable=protected-access

    def test_recipients_parse_comma_separated_addresses(self):
        """Addresses are split on commas with surrounding whitespace and empty entries dropped."""
        with patch.dict(os.environ, {
                "SHARK_NOTIFICATION_EMAILS": " a@example.org, ,b@example.org,"}):
            self.assertEqual(
                src.config.config._notification_emails_from_env(),  # pylint: disable=protected-access
                ["a@example.org", "b@example.org"],
            )


if __name__ == "__main__":
    unittest.main()
