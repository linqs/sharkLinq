"""Schema-validation tests verifying all entries in TRAINED_FUSION_CONFIGS satisfy FullConfig constraints."""
import unittest

import experimental.multimodal.trained_fusion.configs as mm_config
from src.core import utils


class TestFusionConfigSchemaValidation(unittest.TestCase):

    def test_every_fusion_config_validates(self):
        """Every entry in TRAINED_FUSION_CONFIGS must pass FullConfig validation."""
        self.assertGreater(len(mm_config.TRAINED_FUSION_CONFIGS), 0,
                           "TRAINED_FUSION_CONFIGS is empty — nothing to validate.")
        for key, cfg in mm_config.TRAINED_FUSION_CONFIGS.items():
            with self.subTest(config=key):
                utils.ensure_config_schema(cfg)


if __name__ == "__main__":
    unittest.main()
