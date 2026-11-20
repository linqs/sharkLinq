"""Tests for the MultimodalFusionModel architecture across supported fusion methods."""
import unittest

import torch

import experimental.multimodal.trained_fusion.fusion_model as fusion_model


class TestMultimodalFusionModel(unittest.TestCase):
    """Tests for the MultimodalFusionModel class."""

    def _create_model(self, fusion_method, image_dim=128, metadata_dim=32,
                      shared_dim=64, final_dim=32, expert_depth=1,
                      residual_hidden_dim=64):
        return fusion_model.MultimodalFusionModel(
            image_dim=image_dim,
            metadata_dim=metadata_dim,
            shared_dim=shared_dim,
            final_dim=final_dim,
            fusion_method=fusion_method,
            expert_dropout=0.0,
            projection_dropout=0.0,
            expert_depth=expert_depth,
            residual_hidden_dim=residual_hidden_dim,
        )

    def _inputs(self, batch=4, image_dim=128, metadata_dim=32):
        return (torch.randn(batch, image_dim),
                torch.randn(batch, metadata_dim))

    # --- concatenate ---

    def test_concatenate_output_shape(self):
        """Concatenation fusion projects down to final_dim."""
        model = self._create_model("concatenate", final_dim=32).eval()
        image, metadata = self._inputs()
        out = model(image, metadata)
        self.assertEqual(out.shape, (4, 32))

    def test_concatenate_respects_final_dim(self):
        model = self._create_model("concatenate", final_dim=16).eval()
        image, metadata = self._inputs()
        self.assertEqual(model(image, metadata).shape, (4, 16))

    def test_deeper_expert_network(self):
        """expert_depth adds layers to each expert without changing the output shape."""
        shallow = self._create_model("concatenate", expert_depth=1)
        deep = self._create_model("concatenate", expert_depth=3)
        self.assertGreater(len(deep.vfe), len(shallow.vfe))
        self.assertGreater(len(deep.tme), len(shallow.tme))

        deep.eval()
        image, metadata = self._inputs()
        self.assertEqual(deep(image, metadata).shape, (4, 32))

    # --- residual ---

    def test_residual_outputs_image_dim_not_final_dim(self):
        """Residual ignores final_dim: the addition is only defined at image_dim."""
        model = self._create_model(
            "residual", image_dim=128, final_dim=32).eval()
        image, metadata = self._inputs()
        out = model(image, metadata)
        self.assertEqual(out.shape, (4, 128))

    def test_residual_is_identity_at_initialization(self):
        """The residual delta is zero-initialized, so an untrained residual model
        reproduces the image-only baseline exactly.

        This is the property that makes residual fusion safe to start training
        from: at step 0 it cannot be worse than image-only retrieval.
        """
        model = self._create_model("residual", image_dim=128).eval()
        image, metadata = self._inputs()
        with torch.no_grad():
            out = model(image, metadata)
        expected = torch.nn.functional.normalize(image, p=2, dim=1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    # --- shared contract ---

    def test_output_is_l2_normalized(self):
        for method in ("concatenate", "residual"):
            with self.subTest(method=method):
                model = self._create_model(method).eval()
                image, metadata = self._inputs()
                with torch.no_grad():
                    out = model(image, metadata)
                norms = out.norm(p=2, dim=1)
                self.assertTrue(
                    torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_all_methods_batch_size_one(self):
        for method in ("concatenate", "residual"):
            with self.subTest(method=method):
                model = self._create_model(method).eval()
                image, metadata = self._inputs(batch=1)
                self.assertEqual(model(image, metadata).shape[0], 1)

    def test_different_input_dimensions(self):
        cases = [
            (64, 16, 32, 16),
            (256, 128, 128, 64),
            (512, 8, 256, 128),
        ]
        for image_dim, metadata_dim, shared_dim, final_dim in cases:
            with self.subTest(image_dim=image_dim, metadata_dim=metadata_dim,
                              shared_dim=shared_dim, final_dim=final_dim):
                model = self._create_model(
                    "concatenate", image_dim=image_dim,
                    metadata_dim=metadata_dim, shared_dim=shared_dim,
                    final_dim=final_dim).eval()
                image, metadata = self._inputs(
                    image_dim=image_dim, metadata_dim=metadata_dim)
                self.assertEqual(model(image, metadata).shape, (4, final_dim))

    def test_unknown_fusion_method_raises(self):
        # 'gate', 'cross-attention', 'temp_scaling' and 'attention_pooling' were
        # all removed; they now land in this branch like any other bad value.
        for method in ("gate", "cross-attention", "bogus"):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    self._create_model(method)

    def test_registry_methods_all_construct(self):
        """Every fusion_method in TRAINED_FUSION_CONFIGS must be constructible."""
        import experimental.multimodal.trained_fusion.configs as configs
        for key, cfg in configs.TRAINED_FUSION_CONFIGS.items():
            with self.subTest(config=key):
                method = cfg["model"]["fusion_method"]
                self._create_model(method)


if __name__ == "__main__":
    unittest.main()
