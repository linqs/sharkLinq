
import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

try:
    from src.models.miewid_backbone import MiewIDBackbone
except ImportError:
    # Handle case where MiewIDBackbone might not be importable in some environments (e.g. missing timm)
    MiewIDBackbone = None

class TestMiewIDBackbone(unittest.TestCase):
    def setUp(self):
        if MiewIDBackbone is None:
            self.skipTest("MiewIDBackbone could not be imported")

    def test_miewid_output(self):
        print("\n--- Initializing MiewID Backbone ---")
        # Initialize with the same config as training
        # Note: pretrained_model_name_or_path might need to be available locally or downloaded
        # We wrap this in try-except to avoid failing if internet/model is missing during simple regression
        try:
            model = MiewIDBackbone(
                pretrained_model_name_or_path="conservationxlabs/miewid-msv2", 
                in_channels=3
            )
        except Exception as e:
            self.skipTest(f"Could not load MiewID model (network/path issue?): {e}")
            return

        model.eval()
        
        # Create dummy batch: [Batch=2, Channels=3, Height=440, Width=440]
        # MiewID expects 440x440
        dummy_input = torch.randn(2, 3, 440, 440)
        
        print("\n--- Running Forward Pass ---")
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\n[Output Type]: {type(output)}")
        
        # Verify output is a Tensor
        self.assertIsInstance(output, torch.Tensor, f"Output should be a Tensor, got {type(output)}")
        
        # Verify shape [Batch, OutputDim]
        print(f"[Output Shape]: {output.shape}")
        self.assertEqual(output.shape[0], 2)
        
        # KEY CHECK: What is the raw norm of the embeddings?
        norms = torch.norm(output, p=2, dim=1)
        print(f"[Raw Embedding Norms]: {norms}")
        print(f"   -> Mean Norm: {norms.mean().item():.4f}")
        
        if norms.mean().item() > 5.0:
            print("\n[DIAGNOSIS]: Norms are LARGE. A triplet margin of 0.2 is likely negligible.")
            print("             You MUST normalize embeddings or increase the margin significantly.")


class TestMiewIDRevisionPin(unittest.TestCase):
    """Every MiewID hub load, including its remote code, is pinned to one commit."""

    def _assert_pinned(self, mocks):
        from src.core import constants
        for mock in mocks:
            self.assertEqual(mock.call_args.kwargs.get("revision", mock.call_args.kwargs.get("code_revision")),
                             constants.MIEWID_REVISION)

    def test_backbone_pins_config_code_and_weights(self):
        from src.models.miewid_backbone import MiewIDBackbone
        with patch("transformers.AutoConfig.from_pretrained") as config_load, \
                patch("transformers.AutoModel.from_config") as from_config, \
                patch("transformers.AutoModel.from_pretrained") as weights_load:
            MiewIDBackbone(pretrained_model_name_or_path="conservationxlabs/miewid-msv2")
        self._assert_pinned([config_load, from_config, weights_load])

    def test_embedding_model_loader_pins_config_code_and_weights(self):
        from src.core import utils
        with patch("transformers.AutoConfig.from_pretrained") as config_load, \
                patch("transformers.AutoModel.from_config") as from_config, \
                patch("transformers.AutoModel.from_pretrained") as weights_load:
            utils.load_embedding_model("miew", False, None, torch.device("cpu"))
        self._assert_pinned([config_load, from_config, weights_load])


class TestMiewIDGradientCheckpointing(unittest.TestCase):
    """Enabling checkpointing reaches the timm network and keeps gradients intact."""

    def _wrapper_around(self, timm_net):
        import types
        wrapper = MiewIDBackbone.__new__(MiewIDBackbone)
        torch.nn.Module.__init__(wrapper)
        wrapper.backbone = types.SimpleNamespace(backbone=timm_net)
        return wrapper

    def test_enable_turns_on_timm_checkpointing_without_changing_gradients(self):
        import timm
        torch.manual_seed(0)
        plain = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        checkpointed = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        checkpointed.load_state_dict(plain.state_dict())
        checkpointed.eval()
        plain.eval()

        self._wrapper_around(checkpointed).gradient_checkpointing_enable()

        self.assertTrue(checkpointed.grad_checkpointing)
        x = torch.randn(2, 3, 64, 64)
        grads = []
        for net in (plain, checkpointed):
            net.zero_grad()
            net(x).pow(2).sum().backward()
            grads.append([p.grad.clone() for p in net.parameters()])
        for plain_grad, checkpointed_grad in zip(*grads):
            self.assertTrue(torch.allclose(plain_grad, checkpointed_grad, atol=1e-6))
