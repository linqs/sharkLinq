"""MiewID feature extractor backbone wrapper for re-identification."""

import warnings

import torch
import transformers

from src.core import constants


class MiewIDBackbone(torch.nn.Module):
    """Model wrapper for the Hugging Face MiewID model used as a feature extractor backbone."""

    def __init__(self, **kwargs):
        """Initialize MiewIDBackbone and load pretrained weights."""
        super().__init__()

        pretrained_model_name = kwargs.get("pretrained_model_name_or_path", "conservationxlabs/miewid-msv2")
        self.num_channels = kwargs.get("in_channels", 3)

        # Standardize loading with trust_remote_code to prevent meta tensor and Peft issues.
        try:
            # Initialize empty model on Meta device from config.
            config = transformers.AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True, revision=constants.MIEWID_REVISION)
            with transformers.modeling_utils.no_init_weights():
                self.backbone = transformers.AutoModel.from_config(
                    config, trust_remote_code=True, code_revision=constants.MIEWID_REVISION)

            # Load state dict manually to CPU.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state_dict = transformers.AutoModel.from_pretrained(
                    pretrained_model_name, trust_remote_code=True, low_cpu_mem_usage=False,
                    revision=constants.MIEWID_REVISION
                ).state_dict()

            # Assign weights to materialize meta tensors.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.backbone.load_state_dict(state_dict, assign=True)

        except Exception as e:
            print(f"Error loading MiewID model: {e}")
            raise

        # MiewID feature extractor outputs 2152 dimensions (2048 backbone + 104 auxiliary features).
        self.output_dim = 2152

        if self.num_channels != 3:
            raise ValueError(f"Unsupported number of channels: {self.num_channels}. Must be 3.")

    def forward(self, x):
        """Extract feature embeddings from input images."""
        return self.backbone(x)

    def get_output_dim(self):
        """Return the output dimension of the backbone."""
        return self.output_dim

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the timm network inside the MiewID model."""
        del gradient_checkpointing_kwargs  # timm chooses its own checkpoint mode.
        self.backbone.backbone.set_grad_checkpointing(True)
