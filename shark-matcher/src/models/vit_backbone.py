"""Vision Transformer (ViT) feature extractor backbone wrapper for image embeddings."""

from typing import Any, Union

import torch
import transformers

import src.config.config_schema


class ViTBackbone(torch.nn.Module):
    """Vision Transformer backbone module extracting CLS token embeddings."""

    def __init__(self, config: Union[src.config.config_schema.ComponentConfig, dict, Any] = None, **kwargs):
        """Initialize ViTBackbone and load pretrained weights."""
        super().__init__()

        # Normalize config
        if config is None:
            config = kwargs

        if isinstance(config, (src.config.config_schema.ComponentConfig, src.config.config_schema.BaseModel)):
            options = config.options
        elif isinstance(config, dict):
            options = config
        else:
            # Fallback if somehow we get something else but it has options attribute
            options = getattr(config, "options", kwargs)

        pretrained_model_name = options.get("pretrained_model_name_or_path")
        self.num_channels = options.get("in_channels", 3)

        # Load the pre-trained ViT model
        self.backbone = transformers.ViTModel.from_pretrained(
            pretrained_model_name,
            add_pooling_layer=False  # Kept from original, to get sequence of hidden states
        )

        # Set and validate output_dim. It's expected to match the backbone's natural hidden size.
        self.output_dim = options.get("output_dim")
        if self.output_dim != self.backbone.config.hidden_size:
            raise ValueError(
                f"output_dim in kwargs ({self.output_dim}) does not match "
                f"the backbone's hidden_size ({self.backbone.config.hidden_size}). "
                "Ensure these are consistent in your configuration."
            )

        if self.num_channels == 3:
            # Configure image processor for 3 channels
            self.image_processor = transformers.AutoImageProcessor.from_pretrained(
                pretrained_model_name,
                do_resize=False,
                do_rescale=False,
                do_normalize=False
            )
        else:
            raise ValueError(f"Unsupported number of channels: {self.num_channels}. Must be 3.")

    def forward(self, x):
        """Forward pass extracting CLS token embedding from processed images."""
        # The image_processor handles resizing, normalization (if applicable), and tensor conversion.
        # Explicitly provide input_data_format for robustness.
        pixel_values = self.image_processor(
            x,
            return_tensors="pt",
            input_data_format="channels_first"  # Assumes x is [B, C, H, W] or list of such
        )["pixel_values"]
        pixel_values = pixel_values.to(self.backbone.device)  # Ensure tensor is on the same device as the model

        # Pass the processed tensor to the backbone
        # ViTModel expects 'pixel_values' as the argument name for the input tensor
        outputs = self.backbone(pixel_values=pixel_values)

        # Extract the CLS token embedding (first token of the sequence)
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        cls_token_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        return torch.flatten(cls_token_embedding, start_dim=1, end_dim=-1)

    def get_output_dim(self):
        """Return the output dimension specified in kwargs, validated against backbone hidden_size."""
        return self.output_dim

    def save_pretrained(self, path):
        """Save the modified backbone to the specified directory."""
        self.backbone.save_pretrained(path)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate gradient checkpointing enable to the underlying Hugging Face backbone."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        else:
            print("Warning: Underlying backbone in ViTBackbone does not support gradient_checkpointing_enable.")
