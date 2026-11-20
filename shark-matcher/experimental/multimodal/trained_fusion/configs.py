"""Trained fusion model and training configurations for concatenation and residual architectures."""

import torch

import src.multimodal.multimodal_config
import experimental.multimodal.trained_fusion.multimodal_evaluator
import src.learning.loss.triplet

# Shared training hyperparameters.
EPOCHS = 50
BATCH_SIZE = 32  # P × K, target.
K_VALUE = 4

# Reuse the same metadata feature spec as the late-fusion path so the
# trained-fusion ablation rows are directly comparable.
DEFAULT_METADATA_FEATURES = \
    src.multimodal.multimodal_config.DEFAULT_METADATA_FEATURES

def _base_training_block(*, warmup_epochs: int, early_stopping_enabled: bool,
                         early_stopping_patience: int = 8) -> dict:
    """Build shared training hyperparameter dictionary with custom warmup and early stopping settings."""
    return {
        "gradient_accumulation_steps": 1,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "p_value": None,
        "k_value": K_VALUE,
        "filter_incomplete_metadata": False,
        "evaluator":
            experimental.multimodal.trained_fusion.multimodal_evaluator.multimodal_hits_at_k,
        "loss": {
            "name": "triplet",
            "constructor": src.learning.loss.triplet.create_triplet_loss,
            "options": {"margin": 0.2, "mining_strategy": "batch_hard"},
        },
        "optimizer": {
            "name": "AdamW",
            "constructor": torch.optim.AdamW,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
        },
        "scheduler": {
            "name": "CosineAnnealingLR",
            "constructor": torch.optim.lr_scheduler.CosineAnnealingLR,
            "options": {"T_max": EPOCHS},
        },
        "warmup_epochs": warmup_epochs,
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "patience": early_stopping_patience,
            "min_delta": 0.001,
        },
    }
TRAINED_FUSION_CONFIGS = {
    # Naive concatenation fusion: projected embeddings concatenated and passed through a projection head.
    "concatenate": {
        "training": _base_training_block(
            warmup_epochs=5, early_stopping_enabled=False),
        "metadata_features": DEFAULT_METADATA_FEATURES,
        "model": {
            "fusion_method": "concatenate",
            "shared_dim": 512,
            "final_dim": 256,
            "expert_dropout": 0.0,
            "projection_dropout": 0.0,
            "expert_depth": 1,
        },
    },

    # Residual identity fusion: zero-initialized residual branch added to base image embeddings.
    "residual": {
        "training": _base_training_block(
            warmup_epochs=5, early_stopping_enabled=True,
            early_stopping_patience=8),
        "metadata_features": DEFAULT_METADATA_FEATURES,
        "model": {
            "fusion_method": "residual",
            "shared_dim": 512,            # unused; schema parity.
            # final_dim satisfies schema validation and matches base image dimension at runtime.
            "final_dim": 2152,
            "expert_dropout": 0.0,
            "projection_dropout": 0.2,
            "expert_depth": 1,
            "residual_hidden_dim": 256,
        },
    },
}
