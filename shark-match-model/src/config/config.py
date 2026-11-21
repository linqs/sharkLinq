"""Training configuration registry, augmentations, and model architecture definitions."""

import os

import peft
import torch

import src.learning.evaluator.evaluators
import src.learning.loss.triplet
import src.models.arcface_heads
import src.models.augmentations
import src.models.miewid_backbone
import src.models.mlp
import src.models.vit_backbone

NUM_AUGMENTATIONS = 2


def _num_workers_from_env():
    """Return the DataLoader worker count from SHARK_NUM_WORKERS, defaulting to 4."""
    raw = os.getenv("SHARK_NUM_WORKERS", "4")
    if not raw.isdigit():
        raise ValueError(f"SHARK_NUM_WORKERS must be a non-negative integer, got {raw!r}.")
    return int(raw)


# DataLoader worker processes for data loading and augmentation; each costs memory.
NUM_WORKERS = _num_workers_from_env()

AUGMENTATION_CONSTRUCTORS = src.models.augmentations.AUGMENTATION_CONSTRUCTORS


def _notification_emails_from_env():
    """Return training-completion recipients from comma-separated SHARK_NOTIFICATION_EMAILS."""
    raw = os.getenv("SHARK_NOTIFICATION_EMAILS", "")
    return [address.strip() for address in raw.split(",") if address.strip()]


# An empty list disables notifications; `--no-email` on the trainer disables them for one run.
NOTIFICATION_EMAILS = _notification_emails_from_env()

# The four-transform default, also used by the ViT triplet config.
AUGMENTATIONS = [
    ("ColorJitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1}),
    ("RandomHorizontalFlip", {"p": 0.5}),
    ("RandomPerspective", {"distortion_scale": 0.33, "p": 0.75}),
    ("RandomRotation", {"degrees": 30}),
]

MIEW_AUGMENTATIONS = [
    ("RandomPerspective", {"distortion_scale": 0.33, "p": 0.75}),
    ("RandomRotation", {"degrees": 30}),
    ("RandomAdjustSharpness", {"sharpness_factor": 2, "p": 0.5}),
    ("CLAHE", {"clip_limit": 4.0, "tile_grid_size": (8, 8)}),
    ("RandomAffine", {"degrees": 15, "translate": (0.25, 0.25), "scale": (0.8, 1.2)}),
    ("ColorJitter", {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.1}),
    ("RandomHorizontalFlip", {"p": 0.5})
]

MIEW_AUG_BOTH = MIEW_AUGMENTATIONS + [
    ("RandomErasing", {"p": 0.3, "scale": (0.02, 0.15)}),
    ("GaussianBlur", {"kernel_size": 5, "sigma": (0.5, 2.0)})
]

PROJECTION_HEAD_CONSTRUCTORS = {
    "MLP": src.models.mlp.MLP,
    "Identity": torch.nn.Identity,
    "ArcMarginProduct": src.models.arcface_heads.ArcMarginProduct,
    "ArcFaceSubCenterDynamic": src.models.arcface_heads.ArcFaceSubCenterDynamic,
}

EARLY_STOPPING_DEFAULT = {
    "enabled": False,
    "patience": 10,
    "min_delta": 0.001,
    "mode": "max"
}

MIEWID_BACKBONE_NAME = "conservationxlabs/miewid-msv2"
VIT_BACKBONE_NAME = "google/vit-large-patch16-384"
MIEWID_LORA_TARGETS = ["conv_pw", "conv_pwl", "conv_dw", "final",
                       "se_conv_reduce", "se_conv_expand", "conv_exp"]
VIT_LORA_TARGETS = ["query", "key", "value", "dense"]


def _adamw(learning_rate, weight_decay):
    """Return a fresh AdamW optimizer block."""
    return {
        "name": "AdamW",
        "constructor": torch.optim.AdamW,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay
    }


def _cosine_schedule(t_max):
    """Return a fresh cosine-annealing scheduler block; T_max should equal the epoch count."""
    return {
        "name": "CosineAnnealingLR",
        "constructor": torch.optim.lr_scheduler.CosineAnnealingLR,
        "options": {"T_max": t_max}
    }


def _lora(r, lora_alpha, lora_dropout, target_modules):
    """Return a fresh LoRA adapter block."""
    return {
        "constructor": peft.LoraConfig,
        "options": {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": list(target_modules)
        }
    }


def _imagenet_data(image_size):
    """Return a fresh data block with ImageNet normalization and squash resizing."""
    return {
        "image_size": list(image_size),
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize_mode": "squash"
        }
    }


# Configs that exist for tests and are skipped when the trainer runs every config.
TEST_ONLY_CONFIGS = ("test_config",)

CONFIGS = {
    # MiewID with ArcFace (production config).
    "miewid_arcface": {
        "augmentations": MIEW_AUG_BOTH,
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "grad_checkpoint": True,
            "epochs": 50,
            "early_stopping": EARLY_STOPPING_DEFAULT,
            "evaluator": src.learning.evaluator.evaluators.hits_at_k,
            "loss": {"name": "arcface"},
            "freeze_backbone": False,
            "freeze_bn_stats": True,
            "num_augmentations": 2,
            "rebalance_augmentations": False,
            "optimizer": _adamw(learning_rate=5e-4, weight_decay=0.5),
            "scheduler": _cosine_schedule(50),
            "include_original": True
        },
        "lora": _lora(r=8, lora_alpha=8, lora_dropout=0.1, target_modules=MIEWID_LORA_TARGETS),
        "model": {
            "backbone": {
                "name": "MiewIDBackbone",
                "constructor": src.models.miewid_backbone.MiewIDBackbone,
                "options": {
                    "pretrained_model_name_or_path": MIEWID_BACKBONE_NAME,
                    "in_channels": 3
                }
            },
            "projection_head": {
                "name": "ArcFaceSubCenterDynamic",
                "constructor": src.models.arcface_heads.ArcFaceSubCenterDynamic,
                "options": {
                    "s": 51.5,
                    "k": 3,
                    "margins": None,
                }
            }
        },
        "data": _imagenet_data([440, 440])
    },

    "miewid_triplet": { # MiewID backbone trained with triplet loss; underperforms miewid_arcface.
        "augmentations": MIEW_AUG_BOTH,
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "grad_checkpoint": True,
            "epochs": 100,
            "early_stopping": EARLY_STOPPING_DEFAULT,
            "evaluator": src.learning.evaluator.evaluators.hits_at_k,
            "loss": {
                "name": "triplet",
                "constructor": src.learning.loss.triplet.create_triplet_loss,
                "options": {
                    "normalize_embeddings": True,
                },
            },
            "freeze_backbone": False,
            "freeze_bn_stats": True,
            "num_augmentations": 2,
            "rebalance_augmentations": False,
            "optimizer": _adamw(learning_rate=5e-5, weight_decay=0.05),
            "scheduler": _cosine_schedule(100),
            "identities_per_batch": 2, # P
            "samples_per_identity": 2,  # K
            "include_original": True
        },
        "lora": _lora(r=8, lora_alpha=8, lora_dropout=0.3, target_modules=MIEWID_LORA_TARGETS),
        "model": {
            "backbone": {
                "name": "MiewIDBackbone",
                "constructor": src.models.miewid_backbone.MiewIDBackbone,
                "options": {
                    "pretrained_model_name_or_path": MIEWID_BACKBONE_NAME,
                    "in_channels": 3
                }
            },
            "projection_head": {
                "name": "Identity",
                "constructor": torch.nn.Identity,
                "options": {}
            }
        },
        "data": _imagenet_data([440, 440])
    },

    "vit_arcface": { # ViT backbone trained with ArcFace.
        "augmentations": MIEW_AUG_BOTH,
        "training": {
            "batch_size": 32,
            "gradient_accumulation_steps": 2,
            "grad_checkpoint": True,
            "epochs": 50,
            "early_stopping": EARLY_STOPPING_DEFAULT,
            "evaluator": src.learning.evaluator.evaluators.hits_at_k,
            "loss": {"name": "arcface"},
            "freeze_backbone": False,
            "num_augmentations": 2,
            "rebalance_augmentations": False,
            "optimizer": _adamw(learning_rate=6.25e-5, weight_decay=0.5),
            "scheduler": _cosine_schedule(50),
            "include_original": True
        },
        "lora": _lora(r=8, lora_alpha=8, lora_dropout=0.1, target_modules=VIT_LORA_TARGETS),
        "model": {
            "backbone": {
                "name": "ViTModel",
                "constructor": src.models.vit_backbone.ViTBackbone,
                "options": {
                    "pretrained_model_name_or_path": VIT_BACKBONE_NAME,
                    "in_channels": 3,
                    "output_dim": 1024
                }
            },
            "projection_head": {
                "name": "ArcFaceSubCenterDynamic",
                "constructor": src.models.arcface_heads.ArcFaceSubCenterDynamic,
                "options": {
                    "s": 51.5,
                    "k": 3,
                    "margins": None,
                }
            }
        },
        "data": _imagenet_data([384, 384])
    },

    "vit_triplet": { # ViT backbone trained with triplet loss; the best ViT config.
        "augmentations": AUGMENTATIONS,
        "training": {
            "batch_size": 32,
            "gradient_accumulation_steps": 2,
            "grad_checkpoint": True,
            "epochs": 100,
            "early_stopping": EARLY_STOPPING_DEFAULT,
            "evaluator": src.learning.evaluator.evaluators.hits_at_k,
            "loss": {
                "name": "triplet",
                "constructor": src.learning.loss.triplet.create_triplet_loss,
                "options": {
                    "normalize_embeddings": True,
                },
            },
            "freeze_backbone": False,
            "num_augmentations": 2,
            "rebalance_augmentations": False,
            "optimizer": _adamw(learning_rate=6.25e-5, weight_decay=0.5),
            "scheduler": _cosine_schedule(100),
            "identities_per_batch": 8, # P
            "samples_per_identity": 4, # K
            "include_original": True
        },
        "lora": _lora(r=8, lora_alpha=8, lora_dropout=0.3, target_modules=VIT_LORA_TARGETS),
        "model": {
            "backbone": {
                "name": "ViTModel",
                "constructor": src.models.vit_backbone.ViTBackbone,
                "options": {
                    "pretrained_model_name_or_path": VIT_BACKBONE_NAME,
                    "output_dim": 1024,
                    "in_channels": 3
                }
            },
            "projection_head": {
                "name": "MLP",
                "constructor": src.models.mlp.MLP,
                "options": {
                    "hidden_dim": 256,
                    "output_dim": 1024,
                    "num_layers": 1,
                    "dropout_rate": 0.3
                }
            }
        },
        "data": _imagenet_data([384, 384])
    },

    # Small config for unit tests and the parity harness.
    "test_config": {
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "grad_checkpoint": True,
            "epochs": 5,
            "early_stopping": EARLY_STOPPING_DEFAULT,
            "evaluator": src.learning.evaluator.evaluators.hits_at_k,
            "loss": {
                "name": "triplet",
                "constructor": src.learning.loss.triplet.create_triplet_loss,
                "options": {
                    "normalize_embeddings": True,
                    "margin": 0.5,
                },
            },
            "freeze_backbone": False,
            "freeze_bn_stats": True,
            "num_augmentations": NUM_AUGMENTATIONS,
            "rebalance_augmentations": False,
            "optimizer": _adamw(learning_rate=5e-6, weight_decay=0.05),
            "scheduler": _cosine_schedule(5),
            "identities_per_batch": 2, # P
            "samples_per_identity": 2, # K
            "include_original": True
        },
        "lora": _lora(r=2, lora_alpha=2, lora_dropout=0.1,
                      target_modules=["conv_pw", "conv_pwl", "final"]),
        "model": {
            "backbone": {
                "name": "MiewIDBackbone",
                "constructor": src.models.miewid_backbone.MiewIDBackbone,
                "options": {
                    "pretrained_model_name_or_path": MIEWID_BACKBONE_NAME,
                    "in_channels": 3
                }
            },
            "projection_head": {
                "name": "Identity",
                "constructor": torch.nn.Identity,
                "options": {}
            }
        },
        "data": _imagenet_data([100, 100])
    },
}
