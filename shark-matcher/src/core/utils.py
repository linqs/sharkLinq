"""
Utility functions for the shark-matcher src.
"""
import argparse
import base64
import csv
import datetime
import functools
import importlib
import importlib.abc
import importlib.util
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import google.auth.transport.requests
import google.oauth2.credentials
import google_auth_oauthlib.flow
import googleapiclient.discovery
import numpy
import peft
import timm
import torch
import transformers

from src.core import constants
import src.config.config_schema
import src.models.miewid_backbone
import src.models.vit_backbone

# --- Constants & Regex Patterns ---

# Matches 8 digits (typically yymmddXX)
DATE_RE = re.compile(r"(\d{8})")

# Matches media filenames: Location, Date, optional Sex, Length, Extension
# Example: PR19111801_M_3.5.jpeg
MEDIA_NAME_RE = re.compile(
    r"^([A-Z][A-Z]+)"          # Location (two or more caps)
    r"(\d{8})"                 # Numerical date/ID (8 digits)
    r"(?:_*([A-Za-z]))?"      # Optional sex (after underscore)
    r"(?:_*(\d[.\d]*))?"      # Optional length (after underscore)
    r"\.?([A-Za-z]+)?$"       # Optional extension
)

# Matches latitude/longitude strings: +1.2-3.4 or 1.2+3.4
LAT_LONG_RE = re.compile(r'^((?:\+|-)?\d+\.\d+)((?:\+|-)?\d+\.\d+)/?$')

# --- Pickle Compatibility ---

# Earlier module paths that stored pickles may name, mapped to their replacements (None: an empty parent package).
PICKLE_MODULE_ALIASES = {
    "backend": "src",
    "backend.individual_dataset": "src.individual_dataset",
    "backend.learning": "src.learning",
    "backend.learning.augmented_dataset": "src.learning.augmented_dataset",
    "src.sharkfin_dataset": "src.individual_dataset",
    "src.learning.contrastive_learning_sharkfin_dataset": "src.learning.augmented_dataset",
    "src.learning.contrastive_learning_individual_dataset": "src.learning.augmented_dataset",
    "src.experiments": None,
    "src.experiments.multimodal": "src.multimodal",
    "src.experiments.multimodal.fusion_dataset": "src.multimodal.fusion_dataset",
    "src.experiments.multimodal.metadata_features": "src.multimodal.metadata_features",
    "src.experiments.multimodal.multimodal_config": "src.multimodal.multimodal_config",
    "backend.experiments": None,
    "backend.experiments.multimodal": "src.multimodal",
    "backend.experiments.multimodal.fusion_dataset": "src.multimodal.fusion_dataset",
    "backend.experiments.multimodal.metadata_features": "src.multimodal.metadata_features",
    "backend.experiments.multimodal.multimodal_config": "src.multimodal.multimodal_config",
}


class _PickleModuleAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import an aliased module name as the module that replaced it, so unpickling finds its classes."""

    def __init__(self):
        self._replaced_specs = {}

    def find_spec(self, fullname, path, target=None):  # pylint: disable=unused-argument
        """Return a spec only for aliased module names."""
        if fullname not in PICKLE_MODULE_ALIASES:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        """Return the replacement module itself, or an empty package for a parent-only alias."""
        replacement = PICKLE_MODULE_ALIASES[spec.name]
        if replacement is None:
            spec.submodule_search_locations = []
            return None
        module = importlib.import_module(replacement)
        self._replaced_specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module):
        """Restore a replacement module's own spec, which importlib overwrites with the alias spec."""
        if module.__spec__.name in self._replaced_specs:
            module.__spec__ = self._replaced_specs.pop(module.__spec__.name)


def _register_pickle_module_aliases():
    """Install the alias finder once, ahead of the default finders so aliases keep precedence."""
    if not any(isinstance(finder, _PickleModuleAliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _PickleModuleAliasFinder())


_register_pickle_module_aliases()

# --- Application Configuration ---

def resolve_directory_path(input_path: str, fallback_base_dir: str) -> str:
    """Resolves a directory path directly or relative to a fallback base directory."""
    if input_path is None:
        print(f"Resolved (no input) -> {fallback_base_dir} (default base directory).")
        return fallback_base_dir

    if os.path.exists(input_path):
        resolved = os.path.abspath(input_path)
        print(f"Resolved '{input_path}' -> {resolved} (direct match).")
        return resolved

    fallback_path = os.path.join(fallback_base_dir, input_path)
    if os.path.exists(fallback_path):
        resolved = os.path.abspath(fallback_path)
        print(f"Resolved '{input_path}' -> {resolved} (fallback under {fallback_base_dir}).")
        return resolved

    raise ValueError(
        f"Directory not found. Tried exact match '{input_path}' and "
        f"fallback '{fallback_path}'."
    )


def relative_to_data_root(path: str | os.PathLike, root: str | os.PathLike) -> str | None:
    """Return path relative to root with forward slashes, or None when path is outside root."""
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return None
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return None
    return rel.replace(os.sep, "/")


def portable_csv_dir(csv_dir: str | os.PathLike, include_snapshots: bool = False) -> str | None:
    """Return csv_dir relative to the database-format, data, or (optionally) snapshot root, else None."""
    rel = (relative_to_data_root(csv_dir, constants.DB_DATA_DIR)
           or relative_to_data_root(csv_dir, constants.BASE_DATA_DIR))
    if rel is None and include_snapshots:
        rel_snapshot = relative_to_data_root(csv_dir, constants.INFERENCE_SNAPSHOT_ROOT)
        if rel_snapshot is not None:
            rel = f"inference-snapshots/{rel_snapshot}"
    return rel


def resolve_relative_csv_dir(path: str | os.PathLike | None) -> str | None:
    """Resolve an exact or relative CSV directory against standard data roots."""
    if path is None:
        return None
    raw = os.fspath(path)
    if os.path.isabs(raw):
        return raw
    if os.path.isdir(raw):
        return os.path.abspath(raw)
    candidate_db = os.path.join(constants.DB_DATA_DIR, raw)
    if os.path.isdir(candidate_db):
        return os.path.abspath(candidate_db)
    candidate_base = os.path.join(constants.BASE_DATA_DIR, raw)
    if os.path.isdir(candidate_base):
        return os.path.abspath(candidate_base)
    candidate_snapshot = os.path.join(constants.INFERENCE_SNAPSHOT_ROOT, raw)
    if os.path.isdir(candidate_snapshot):
        return os.path.abspath(candidate_snapshot)
    if raw.startswith("inference-snapshots" + os.sep) or raw.startswith("inference-snapshots/"):
        sub = raw.split("/", 1)[1] if "/" in raw else raw.split(os.sep, 1)[1]
        candidate_sub = os.path.join(constants.INFERENCE_SNAPSHOT_ROOT, sub)
        if os.path.isdir(candidate_sub):
            return os.path.abspath(candidate_sub)
    return raw


def resolve_csv_dir(args: argparse.Namespace, dataset=None) -> str | None:
    """Resolves the source CSV directory from CLI arguments, a stored dataset, or an annotations sidecar."""
    if getattr(args, 'csv_dir', None):
        return resolve_directory_path(args.csv_dir, constants.DB_DATA_DIR)

    if dataset is not None:
        csv_dir = getattr(dataset, 'csv_dir', None)
        if csv_dir:
            resolved = resolve_relative_csv_dir(csv_dir)
            if resolved and os.path.isdir(resolved):
                print(f"Using CSV dir from stored dataset: {resolved}")
                return resolved
            print(
                f"\nWARNING: The CSV directory recorded on this dataset was not found on this machine:\n"
                f"  {csv_dir}\n"
                f"This occurs when embeddings or datasets were serialized on a different machine or path layout.\n"
                f"Please specify the correct directory using --csv-dir (e.g. --csv-dir database-format/thesis).\n",
                file=sys.stderr,
            )
        print("ERROR: No CSV directory could be resolved. Pass --csv-dir or "
              "regenerate the dataset with a current build of "
              "src.retrieval.embeddings so csv_dir is pinned.")
        sys.exit(1)

    if getattr(args, 'annotations_version', None):
        version_dir = os.path.join(constants.ANNOTATIONS_DIR, args.annotations_version)
        sidecar = os.path.join(version_dir, constants.SOURCE_CSV_DIR_SIDECAR)
        if os.path.exists(sidecar):
            with open(sidecar, 'r', encoding='utf-8') as f:
                recorded = f.read().strip()
            if recorded:
                resolved = resolve_relative_csv_dir(recorded)
                if resolved and os.path.isdir(resolved):
                    print(f"Using CSV dir from annotations sidecar: {resolved}")
                    return resolved
                print(f"WARNING: sidecar at {sidecar} points to a non-existent "
                      f"directory: {recorded!r}. Use --csv-dir to override.",
                      file=sys.stderr)

    return None


def ensure_config_schema(config_dict: Dict[str, Any]) -> src.config.config_schema.FullConfig:
    """Validates and converts a raw dictionary to a FullConfig Pydantic object."""
    try:
        return src.config.config_schema.FullConfig(**config_dict)
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        raise e

# --- Subdirectory Management ---
def resolve_subdirs(name: str, base_dir: str, marker_files, skip_dirs=()) -> List[str]:
    """Resolves a directory name to the subdirectories holding any marker file, never entering skip_dirs."""
    marker_files = (marker_files,) if isinstance(marker_files, str) else tuple(marker_files)
    target_dir = os.path.join(base_dir, name)

    # Case 1: Direct directory with a marker file.
    if any(os.path.exists(os.path.join(target_dir, marker)) for marker in marker_files):
        print(f"Resolved '{name}' -> 1 directory (direct match).")
        return [name]

    # Case 2: Parent directory with marker-containing subdirectories.
    if os.path.isdir(target_dir):
        subdirs = []
        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [directory for directory in dirs if directory not in skip_dirs]
            if any(marker in files for marker in marker_files):
                rel_path = os.path.relpath(root, base_dir)
                subdirs.append(rel_path)
        if subdirs:
            subdirs.sort()
            print(f"\nResolved '{name}' -> {len(subdirs)} directories:")
            for i, subdir in enumerate(subdirs, 1):
                print(f"  {i}. {subdir}")
            print()
            return subdirs

    # Case 3: No marker file found anywhere.
    print(f"Warning: No '{' or '.join(marker_files)}' found in '{name}'. Returning as-is.")
    return [name]

def get_torch_device():
    """Determines and returns the available PyTorch device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_trainable_parameters(model: torch.nn.Module):
    """Print how many of the model's parameters are trainable."""
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    pct = 0.0
    if all_param > 0:
        pct = 100 * trainable_params / all_param

    # pylint: disable=line-too-long
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {pct:.2f}"
    )


def save_model_state(model: torch.nn.Module, out_directory: str, filename):
    """Save a model's state dict to out_directory/filename without DataParallel 'module.' prefixes."""
    formatted_model_state_dict = {
        re.sub(r"^module\.", "", key).strip(): model.state_dict()[key]
        for key in model.state_dict()
    }
    torch.save(formatted_model_state_dict, os.path.join(out_directory, filename))


def resolve_model_dirs(model_name: str) -> List[str]:
    """Resolves a model name to a list of config subdirectory paths under constants.TRAINED_MODELS_DIR."""
    return resolve_subdirs(model_name, constants.TRAINED_MODELS_DIR, "config.json")

def resolve_embedding_dirs(embeddings_name: str) -> List[str]:
    """Resolves an embeddings name to its embedding set directories, each read through its active generation."""
    return resolve_subdirs(embeddings_name, constants.EMBEDDINGS_DIR,
                           ("stored_embeddings.pt", "current.json"), skip_dirs=("generations",))


ALL_BACKBONE_CONSTRUCTORS = {
    "ViTBackbone": src.models.vit_backbone.ViTBackbone,
    "MiewIDBackbone": src.models.miewid_backbone.MiewIDBackbone,
}

def load_model(
        model_name: str,
        backbone_prefix: str = "best") -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Loads a pretrained model and returns its projection head and backbone."""
    # pylint: disable=import-outside-toplevel
    device = get_torch_device()

    import src.config.config

    print(f"Loading trained model: {model_name}")
    model_dir = os.path.join(constants.TRAINED_MODELS_DIR, model_name)
    model_config_path = os.path.join(model_dir, "config.json")
    model_config = load_json_file(model_config_path)

    backbone_config = model_config["model"]["backbone"]
    projection_head_config = model_config["model"]["projection_head"]

    print(f"  Backbone: {backbone_config['constructor']}")

    constructor_name = backbone_config["constructor"]
    base_backbone_model = ALL_BACKBONE_CONSTRUCTORS[constructor_name](
        **backbone_config["options"]).to(device)

    print(f"  Backbone checkpoint: {backbone_prefix}")
    lora_adapter_path = os.path.join(model_dir, f"{backbone_prefix}-backbone-lora")
    frozen_state_dict_path = os.path.join(model_dir, f"{backbone_prefix}-backbone-state.pt")

    model_backbone = base_backbone_model
    state_dict = None
    if os.path.isfile(frozen_state_dict_path):
        print(f"  Found weights at {os.path.basename(frozen_state_dict_path)}")
        # We use weights_only=False as defined in constants.PY logic for compatibility
        state_dict = torch.load(frozen_state_dict_path, map_location=device, weights_only=False)

    has_lora_keys = state_dict is not None and any("lora" in k for k in state_dict.keys())

    if os.path.isdir(lora_adapter_path):
        print(f"  Found LoRA adapters at {os.path.basename(lora_adapter_path)}")
        if has_lora_keys:
            print("  State dict contains LoRA keys. Wrapping with PeftModel BEFORE loading.")
            model_backbone = peft.PeftModel.from_pretrained(
                model_backbone, lora_adapter_path).to(device)

            new_state_dict = {"base_model.model." + k: v for k, v in state_dict.items()}
            print(f"Loading state dict with {len(new_state_dict)} keys into PeftModel.")
            model_backbone.load_state_dict(new_state_dict)
        else:
            if state_dict is not None:
                print("State dict does not contain LoRA keys. Loading into base model FIRST.")
                model_backbone.load_state_dict(state_dict)

            print("Loading PeftModel adapters.")
            model_backbone = peft.PeftModel.from_pretrained(
                model_backbone, lora_adapter_path).to(device)

    elif state_dict is not None:
        if has_lora_keys:
            print("State dict has LoRA keys but no adapter path found.")
        model_backbone.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"No backbone weights for '{model_name}': expected "
            f"{os.path.basename(frozen_state_dict_path)} or a "
            f"{os.path.basename(lora_adapter_path)} folder in {model_dir}.")

    model_backbone.eval()

    # Projection Head resolution
    proj_head_path = os.path.join(model_dir, "best-projection-head.pt")
    if not os.path.exists(proj_head_path):
        proj_head_path = os.path.join(model_dir, "last-projection-head.pt")

    arcface_head_path = os.path.join(model_dir, "best-arcface-head.pt")
    if not os.path.exists(arcface_head_path):
        arcface_head_path = os.path.join(model_dir, "last-arcface-head.pt")

    if os.path.exists(arcface_head_path):
        print("  ArcFace head detected. Using Identity head for inference/embeddings.")
        projection_head = torch.nn.Identity().to(device)
    else:
        proj_head_name = projection_head_config["name"]
        proj_head_constructor = src.config.config.PROJECTION_HEAD_CONSTRUCTORS[proj_head_name]
        try:
            if proj_head_name == "MLP":
                projection_head = proj_head_constructor(
                    input_dim=model_backbone.get_output_dim(),
                    **projection_head_config["options"]).to(device)

                if os.path.exists(proj_head_path):
                    projection_head.load_state_dict(
                        torch.load(proj_head_path, map_location=device, weights_only=False))
                else:
                    print(f"No projection head weights found at {proj_head_path}.")
                projection_head.eval()

            elif proj_head_name in ["ArcMarginProduct", "ArcFaceSubCenterDynamic"]:
                print(f"ArcFace head '{proj_head_name}' detected. Using Identity head.")
                projection_head = torch.nn.Identity().to(device)

            else: # Handles nn.Identity and other parameter-less heads
                projection_head = proj_head_constructor().to(device)
                if os.path.exists(proj_head_path):
                    projection_head.load_state_dict(
                        torch.load(proj_head_path, map_location=device, weights_only=False))
                projection_head.eval()

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Failed to initialize projection head '{proj_head_name}'. "
                  f"Defaulting to Identity. Error: {e}")
            projection_head = torch.nn.Identity().to(device)

    return projection_head, model_backbone

def get_model_metadata(model_name):
    """Retrieves preprocessing metadata (is_finetuned, config_path, output_image_size, normalization_config) for a model."""
    model_config_path = os.path.join(constants.TRAINED_MODELS_DIR, model_name, "config.json")
    is_finetuned_model = os.path.exists(model_config_path)

    output_image_size = None
    normalization_config = None

    if not is_finetuned_model:
        if model_name.strip().lower() == "miew":
            print("Using miewid model configuration.")
            # MiewID expects 440x440 and ImageNet normalization with squashing (no aspect-ratio
            # preservation), matching the original training procedure.
            output_image_size = (440, 440)
            normalization_config = {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize_mode": "squash"
            }
        elif model_name.strip().lower() == "mega":
            print("Using MegaDescriptor model configuration.")
            output_image_size = (384, 384)
            normalization_config = {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "resize_mode": "squash"
            }
        elif model_name.strip().lower() == "vit":
            print("Using base ViT backbone configuration.")
            # Uses 384x384 ImageNet normalization applied at dataset level to match ViT training config.
            output_image_size = (384, 384)
            normalization_config = {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize_mode": "squash"
            }
        else:
            raise FileNotFoundError(
                f"Could not find a config file for fine-tuned model '{model_name}' "
                f"at '{model_config_path}', and it's not a known pre-canned model "
                f"(e.g., 'miew', 'mega', 'vit').")
    else:
        try:
            config = load_json_file(model_config_path)
            data_config = config.get("data", {})
            if "image_size" in data_config:
                output_image_size = tuple(data_config["image_size"])
            if "normalization" in data_config:
                normalization_config = data_config["normalization"]
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Warning: Failed to load config for {model_name}: {e}")

    return is_finetuned_model, model_config_path, output_image_size, normalization_config


def load_embedding_model(model_name, is_finetuned_model, model_config_path, device,
                         backbone_prefix="best"):
    """Loads and returns a model ready for embedding generation."""
    print_function_banner("Building Model")
    model = None
    if not is_finetuned_model:
        if model_name.strip().lower() == "miew":
            print("Initializing MiewID backbone...")
            model_tag = "conservationxlabs/miewid-msv2"
            config = transformers.AutoConfig.from_pretrained(
                model_tag, trust_remote_code=True, revision=constants.MIEWID_REVISION)
            with transformers.modeling_utils.no_init_weights():
                model = transformers.AutoModel.from_config(
                    config, trust_remote_code=True, code_revision=constants.MIEWID_REVISION)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state_dict = transformers.AutoModel.from_pretrained(
                    model_tag, trust_remote_code=True, low_cpu_mem_usage=False,
                    revision=constants.MIEWID_REVISION).state_dict()
                model.load_state_dict(state_dict, assign=True)
            model = model.to(device)
            model.eval()
        elif model_name.strip().lower() == "mega":
            print("Initializing MegaDescriptor backbone...")
            model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
            model = model.to(device)
            model.eval()
        elif model_name.strip().lower() == "vit":
            print("Initializing base ViT backbone (google/vit-large-patch16-384)...")
            # Instantiate ViTBackbone directly — no projection head, so the untrained MLP
            # used during fine-tuning is never attached and cannot affect the embeddings.
            model = src.models.vit_backbone.ViTBackbone(config={
                "pretrained_model_name_or_path": "google/vit-large-patch16-384",
                "output_dim": 1024,
                "in_channels": 3,
            })
            model = model.to(device)
            model.eval()
    else:
        print(f"Loading fine-tuned model '{model_name}'...")
        projection_head, model_backbone = load_model(model_name, backbone_prefix=backbone_prefix)
        projection_head.eval()
        model_backbone.eval()
        model = torch.nn.Sequential(model_backbone, projection_head).to(device)

    return model


def resolve_resume_dir(resume_path):
    """Resolve a resume path given absolutely, relative to the cwd, or under the trained-models root."""
    if not os.path.isabs(resume_path) and not os.path.exists(resume_path):
        resolved_path = os.path.join(constants.TRAINED_MODELS_DIR, resume_path)
        if os.path.exists(resolved_path):
            return resolved_path
        # Also try without the 'trained-models' prefix if the user included it manually
        # but it is already part of the constants path.
        if resume_path.startswith("trained-models/"):
            sub_path = resume_path.replace("trained-models/", "", 1)
            resolved_path = os.path.join(constants.TRAINED_MODELS_DIR, sub_path)
            if os.path.exists(resolved_path):
                return resolved_path
    return resume_path


def check_resume_checkpoint(resume_path):
    """Return the resolved resume directory, raising FileNotFoundError unless it holds loadable weights."""
    resume_path = resolve_resume_dir(resume_path)
    if not os.path.isdir(resume_path):
        raise FileNotFoundError(f"Resume checkpoint directory not found: {resume_path}")
    if not os.path.isfile(os.path.join(resume_path, "best-backbone-state.pt")) and \
            not os.path.isdir(os.path.join(resume_path, "best-backbone-lora")):
        raise FileNotFoundError(f"No backbone weights (best-backbone-state.pt or "
                                f"best-backbone-lora) found in {resume_path}")
    return resume_path


def load_model_for_resume(resume_path, base_backbone_model, projection_head):
    """Loads model weights from a checkpoint directory for resuming training."""
    resume_path = check_resume_checkpoint(resume_path)

    device = next(projection_head.parameters()).device
    print(f"Resuming training from checkpoint: {resume_path}")

    lora_adapter_path = os.path.join(resume_path, "best-backbone-lora")
    frozen_state_dict_path = os.path.join(resume_path, "best-backbone-state.pt")

    if os.path.isfile(frozen_state_dict_path):
        print(f"Found and loading backbone state_dict from {frozen_state_dict_path}")
        target_model = base_backbone_model
        if isinstance(base_backbone_model, peft.PeftModel):
            target_model = base_backbone_model.get_base_model()
        target_model.load_state_dict(
            torch.load(frozen_state_dict_path, map_location=device, weights_only=True))

    if os.path.isdir(lora_adapter_path):
        if not isinstance(base_backbone_model, peft.PeftModel):
            raise ValueError(f"{lora_adapter_path} holds LoRA adapters but the configured backbone "
                             "has no LoRA; resume with the config the model was trained with.")
        print(f"Found and loading LoRA adapters from {lora_adapter_path}")
        base_backbone_model.load_adapter(lora_adapter_path, adapter_name="default")

    # Try standard projection head or ArcFace head name.
    proj_head_path = os.path.join(resume_path, "best-projection-head.pt")
    if not os.path.exists(proj_head_path):
        proj_head_path = os.path.join(resume_path, "best-arcface-head.pt")
    if not os.path.exists(proj_head_path):
        raise FileNotFoundError(f"No projection or ArcFace head weights found in {resume_path}")

    projection_head.load_state_dict(torch.load(proj_head_path, map_location=device, weights_only=True))
    print(f"Loaded projection head weights from {proj_head_path}")

def enable_gradient_checkpointing(trainable_backbone, use_grad_checkpoint=False):
    """Enables gradient checkpointing on the given backbone model if configured."""
    if not use_grad_checkpoint:
        print("Gradient checkpointing is disabled via config.")
        return

    checkpointable_model_part = None
    if isinstance(trainable_backbone, peft.PeftModel):
        checkpointable_model_part = trainable_backbone
        print("Attempting to enable gradient checkpointing on PeftModel instance.")
    elif hasattr(trainable_backbone, 'backbone') and \
            hasattr(trainable_backbone.backbone, 'gradient_checkpointing_enable'):
        checkpointable_model_part = trainable_backbone.backbone
        print("Attempting to enable gradient checkpointing on underlying model.")
    elif hasattr(trainable_backbone, 'gradient_checkpointing_enable'):
        checkpointable_model_part = trainable_backbone
        print("Attempting to enable gradient checkpointing directly on instance.")

    if checkpointable_model_part:
        try:
            checkpointable_model_part.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("Successfully enabled gradient checkpointing with use_reentrant=False.")
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Failed to enable gradient checkpointing with use_reentrant=False: {e}")
            try:
                checkpointable_model_part.gradient_checkpointing_enable()
                print("Successfully enabled gradient checkpointing (standard).")
            except Exception as e2: # pylint: disable=broad-exception-caught
                print(f"Failed to enable gradient checkpointing (standard): {e2}")
    else:
        print("Warning: 'grad_checkpoint' is True, but no suitable model found.")

# --- Data & Metadata Parsing ---
# TODO: Transition to pathlib.Path for modern path handling.
def parse_time_from_name(name: str) -> Optional[int]:
    """Parses date and time digits from a media name and returns a unix timestamp."""
    match = DATE_RE.search(name)
    if not match:
        return None

    numerical = match.group(1)

    try:
        year = int(numerical[0:2])
        month = int(numerical[2:4])
        day = int(numerical[4:6])
    except ValueError:
        return None

    full_year = 2000 + year if year <= 50 else 1900 + year
    if month > 12:
        month_index = month - 1
        full_year += month_index // 12
        month = (month_index % 12) + 1

    date_str = f"{month:02d}-{day:02d}-{full_year}"
    return convert_date_to_unix(date_str)

def parse_media_name(media_name: str) -> Dict[str, Any]:
    """Parses a media filename for structured metadata such as location, time, sex, and length."""
    metadata_dict = {
        "name": media_name,
        "media_name": None,
        "location": None,
        "time": None,
        "length": None,
        "sex": None,
        "extension": None,
    }

    match = MEDIA_NAME_RE.search(media_name)
    if match is not None:
        metadata_dict["media_name"] = match.group(1) + match.group(2)
        metadata_dict["location"] = match.group(1)
        numerical = match.group(2)
        metadata_dict["sex"] = match.group(3)
        metadata_dict["length"] = match.group(4)
        metadata_dict["extension"] = match.group(5)
    else:
        return metadata_dict

    # Robustly parse the time using existing helper.
    unix_time = parse_time_from_name(numerical)
    if unix_time:
        metadata_dict["time"] = datetime.datetime.fromtimestamp(unix_time)

    return metadata_dict

def get_all_metadata(path):
    """Retrieves JSON format and stream metadata from a media file using ffprobe."""
    args = [
        _get_path('ffprobe'), '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', path
    ]
    stdout, _ = _run(args, 'ffprobe')
    return json.loads(stdout)

def get_key_metadata(path: str, override_with_shark_data: bool = True) -> Tuple[Dict, Dict]:
    """Extracts key metadata (location, time, shark tags) from an image file."""
    shark_tag_key = 'shark-clipper'
    data = {}
    all_metadata = get_all_metadata(path)

    if 'format' in all_metadata:
        tags = all_metadata['format'].get('tags', {})

        if 'location' in tags:
            text_loc = tags['location'].strip()
            data['location_raw'] = text_loc
            match = LAT_LONG_RE.search(text_loc)
            if match is not None:
                data['location'] = {'latitude': match.group(1), 'longitude': match.group(2)}

        if 'creation_time' in tags:
            text_time = tags['creation_time'].strip()
            data['start_time_raw'] = text_time
            try:
                parsed_time = datetime.datetime.fromisoformat(text_time)
                data['start_time_unix'] = int(parsed_time.timestamp())
            except Exception as ex: # pylint: disable=broad-exception-caught
                print(f"Invalid creation time in metadata: {text_time} | Exception: {ex}")

        if override_with_shark_data and shark_tag_key in tags:
            shark_json = tags[shark_tag_key].strip()
            try:
                data.update(json.loads(shark_json))
            except Exception as ex: # pylint: disable=broad-exception-caught
                print(f"Invalid (json) shark data in metadata. | Exception: {ex}")

    return data, all_metadata

def get_annotation_paths(annotations_version=None):
    """Returns training and validation annotation paths for a given or default annotations version."""
    if annotations_version is None:
        annotations_version = "annotations"

    print(f"Using annotations version: {annotations_version}")
    version_dir = os.path.join(constants.ANNOTATIONS_DIR, annotations_version)
    train_path = os.path.join(version_dir, constants.TRAIN_ANNOTATIONS)
    valid_path = os.path.join(version_dir, constants.VALID_ANNOTATIONS)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train annotations not found at {train_path}. "
                                f"Please ensure an annotations version is specified or created.")
    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"Valid annotations not found at {valid_path}. "
                                f"Please ensure an annotations version is specified or created.")

    return train_path, valid_path

# --- Data Transfer & Normalization ---

def convert_date_to_unix(date_input: Union[str, int, float]) -> Union[int, float]:
    """Converts a date string in 'mm-dd-yyyy' format to a unix timestamp."""
    if isinstance(date_input, (int, float)):
        return date_input

    if isinstance(date_input, str):
        try:
            dt = datetime.datetime.strptime(date_input, "%m-%d-%Y")
            return int(dt.timestamp())
        except ValueError:
            pass

    return date_input

def clean_input_bool(input_val: Any) -> Optional[bool]:
    """Cleans boolean input strings or types, returning a bool or None."""
    if input_val is None:
        return None
    if isinstance(input_val, bool):
        return input_val
    if isinstance(input_val, str):
        lower_val = input_val.lower()
        if lower_val == 'true':
            return True
        if lower_val == 'false':
            return False
    return None

def normalize_list_params(kwargs: Dict[str, Any],
                          list_params: Sequence[str],
                          splitter: str = ",") -> Dict[str, Any]:
    """Returns a copy of kwargs with specified list-type parameters normalized."""
    normalized = kwargs.copy()
    for key in list_params:
        value = normalized.get(key)
        final_val = []

        if value is None:
            final_val = []
        elif isinstance(value, str):
            if value.strip():
                # Efficient list comprehension instead of manual splitting + cleaning
                final_val = [item.strip() for item in value.split(splitter) if item.strip()]
        elif isinstance(value, list):
            final_val = [item.strip() if isinstance(item, str) else item for item in value]
        elif isinstance(value, (int, float)):
            final_val = [value]
        else:
            final_val = [value] if value else []

        normalized[key] = final_val if final_val else None

    return normalized

def normalize_scalar_params(kwargs: Dict[str, Any],
                            scalar_params: Sequence[str]) -> Dict[str, Any]:
    """Returns a copy of kwargs with specified scalar parameters normalized."""
    normalized = kwargs.copy()
    for key in scalar_params:
        val = normalized.get(key)
        if key == "allow_unknown":
            normalized[key] = clean_input_bool(val)
        else:
            normalized[key] = None if val == "" else val
    return normalized

# --- Logging & Notifications ---

def send_email(recipient_email, subject, body):
    """Sends an email using the Gmail API with OAuth 2.0 or environment credentials."""
    # pylint: disable=invalid-name
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    creds = None

    client_id = os.getenv("GMAIL_CLIENT_ID")
    client_secret = os.getenv("GMAIL_CLIENT_SECRET")
    refresh_token = os.getenv("GMAIL_REFRESH_TOKEN")

    if client_id and client_secret and refresh_token:
        # Build credentials directly from environment variables.
        creds = google.oauth2.credentials.Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES,
        )
        try:
            creds.refresh(google.auth.transport.requests.Request())
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Warning: Failed to refresh Gmail credentials from env vars: {e}")
            return None
    else:
        # Fall back to file-based credential storage.
        if os.path.exists('token.json'):
            creds = google.oauth2.credentials.Credentials.from_authorized_user_file(
                'token.json', SCOPES)
        else:
            print("Warning: 'token.json' not found and Gmail env vars not set.")

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(google.auth.transport.requests.Request())
            else:
                if not os.path.exists('credentials.json'):
                    print("Warning: 'credentials.json' not found. Skipping email notification.")
                    return None
                flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)

            with open('token.json', 'w', encoding='utf-8') as token:
                token.write(creds.to_json())

    try:
        service = googleapiclient.discovery.build('gmail', 'v1', credentials=creds)

        message = {
            'raw': base64.urlsafe_b64encode(
                f"To: {recipient_email}\r\n"
                f"Subject: {subject}\r\n"
                f"Content-Type: text/plain; charset=utf-8\r\n"  # Specify content type
                f"\r\n"  # Blank line to separate headers from body
                f"{body}"
                .encode('utf-8')  # Encode the entire message (including headers)
            ).decode()
        }

        # pylint: disable=no-member
        message = (service.users().messages().send(userId='me', body=message).execute())
        print(f"Sent email to {recipient_email} ID: {message['id']}")
        return message
    except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"An error occurred: {error}")
        return None

def record_embeddings_metadata(
        embeddings_name: str, start_time: float, end_time: float,
        num_images: int, output_dir=None, generation_id=None,
        snapshot_id=None):
    """Records metadata about an embeddings update alongside its artifacts."""
    update_duration = end_time - start_time
    start_time_readable = datetime.datetime.fromtimestamp(start_time).strftime(
        '%Y-%m-%d %H:%M:%S')

    hours, remainder = divmod(update_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_parts = []
    if hours > 0:
        duration_parts.append(f"{int(hours)} hours")
    if minutes > 0:
        duration_parts.append(f"{int(minutes)} minutes")
    duration_parts.append(f"{seconds:.2f} seconds")
    update_time_readable = ", ".join(duration_parts)

    save_dir = output_dir or os.path.join(constants.EMBEDDINGS_DIR, embeddings_name)
    os.makedirs(save_dir, exist_ok=True)

    metadata = {
        "num_images": num_images,
        "start_time": start_time,
        "start_time_readable": start_time_readable,
        "update_time": update_duration,
        "update_time_readable": update_time_readable,
    }
    if generation_id is not None:
        metadata["generation_id"] = generation_id
        metadata["snapshot_id"] = snapshot_id

    metadata_path = os.path.join(save_dir, "embeddings_metadata.json")
    with open(metadata_path, "w", encoding='utf-8') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)
    return metadata

def log_training_history_csv(training_history, trained_models_dir, epoch, loss, metrics_dict=None):
    """Logs epoch training history and metrics to a CSV file."""
    metrics_dict = metrics_dict or {}
    row = [epoch, loss]
    if metrics_dict:
        row.extend(metrics_dict.values())
    elif training_history:
        max_cols = max(len(r) for r in training_history)
        num_metrics = max_cols - 2
        if num_metrics > 0:
            row.extend([float('nan')] * num_metrics)
    training_history.append(row)

    os.makedirs(trained_models_dir, exist_ok=True)
    write_csv_file(os.path.join(trained_models_dir, "training_history.csv"),
                              training_history)

def send_completion_notice(recipient_email, start_time, model_save_dir_path, subject_prefix=""):
    """Sends an email notification when training completes."""
    end_time = time.time()
    duration_str = f"{(end_time - start_time) / 3600:.2f} hours"
    run_id = os.path.basename(os.path.dirname(model_save_dir_path))
    config_id = os.path.basename(model_save_dir_path)

    subject = f"Training Complete: {run_id}/{config_id}"
    if subject_prefix:
        subject = f"Training Complete ({subject_prefix}): {run_id}/{config_id}"
    body = (f"Training completed.\n\nRun Identifier: {run_id}\n"
            f"Config Key: {config_id}\nTraining duration: {duration_str}")
    send_email(recipient_email, subject, body)

def get_notification_recipients():
    """Returns the configured notification subscriber list from src.config.config."""
    try:
        import src.config.config  # pylint: disable=import-outside-toplevel
        return list(getattr(src.config.config, "NOTIFICATION_EMAILS", []) or [])
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: could not read NOTIFICATION_EMAILS: {e}")
        return []


def notify_completion(start_time, save_dir_path, no_email=False, subject_prefix=""):
    """Notifies all configured subscribers that a run has finished."""
    if no_email:
        print("Completion notification suppressed (--no-email).")
        return

    recipients = get_notification_recipients()
    if not recipients:
        print("No notification recipients configured; skipping completion email.")
        return

    for recipient in recipients:
        try:
            send_completion_notice(recipient, start_time, save_dir_path, subject_prefix)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Warning: could not notify {recipient}: {e}")

# --- UI & Formatting Helpers ---
def print_script_banner(script_name, args=None):
    """Prints a standardized banner showing the script name and arguments."""
    print("-" * 60)
    print(f"Running: {script_name}")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args:
        print("Arguments:")
        if hasattr(args, '__dict__'):
            dct = vars(args)
        elif isinstance(args, dict):
            dct = args
        else:
            dct = {"args": str(args)}

        for key, value in dct.items():
            print(f"  {key}: {value}")
    print("-" * 60)

def print_key_value_section(title, data):
    """Prints a formatted section header and aligned key-value pairs."""
    print(f"\n=== {title} ===")
    if not data:
        print("No data.")
        return
    mx = max(len(str(k)) for k in data.keys())
    for key, value in data.items():
        print(f"{str(key):<{mx}} : {value}")

def print_table(headers, rows):
    """Prints a formatted text table with aligned column headers and data rows."""
    if not rows:
        print("No data to display.")
        return
    wds = [len(str(h)) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if i < len(wds):
                wds[i] = max(wds[i], len(str(val)))
            else:
                wds.append(len(str(val)))
    fmt = "  ".join([f"{{:<{w}}}" for w in wds])
    sep = "-" * (sum(wds) + 2 * (len(wds) - 1))
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        row_str = [str(val) for val in row]
        while len(row_str) < len(headers):
            row_str.append("")
        print(fmt.format(*row_str))
    print(sep)

def print_function_banner(function_name, description=None):
    """Prints an execution banner for a function with an optional description."""
    print(f"\n=== Running: {function_name} ===")
    if description:
        print(f"{description}")
    print("-" * 40)

# --- Internal Helpers ---

def seed_everything(seed=None):
    """Seeds random number generators and optionally pins deterministic algorithms if enabled."""
    if seed is None:
        seed = constants.RANDOM_SEED

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if constants.STRICT_DETERMINISM:
        _enable_strict_determinism(seed)


def _enable_strict_determinism(seed):
    """Enables deterministic PyTorch operations and cuDNN settings with warnings on non-deterministic ops.
    Sets PYTHONHASHSEED so spawned DataLoader workers inherit deterministic string hashing."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if os.environ.get("_SHARK_HASHSEED_CHECKED") != "1":
        os.environ["_SHARK_HASHSEED_CHECKED"] = "1"
        if sys.flags.hash_randomization:
            print("Note: PYTHONHASHSEED was not set before interpreter start, so "
                  "this process keeps randomized string hashing. Export "
                  f"PYTHONHASHSEED={seed} before launching for full strictness; "
                  "spawned DataLoader workers will inherit it either way.")

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def timeit(func):
    """Decorator that measures and prints execution time of the wrapped function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(f"'{func.__qualname__}' executed in {time.time() - start:.4f}s")
        return res
    return wrapper

def _run(args, name):
    """Executes a subprocess command, returning its stdout and stderr or raising on non-zero exit."""
    print(shlex.join(args))
    res = subprocess.run(args, capture_output=True, check=False)
    if res.returncode != 0:
        out = res.stdout.decode('utf-8', errors='replace')
        err = res.stderr.decode('utf-8', errors='replace')
        raise ValueError(f"{name} did not exit cleanly.\n--- stdout ---\n{out}\n"
                         f"---\n--- stderr ---{err}\n---")
    return res.stdout, res.stderr

def _get_path(name):
    """Finds the absolute executable path of a binary on the system PATH, or None."""
    return shutil.which(name)

def save_config(model_save_dir: str, config: dict):
    """Serializes the configuration and copies config source code into the model directory."""
    import copy # pylint: disable=import-outside-toplevel
    shutil.copyfile(os.path.join(constants.BACKEND_DIR, "config", "config.py"),
                    os.path.join(model_save_dir, "config_source.py"))

    string_config = copy.deepcopy(config)
    def recursive_to_str(_config):
        for key in _config:
            if key in ["constructor", "evaluator"] and _config[key] is not None:
                _config[key] = _config[key].__name__
            elif isinstance(_config[key], dict):
                recursive_to_str(_config[key])
    recursive_to_str(string_config)

    proj_head = string_config.get("model", {}).get("projection_head", {})
    if proj_head and proj_head.get("constructor") in ["Identity", "torch.nn.Identity"]:
        proj_head["name"] = "Identity"

    with open(os.path.join(model_save_dir, "config.json"), "w", encoding='utf-8') as f:
        json.dump(string_config, f, indent=4, sort_keys=True)

# --- Data Integrity & Verification ---
# TODO: Transition to pathlib.Path for modern path handling.
def verify_train_test_split_integrity(train_dataset: Any,
                                      valid_dataset: Any,
                                      dataset_name: str = "Dataset") -> Dict[str, int]:
    """Verifies that train and validation datasets share no sample IDs and reports overlap metrics."""
    train_ids = set(train_dataset.get_ids())
    test_ids = set(valid_dataset.get_ids())
    train_labels = train_dataset.get_labels()
    test_labels = valid_dataset.get_labels()

    tr_set = set(train_labels.cpu().numpy() if isinstance(train_labels, torch.Tensor)
                 else train_labels)
    te_set = set(test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor)
                 else test_labels)

    overlap = train_ids.intersection(test_ids)
    if len(overlap) > 0:
        print(f"Leakage in {dataset_name}! First 10 overlap IDs: {list(overlap)[:10]}")
        raise ValueError(f"CRITICAL: Leakage in {dataset_name}! IDs: {list(overlap)[:10]}")

    label_overlap = tr_set.intersection(te_set)
    if len(label_overlap) > 0:
        print(f"{len(label_overlap)} sharks found in both sets (identification expected).")

    print(f"Data leakage check passed for {dataset_name}.")

    return {
        "overlapping_ids": len(overlap),
        "overlapping_labels": len(label_overlap),
        "train_ids_count": len(train_ids),
        "test_ids_count": len(test_ids),
        "train_labels_count": len(tr_set),
        "test_labels_count": len(te_set)
    }

def verify_dataset_split_integrity(dataset: Any,
                                   split: str,
                                   included_ids: Sequence[int],
                                   train_ids_source: Sequence[int],
                                   test_ids_source: Sequence[int],
                                   dataset_name: str = "Dataset") -> None:
    """Verifies that a dataset split contains only IDs belonging to its designated partition."""
    # pylint: disable=unused-argument
    inc = set(included_ids)
    tr = set(train_ids_source)
    te = set(test_ids_source)

    if split == 'train' and inc.intersection(te):
        overlap = list(inc.intersection(te))[:10]
        print(f"Test IDs found in train split ({dataset_name}): {overlap}")
        raise ValueError(f"CRITICAL: Leakage! Test IDs in train: {overlap}")

    if split == 'test' and inc.intersection(tr):
        overlap = list(inc.intersection(tr))[:10]
        print(f"Train IDs found in test split ({dataset_name}): {overlap}")
        raise ValueError(f"CRITICAL: Leakage! Train IDs in test: {overlap}")

    print(f"Split integrity verified for {dataset_name} ({split} split).")


# --- File I/O Utilities ---

def load_csv_file(path, delimiter=','):
    """Return a CSV file's rows, or None after reporting a missing file."""
    try:
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimiter)
            return list(reader)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None


def load_json_file(path):
    """Return a JSON file's parsed contents, or None after reporting a missing file."""
    try:
        with open(path, "r", encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None


def load_psl_file(path, dtype=str):
    """Return a tab-separated PSL file's non-empty rows cast to dtype, or None after reporting a missing file."""
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line == '':
                    continue

                data.append(list(map(dtype, line.split("\t"))))
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None


def load_txt_file(path):
    """Return a text file's contents, or None after reporting a missing file."""
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None


def write_csv_file(path, data, delimiter=','):
    """Write rows to a CSV file."""
    with open(path, 'w', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerows(data)



def write_json_file(path, data, indent=4):
    """Write data to a JSON file, indented unless indent is None."""
    with open(path, "w", encoding='utf-8') as file:
        if indent is None:
            json.dump(data, file)
        else:
            json.dump(data, file, indent=indent)



def write_psl_file(path, data):
    """Write rows to a tab-separated PSL file."""
    with open(path, 'w', encoding='utf-8') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")
