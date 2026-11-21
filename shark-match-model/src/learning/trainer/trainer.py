# pylint: disable=no-member, too-many-arguments, too-many-locals, too-many-statements, too-many-instance-attributes, broad-exception-caught, import-error, duplicate-code, wrong-import-position, line-too-long, no-name-in-module, import-outside-toplevel, ungrouped-imports, too-many-nested-blocks, too-many-branches, unused-variable, bare-except, unused-import, not-callable, unnecessary-pass, too-few-public-methods, too-many-positional-arguments, protected-access
"""Unified RobustTrainer for contrastive and classification-based metric learning."""

import abc
import argparse
import copy
import datetime
import json
import os
import shutil
import socket
import sys
import time
from typing import Any, Dict, List, Tuple

# Must be set before torch is imported to take effect (cdist backward in triplet loss on MPS).
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import peft
import sklearn.preprocessing
import torch
import torch.utils.data
import torch.utils.tensorboard
import torchvision.utils
import tqdm

from src.core import constants
import src.config.config
import src.config.config_schema
import src.individual_dataset
import src.learning.augmented_dataset
import src.learning.pk_sampler
from src.core import utils

# Repo-wide defaults for epoch checkpoint selection and early-stopping patience signals.
CHECKPOINT_METRIC = constants.CHECKPOINT_METRIC
EARLY_STOPPING_METRIC = constants.EARLY_STOPPING_METRIC


def metric_learning_collate(batch):
    """Collates single-view or multi-view metric learning samples into stacked 4-D image tensors."""
    all_images = []
    all_labels = []
    for images, labels in batch:
        if isinstance(images, list):
            # Multi-view: one label covers every view of the image.
            all_images.extend(images)
            all_labels.extend([labels] * len(images))
        else:
            all_images.append(images)
            all_labels.append(labels)
    return torch.stack(all_images), torch.stack(all_labels)


def print_config_summary(config_key: str, config: src.config.config_schema.FullConfig):
    """Prints a summary of the configuration for the current run."""
    utils.print_function_banner(f"Training Config: {config_key}")

    # Compute effective batch sizes including augmented copies and unaugmented originals.
    views_per_image = config.training.num_augmentations + (
        1 if config.training.include_original else 0
    )
    effective_images_per_step = config.training.batch_size * views_per_image
    effective_grad_batch = (
        effective_images_per_step * config.training.gradient_accumulation_steps
    )

    # Resolve effective num_workers (mirrors _build_dataloader_kwargs logic).
    effective_num_workers = config.training.num_workers
    if effective_num_workers is None:
        effective_num_workers = src.config.config.NUM_WORKERS

    # Prepare key details for display.
    config_details = {
        "Batch Size": config.training.batch_size,
        "Accumulation Steps": config.training.gradient_accumulation_steps,
        "Views/Image": f"{views_per_image} ({config.training.num_augmentations} aug"
                       f" + {'original' if config.training.include_original else 'no original'})",
        "Effective Images/Step": effective_images_per_step,
        "Effective Grad Batch": effective_grad_batch,
        "Epochs": config.training.epochs,
        "Learning Rate": config.training.optimizer.learning_rate,
        "Loss Function": config.training.loss.name,
        "DataLoader Workers": effective_num_workers,
    }

    # P×K sampler details.
    if config.training.identities_per_batch is not None:
        config_details["P (Identities/Batch)"] = config.training.identities_per_batch
        config_details["K (Samples/Identity)"] = config.training.samples_per_identity

    # Specific check for normalization.
    if config.training.loss.options and "normalize_embeddings" in config.training.loss.options:
        config_details["Normalize Embeddings"] = config.training.loss.options["normalize_embeddings"]

    # Add LoRA details if present.
    if config.lora:
        config_details["LoRA Rank (r)"] = config.lora.options.get("r", "N/A")
        config_details["LoRA Alpha"] = config.lora.options.get("lora_alpha", "N/A")

    utils.print_key_value_section("Configuration Details", config_details)
    print("-" * 40 + "\n")


def perform_dry_run_checks(config_key: str, model_save_dir: str, args: argparse.Namespace):
    """Print path verification for a dry run and return False when a required path is missing."""
    path_checks = {}
    # Verify annotations.
    try:
        train_ann, valid_ann = utils.get_annotation_paths(args.annotations_version)
        if hasattr(args, 'split_strategy') and args.split_strategy == 'none':
            version = args.annotations_version if args.annotations_version else "annotations"
            train_ann = os.path.join(constants.ANNOTATIONS_DIR, version, constants.FULL_ANNOTATIONS)
            valid_ann = None

        path_checks["Train Annotations"] = f"{train_ann} ({'OK' if os.path.exists(train_ann) else 'MISSING'})"
        if valid_ann:
            path_checks["Valid Annotations"] = f"{valid_ann} ({'OK' if os.path.exists(valid_ann) else 'MISSING'})"
    except FileNotFoundError as e:
        path_checks["Annotations"] = f"ERROR: {e}"

    path_checks["Images Directory"] = f"{constants.IMAGES_DIR} ({'OK' if os.path.exists(constants.IMAGES_DIR) else 'MISSING'})"
    path_checks["Model Save Directory"] = model_save_dir
    if saved_weights(model_save_dir) and not args.overwrite:
        path_checks["Model Save Directory"] += " (holds trained weights; a real run needs --overwrite)"

    if args.resume_from_model:
        try:
            path_checks["Resume From"] = f"{utils.check_resume_checkpoint(args.resume_from_model)} (OK)"
        except FileNotFoundError as error:
            path_checks["Resume From"] = f"MISSING: {error}"

    utils.print_key_value_section("Path Verification", path_checks)
    print(f"\nDry run complete for config '{config_key}'. Skipping training.\n")
    return not any("MISSING" in str(check) or "ERROR" in str(check)
                   for check in path_checks.values())


def setup_augmentations(raw_config):
    """Sets up augmentations based on the raw config strings."""
    utils.print_function_banner("Setting up Augmentations")

    # Load custom augmentations from config or fall back to global defaults.
    aug_list = raw_config.get("augmentations", src.config.config.AUGMENTATIONS)
    print(f"Using {len(aug_list)} augmentations.")

    augmentations = []
    for aug_name, aug_options in aug_list:
        if aug_name in src.config.config.AUGMENTATION_CONSTRUCTORS:
            constructor = src.config.config.AUGMENTATION_CONSTRUCTORS[aug_name]
            augmentations.append(constructor(**aug_options))
            print(f"  Added: {aug_name} with {aug_options}")
        else:
            print(f"  Warning: Augmentation '{aug_name}' not found. Skipping.")

    return augmentations


def setup_datasets_and_augmentations(config, raw_config, args):
    """Sets up IndividualDataset objects and augmentations."""
    utils.print_function_banner("Loading Datasets")

    train_path, valid_path = utils.get_annotation_paths(args.annotations_version)

    if hasattr(args, 'split_strategy') and args.split_strategy == 'none':
        version = args.annotations_version if args.annotations_version else "annotations"
        train_path = os.path.join(constants.ANNOTATIONS_DIR, version, constants.FULL_ANNOTATIONS)
        valid_path = None
        print("Split strategy is 'none'. Using full dataset for training without validation holdout.")

    train_base_ds = src.individual_dataset.IndividualDataset(
        constants.IMAGES_DIR, train_path,
        train_annotations_path=train_path, # Explicitly pass resolved paths
        test_annotations_path=valid_path,
        output_image_size=config.data.image_size,
        normalization=config.data.normalization
    )
    if len(train_base_ds) == 0:
        raise ValueError(f"No training samples found in {train_path}. For annotations created with "
                         "--no-test, train with --split-strategy none.")

    valid_base_ds = None
    if valid_path and os.path.exists(valid_path):
        valid_base_ds = src.individual_dataset.IndividualDataset(
            constants.IMAGES_DIR, valid_path,
            train_annotations_path=train_path,
            test_annotations_path=valid_path,
            output_image_size=config.data.image_size,
            normalization=config.data.normalization
        )

    train_base_ds.check_images_readable()
    if valid_base_ds is not None:
        valid_base_ds.check_images_readable()

    # Setup augmentations.
    augmentations = setup_augmentations(raw_config)

    # Optional RAM cache: pre-load images into memory to eliminate per-sample disk I/O.
    # train_base_ds.load_images()
    # if valid_base_ds is not None:
    #     valid_base_ds.load_images()

    return train_base_ds, valid_base_ds, augmentations


def _build_dataloader_kwargs(config):
    """Builds common DataLoader keyword arguments from config."""
    # Resolve num_workers: per-config value takes precedence; None falls back to the
    # global constant in src.config.config so the caller controls it from one place.
    num_workers = config.training.num_workers
    if num_workers is None:
        num_workers = src.config.config.NUM_WORKERS
    kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": metric_learning_collate,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        # Forked workers freeze inside torch operations on Linux; spawn is already the macOS default.
        kwargs["multiprocessing_context"] = "spawn"

    # Seed DataLoader generator directly to decouple worker augmentations from model RNG draws.
    if constants.DETERMINISTIC_DATALOADER:
        generator = torch.Generator()
        generator.manual_seed(constants.RANDOM_SEED)
        kwargs["generator"] = generator

    return kwargs


def resolve_pk_batch_size(config):
    """Forces batch_size to P*K for metric learning so batches match the sampler identity structure."""
    if config.training.loss.name == "arcface":
        return
    p = config.training.identities_per_batch
    k = config.training.samples_per_identity
    if p is None or k is None:
        return
    pk_batch_size = p * k
    if config.training.batch_size != pk_batch_size:
        print(f"\nWARNING: Config batch_size ({config.training.batch_size}) "
              f"does not match PxK ({p}x{k} = {pk_batch_size}). "
              f"Overriding batch_size to {pk_batch_size} to preserve metric learning batch structure.\n")
        config.training.batch_size = pk_batch_size


def encode_labels(label_encoder, labels):
    """Encodes labels with a fitted encoder, giving unseen identities fresh codes above the trained class range."""
    codes = {label: code for code, label in enumerate(label_encoder.classes_.tolist())}
    first_unseen_code = len(codes)
    for offset, label in enumerate(sorted(set(labels) - codes.keys())):
        codes[label] = first_unseen_code + offset
    return torch.tensor([codes[label] for label in labels], dtype=torch.long)


def _build_train_dataset(config, train_base_ds, augmentations):
    """Wraps the base dataset in AugmentedDataset using the configured view settings."""
    return src.learning.augmented_dataset.AugmentedDataset(
        train_base_ds, augmentations,
        num_augmentations=config.training.num_augmentations,
        include_original=config.training.include_original,
        rebalance_augmentations=config.training.rebalance_augmentations
    )


def build_model_and_strategy(config, device, train_base_ds, valid_base_ds, augmentations):
    """Builds the model, strategy, and data loader based on config."""
    utils.print_function_banner("Building Model")
    backbone_cls = config.model.backbone.constructor
    backbone = backbone_cls(**config.model.backbone.options).to(device)

    utils.print_function_banner("Attempting Model Optimizations")
    if config.lora:
        lora_conf = config.lora.constructor(**config.lora.options)
        backbone = peft.get_peft_model(backbone, lora_conf).to(device)
        if config.training.grad_checkpoint:
            utils.enable_gradient_checkpointing(backbone, True)
    else:
        print("No optimizations (LoRA/Checkpointing) specified in config.")

    is_arcface = config.training.loss.name == "arcface"

    resolve_pk_batch_size(config)

    loader_kwargs = _build_dataloader_kwargs(config)

    if is_arcface:
        strategy = SupervisedStrategy()
        # Label Encoding.
        le = sklearn.preprocessing.LabelEncoder()
        train_labels = train_base_ds.get_labels().tolist()
        le.fit(train_labels)
        train_base_ds.labels = torch.tensor(le.transform(train_labels), dtype=torch.long)
        if valid_base_ds is not None:
            valid_base_ds.labels = encode_labels(le, valid_base_ds.get_labels().tolist())

        # Dynamic margins.
        if "margins" in config.model.projection_head.options:
            print("Computing Dynamic Margins based on class frequency...")
            class_counts = train_base_ds.get_class_counts()
            class_counts = class_counts.float() + 1e-6
            margins = class_counts.pow(-0.25)
            margins = margins * (0.5 / margins.mean())
            config.model.projection_head.options["margins"] = margins
            print(f"Dynamic margins computed. Mean: {margins.mean():.4f}")

        # Built after label encoding: AugmentedDataset reads the dataset's
        # labels when rebalance_augmentations is enabled.
        train_dataset = _build_train_dataset(config, train_base_ds, augmentations)
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        head_cls = config.model.projection_head.constructor
        head = head_cls(in_features=backbone.get_output_dim(), out_features=len(le.classes_), **config.model.projection_head.options).to(device)
        model = ArcFaceModelWrapper(backbone, head, torch.nn.CrossEntropyLoss())
    else:
        strategy = ContrastiveStrategy(loss_fn=config.training.loss.constructor(**config.training.loss.options))

        train_dataset = _build_train_dataset(config, train_base_ds, augmentations)

        # P×K sampler for metric learning.
        p = config.training.identities_per_batch
        k = config.training.samples_per_identity
        if p is not None and k is not None:
            sampler = src.learning.pk_sampler.PKSampler(train_base_ds, p=p, k=k)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, sampler=sampler, **loader_kwargs
            )
        else:
            # Fallback to shuffle if P×K not configured.
            train_loader = torch.utils.data.DataLoader(
                train_dataset, shuffle=True, **loader_kwargs
            )

        head_cls = config.model.projection_head.constructor
        head = head_cls(input_dim=backbone.get_output_dim(), **config.model.projection_head.options).to(device)
        model = torch.nn.Sequential(backbone, head).to(device)

    print(f"Created DataLoader with {len(train_loader)} batches (batch_size={config.training.batch_size}).")
    return model, strategy, train_loader


WEIGHT_FILE_PREFIXES = ("best-", "last-", "init-")


def saved_weights(model_save_dir):
    """Return the names of trained-weight files already stored in a model directory."""
    if not os.path.isdir(model_save_dir):
        return []
    return sorted(name for name in os.listdir(model_save_dir)
                  if name.startswith(WEIGHT_FILE_PREFIXES))


PREVIOUS_RUN_RECORDS = ("training_summary.json", "training_history.csv", "logs")


def clear_previous_run(model_save_dir):
    """Delete an earlier run's weights, records and logs so an overwritten run never mixes with them."""
    for name in saved_weights(model_save_dir) + list(PREVIOUS_RUN_RECORDS):
        path = os.path.join(model_save_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)


def check_output_directories(config_keys, run_identifier, args):
    """Exit before any training when a run would replace trained weights without --overwrite."""
    if args.overwrite:
        return
    resume_dir = (os.path.realpath(utils.resolve_resume_dir(args.resume_from_model))
                  if args.resume_from_model else None)
    problems = []
    for config_key in config_keys:
        model_save_dir = os.path.join(constants.TRAINED_MODELS_DIR, run_identifier, config_key)
        if not saved_weights(model_save_dir):
            continue
        if resume_dir == os.path.realpath(model_save_dir):
            problems.append(f"Resuming from {model_save_dir} would overwrite the weights it "
                            "resumes from; resume into a new --model-name or pass --overwrite.")
        else:
            problems.append(f"Trained weights already exist in {model_save_dir}; choose a new "
                            "--model-name or pass --overwrite to replace them.")
    if problems:
        for problem in problems:
            print(f"Error: {problem}")
        sys.exit(1)


def apply_epochs_override(raw_config, epochs):
    """Return a copy of a config entry with a new epoch count and, if it matched, a rescaled T_max."""
    overridden = copy.deepcopy(raw_config)
    training = overridden["training"]
    previous_epochs = training["epochs"]
    training["epochs"] = epochs

    scheduler = training.get("scheduler")
    t_max = scheduler["options"].get("T_max") if scheduler else None
    if t_max is not None and epochs > 0:
        if t_max == previous_epochs:
            scheduler["options"]["T_max"] = epochs
            print(f"Scheduler T_max rescaled from {t_max} to {epochs} to match --epochs.")
        else:
            print(f"Scheduler T_max stays at {t_max} (it differs from the configured "
                  f"{previous_epochs} epochs).")
    return overridden


def run_single_config(config_key, run_identifier, args):
    """The top-level orchestrator for training one configuration."""
    start_time = time.time()
    raw_config = src.config.config.CONFIGS[config_key]
    if isinstance(getattr(args, "epochs", None), int):
        raw_config = apply_epochs_override(raw_config, args.epochs)

    try:
        config = utils.ensure_config_schema(raw_config)
    except Exception as e:
        print(f"Error: configuration '{config_key}' is invalid: {e}")
        sys.exit(1)

    model_save_dir = os.path.join(constants.TRAINED_MODELS_DIR, run_identifier, config_key)
    if not args.dry_run:
        os.makedirs(model_save_dir, exist_ok=True)
        utils.save_config(model_save_dir, raw_config)

    # Resolve P*K before printing, so the summary reports the batch size that
    # will actually be used rather than the pre-override config value.
    resolve_pk_batch_size(config)
    print_config_summary(config_key, config)

    if args.dry_run:
        if not perform_dry_run_checks(config_key, model_save_dir, args):
            sys.exit(1)
        return

    # Setup hardware and seeds.
    device = utils.get_torch_device()
    utils.seed_everything()

    # Load data.
    train_base_ds, valid_base_ds, augmentations = setup_datasets_and_augmentations(config, raw_config, args)

    # Build model.
    model, strategy, train_loader = build_model_and_strategy(config, device, train_base_ds, valid_base_ds, augmentations)

    # Resume Logic.
    if args.resume_from_model:
        is_arcface = config.training.loss.name == "arcface"
        r_backbone = model.backbone if is_arcface else model[0]
        r_head = model.head if is_arcface else model[1]
        utils.load_model_for_resume(args.resume_from_model, r_backbone, r_head)

    # Cleared only after resume weights are in memory, so resuming in place stays safe.
    if args.overwrite:
        clear_previous_run(model_save_dir)

    # Train.
    trainer = RobustTrainer(model, strategy, device, config, train_base_ds, valid_base_ds,
                            config_key=config_key, run_identifier=run_identifier)
    trainer.train(train_loader, model_save_dir)

    utils.notify_completion(start_time, model_save_dir,
                            no_email=getattr(args, "no_email", False))


class ArcFaceModelWrapper(torch.nn.Module):
    """Wrapper model to handle ArcFace's requirement for labels in the forward pass."""
    def __init__(self, backbone, head, loss_fn=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.loss_fn = loss_fn
        self.forward_needs_labels = True

    def forward(self, images, labels=None):
        """Forward pass returning loss, output, and auxiliary logs if labels are provided."""
        features = self.backbone(images)
        if labels is None:
            # Return normalized features in inference mode.
            return torch.nn.functional.normalize(features, p=2, dim=1)

        output = self.head(features, labels)

        # Handle heads returning (loss, logits) or bare logits needing external loss evaluation.
        loss = None
        if isinstance(output, tuple):
            loss, output = output

        # Calculate Average Theta (Angle to truth).
        aux_logs = {}
        try:
            # Identify the weight matrix.
            weight = None
            if hasattr(self.head, "weight"):
                weight = self.head.weight

            if weight is not None:
                with torch.no_grad():
                    # Normalize features and weights.
                    x = torch.nn.functional.normalize(features, p=2, dim=1)
                    w = torch.nn.functional.normalize(weight, p=2, dim=1)

                    # Calculate cosine similarity between normalized features and class weight vectors.
                    # ArcMarginProduct stores weight as [out, in].
                    cos_theta = torch.nn.functional.linear(x, w) # [B, C].

                    # Gather cosine similarity for ground truth class labels.
                    cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7) # stability.
                    target_cos = cos_theta.gather(1, labels.view(-1, 1))

                    # Convert arccos to degrees.
                    target_theta = torch.acos(target_cos) * (180.0 / 3.14159265)
                    aux_logs["Metric/AvgTheta"] = target_theta.mean().item()
        except Exception:
            # Fail silently on unrecognized head configurations (e.g. SubCenter).
            pass

        if loss is not None:
            return loss, output, aux_logs

        if self.loss_fn is not None:
            loss = self.loss_fn(output, labels)
            return loss, output, aux_logs
        return output


class TrainingStrategy(abc.ABC):
    """Abstract base class for training strategies."""

    # Indicates whether strategy loss is sum-reduced so _base_logs can normalize to per-sample mean.
    LOSS_IS_SUM_REDUCED = False

    @abc.abstractmethod
    def process_batch(self, model: torch.nn.Module, batch: Any, device: torch.device) -> Tuple[torch.Tensor, Any, Dict[str, float]]:
        """Process a single batch and return (loss, outputs, logs_dict)."""
        pass

    def _base_logs(self, loss: torch.Tensor, images: torch.Tensor,
                   labels: torch.Tensor) -> Dict[str, float]:
        """Build normalized per-sample log entries common to all training strategies."""
        per_sample_loss = loss.item()
        if self.LOSS_IS_SUM_REDUCED:
            per_sample_loss /= images.size(0)
        return {
            "loss": per_sample_loss,
            "Data/UniqueClasses": float(len(torch.unique(labels))),
        }

    def get_validation_metric(self, evaluator_result: Any) -> float:
        """Extract the primary metric for model saving from evaluator output tuples."""
        if isinstance(evaluator_result, tuple):
            return evaluator_result[0]
        return 0.0


class ContrastiveStrategy(TrainingStrategy):
    """Strategy for Contrastive Learning (Triplet Loss)."""

    # The triplet loss sums over anchors, so logging divides by batch size.
    LOSS_IS_SUM_REDUCED = True

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def process_batch(self, model: torch.nn.Module, batch: Any, device: torch.device) -> Tuple[torch.Tensor, Any, Dict[str, float]]:
        """Compute the triplet loss for one stacked batch and log batch composition and embedding norms."""
        images, labels = batch

        # metric_learning_collate produces stacked tensors; move to device.
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        loss_output = self.loss_fn(embeddings, labels)

        # Loss functions return either the loss alone or (loss, diagnostics).
        if isinstance(loss_output, tuple):
            loss, mining_diagnostics = loss_output
        else:
            loss = loss_output
            mining_diagnostics = {}

        logs = self._base_logs(loss, images, labels)

        # Batch composition diagnostics.
        unique_classes = len(torch.unique(labels))
        logs["Data/ImagesPerClass"] = float(images.size(0)) / unique_classes if unique_classes > 0 else 0.0

        # Average embedding norm.
        avg_norm = torch.norm(embeddings, p=2, dim=1).mean().item()
        logs["Metric/AvgEmbeddingNorm"] = avg_norm

        # Merge mining diagnostics if present.
        logs.update(mining_diagnostics)

        return loss, embeddings, logs


class SupervisedStrategy(TrainingStrategy):
    """Strategy for supervised ArcFace learning with loss evaluated by the wrapper or head."""

    def process_batch(self, model: torch.nn.Module, batch: Any, device: torch.device) -> Tuple[torch.Tensor, Any, Dict[str, float]]:
        """Compute the wrapper's ArcFace loss for one stacked batch and log training accuracy."""
        images, labels = batch

        # metric_learning_collate produces stacked tensors; move to device.
        images = images.to(device)
        labels = labels.to(device)

        # ArcFace wrapper handles loss calculation.
        loss, logits, aux_logs = model(images, labels)

        logs = self._base_logs(loss, images, labels)
        if aux_logs:
            logs.update(aux_logs)

        # Train Accuracy.
        # logits: [B, NumClasses].
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()
        logs["Metric/TrainAccuracy"] = acc

        return loss, None, logs


class EarlyStopping:
    """Early stopping helper to stop training when metric yields no improvement."""
    def __init__(self, patience=10, min_delta=0.0, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        else:
            if self.mode == "max":
                improvement = current_score - self.best_score
            else: # min
                improvement = self.best_score - current_score

            if improvement > self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True


class RobustTrainer:
    """Unified trainer handling the training loop, lifecycle, and observability."""
    def __init__(self,
                 model: torch.nn.Module,
                 strategy: TrainingStrategy,
                 device: torch.device,
                 config: src.config.config_schema.FullConfig,
                 train_dataset: torch.utils.data.Dataset,
                 valid_dataset: torch.utils.data.Dataset,
                 config_key: str = "",
                 run_identifier: str = ""):

        self.model = model
        self.strategy = strategy
        self.device = device
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.config_key = config_key
        self.run_identifier = run_identifier

        # Configure checkpoint selection and early-stopping metrics with repository defaults.
        self.checkpoint_metric = config.training.checkpoint_metric or CHECKPOINT_METRIC
        self.early_stopping_metric = (
            config.training.early_stopping_metric or EARLY_STOPPING_METRIC)
        if self.early_stopping_metric != self.checkpoint_metric:
            print(f"Checkpointing on '{self.checkpoint_metric}', "
                  f"early stopping on '{self.early_stopping_metric}'.")

        # Set up key components based on config.
        self.optimizer = config.training.optimizer.constructor(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.training.optimizer.learning_rate,
            weight_decay=config.training.optimizer.weight_decay
        )
        if config.training.scheduler:
            self.scheduler = config.training.scheduler.constructor(
                self.optimizer, **config.training.scheduler.options
            )
        else:
            self.scheduler = None

        # Mixed Precision.
        self.use_amp = self.device.type in ['cuda', 'mps']
        self.amp_dtype = torch.float16
        if self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        elif self.device.type == 'mps':
            # MPS supports float16 and bfloat16.
            self.amp_dtype = torch.bfloat16

        # Unified GradScaler (MPS handles scaling internally or via bfloat16, but we use unified API).
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=(self.use_amp and self.amp_dtype == torch.float16))

    @staticmethod
    def _resolve_metric(eval_dict, metric_name, fallback, role):
        """Read metric_name out of eval_dict, warning and falling back if absent."""
        if metric_name not in eval_dict:
            print(f"Warning: {role} metric '{metric_name}' not found in "
                  f"eval_dict; using default metric.")
            return fallback
        return eval_dict[metric_name]

    def save_model(self, trained_models_dir: str, prefix: str = "best"):
        """Save the model components (backbone, head) to disk."""
        os.makedirs(trained_models_dir, exist_ok=True)

        # Check if model is wrapped (ArcFace) or Sequential.
        if hasattr(self.model, "backbone"):
            backbone = self.model.backbone
            head = self.model.head if hasattr(self.model, "head") else None
        elif isinstance(self.model, torch.nn.Sequential):
            backbone = self.model[0]
            head = self.model[1] if len(self.model) > 1 else None
        else:
            backbone = self.model
            head = None

        # Save backbone.
        if isinstance(backbone, peft.PeftModel):
            backbone.save_pretrained(os.path.join(trained_models_dir, f"{prefix}-backbone-lora"))
            torch.save(backbone.get_base_model().state_dict(), os.path.join(trained_models_dir, f"{prefix}-backbone-state.pt"))
        else:
            torch.save(backbone.state_dict(), os.path.join(trained_models_dir, f"{prefix}-backbone-state.pt"))

        # Save head.
        if head:
            torch.save(head.state_dict(), os.path.join(trained_models_dir, f"{prefix}-projection-head.pt"))

    def _save_training_summary(
            self,
            trained_models_dir: str,
            epoch_times: List[float],
            epochs_completed: int,
            early_stopped: bool,
            early_stopped_at_epoch,
            best_metric: float,
            best_epoch: int,
            initial_loss: float,
            final_loss: float,
            global_step: int,
            batches_per_epoch: int,
            training_start_time: float
    ):
        """Saves a training_summary.json file to the output directory."""
        training_end_time = time.time()
        total_seconds = training_end_time - training_start_time
        total_hours = total_seconds / 3600.0
        epoch_hours = [t / 3600.0 for t in epoch_times]

        # Compute throughput.
        batch_size = self.config.training.batch_size
        total_samples = batches_per_epoch * batch_size * epochs_completed
        samples_per_second = total_samples / total_seconds if total_seconds > 0 else 0.0

        # Calculate images_per_second (accounting for augmentations).
        views_per_image = self.config.training.num_augmentations + (
            1 if self.config.training.include_original else 0
        )
        images_per_second = samples_per_second * views_per_image

        # Determine mixed precision description.
        if not self.use_amp:
            amp_desc = "disabled"
        else:
            amp_desc = str(self.amp_dtype).replace("torch.", "")

        summary = {
            "timing": {
                "total_training_time_hours": round(total_hours, 3),
                "time_per_epoch_hours": [round(t, 3) for t in epoch_hours],
                "average_time_per_epoch_hours": round(
                    sum(epoch_hours) / len(epoch_hours), 3
                ) if epoch_hours else 0.0,
                "fastest_epoch_hours": round(min(epoch_hours), 3) if epoch_hours else 0.0,
                "slowest_epoch_hours": round(max(epoch_hours), 3) if epoch_hours else 0.0,
                "estimated_time_per_epoch_seconds": round(
                    sum(epoch_times) / len(epoch_times), 3
                ) if epoch_times else 0.0,
            },
            "training_outcome": {
                "epochs_completed": epochs_completed,
                "epochs_configured": self.config.training.epochs,
                "early_stopped": early_stopped,
                "early_stopped_at_epoch": early_stopped_at_epoch,
                "best_validation_metric": round(best_metric, 3) if best_metric != -float('inf') else None,
                "best_epoch": best_epoch if best_epoch >= 0 else None,
                "initial_training_loss": round(initial_loss, 3) if initial_loss is not None else None,
                "final_training_loss": round(final_loss, 3) if final_loss is not None else None,
            },
            "throughput": {
                "device": self.device.type,
                "mixed_precision_dtype": amp_desc,
                "total_training_steps": global_step,
                "batches_per_epoch": batches_per_epoch,
                "samples_per_second": round(samples_per_second, 3),
                "images_per_second": round(images_per_second, 3),
            },
            "metadata": {
                "config_key": self.config_key,
                "run_identifier": self.run_identifier,
                "training_started_at": datetime.datetime.fromtimestamp(
                    training_start_time
                ).isoformat(),
                "training_finished_at": datetime.datetime.fromtimestamp(
                    training_end_time
                ).isoformat(),
                "hostname": socket.gethostname(),
            },
        }

        summary_path = os.path.join(trained_models_dir, "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print(f"Training summary saved to {summary_path}")

    def train(self, train_loader: torch.utils.data.DataLoader, trained_models_dir: str):
        """Runs the training loop and manages checkpoints, logging, and evaluation."""
        # Reseed RNG before training to decouple dataset augmentations from model construction draws.
        if constants.RESEED_BEFORE_TRAIN:
            utils.seed_everything()

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=os.path.join(trained_models_dir, "logs"))
        epochs = self.config.training.epochs
        accumulation_steps = self.config.training.gradient_accumulation_steps
        # Fixed for the run; used to size the trailing accumulation group.
        num_batches = len(train_loader)

        # Early Stopping Setup.
        early_stopping = None
        if self.config.training.early_stopping and self.config.training.early_stopping.enabled:
            print(f"Early Stopping Enabled (Patience: {self.config.training.early_stopping.patience})")
            early_stopping = EarlyStopping(
                patience=self.config.training.early_stopping.patience,
                min_delta=self.config.training.early_stopping.min_delta,
                mode=self.config.training.early_stopping.mode
            )

        best_metric = -float('inf')
        best_epoch = -1
        global_step = 0
        training_history = []
        epoch_times: List[float] = []
        initial_loss = None
        final_loss = None
        early_stopped = False
        early_stopped_at_epoch = None
        training_start_time = time.time()

        # Save initialization state.
        self.save_model(trained_models_dir, "init")

        # If epochs is 0, we still want to save the "last" and "best" model states
        # as they represent the initialized (and potentially LoRA-wrapped) model.
        if epochs == 0:
            print("Epochs set to 0. Saving initialized model as 'last' and 'best' and exiting training loop.")
            self.save_model(trained_models_dir, "last")
            self.save_model(trained_models_dir, "best")

        # Training loop.
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.model.train()

            # If backbone is frozen, ensure it stays in eval mode.
            if self.config.training.freeze_backbone:
                backbone_to_freeze = self.model
                # Unwrap PeftModel if present.
                if isinstance(backbone_to_freeze, peft.PeftModel):
                    backbone_to_freeze = backbone_to_freeze.base_model.model

                if hasattr(backbone_to_freeze, "backbone"):
                    backbone_to_freeze.backbone.eval()
                elif isinstance(backbone_to_freeze, torch.nn.Sequential):
                    # Usually [0] is backbone.
                    backbone_to_freeze[0].eval()

            # Feature: Freeze Batch Norm Statistics (keep weights trainable if needed).
            if self.config.training.freeze_bn_stats:
                def set_bn_eval(m):
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.eval()
                self.model.apply(set_bn_eval)

            self.optimizer.zero_grad()

            epoch_total_loss = 0.0
            epoch_steps = 0

            with tqdm.tqdm(train_loader, desc=f"Epoch {epoch}") as tq:
                for step, batch in enumerate(tq):
                    # Log the first batch's augmented image grid to TensorBoard
                    # once per epoch so we can sanity-check the augmentation
                    # pipeline without blocking the run.
                    if step == 0:
                        debug_images = batch[0][:8].to(self.device)
                        grid = torchvision.utils.make_grid(debug_images, normalize=True)
                        writer.add_image("Input/AugmentedBatch", grid, epoch)

                    with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                        loss, outputs, logs = self.strategy.process_batch(self.model, batch, self.device)

                    epoch_total_loss += logs["loss"]
                    epoch_steps += 1

                    # Scale loss by the actual accumulation group size so trailing partial batches are not under-weighted.
                    group_start = (step // accumulation_steps) * accumulation_steps
                    group_size = min(group_start + accumulation_steps, num_batches) - group_start
                    if group_size > 1:
                        loss = loss / group_size

                    self.scaler.scale(loss).backward()

                    if (step + 1) % accumulation_steps == 0 or (step + 1) == num_batches:
                        self.scaler.unscale_(self.optimizer)

                        # Log gradient norm.
                        grads = [torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if p.grad is not None]
                        if grads:
                            total_norm = torch.norm(torch.stack(grads), 2)
                            writer.add_scalar("System/GradientNorm", total_norm, global_step)

                        # Clips every parameter, not just trainable ones, which
                        # is the long-standing behaviour; only the threshold is
                        # configurable.
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.training.grad_clip_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        global_step += 1

                        # Step logging.
                        if global_step % 10 == 0:
                            writer.add_scalar("Loss/Train", logs["loss"], global_step)
                            writer.add_scalar("System/LearningRate", self.optimizer.param_groups[0]['lr'], global_step)

                            # Log optional enhanced metrics if present.
                            for key, val in logs.items():
                                if key != "loss": # Loss/Train is already handled.
                                    writer.add_scalar(key, val, global_step)

                            # Debug: Log Batch Norm running stats to monitor drift.
                            if not self.config.training.freeze_bn_stats:
                                bn_modules = []
                                for name, module in self.model.named_modules():
                                    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                                        if module.running_mean is not None:
                                            bn_modules.append((name, module))

                                if bn_modules:
                                    indices_to_log = {0, len(bn_modules) // 2, len(bn_modules) - 1}
                                    for idx in sorted(indices_to_log):
                                        name, module = bn_modules[idx]
                                        writer.add_scalar(
                                            f"BN_Stats/{name}.running_mean_avg",
                                            module.running_mean.mean().item(),
                                            global_step
                                        )

                    tq.set_postfix(loss=logs["loss"])

            # Validation.
            eval_dict = {}
            if self.config.training.evaluator:
                self.model.eval()
                # Resolve evaluator callable with signature (model, train_dataset, valid_dataset, device).
                evaluator = self.config.training.evaluator

                metric, eval_dict = evaluator(self.model, self.train_dataset, self.valid_dataset, self.device)
                val_metric = self.strategy.get_validation_metric((metric, eval_dict))

                writer.add_scalar("Metric/ValidationHits@1", val_metric, epoch)
                # One add_scalar per key rather than add_scalars, which opens a
                # nested SummaryWriter per tag and leaks file handles.
                for metric_name, metric_value in eval_dict.items():
                    writer.add_scalar(f"Validation/Metrics/{metric_name}", metric_value, epoch)

                print(f"Epoch {epoch} Validation: {eval_dict}")

                # Resolve the checkpoint and early-stopping metrics separately;
                # by default both are constants.CHECKPOINT_METRIC.
                checkpoint_val = self._resolve_metric(
                    eval_dict, self.checkpoint_metric, val_metric, "checkpoint")
                early_stop_val = checkpoint_val
                if self.early_stopping_metric != self.checkpoint_metric:
                    early_stop_val = self._resolve_metric(
                        eval_dict, self.early_stopping_metric, val_metric, "early-stopping")

                if checkpoint_val > best_metric:
                    best_metric = checkpoint_val
                    best_epoch = epoch
                    self.save_model(trained_models_dir, "best")

                # Check early stopping condition while allowing remaining epoch bookkeeping to run.
                if early_stopping:
                    early_stopping(early_stop_val)
                    if early_stopping.early_stop:
                        print(f"Early Stopping triggered at epoch {epoch}")
                        early_stopped = True
                        early_stopped_at_epoch = epoch

            # End-of-epoch bookkeeping: loss history, timing, CSV, "last" checkpoint.
            avg_epoch_loss = epoch_total_loss / epoch_steps if epoch_steps > 0 else 0
            if initial_loss is None:
                initial_loss = avg_epoch_loss
            final_loss = avg_epoch_loss
            epoch_times.append(time.time() - epoch_start_time)
            utils.log_training_history_csv(training_history, trained_models_dir, epoch, avg_epoch_loss, eval_dict)

            self.save_model(trained_models_dir, "last")

            if early_stopped:
                break

            if self.scheduler:
                self.scheduler.step()

        # Save training summary.
        self._save_training_summary(
            trained_models_dir=trained_models_dir,
            epoch_times=epoch_times,
            epochs_completed=len(epoch_times),
            early_stopped=early_stopped,
            early_stopped_at_epoch=early_stopped_at_epoch,
            best_metric=best_metric,
            best_epoch=best_epoch,
            initial_loss=initial_loss,
            final_loss=final_loss,
            global_step=global_step,
            batches_per_epoch=len(train_loader),
            training_start_time=training_start_time,
        )
        writer.close()

def _non_negative_int(text):
    """Parse a command-line integer that must not be negative."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def main():
    """Trains re-identification models according to specified configurations."""
    parser = argparse.ArgumentParser(
        description="Train a re-identification model using configurations "
                    "defined in src/config/config.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m src.learning.trainer.trainer "
               "--model-name test-model "
               "--annotations-version test-annotations "
               "--configs miewid_arcface\n")
    parser.add_argument('--model-name', type=str, default='robust_run',
                        help='Name for the output model directory under '
                             'sm-data/trained-models/. If not provided, a '
                             'timestamp is used. Default: "robust_run".')
    parser.add_argument('--configs', nargs='*',
                        help='Space-separated list of config keys from '
                             'src/config/config.py to train (e.g. "miewid_arcface '
                             'vit_arcface"). If omitted, every config except '
                             'test_config is trained sequentially.')
    parser.add_argument('--resume-from-model', type=str, default=None,
                        help='Path to a previously trained model directory '
                             '(relative to sm-data/trained-models/) to resume '
                             'training from. E.g. "my_model/vit_triplet_1".')
    parser.add_argument('--annotations-version', type=str, default=None,
                        help='Annotations version directory name under '
                             'sm-data/annotations/ (e.g. "test-annotations"). '
                             'Must match the version used in convert_data.py.')
    parser.add_argument('--split-strategy', choices=['train-test', 'none'],
                        default='train-test',
                        help='Strategy for handling dataset splits. '
                             '"train-test" (default) uses a split. "none" '
                             'treats the entire dataset as a single set '
                             'for training without a held-out validation set.')
    parser.add_argument('--epochs', type=_non_negative_int, default=None,
                        help='Train for this many epochs instead of the configured '
                             'count. A scheduler T_max equal to the configured epochs '
                             'is rescaled to match; the saved config.json records the '
                             'value used.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow the run to replace trained weights that already '
                             'exist in the output folder, including when resuming '
                             'from that same folder.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate configuration and paths without '
                             'starting training. Prints a summary of '
                             'resolved settings and exits non-zero when a '
                             'required path is missing.')
    parser.add_argument('--no-email', action='store_true',
                        help='Do not send completion notifications for this '
                             'run. Recipients come from the comma-separated '
                             'SHARK_NOTIFICATION_EMAILS environment variable.')
    args = parser.parse_args()

    utils.print_script_banner(__file__, args)

    run_identifier = args.model_name
    if args.model_name == 'robust_run':
        run_identifier = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    configs_to_run = args.configs or [key for key in src.config.config.CONFIGS
                                      if key not in src.config.config.TEST_ONLY_CONFIGS]

    unknown_keys = [key for key in configs_to_run if key not in src.config.config.CONFIGS]
    if unknown_keys:
        print(f"Error: configuration(s) not found in src/config/config.py: {', '.join(unknown_keys)}")
        sys.exit(1)

    # --- Early Path Validation ---
    if not os.path.exists(constants.IMAGES_DIR):
        print(f"Error: Base images directory not found at {constants.IMAGES_DIR}")
        sys.exit(1)

    if args.annotations_version:
        annotations_dir = os.path.join(constants.ANNOTATIONS_DIR, args.annotations_version)
        if not os.path.exists(annotations_dir):
            print(f"Error: Annotations version '{args.annotations_version}' not found at {annotations_dir}")
            sys.exit(1)
    elif not os.path.exists(constants.ANNOTATIONS_DIR):
        print(f"Error: Base annotations directory not found at {constants.ANNOTATIONS_DIR}")
        sys.exit(1)

    if not args.dry_run:
        check_output_directories(configs_to_run, run_identifier, args)

    for config_key in configs_to_run:
        run_single_config(config_key, run_identifier, args)

    print(f"\n{'='*60}")
    print(f"All requested configurations completed for: {run_identifier}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
