# pylint: disable=no-member, too-many-arguments, too-many-locals, too-many-statements, too-many-instance-attributes, broad-exception-caught, import-error, duplicate-code, line-too-long, no-name-in-module, import-outside-toplevel, too-many-positional-arguments, too-many-branches
"""Trains a multimodal fusion head over frozen image embeddings and live CSV metadata features."""

import argparse
import copy
import datetime
import json
import os
import socket
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Enable MPS fallback for ops that may not be implemented on Apple GPUs.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.utils.data
import torch.utils.tensorboard
import tqdm

from src.core import constants
import src.config.config
import src.multimodal.fusion_dataset
import src.multimodal.metadata_features
import experimental.multimodal.trained_fusion.configs
import experimental.multimodal.trained_fusion.fusion_model
import src.learning.pk_sampler
from src.core import utils

# Drives best-checkpoint selection and early stopping (matches new_trainer.py).
CHECKPOINT_METRIC = constants.CHECKPOINT_METRIC
EARLY_STOPPING_METRIC = constants.CHECKPOINT_METRIC


class EarlyStopping:
    """Stop training when the early-stopping metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float):
        if self.best_score is None:
            self.best_score = current_score
            return
        improvement = (current_score - self.best_score) if self.mode == "max" \
            else (self.best_score - current_score)
        if improvement > self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ---------------------------------------------------------------------------
# Setup helpers.
# ---------------------------------------------------------------------------

def print_config_summary(config_key: str, config: Dict[str, Any], image_embeddings_name: str):
    utils.print_function_banner(f"Training Config: {config_key}")
    train_cfg = config["training"]
    model_cfg = config["model"]
    spec = config.get("metadata_features", {})

    p_value = train_cfg.get("p_value")
    k_value = train_cfg.get("k_value", 4)
    batch_size = train_cfg.get("batch_size", 32)
    if p_value is None and k_value:
        p_value = batch_size // k_value
    effective_batch = (p_value or 0) * (k_value or 0)

    config_details = {
        "Image Embeddings": image_embeddings_name,
        "Fusion Method": model_cfg.get("fusion_method", "concatenate"),
        "Shared Dim": model_cfg.get("shared_dim"),
        "Final Dim": model_cfg.get("final_dim"),
        "Batch Size (P*K)": effective_batch if effective_batch else batch_size,
        "P (Identities/Batch)": p_value,
        "K (Samples/Identity)": k_value,
        "Accumulation Steps": train_cfg.get("gradient_accumulation_steps", 1),
        "Epochs": train_cfg.get("epochs"),
        "Learning Rate": train_cfg["optimizer"]["learning_rate"],
        "Loss Function": train_cfg["loss"]["name"],
        "Encoding Mode": "soft" if spec.get("use_soft_encoding") else "one-hot",
    }
    if spec:
        included = [k.replace("include_", "") for k, v in spec.items()
                    if k.startswith("include_") and v]
        config_details["Included Metadata Features"] = ", ".join(included) or "(none)"

    utils.print_key_value_section("Configuration Details", config_details)
    print("-" * 40 + "\n")


def setup_datasets(args: argparse.Namespace, config: Dict[str, Any]
                   ) -> Tuple[Any, Optional[Any]]:
    """Load stored image-only datasets and embeddings, wrapping them in train and validation FusionDataset instances."""
    utils.print_function_banner("Loading Datasets")
    embeddings_dir = os.path.join(constants.EMBEDDINGS_DIR, args.image_embeddings_name)
    stored_dataset_path = os.path.join(embeddings_dir, "stored_dataset.pt")
    image_embeddings_path = os.path.join(embeddings_dir, "stored_embeddings.pt")
    if not os.path.exists(stored_dataset_path):
        raise FileNotFoundError(f"Stored dataset not found at {stored_dataset_path}.")
    if not os.path.exists(image_embeddings_path):
        raise FileNotFoundError(f"Stored image embeddings not found at {image_embeddings_path}.")

    stored_dataset = torch.load(
        stored_dataset_path, map_location='cpu', weights_only=False)
    image_embeddings = torch.load(
        image_embeddings_path, map_location='cpu', weights_only=False)

    spec = src.multimodal.metadata_features.MetadataFeatureSpec.from_dict(
        config.get("metadata_features"))

    if args.split_strategy == "none":
        print("Split strategy is 'none'. Training over the full corpus without a held-out validation set.")
        train_dataset = src.multimodal.fusion_dataset.FusionDataset(
            stored_dataset, image_embeddings, spec, split=None,
            fit_split=args.fit_stats_split,
        )
        valid_dataset = None
    else:
        train_dataset = src.multimodal.fusion_dataset.FusionDataset(
            stored_dataset, image_embeddings, spec, split='train',
            fit_split=args.fit_stats_split,
        )
        valid_dataset = src.multimodal.fusion_dataset.FusionDataset(
            stored_dataset, image_embeddings, spec, split='test',
            norm_stats=train_dataset.norm_stats,
        )
        utils.verify_train_test_split_integrity(
            train_dataset=train_dataset, valid_dataset=valid_dataset,
            dataset_name="Multimodal Fusion")

    print(f"Train dataset: {len(train_dataset)} samples")
    if valid_dataset is not None:
        print(f"Validation dataset: {len(valid_dataset)} samples")
    print(f"Image embedding dim: {train_dataset.image_dim()}")
    print(f"Metadata feature dim: {train_dataset.metadata_dim()}")
    print(f"Fit stats split: {args.fit_stats_split}")
    return train_dataset, valid_dataset


def build_model(config: Dict[str, Any], train_dataset: Any, device: torch.device
                ) -> torch.nn.Module:
    utils.print_function_banner("Building Model")
    model_config = config["model"]
    model = experimental.multimodal.trained_fusion.fusion_model.MultimodalFusionModel(
        image_dim=train_dataset.image_dim(),
        metadata_dim=train_dataset.metadata_dim(),
        shared_dim=model_config["shared_dim"],
        final_dim=model_config["final_dim"],
        fusion_method=model_config["fusion_method"],
        expert_dropout=model_config.get("expert_dropout", 0.0),
        projection_dropout=model_config.get("projection_dropout", 0.0),
        expert_depth=model_config.get("expert_depth", 1),
        residual_hidden_dim=model_config.get("residual_hidden_dim", 256),
    ).to(device)
    print(f"Created MultimodalFusionModel "
          f"(image_dim={train_dataset.image_dim()}, "
          f"metadata_dim={train_dataset.metadata_dim()}).")
    return model


def build_data_loader(train_dataset: Any, train_config: Dict[str, Any]
                      ) -> torch.utils.data.DataLoader:
    batch_size = train_config.get("batch_size", 32)
    p_value = train_config.get("p_value")
    k_value = train_config.get("k_value", 4)
    if p_value is None:
        p_value = batch_size // k_value
        if p_value < 1:
            raise ValueError(f"batch_size ({batch_size}) must be >= k_value ({k_value}).")
        print(f"Calculated P={p_value} from batch_size={batch_size} and K={k_value}")

    actual_batch_size = p_value * k_value
    if batch_size != actual_batch_size:
        print(f"WARNING: Config batch_size ({batch_size}) does not match P×K "
              f"({p_value}×{k_value} = {actual_batch_size}). Overriding.")
        batch_size = actual_batch_size

    num_workers = train_config.get("num_workers")
    if num_workers is None:
        num_workers = src.config.config.NUM_WORKERS

    sampler = src.learning.pk_sampler.PKSampler(train_dataset, p=p_value, k=k_value)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler, batch_size=batch_size, drop_last=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    print(f"PKSampler will yield {len(sampler) // batch_size} batches per epoch "
          f"(P={p_value}, K={k_value}, batch_size={batch_size}).")
    return train_loader


def setup_training_components(train_config: Dict[str, Any], model: torch.nn.Module
                              ) -> Tuple[Any, Any, Any, Any]:
    print("Initializing loss function, optimizer, and scheduler...")
    loss_fn = train_config["loss"]["constructor"](**train_config["loss"]["options"])
    optimizer = train_config["optimizer"]["constructor"](
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_config["optimizer"]["learning_rate"],
        weight_decay=train_config["optimizer"]["weight_decay"],
    )
    scheduler = None
    if "scheduler" in train_config:
        scheduler = train_config["scheduler"]["constructor"](
            optimizer, **train_config["scheduler"]["options"])
    evaluator = train_config.get("evaluator")
    return loss_fn, optimizer, scheduler, evaluator


# ---------------------------------------------------------------------------
# Trainer.
# ---------------------------------------------------------------------------

class FusionTrainer:
    def __init__(
            self, model, device, config, train_dataset, valid_dataset,
            loss_fn, optimizer, scheduler, evaluator, config_key, run_identifier):
        self.model = model
        self.device = device
        self.config = config
        self.train_config = config["training"]
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.config_key = config_key
        self.run_identifier = run_identifier

        self.use_amp = self.device.type in ('cuda', 'mps')
        self.amp_dtype = torch.float16
        if self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        elif self.device.type == 'mps':
            self.amp_dtype = torch.bfloat16
        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=(self.use_amp and self.amp_dtype == torch.float16),
        )

    def save_model(self, save_dir: str, prefix: str = "best"):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{prefix}-fusion_model.pt")
        torch.save(self.model.state_dict(), path)

    def _save_training_summary(self, save_dir, epoch_times, epochs_completed,
                               early_stopped, early_stopped_at_epoch, best_metric,
                               best_epoch, initial_loss, final_loss, global_step,
                               batches_per_epoch, training_start_time):
        training_end_time = time.time()
        total_seconds = training_end_time - training_start_time
        total_hours = total_seconds / 3600.0
        epoch_hours = [t / 3600.0 for t in epoch_times]

        batch_size = self.train_config.get("batch_size", 0)
        total_samples = batches_per_epoch * batch_size * epochs_completed
        samples_per_second = total_samples / total_seconds if total_seconds > 0 else 0.0
        amp_desc = "disabled" if not self.use_amp else str(self.amp_dtype).replace("torch.", "")

        summary = {
            "timing": {
                "total_training_time_hours": round(total_hours, 3),
                "time_per_epoch_hours": [round(t, 3) for t in epoch_hours],
                "average_time_per_epoch_hours": round(
                    sum(epoch_hours) / len(epoch_hours), 3
                ) if epoch_hours else 0.0,
                "fastest_epoch_hours": round(min(epoch_hours), 3) if epoch_hours else 0.0,
                "slowest_epoch_hours": round(max(epoch_hours), 3) if epoch_hours else 0.0,
            },
            "training_outcome": {
                "epochs_completed": epochs_completed,
                "epochs_configured": self.train_config.get("epochs"),
                "early_stopped": early_stopped,
                "early_stopped_at_epoch": early_stopped_at_epoch,
                "best_validation_metric": round(best_metric, 3) if best_metric != -float('inf') else None,
                "best_epoch": best_epoch if best_epoch >= 0 else None,
                "initial_training_loss": round(initial_loss, 6) if initial_loss is not None else None,
                "final_training_loss": round(final_loss, 6) if final_loss is not None else None,
            },
            "throughput": {
                "device": self.device.type,
                "mixed_precision_dtype": amp_desc,
                "total_training_steps": global_step,
                "batches_per_epoch": batches_per_epoch,
                "samples_per_second": round(samples_per_second, 3),
            },
            "metadata": {
                "config_key": self.config_key,
                "run_identifier": self.run_identifier,
                "training_started_at": datetime.datetime.fromtimestamp(training_start_time).isoformat(),
                "training_finished_at": datetime.datetime.fromtimestamp(training_end_time).isoformat(),
                "hostname": socket.gethostname(),
            },
        }
        with open(os.path.join(save_dir, "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print(f"Training summary saved to {save_dir}/training_summary.json")

    def _process_batch(self, batch):
        image_embeds, metadata_embeds, labels = batch
        image_embeds = image_embeds.to(self.device)
        metadata_embeds = metadata_embeds.to(self.device)
        labels = labels.to(self.device)

        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.use_amp):
            embeddings = self.model(image_embeds, metadata_embeds)
            loss_output = self.loss_fn(embeddings, labels)

        if isinstance(loss_output, tuple):
            loss, mining_diagnostics = loss_output
        else:
            loss, mining_diagnostics = loss_output, {}

        logs = {"loss": loss.item() / max(image_embeds.size(0), 1)}
        unique_classes = len(torch.unique(labels))
        logs["Data/UniqueClasses"] = float(unique_classes)
        logs["Data/ImagesPerClass"] = float(image_embeds.size(0)) / unique_classes \
            if unique_classes > 0 else 0.0
        logs["Metric/AvgEmbeddingNorm"] = torch.norm(embeddings, p=2, dim=1).mean().item()
        logs.update(mining_diagnostics)
        return loss, logs

    def train(self, train_loader, save_dir):
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=os.path.join(save_dir, "logs"))
        epochs = self.train_config.get("epochs", 0)
        accum_steps = self.train_config.get("gradient_accumulation_steps", 1)
        warmup_epochs = self.train_config.get("warmup_epochs", 0)
        base_lr = self.train_config["optimizer"]["learning_rate"]

        es_cfg = self.train_config.get("early_stopping") or {}
        early_stopping = None
        if es_cfg.get("enabled"):
            print(f"Early Stopping Enabled (Patience: {es_cfg.get('patience', 10)}).")
            early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 10),
                min_delta=es_cfg.get("min_delta", 0.0),
                mode=es_cfg.get("mode", "max"))

        best_metric = -float('inf')
        best_epoch = -1
        global_step = 0
        training_history: List[List[Any]] = []
        epoch_times: List[float] = []
        initial_loss: Optional[float] = None
        final_loss: Optional[float] = None
        early_stopped = False
        early_stopped_at_epoch: Optional[int] = None
        training_start_time = time.time()

        self.save_model(save_dir, "init")

        if epochs == 0:
            print("Epochs set to 0. Saving initialized model as 'last' and 'best'; exiting.")
            self.save_model(save_dir, "last")
            self.save_model(save_dir, "best")
            writer.close()
            return

        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()

            if warmup_epochs > 0 and epoch < warmup_epochs:
                warmup_lr = base_lr * (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            epoch_loss_sum = 0.0
            epoch_steps = 0

            with tqdm.tqdm(train_loader, desc=f"Epoch {epoch}") as tq:
                for step, batch in enumerate(tq):
                    loss, logs = self._process_batch(batch)
                    epoch_loss_sum += logs["loss"]
                    epoch_steps += 1

                    scaled_loss = loss / accum_steps if accum_steps > 1 else loss
                    self.scaler.scale(scaled_loss).backward()

                    if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                        self.scaler.unscale_(self.optimizer)
                        grads = [torch.norm(p.grad.detach(), 2)
                                 for p in self.model.parameters() if p.grad is not None]
                        if grads:
                            total_norm = torch.norm(torch.stack(grads), 2)
                            writer.add_scalar("System/GradientNorm", total_norm, global_step)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        global_step += 1

                        if global_step % 10 == 0:
                            writer.add_scalar("Loss/Train", logs["loss"], global_step)
                            writer.add_scalar("System/LearningRate",
                                              self.optimizer.param_groups[0]["lr"], global_step)
                            for key, val in logs.items():
                                if key != "loss":
                                    writer.add_scalar(key, val, global_step)

                    tq.set_postfix(loss=logs["loss"])

            avg_epoch_loss = epoch_loss_sum / epoch_steps if epoch_steps > 0 else 0.0
            if initial_loss is None:
                initial_loss = avg_epoch_loss
            final_loss = avg_epoch_loss
            epoch_times.append(time.time() - epoch_start_time)

            eval_dict: Dict[str, float] = {}
            if self.evaluator and self.valid_dataset is not None:
                self.model.eval()
                metric, eval_dict = self.evaluator(
                    self.model, self.train_dataset, self.valid_dataset, self.device)
                self.model.train()

                writer.add_scalar("Metric/ValidationHits@1", metric, epoch)
                writer.add_scalars("Validation/Metrics", eval_dict, epoch)
                print(f"Epoch {epoch} Validation: {eval_dict}")

                checkpoint_val = eval_dict.get(CHECKPOINT_METRIC, metric)
                early_stop_val = eval_dict.get(EARLY_STOPPING_METRIC, metric)

                if checkpoint_val > best_metric:
                    best_metric = checkpoint_val
                    best_epoch = epoch
                    self.save_model(save_dir, "best")

                if early_stopping is not None:
                    early_stopping(early_stop_val)
                    if early_stopping.early_stop:
                        print(f"Early stopping triggered at epoch {epoch}.")
                        early_stopped = True
                        early_stopped_at_epoch = epoch
                        utils.log_training_history_csv(
                            training_history, save_dir, epoch, avg_epoch_loss, eval_dict)
                        self.save_model(save_dir, "last")
                        break

            utils.log_training_history_csv(
                training_history, save_dir, epoch, avg_epoch_loss, eval_dict)
            self.save_model(save_dir, "last")

            if self.scheduler is not None and epoch >= warmup_epochs:
                self.scheduler.step()

        self._save_training_summary(
            save_dir=save_dir, epoch_times=epoch_times,
            epochs_completed=len(epoch_times), early_stopped=early_stopped,
            early_stopped_at_epoch=early_stopped_at_epoch, best_metric=best_metric,
            best_epoch=best_epoch, initial_loss=initial_loss, final_loss=final_loss,
            global_step=global_step, batches_per_epoch=len(train_loader),
            training_start_time=training_start_time)
        writer.close()


# ---------------------------------------------------------------------------
# Top-level orchestration.
# ---------------------------------------------------------------------------

def run_single_config(config_key: str, run_identifier: str, args: argparse.Namespace):
    start_time = time.time()
    config = experimental.multimodal.trained_fusion.configs.TRAINED_FUSION_CONFIGS[config_key]

    try:
        utils.ensure_config_schema(config)
    except Exception as exc:
        print(f"Skipping {config_key} due to config error: {exc}")
        return

    save_dir = os.path.join(constants.TRAINED_MODELS_DIR, run_identifier, config_key)
    os.makedirs(save_dir, exist_ok=True)

    # Embed runtime info in saved config so the generator can reconstruct the spec.
    config_for_save = copy.deepcopy(config)
    config_for_save["data_sources"] = {
        "image_embeddings_name": args.image_embeddings_name,
        "fit_stats_split": args.fit_stats_split,
    }
    utils.save_config(save_dir, config_for_save)

    print_config_summary(config_key, config, args.image_embeddings_name)

    device = utils.get_torch_device()
    utils.seed_everything()

    try:
        train_dataset, valid_dataset = setup_datasets(args, config)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return

    if len(train_dataset) == 0:
        print("ERROR: No training samples loaded; skipping config.")
        return

    model = build_model(config, train_dataset, device)
    train_loader = build_data_loader(train_dataset, config["training"])
    loss_fn, optimizer, scheduler, evaluator = setup_training_components(
        config["training"], model)

    trainer = FusionTrainer(
        model=model, device=device, config=config,
        train_dataset=train_dataset, valid_dataset=valid_dataset,
        loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, evaluator=evaluator,
        config_key=config_key, run_identifier=run_identifier)
    trainer.train(train_loader, save_dir)
    elapsed = time.time() - start_time
    print(f"Trained '{config_key}' in {elapsed/60:.1f} min. Saved to {save_dir}.")


def main():
    parser = argparse.ArgumentParser(
        description="Train a small fusion head over frozen image embeddings + "
                    "engineered metadata features. Pared down to two methods "
                    "(concatenate, residual) used as ablation rows in the paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n"
               "  python3 -m experimental.multimodal.trained_fusion.train_multimodal "
               "--image-embeddings-name thesis_unmatch_random_out "
               "--configs concatenate residual "
               "--model-name thesis_trained_fusion_p3\n")
    parser.add_argument('--model-name', type=str, default='trained_fusion_run',
                        help='Name for the output run directory under '
                             'sm-data/trained-models/. If left as the default, '
                             'a timestamp is used instead.')
    parser.add_argument('--image-embeddings-name', type=str, required=True,
                        help='Name of the pre-computed image embeddings '
                             'directory under sm-data/embeddings/.')
    parser.add_argument('--configs', nargs='*',
                        help='Space-separated list of config keys from '
                             'TRAINED_FUSION_CONFIGS (e.g. "concatenate residual"). '
                             'If omitted, all configs are trained sequentially.')
    parser.add_argument('--split-strategy', choices=['train-test', 'none'],
                        default='train-test',
                        help='"train-test" (default) trains on the train split and '
                             'validates on the test split. "none" trains over the '
                             'full corpus without a held-out validation set.')
    parser.add_argument('--fit-stats-split', choices=('train', 'all'),
                        default='train',
                        help='Subset used to fit metadata normalization '
                             'statistics. Default "train" prevents test data '
                             'from influencing vocabularies/min-max/RBF/proportions.')
    args = parser.parse_args()
    utils.print_script_banner(__file__, args)

    run_identifier = args.model_name
    if args.model_name == 'trained_fusion_run':
        run_identifier = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    fusion_configs = experimental.multimodal.trained_fusion.configs.TRAINED_FUSION_CONFIGS
    configs_to_run = args.configs if args.configs else list(fusion_configs.keys())

    for config_key in configs_to_run:
        if config_key not in fusion_configs:
            print(f"Warning: Configuration '{config_key}' not found. "
                  f"Available: {list(fusion_configs.keys())}. Skipping.")

    embeddings_dir = os.path.join(constants.EMBEDDINGS_DIR, args.image_embeddings_name)
    if not os.path.exists(embeddings_dir):
        print(f"Error: Image embeddings dir not found at {embeddings_dir}.")
        sys.exit(1)

    for config_key in configs_to_run:
        if config_key in fusion_configs:
            run_single_config(config_key, run_identifier, args)

    print(f"\n{'='*60}")
    print(f"All requested fusion configurations completed for: {run_identifier}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
