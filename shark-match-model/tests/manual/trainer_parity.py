"""Trainer parity harness that fingerprints training cycles on pseudo data to verify refactoring preserves behavior."""
# pylint: disable=wrong-import-position, too-many-locals, broad-exception-caught

import argparse
import os
import random
import sys
import tempfile

# Set MPS fallback before importing torch so aten::_cdist_backward is supported if executed on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import constants
import src.config.config
import src.learning.augmented_dataset
import src.learning.pk_sampler
import src.models.arcface_heads
from src.core import utils
from src.learning.trainer import new_trainer

SEED = 42

# 8 identities gives 4 PK batches per epoch; accumulation=2 exercises both accumulate and flush steps.
NUM_IDENTITIES = 8
IMAGES_PER_IDENTITY = 3
IMAGE_SIZE = 32
EMBEDDING_DIM = 32
EPOCHS = 2
ACCUMULATION_STEPS = 2

# Elevate learning rate so behavioural regressions produce pronounced numerical differences in reports.
LEARNING_RATE = 1e-2


def seed_all(seed=SEED):
    """Pin every RNG the pipeline touches, and forbid nondeterministic kernels."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class PseudoIndividualDataset(torch.utils.data.Dataset):
    """In-memory deterministic dataset matching the IndividualDataset interface."""

    def __init__(self, num_identities=NUM_IDENTITIES,
                 images_per_identity=IMAGES_PER_IDENTITY,
                 image_size=IMAGE_SIZE, seed=SEED):
        generator = torch.Generator().manual_seed(seed)
        self.images = []
        self.labels = []
        for identity in range(num_identities):
            for _ in range(images_per_identity):
                # rand (not randn) keeps values in [0, 1], which is what the
                # augmentation stack expects from a loaded image.
                self.images.append(
                    torch.rand(3, image_size, image_size, generator=generator))
                self.labels.append(identity)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.filenames = [f"pseudo_{i:03d}.jpeg" for i in range(len(self.images))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def get_labels(self):
        return self.labels

    def get_ids(self):
        return list(range(len(self.images)))

    def get_class_counts(self):
        return torch.bincount(self.labels)


class TinyBackbone(torch.nn.Module):
    """Small CNN backbone with BatchNorm2d to exercise freeze_bn_stats and BN logging."""

    def __init__(self, output_dim=EMBEDDING_DIM):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.norm = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(16, output_dim)
        self.output_dim = output_dim

    def get_output_dim(self):
        return self.output_dim

    def forward(self, images):
        features = torch.nn.functional.relu(self.norm(self.conv1(images)))
        features = torch.nn.functional.relu(self.conv2(features))
        return self.fc(self.pool(features).flatten(1))


def scripted_evaluator(scores, metric_key=constants.CHECKPOINT_METRIC):
    """Deterministic evaluator yielding predetermined scores matching the hits_at_k return signature."""
    state = {"calls": 0}

    def evaluate(_model, _train, _valid, _device):
        score = scores[min(state["calls"], len(scores) - 1)]
        state["calls"] += 1
        return score, {metric_key: score}

    return evaluate


def build_config(loss_name, single_view=False, early_stop=False,
                 via_build=False, missing_metric=False, pk_mismatch=False,
                 no_pk=False, partial_accum=False):
    """Build a validated FullConfig customized from test_config to exercise specific trainer code paths."""
    import copy  # pylint: disable=import-outside-toplevel

    raw = copy.deepcopy(src.config.config.CONFIGS["test_config"])
    raw["training"]["epochs"] = EPOCHS
    raw["training"]["gradient_accumulation_steps"] = ACCUMULATION_STEPS
    raw["training"]["batch_size"] = 4          # P*K = 2*2, matches test_config.
    raw["training"]["optimizer"]["learning_rate"] = LEARNING_RATE
    raw["training"]["scheduler"]["options"]["T_max"] = EPOCHS
    raw["training"]["num_workers"] = 0         # pinned: worker seeding is variance.
    raw["data"]["image_size"] = [IMAGE_SIZE, IMAGE_SIZE]
    raw["training"]["loss"]["name"] = loss_name
    # LoRA targets MiewID-specific module names; the stub backbone has none.
    raw.pop("lora", None)

    if single_view:
        raw["training"]["num_augmentations"] = 1
        raw["training"]["include_original"] = False

    if early_stop:
        raw["training"]["epochs"] = 3
        raw["training"]["early_stopping"] = {
            "enabled": True, "patience": 1, "min_delta": 0.001, "mode": "max",
        }
        # Rises then falls: epoch 1 takes the improvement branch, epoch 2
        # exhausts patience and takes the break path.
        raw["training"]["evaluator"] = scripted_evaluator([0.5, 0.9, 0.1])

    if missing_metric:
        raw["training"]["evaluator"] = scripted_evaluator(
            [0.5, 0.6], metric_key="some_metric_that_is_not_the_checkpoint_one")

    if partial_accum:
        # Uneven batch grouping exercises trailing partial-group gradient accumulation flushing.
        raw["training"]["gradient_accumulation_steps"] = 3

    if pk_mismatch:
        # batch_size deliberately disagrees with P*K so build_model_and_strategy
        # takes its warn-and-override branch.
        raw["training"]["batch_size"] = 7

    if no_pk:
        # No P/K configured at all -> the shuffle fallback instead of PKSampler.
        raw["training"]["identities_per_batch"] = None
        raw["training"]["samples_per_identity"] = None

    if via_build:
        raw["model"]["backbone"] = {
            "name": "TinyBackbone", "constructor": TinyBackbone, "options": {},
        }
        if loss_name == "arcface":
            # margins present -> the dynamic-margin block runs for real.
            raw["model"]["projection_head"] = {
                "name": "ArcFaceSubCenterDynamic",
                "constructor": src.models.arcface_heads.ArcFaceSubCenterDynamic,
                "options": {"margins": None},
            }

    return utils.ensure_config_schema(raw)


def build_scenario(loss_name, device, single_view=False, early_stop=False,
                   via_build=False, missing_metric=False, pk_mismatch=False,
                   no_pk=False, partial_accum=False):
    """Construct the model, strategy, loader, and dataset for a parity test scenario."""
    config = build_config(loss_name, single_view=single_view,
                          early_stop=early_stop, via_build=via_build,
                          missing_metric=missing_metric,
                          pk_mismatch=pk_mismatch, no_pk=no_pk,
                          partial_accum=partial_accum)
    base_dataset = PseudoIndividualDataset()
    augmentations = new_trainer.setup_augmentations(
        {"augmentations": src.config.config.AUGMENTATIONS})

    if via_build:
        model, strategy, loader = new_trainer.build_model_and_strategy(
            config, device, base_dataset, base_dataset, augmentations)
        return config, model, strategy, loader, base_dataset

    train_dataset = src.learning.augmented_dataset.AugmentedDataset(
        base_dataset, augmentations,
        num_augmentations=config.training.num_augmentations,
        include_original=config.training.include_original,
        rebalance_augmentations=config.training.rebalance_augmentations,
    )

    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": 0,            # pinned: worker seeding is a variance source.
        "pin_memory": False,
        "collate_fn": new_trainer.metric_learning_collate,
    }

    backbone = TinyBackbone().to(device)

    if loss_name == "arcface":
        head = src.models.arcface_heads.ArcMarginProduct(
            in_features=backbone.get_output_dim(),
            out_features=NUM_IDENTITIES,
        ).to(device)
        model = new_trainer.ArcFaceModelWrapper(
            backbone, head, torch.nn.CrossEntropyLoss())
        strategy = new_trainer.SupervisedStrategy()
        generator = torch.Generator().manual_seed(SEED)
        loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    else:
        loss_fn = config.training.loss.constructor(**config.training.loss.options)
        strategy = new_trainer.ContrastiveStrategy(loss_fn=loss_fn)
        model = torch.nn.Sequential(backbone, torch.nn.Identity()).to(device)
        sampler = src.learning.pk_sampler.PKSampler(
            base_dataset,
            p=config.training.identities_per_batch,
            k=config.training.samples_per_identity,
        )
        loader = torch.utils.data.DataLoader(
            train_dataset, sampler=sampler, **loader_kwargs)

    return config, model, strategy, loader, base_dataset


def capture_pipeline(loader):
    """Record the exact batches produced by the loader so data-pipeline changes are isolated from training."""
    seed_all()
    images, labels = [], []
    for batch_images, batch_labels in loader:
        images.append(batch_images.clone())
        labels.append(batch_labels.clone())
    return {"images": images, "labels": labels}


def run_scenario(name, device, workdir, **options):
    """Run one scenario and record each stage separately so --compare pinpoints the first divergence."""
    seed_all()
    loss_name = options.pop("loss")
    config, model, strategy, loader, base_dataset = build_scenario(
        loss_name, device, **options)

    # Snapshot initial model state to detect initialization RNG divergence.
    post_build = {k: v.clone() for k, v in model.state_dict().items()}

    # Stage 2: the data pipeline in isolation.
    pipeline = capture_pipeline(loader)

    # Wrap strategy and optimizer to record loss and grad norms during training.
    seed_all()
    losses = []
    original_process_batch = strategy.process_batch

    def recording_process_batch(model_arg, batch, device_arg):
        loss, outputs, logs = original_process_batch(model_arg, batch, device_arg)
        losses.append(float(loss.detach().item()))
        return loss, outputs, logs

    strategy.process_batch = recording_process_batch

    trainer = new_trainer.RobustTrainer(
        model, strategy, device, config,
        train_dataset=base_dataset,
        valid_dataset=base_dataset,
        config_key=f"parity_{name}",
        run_identifier="parity",
    )
    # AMP is off on CPU already; assert it so a device mix-up fails loudly
    # instead of silently changing numerics.
    assert not trainer.use_amp, "parity runs must be CPU/no-AMP"

    grad_norms = []
    original_step = trainer.optimizer.step

    def recording_step(*args, **kwargs):
        total = torch.norm(torch.stack([
            torch.norm(p.grad.detach(), 2)
            for p in model.parameters() if p.grad is not None
        ]), 2)
        grad_norms.append(float(total.item()))
        return original_step(*args, **kwargs)

    trainer.optimizer.step = recording_step

    trainer.train(loader, os.path.join(workdir, name))

    # Stage 4: final weights and logits on a fixed probe batch.
    post_train = {k: v.clone() for k, v in model.state_dict().items()}
    probe = torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE,
                       generator=torch.Generator().manual_seed(SEED + 1)).to(device)
    model.eval()
    with torch.no_grad():
        probe_logits = model(probe)

    return {
        "post_build": post_build,
        "pipeline": pipeline,
        "losses": losses,
        "grad_norms": grad_norms,
        "post_train": post_train,
        "probe_logits": probe_logits,
    }


# Scenarios covering triplet/arcface, single-view, early stopping, missing metrics, and build_model_and_strategy.
SCENARIOS = (
    ("triplet",            {"loss": "triplet"}),
    ("arcface",            {"loss": "arcface"}),
    ("triplet_singleview", {"loss": "triplet", "single_view": True}),
    ("triplet_earlystop",  {"loss": "triplet", "early_stop": True}),
    ("triplet_nometric",   {"loss": "triplet", "missing_metric": True}),
    ("build_triplet",      {"loss": "triplet", "via_build": True}),
    ("build_arcface",      {"loss": "arcface", "via_build": True}),
    ("build_pk_mismatch",  {"loss": "triplet", "via_build": True, "pk_mismatch": True}),
    ("build_no_pk",        {"loss": "triplet", "via_build": True, "no_pk": True}),
    ("triplet_partial_accum", {"loss": "triplet", "partial_accum": True}),
)


def run_all(device=torch.device("cpu")):
    """Run every scenario and return ``{scenario_name: fingerprint}``."""
    results = {}
    with tempfile.TemporaryDirectory() as workdir:
        for name, options in SCENARIOS:
            print(f"\n=== scenario: {name} ===")
            results[name] = run_scenario(name, device, workdir, **dict(options))
    return results


# --- Comparison -----------------------------------------------------------

def _diff(left, right, path=""):
    """Recursively compare two fingerprints and return a list of differences."""
    if isinstance(left, dict):
        problems = []
        for key in sorted(set(left) | set(right)):
            if key not in left:
                problems.append(f"{path}/{key}: missing in baseline")
            elif key not in right:
                problems.append(f"{path}/{key}: missing in current")
            else:
                problems.extend(_diff(left[key], right[key], f"{path}/{key}"))
        return problems

    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} vs {len(right)}"]
        problems = []
        for index, (a, b) in enumerate(zip(left, right)):
            problems.extend(_diff(a, b, f"{path}[{index}]"))
        return problems

    if isinstance(left, torch.Tensor):
        if left.shape != right.shape:
            return [f"{path}: shape {tuple(left.shape)} vs {tuple(right.shape)}"]
        if torch.equal(left, right):
            return []
        delta = (left.float() - right.float()).abs().max().item()
        return [f"{path}: differs (max abs {delta:.3e})"]

    if isinstance(left, float):
        if left != right:
            return [f"{path}: {left!r} vs {right!r} (delta {abs(left - right):.3e})"]
        return []

    return [] if left == right else [f"{path}: {left!r} vs {right!r}"]


# Order matters: the first stage listed that differs is the one worth
# debugging, because a divergence at an early stage explains every later one.
STAGE_ORDER = ["post_build", "pipeline", "losses", "grad_norms",
               "post_train", "probe_logits"]


def report(baseline, current):
    """Print a stage-by-stage comparison. Returns True when fully identical."""
    identical = True
    for scenario in sorted(set(baseline) | set(current)):
        print(f"\n--- {scenario} ---")
        if scenario not in baseline or scenario not in current:
            print(f"  scenario missing from {'baseline' if scenario not in baseline else 'current'}")
            identical = False
            continue

        first_divergence = None
        for stage in STAGE_ORDER:
            problems = _diff(baseline[scenario].get(stage),
                             current[scenario].get(stage), stage)
            if problems:
                identical = False
                marker = "FAIL"
                if first_divergence is None:
                    first_divergence = stage
            else:
                marker = " OK "
            print(f"  [{marker}] {stage}")
            for problem in problems[:6]:
                print(f"           {problem}")
            if len(problems) > 6:
                print(f"           ... and {len(problems) - 6} more")

        if first_divergence:
            print(f"  -> first divergence at '{first_divergence}'")
    return identical


def main():
    """Run the CLI to save, compare, or self-check training fingerprints."""
    parser = argparse.ArgumentParser(
        description="Verify the training pipeline behaves identically before "
                    "and after a change.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", metavar="PATH",
                       help="Run and write a baseline fingerprint to PATH.")
    group.add_argument("--compare", metavar="PATH",
                       help="Run and compare against the baseline at PATH. "
                            "Exits non-zero on any difference.")
    group.add_argument("--check", action="store_true",
                       help="Run twice in-process and assert the two runs are "
                            "bitwise identical (validates the harness itself).")
    args = parser.parse_args()

    torch.set_num_threads(1)  # thread count changes float reduction order.

    if args.check:
        print("Run 1 of 2")
        first = run_all()
        print("\nRun 2 of 2")
        second = run_all()
        print("\n" + "=" * 60)
        print("SELF-CHECK: two runs of identical code")
        print("=" * 60)
        ok = report(first, second)
        print("\n" + ("PASS: harness is deterministic."
                      if ok else
                      "FAIL: harness is NOT deterministic; --compare results "
                      "cannot be trusted until this is fixed."))
        return 0 if ok else 1

    if args.save:
        results = run_all()
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save(results, args.save)
        print(f"\nBaseline written to {args.save}")
        return 0

    baseline = torch.load(args.compare, weights_only=False)
    current = run_all()
    print("\n" + "=" * 60)
    print(f"COMPARE: current code vs {args.compare}")
    print("=" * 60)
    ok = report(baseline, current)
    print("\n" + ("PASS: training pipeline is unchanged."
                  if ok else
                  "FAIL: training pipeline behaviour changed."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
