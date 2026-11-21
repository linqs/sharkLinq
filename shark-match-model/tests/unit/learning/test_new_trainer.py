"""Tests for RobustTrainer, config overrides, resume handling, and overwrite guards."""
import contextlib
import copy
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import peft
import sklearn.preprocessing
import torch
import torch.nn as nn

import src.config.config
import src.learning.trainer.new_trainer as new_trainer
import src.models.arcface_heads
from src.core import utils
from src.core import constants


def _make_config(
        *,
        epochs: int = 1,
        gradient_accumulation_steps: int = 1,
        batch_size: int = 2,
        early_stopping_enabled: bool = False,
        evaluator=None,
):
    """Build a validated FullConfig from test_config with test overrides."""
    raw = copy.deepcopy(src.config.config.CONFIGS["test_config"])
    raw["training"]["epochs"] = epochs
    raw["training"]["gradient_accumulation_steps"] = gradient_accumulation_steps
    raw["training"]["batch_size"] = batch_size
    raw["training"]["num_augmentations"] = 1   # keep views_per_image small.
    raw["training"]["include_original"] = False
    if evaluator is not None:
        raw["training"]["evaluator"] = evaluator
    raw["training"]["early_stopping"]["enabled"] = early_stopping_enabled
    return utils.ensure_config_schema(raw)


class _MockModel(nn.Module):
    """Tiny model: a single Linear layer over 4D image tensors flattened."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 8 * 8, 2)

    def forward(self, x):
        return self.linear(x.flatten(1))


class _MockStrategy(new_trainer.TrainingStrategy):
    """Run the model and return the sum of embeddings as a differentiable loss."""
    def process_batch(self, model, batch, device):
        images, _ = batch
        embeddings = model(images.to(device))
        loss = embeddings.sum()
        return loss, embeddings, {"loss": loss.item()}

    def get_validation_metric(self, evaluator_result):
        return evaluator_result[0]


class _FakeLoader:
    """Iterable yielding ``(images, labels)`` 4D batches for ``num_batches`` steps."""
    def __init__(self, num_batches: int, batch_size: int = 2):
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            yield (
                torch.randn(self.batch_size, 3, 8, 8),
                torch.randint(0, 2, (self.batch_size,)),
            )

    def __len__(self):
        return self.num_batches


class TestNewTrainer(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.test_dir = tempfile.mkdtemp()
        self.evaluator = MagicMock(return_value=(0.9, {constants.CHECKPOINT_METRIC: 0.9}))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _make_trainer(self, *, config=None):
        config = config or _make_config(evaluator=self.evaluator)
        return new_trainer.RobustTrainer(
            _MockModel(),
            _MockStrategy(),
            self.device,
            config,
            train_dataset=MagicMock(),
            valid_dataset=MagicMock(),
            config_key="test_config",
            run_identifier="test_run",
        )

    # ---- Core training loop --------------------------------------------

    def test_train_smoke_test(self):
        """One epoch, three batches: optimizer steps once per batch, scheduler once per epoch."""
        trainer = self._make_trainer()
        # Wrap the real optimizer's step/zero_grad so we can count calls
        # without breaking the gradient pipeline.
        trainer.optimizer.step = MagicMock(side_effect=trainer.optimizer.step)
        trainer.scheduler.step = MagicMock(side_effect=trainer.scheduler.step)

        models_dir = os.path.join(self.test_dir, "models")
        trainer.train(_FakeLoader(num_batches=3), models_dir)

        self.assertEqual(trainer.optimizer.step.call_count, 3)
        self.assertEqual(trainer.scheduler.step.call_count, 1)

    def test_gradient_accumulation(self):
        """With accumulation=2 over 4 batches, optimizer steps only twice."""
        config = _make_config(
            epochs=1, gradient_accumulation_steps=2, evaluator=self.evaluator
        )
        trainer = self._make_trainer(config=config)
        trainer.optimizer.step = MagicMock(side_effect=trainer.optimizer.step)

        trainer.train(_FakeLoader(num_batches=4), os.path.join(self.test_dir, "models"))

        self.assertEqual(trainer.optimizer.step.call_count, 2)

    def _accumulation_divisors(self, *, accumulation_steps, num_batches):
        """Return the divisor applied to each batch's loss by comparing raw and scaled loss values."""
        torch.manual_seed(0)
        config = _make_config(
            epochs=1, gradient_accumulation_steps=accumulation_steps,
            evaluator=self.evaluator)
        trainer = self._make_trainer(config=config)

        raw, scaled = [], []

        original_process = trainer.strategy.process_batch
        def spy_process(model, batch, device):
            loss, outputs, logs = original_process(model, batch, device)
            raw.append(float(loss.detach()))
            return loss, outputs, logs
        trainer.strategy.process_batch = spy_process

        original_scale = trainer.scaler.scale
        def spy_scale(loss):
            scaled.append(float(loss.detach()))
            return original_scale(loss)
        trainer.scaler.scale = spy_scale

        trainer.train(_FakeLoader(num_batches=num_batches),
                      os.path.join(self.test_dir, "models"))

        self.assertEqual(len(raw), num_batches)
        for value in raw:
            # Guard the ratio against a near-zero denominator making the
            # measurement meaningless rather than wrong.
            self.assertGreater(abs(value), 1e-3)
        return [r / s for r, s in zip(raw, scaled)]

    def test_partial_accumulation_group_is_weighted_by_its_own_size(self):
        """Verify trailing partial accumulation groups scale by actual group size rather than accumulation_steps."""
        divisors = self._accumulation_divisors(
            accumulation_steps=3, num_batches=4)
        for i, (actual, expected) in enumerate(zip(divisors, [3, 3, 3, 1])):
            with self.subTest(batch=i):
                self.assertAlmostEqual(actual, expected, places=4)

    def test_full_accumulation_groups_are_unchanged(self):
        """Verify evenly divided accumulation groups maintain constant divisor scaling across all batches."""
        divisors = self._accumulation_divisors(
            accumulation_steps=2, num_batches=4)
        for i, actual in enumerate(divisors):
            with self.subTest(batch=i):
                self.assertAlmostEqual(actual, 2, places=4)

    def test_no_accumulation_never_scales(self):
        """accumulation_steps=1 means every group has size 1: no division."""
        divisors = self._accumulation_divisors(
            accumulation_steps=1, num_batches=3)
        for i, actual in enumerate(divisors):
            with self.subTest(batch=i):
                self.assertAlmostEqual(actual, 1, places=4)

    # ---- Checkpointing -------------------------------------------------

    def test_save_best_model(self):
        """``best`` is saved only when the validation metric improves; ``last`` saves every epoch."""
        # Ramp metric over 3 epochs: 0.5 (improve, save) -> 0.4 (skip) -> 0.6 (improve, save).
        scores = [0.5, 0.4, 0.6]

        def evaluator_side_effect(model, train, valid, device):
            score = scores[evaluator_side_effect.calls]
            evaluator_side_effect.calls += 1
            return score, {constants.CHECKPOINT_METRIC: score}
        evaluator_side_effect.calls = 0

        config = _make_config(epochs=3, evaluator=evaluator_side_effect)
        trainer = self._make_trainer(config=config)
        trainer.save_model = MagicMock()

        trainer.train(_FakeLoader(num_batches=1), os.path.join(self.test_dir, "models"))

        prefixes = [call.args[1] for call in trainer.save_model.call_args_list]
        # 1 init + 2 best (epoch 0 and 2) + 3 last (every epoch).
        self.assertEqual(prefixes.count("init"), 1)
        self.assertEqual(prefixes.count("best"), 2)
        self.assertEqual(prefixes.count("last"), 3)

    # ---- Observability -------------------------------------------------

    @patch('src.core.utils.log_training_history_csv')
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_logging_calls(self, mock_writer_cls, mock_csv_logger):
        """Per-step TB scalars and per-epoch CSV logging both fire."""
        # 11 batches so the "every 10 steps" tag-write triggers at least once.
        trainer = self._make_trainer()
        mock_writer = mock_writer_cls.return_value

        trainer.train(_FakeLoader(num_batches=11), os.path.join(self.test_dir, "models"))

        scalar_tags = [call.args[0] for call in mock_writer.add_scalar.call_args_list]
        self.assertIn("Loss/Train", scalar_tags)
        self.assertIn("System/LearningRate", scalar_tags)
        self.assertIn("System/GradientNorm", scalar_tags)

        # CSV logger fires once per epoch.
        self.assertEqual(mock_csv_logger.call_count, 1)

    def test_training_summary_file(self):
        """``training_summary.json`` is written and has the expected schema."""
        config = _make_config(epochs=2, evaluator=self.evaluator)
        trainer = self._make_trainer(config=config)
        models_dir = os.path.join(self.test_dir, "models")
        trainer.train(_FakeLoader(num_batches=3), models_dir)

        summary_path = os.path.join(models_dir, "training_summary.json")
        self.assertTrue(os.path.exists(summary_path))
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        for section in ("timing", "training_outcome", "throughput", "metadata"):
            self.assertIn(section, summary)

        self.assertEqual(summary["training_outcome"]["epochs_completed"], 2)
        self.assertEqual(summary["training_outcome"]["epochs_configured"], 2)
        self.assertFalse(summary["training_outcome"]["early_stopped"])
        self.assertIsNone(summary["training_outcome"]["early_stopped_at_epoch"])
        self.assertIsNotNone(summary["training_outcome"]["initial_training_loss"])
        self.assertIsNotNone(summary["training_outcome"]["final_training_loss"])

        self.assertEqual(summary["throughput"]["device"], "cpu")
        self.assertEqual(summary["throughput"]["batches_per_epoch"], 3)
        self.assertGreater(summary["throughput"]["samples_per_second"], 0)

        self.assertEqual(summary["metadata"]["config_key"], "test_config")
        self.assertEqual(summary["metadata"]["run_identifier"], "test_run")


class TestSaveConfig(unittest.TestCase):
    """``utils.save_config`` writes the run's config JSON and a copy of the config source."""

    def test_writes_config_json_and_source_copy(self):
        with tempfile.TemporaryDirectory() as out_dir:
            utils.save_config(out_dir, copy.deepcopy(src.config.config.CONFIGS["test_config"]))

            self.assertTrue(os.path.isfile(os.path.join(out_dir, "config.json")))
            copied = os.path.join(out_dir, "config_source.py")
            self.assertTrue(os.path.isfile(copied))
            with open(copied, encoding="utf-8") as copied_file:
                self.assertIn('"test_config"', copied_file.read())


class _LabelledDataset(torch.utils.data.Dataset):
    """Image-free dataset exposing the label API ``build_model_and_strategy`` reads."""

    def __init__(self, labels):
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.zeros(3, 8, 8), self.labels[index]

    def get_labels(self):
        return self.labels

    def get_class_counts(self):
        return torch.bincount(self.labels)


class _TinyBackbone(nn.Module):
    """Smallest backbone satisfying the ``get_output_dim`` contract."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 4)

    def get_output_dim(self):
        return 4

    def forward(self, images):
        return self.fc(images.flatten(1))


def _arcface_config():
    raw = copy.deepcopy(src.config.config.CONFIGS["test_config"])
    raw["training"]["loss"]["name"] = "arcface"
    raw["training"]["num_workers"] = 0
    raw.pop("lora", None)
    raw["model"]["backbone"] = {"name": "Tiny", "constructor": _TinyBackbone, "options": {}}
    raw["model"]["projection_head"] = {
        "name": "ArcMarginProduct",
        "constructor": src.models.arcface_heads.ArcMarginProduct,
        "options": {},
    }
    return utils.ensure_config_schema(raw)


class TestArcFaceValidationLabels(unittest.TestCase):
    """Validation labels are encoded into the training label space, including unseen identities."""

    @staticmethod
    def _fitted_encoder(labels):
        encoder = sklearn.preprocessing.LabelEncoder()
        encoder.fit(labels)
        return encoder

    def test_seen_labels_match_the_encoder(self):
        encoder = self._fitted_encoder([10, 20, 30])
        encoded = new_trainer.encode_labels(encoder, [30, 10, 20, 20])
        self.assertEqual(encoded.tolist(), encoder.transform([30, 10, 20, 20]).tolist())

    def test_unseen_labels_get_codes_above_the_trained_range(self):
        encoder = self._fitted_encoder([10, 20, 30])
        encoded = new_trainer.encode_labels(encoder, [20, 99, 77, 99])
        self.assertEqual(encoded.tolist(), [1, 4, 3, 4])

    def test_build_keeps_head_size_and_separates_unseen_identities(self):
        train_ds = _LabelledDataset([5, 5, 6, 6, 7, 7])
        valid_ds = _LabelledDataset([5, 8])
        model, _, _ = new_trainer.build_model_and_strategy(
            _arcface_config(), torch.device("cpu"), train_ds, valid_ds, [])

        self.assertEqual(model.head.out_features, 3)
        self.assertEqual(valid_ds.labels.tolist(), [0, 3])

    def test_build_without_validation_set_does_not_warn(self):
        train_ds = _LabelledDataset([5, 5, 6, 6, 7, 7])
        with patch("builtins.print") as mock_print:
            new_trainer.build_model_and_strategy(
                _arcface_config(), torch.device("cpu"), train_ds, None, [])

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertNotIn("mismatch", printed)


class _LoraBackbone(nn.Module):
    """Two-layer backbone whose ``fc`` layer is the LoRA target."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.out = nn.Linear(4, 4)

    def forward(self, x):
        return self.out(torch.relu(self.fc(x)))


def _lora_model():
    backbone = peft.get_peft_model(
        _LoraBackbone(), peft.LoraConfig(r=2, lora_alpha=2, target_modules=["fc"]))
    return nn.Sequential(backbone, nn.Linear(4, 2))


class TestLoraResume(unittest.TestCase):
    """Resuming from a LoRA checkpoint restores the adapters and keeps them trainable."""

    def test_resume_restores_adapters_and_keeps_them_trainable(self):
        torch.manual_seed(0)
        saved = _lora_model()
        for name, param in saved[0].named_parameters():
            if "lora_B" in name:
                nn.init.constant_(param, 0.5)

        trainer = new_trainer.RobustTrainer(
            saved, _MockStrategy(), torch.device("cpu"), _make_config(),
            train_dataset=MagicMock(), valid_dataset=MagicMock())
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            trainer.save_model(checkpoint_dir, "best")

            torch.manual_seed(1)
            resumed = _lora_model()
            utils.load_model_for_resume(checkpoint_dir, resumed[0], resumed[1])

        saved_lora = {n: p for n, p in saved[0].named_parameters() if "lora_" in n}
        resumed_lora = {n: p for n, p in resumed[0].named_parameters() if "lora_" in n}
        self.assertTrue(saved_lora)
        for name, param in saved_lora.items():
            self.assertTrue(torch.equal(param, resumed_lora[name]), name)
            self.assertTrue(resumed_lora[name].requires_grad, name)

        resumed(torch.ones(2, 4)).sum().backward()
        self.assertTrue(all(param.grad is not None for param in resumed_lora.values()))


class TestConfigSelectionFailures(unittest.TestCase):
    """Unknown or invalid configurations stop the run with a non-zero exit."""

    def test_unknown_config_key_exits_nonzero(self):
        with patch("sys.argv", ["new_trainer", "--configs", "no_such_config"]), \
                patch.object(new_trainer, "run_single_config") as run_single:
            with self.assertRaises(SystemExit) as raised:
                new_trainer.main()

        self.assertEqual(raised.exception.code, 1)
        run_single.assert_not_called()

    def test_invalid_config_exits_nonzero(self):
        args = MagicMock(dry_run=True)
        with patch.object(utils, "ensure_config_schema", side_effect=ValueError("bad")):
            with self.assertRaises(SystemExit) as raised:
                new_trainer.run_single_config("test_config", "run", args)

        self.assertEqual(raised.exception.code, 1)

    def test_failed_dry_run_checks_exit_nonzero(self):
        args = MagicMock(dry_run=True)
        for passed, expected in ((False, 1), (True, None)):
            with self.subTest(passed=passed):
                with patch.object(new_trainer, "perform_dry_run_checks", return_value=passed), \
                        patch.object(new_trainer, "print_config_summary"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if expected is None:
                        new_trainer.run_single_config("test_config", "run", args)
                    else:
                        with self.assertRaises(SystemExit) as raised:
                            new_trainer.run_single_config("test_config", "run", args)
                        self.assertEqual(raised.exception.code, expected)


class TestEmptyTrainingSet(unittest.TestCase):
    """An empty training split fails with a message that names the split-strategy fix."""

    def test_empty_train_split_names_split_strategy_none(self):
        config = _make_config()
        args = MagicMock(annotations_version=None, split_strategy="train-test")
        with patch.object(utils, "get_annotation_paths", return_value=("train.csv", "valid.csv")), \
                patch.object(new_trainer.src.individual_dataset, "IndividualDataset",
                             return_value=[]):
            with self.assertRaisesRegex(ValueError, "--split-strategy none"):
                new_trainer.setup_datasets_and_augmentations(config, {}, args)


class TestResumeFailures(unittest.TestCase):
    """A checkpoint that cannot be fully restored stops the run instead of training from base weights."""

    def _save_checkpoint(self, model, checkpoint_dir):
        new_trainer.RobustTrainer(
            model, _MockStrategy(), torch.device("cpu"), _make_config(),
            train_dataset=MagicMock(), valid_dataset=MagicMock()).save_model(checkpoint_dir, "best")

    def test_missing_checkpoint_directory_raises(self):
        with tempfile.TemporaryDirectory() as out_dir:
            model = _lora_model()
            with self.assertRaises(FileNotFoundError):
                utils.load_model_for_resume(os.path.join(out_dir, "absent"), model[0], model[1])

    def test_head_shape_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            self._save_checkpoint(_lora_model(), checkpoint_dir)
            wrong_head = nn.Linear(4, 3)
            with self.assertRaises(RuntimeError):
                utils.load_model_for_resume(checkpoint_dir, _lora_model()[0], wrong_head)

    def test_lora_checkpoint_into_plain_backbone_raises(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            self._save_checkpoint(_lora_model(), checkpoint_dir)
            with self.assertRaises((RuntimeError, ValueError)):
                utils.load_model_for_resume(checkpoint_dir, _LoraBackbone(), nn.Linear(4, 2))


class TestHelpText(unittest.TestCase):
    """Every configuration key quoted in the trainer's help exists in the registry."""

    def _help_text(self):
        output = io.StringIO()
        with patch("sys.argv", ["new_trainer", "--help"]), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit):
                new_trainer.main()
        return " ".join(output.getvalue().split())

    def test_help_examples_name_real_configs(self):
        text = self._help_text()
        quoted = re.search(r'e\.g\. "([^"]+)"', text)
        example = re.search(r"--configs ([a-z][a-z_0-9]*)", text)
        self.assertIsNotNone(quoted)
        self.assertIsNotNone(example)
        named = set(quoted.group(1).split()) | {example.group(1)}
        self.assertLessEqual(named, set(src.config.config.CONFIGS))


class TestDryRunResumeCheck(unittest.TestCase):
    """The dry run reports a resume source that training could not load."""

    def _dry_run_output(self, trained_models_dir, resume_from_model):
        return self._dry_run(trained_models_dir, resume_from_model)[1]

    def _dry_run(self, trained_models_dir, resume_from_model):
        args = MagicMock(annotations_version=None, split_strategy="none",
                         resume_from_model=resume_from_model)
        output = io.StringIO()
        with patch.object(constants, "TRAINED_MODELS_DIR", trained_models_dir), \
                contextlib.redirect_stdout(output):
            passed = new_trainer.perform_dry_run_checks("test_config", "save-dir", args)
        return passed, output.getvalue()

    def test_missing_resume_source_fails_the_dry_run(self):
        with tempfile.TemporaryDirectory() as root:
            passed, _ = self._dry_run(root, "model/absent")
        self.assertFalse(passed)

    def test_resume_directory_without_weights_is_reported_missing(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "model", "config"))
            output = self._dry_run_output(root, "model/config")
        self.assertRegex(output, r"Resume From.*MISSING.*backbone weights")

    def test_absent_resume_directory_is_reported_missing(self):
        with tempfile.TemporaryDirectory() as root:
            output = self._dry_run_output(root, "model/absent")
        self.assertRegex(output, r"Resume From.*MISSING")

    def test_complete_resume_checkpoint_is_reported_ok(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint_dir = os.path.join(root, "model", "config")
            os.makedirs(checkpoint_dir)
            new_trainer.RobustTrainer(
                _lora_model(), _MockStrategy(), torch.device("cpu"), _make_config(),
                train_dataset=MagicMock(), valid_dataset=MagicMock()
            ).save_model(checkpoint_dir, "best")
            output = self._dry_run_output(root, "model/config")
        self.assertRegex(output, r"Resume From.*\(OK\)")


class TestEpochsOverride(unittest.TestCase):
    """--epochs changes the epoch count, rescales a matching T_max, and leaves the registry alone."""

    def _entry(self, epochs, t_max):
        entry = copy.deepcopy(src.config.config.CONFIGS["test_config"])
        entry["training"]["epochs"] = epochs
        entry["training"]["scheduler"] = {
            "name": "CosineAnnealingLR",
            "constructor": torch.optim.lr_scheduler.CosineAnnealingLR,
            "options": {"T_max": t_max},
        }
        return entry

    def test_matching_t_max_follows_the_new_epoch_count(self):
        entry = self._entry(epochs=50, t_max=50)
        overridden = new_trainer.apply_epochs_override(entry, 3)
        self.assertEqual(overridden["training"]["epochs"], 3)
        self.assertEqual(overridden["training"]["scheduler"]["options"]["T_max"], 3)

    def test_differing_t_max_is_left_alone_with_a_note(self):
        entry = self._entry(epochs=50, t_max=20)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            overridden = new_trainer.apply_epochs_override(entry, 3)
        self.assertEqual(overridden["training"]["epochs"], 3)
        self.assertEqual(overridden["training"]["scheduler"]["options"]["T_max"], 20)
        self.assertIn("T_max", output.getvalue())

    def test_config_without_a_scheduler_only_changes_epochs(self):
        entry = copy.deepcopy(src.config.config.CONFIGS["test_config"])
        entry["training"]["scheduler"] = None
        overridden = new_trainer.apply_epochs_override(entry, 2)
        self.assertEqual(overridden["training"]["epochs"], 2)
        self.assertIsNone(overridden["training"]["scheduler"])

    def test_zero_epochs_keeps_a_valid_t_max(self):
        entry = self._entry(epochs=50, t_max=50)
        overridden = new_trainer.apply_epochs_override(entry, 0)
        self.assertEqual(overridden["training"]["epochs"], 0)
        self.assertEqual(overridden["training"]["scheduler"]["options"]["T_max"], 50)

    def test_original_entry_is_not_modified(self):
        entry = self._entry(epochs=50, t_max=50)
        new_trainer.apply_epochs_override(entry, 3)
        self.assertEqual(entry["training"]["epochs"], 50)
        self.assertEqual(entry["training"]["scheduler"]["options"]["T_max"], 50)

    def test_negative_epochs_are_rejected_by_the_parser(self):
        with patch("sys.argv", ["new_trainer", "--epochs", "-1"]), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                new_trainer.main()
        self.assertEqual(raised.exception.code, 2)

    def test_run_trains_and_records_the_overridden_epochs(self):
        args = MagicMock(dry_run=False, resume_from_model=None, no_email=True, epochs=1,
                         overwrite=False)
        with tempfile.TemporaryDirectory() as root, \
                patch.object(constants, "TRAINED_MODELS_DIR", root), \
                patch.object(new_trainer, "setup_datasets_and_augmentations",
                             return_value=(MagicMock(), MagicMock(), [])), \
                patch.object(new_trainer, "build_model_and_strategy",
                             return_value=(MagicMock(), MagicMock(), MagicMock())), \
                patch.object(new_trainer, "RobustTrainer") as trainer_class, \
                patch.object(utils, "notify_completion"), \
                contextlib.redirect_stdout(io.StringIO()):
            new_trainer.run_single_config("test_config", "run", args)
            with open(os.path.join(root, "run", "test_config", "config.json"),
                      encoding="utf-8") as saved:
                saved_epochs = json.load(saved)["training"]["epochs"]
        trained_config = trainer_class.call_args.args[3]
        self.assertEqual(trained_config.training.epochs, 1)
        self.assertEqual(saved_epochs, 1)
        self.assertNotEqual(src.config.config.CONFIGS["test_config"]["training"]["epochs"], 1)


class TestOverwriteGuard(unittest.TestCase):
    """Trained weights are never replaced unless --overwrite is passed."""

    def setUp(self):
        self._root = tempfile.TemporaryDirectory()
        self.addCleanup(self._root.cleanup)
        self.models = os.path.join(self._root.name, "models")
        os.makedirs(self.models)
        images = os.path.join(self._root.name, "images")
        os.makedirs(images)
        for name, value in (("TRAINED_MODELS_DIR", self.models), ("IMAGES_DIR", images),
                            ("ANNOTATIONS_DIR", self._root.name)):
            patcher = patch.object(constants, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _make_dir(self, run, config="test_config", files=()):
        directory = os.path.join(self.models, run, config)
        os.makedirs(directory)
        for name in files:
            open(os.path.join(directory, name), "w", encoding="utf-8").close()
        return directory

    def _run_main(self, *extra):
        argv = ["new_trainer", "--model-name", "run", "--configs", "test_config", *extra]
        output = io.StringIO()
        with patch("sys.argv", argv), patch.object(new_trainer, "run_single_config") as run_single, \
                contextlib.redirect_stdout(output):
            try:
                new_trainer.main()
                code = 0
            except SystemExit as raised:
                code = raised.code
        return code, run_single, output.getvalue()

    def test_existing_weights_stop_the_run_naming_the_flag(self):
        self._make_dir("run", files=["best-backbone-state.pt"])
        code, run_single, output = self._run_main()
        self.assertEqual(code, 1)
        run_single.assert_not_called()
        self.assertIn("--overwrite", output)
        self.assertIn(os.path.join("run", "test_config"), output)

    def test_last_and_init_weights_also_count(self):
        for prefix in ("last-projection-head.pt", "init-backbone-state.pt"):
            with self.subTest(file=prefix):
                directory = self._make_dir("run", files=[prefix])
                code, _, _ = self._run_main()
                self.assertEqual(code, 1)
                shutil.rmtree(os.path.dirname(directory))

    def test_folder_holding_only_a_config_does_not_block(self):
        self._make_dir("run", files=["config.json"])
        code, run_single, _ = self._run_main()
        self.assertEqual(code, 0)
        run_single.assert_called_once()

    def test_overwrite_flag_allows_the_run(self):
        self._make_dir("run", files=["best-backbone-state.pt"])
        code, run_single, _ = self._run_main("--overwrite")
        self.assertEqual(code, 0)
        run_single.assert_called_once()

    def test_new_model_name_is_unaffected(self):
        self._make_dir("other", files=["best-backbone-state.pt"])
        code, run_single, _ = self._run_main()
        self.assertEqual(code, 0)
        run_single.assert_called_once()

    def test_conflict_in_a_later_config_stops_before_any_training(self):
        self._make_dir("run", config="vit_arcface", files=["best-backbone-state.pt"])
        argv = ["new_trainer", "--model-name", "run", "--configs", "test_config", "vit_arcface"]
        with patch("sys.argv", argv), patch.object(new_trainer, "run_single_config") as run_single, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                new_trainer.main()
        self.assertEqual(raised.exception.code, 1)
        run_single.assert_not_called()

    def test_resuming_in_place_needs_the_flag(self):
        self._make_dir("run", files=["best-backbone-state.pt"])
        code, run_single, output = self._run_main("--resume-from-model", "run/test_config")
        self.assertEqual(code, 1)
        run_single.assert_not_called()
        self.assertIn("resume", output.lower())
        code, run_single, _ = self._run_main("--resume-from-model", "run/test_config", "--overwrite")
        self.assertEqual(code, 0)
        run_single.assert_called_once()

    def test_resuming_into_a_new_model_name_needs_no_flag(self):
        self._make_dir("source", files=["best-backbone-state.pt"])
        code, run_single, _ = self._run_main("--resume-from-model", "source/test_config")
        self.assertEqual(code, 0)
        run_single.assert_called_once()

    def test_training_every_config_skips_the_test_config(self):
        argv = ["new_trainer", "--model-name", "run"]
        with patch("sys.argv", argv), patch.object(new_trainer, "run_single_config") as run_single, \
                contextlib.redirect_stdout(io.StringIO()):
            new_trainer.main()
        trained = [call.args[0] for call in run_single.call_args_list]
        self.assertNotIn("test_config", trained)
        self.assertIn("miewid_arcface", trained)
        self.assertEqual(len(trained), len(src.config.config.CONFIGS) - 1)

    def _run_single_config(self, args, on_resume=None):
        with patch.object(new_trainer, "setup_datasets_and_augmentations",
                          return_value=(MagicMock(), MagicMock(), [])), \
                patch.object(new_trainer, "build_model_and_strategy",
                             return_value=(MagicMock(), MagicMock(), MagicMock())), \
                patch.object(utils, "load_model_for_resume", side_effect=on_resume), \
                patch.object(new_trainer, "RobustTrainer"), \
                patch.object(utils, "notify_completion"), \
                contextlib.redirect_stdout(io.StringIO()):
            new_trainer.run_single_config("test_config", "run", args)

    def test_overwrite_clears_the_previous_run_but_keeps_unknown_files(self):
        directory = self._make_dir("run", files=[
            "best-backbone-state.pt", "last-projection-head.pt", "init-backbone-state.pt",
            "training_summary.json", "training_history.csv", "notes.txt"])
        os.makedirs(os.path.join(directory, "best-backbone-lora"))
        os.makedirs(os.path.join(directory, "logs"))
        args = MagicMock(dry_run=False, resume_from_model=None, no_email=True, epochs=None,
                         overwrite=True)
        self._run_single_config(args)
        self.assertEqual(sorted(os.listdir(directory)),
                         ["config.json", "config_source.py", "notes.txt"])

    def test_without_overwrite_nothing_is_deleted(self):
        directory = self._make_dir("run", files=["training_summary.json"])
        args = MagicMock(dry_run=False, resume_from_model=None, no_email=True, epochs=None,
                         overwrite=False)
        self._run_single_config(args)
        self.assertIn("training_summary.json", os.listdir(directory))

    def test_resuming_in_place_loads_weights_before_clearing_them(self):
        directory = self._make_dir("run", files=["best-backbone-state.pt"])
        seen_at_resume = []
        args = MagicMock(dry_run=False, resume_from_model="run/test_config", no_email=True,
                         epochs=None, overwrite=True)
        self._run_single_config(args, on_resume=lambda *_: seen_at_resume.extend(os.listdir(directory)))
        self.assertIn("best-backbone-state.pt", seen_at_resume)
        self.assertNotIn("best-backbone-state.pt", os.listdir(directory))

    def test_dry_run_reports_the_conflict_without_exiting(self):
        directory = self._make_dir("run", files=["best-backbone-state.pt"])
        args = MagicMock(annotations_version=None, split_strategy="none", resume_from_model=None,
                         overwrite=False)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            new_trainer.perform_dry_run_checks("test_config", directory, args)
        self.assertRegex(output.getvalue(), r"Model Save Directory.*--overwrite")


class TestDataLoaderWorkers(unittest.TestCase):
    """Worker processes start by spawn so a Linux fork cannot freeze them inside a torch operation."""

    def _kwargs(self, num_workers):
        config = _make_config()
        config.training.num_workers = num_workers
        return new_trainer._build_dataloader_kwargs(config)  # pylint: disable=protected-access

    def test_workers_are_spawned_not_forked(self):
        kwargs = self._kwargs(2)
        self.assertEqual(kwargs["multiprocessing_context"], "spawn")
        self.assertTrue(kwargs["persistent_workers"])

    def test_in_process_loading_sets_no_worker_options(self):
        kwargs = self._kwargs(0)
        self.assertNotIn("multiprocessing_context", kwargs)
        self.assertNotIn("persistent_workers", kwargs)


if __name__ == "__main__":
    unittest.main()
