"""
Tests for ``src.multimodal.train_multimodal.FusionTrainer``.

Mirrors the scenarios in ``tests/unit/learning/test_new_trainer.py`` but
adapted to the fusion trainer's contract: a 2-input model, a 3-tuple
dataset, externally-constructed optimizer/scheduler/loss/evaluator,
plain-dict config (not pydantic), and ``init/best/last-fusion_model.pt``
checkpoint naming.
"""
import copy
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

import experimental.multimodal.trained_fusion.configs as mm_config
import experimental.multimodal.trained_fusion.train_multimodal as tm
from src.core import constants


def _make_config(
        *,
        epochs: int = 1,
        gradient_accumulation_steps: int = 1,
        early_stopping_enabled: bool = False,
        early_stopping_patience: int = 5,
        warmup_epochs: int = 0,
):
    """Build a fusion config dict starting from FUSION_CONFIGS, with overrides."""
    raw = copy.deepcopy(mm_config.TRAINED_FUSION_CONFIGS["concatenate"])
    raw["training"]["epochs"] = epochs
    raw["training"]["gradient_accumulation_steps"] = gradient_accumulation_steps
    raw["training"]["warmup_epochs"] = warmup_epochs
    raw["training"]["early_stopping"] = {
        "enabled": early_stopping_enabled,
        "patience": early_stopping_patience,
        "min_delta": 0.0,
        "mode": "max",
    }
    return raw


class _FusionMockModel(nn.Module):
    """
    Minimal 2-input fusion model: projects each input through a Linear and
    returns their concatenation. Trainable params let the optimizer step.
    """
    def __init__(self, image_dim: int = 8, metadata_dim: int = 4, out_dim: int = 4):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, out_dim)
        self.metadata_proj = nn.Linear(metadata_dim, out_dim)

    def forward(self, image_emb, metadata_emb):
        return torch.cat([self.image_proj(image_emb), self.metadata_proj(metadata_emb)], dim=1)


def _loss_fn(embeddings, _labels):
    """Simple sum loss; differentiable through the model."""
    return embeddings.sum()


class _FakeFusionLoader:
    """Iterable yielding ``(image_emb, metadata, labels)`` 3-tuples per batch."""
    def __init__(self, num_batches: int, batch_size: int = 4,
                 image_dim: int = 8, metadata_dim: int = 4):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.image_dim = image_dim
        self.metadata_dim = metadata_dim

    def __iter__(self):
        for _ in range(self.num_batches):
            yield (
                torch.randn(self.batch_size, self.image_dim),
                torch.randn(self.batch_size, self.metadata_dim),
                torch.randint(0, 2, (self.batch_size,)),
            )

    def __len__(self):
        return self.num_batches


class TestFusionTrainer(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.test_dir = tempfile.mkdtemp()
        # Evaluator returns a metric and a dict whose key matches the
        # CHECKPOINT_METRIC constant; ensures the trainer's best-model logic
        # has the metric it expects rather than falling back to the warning path.
        self.evaluator = MagicMock(return_value=(0.9, {constants.CHECKPOINT_METRIC: 0.9}))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _make_trainer(self, *, config=None, scheduler_enabled: bool = True):
        config = config or _make_config()
        model = _FusionMockModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = None
        if scheduler_enabled:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return tm.FusionTrainer(
            model=model,
            device=self.device,
            config=config,
            train_dataset=MagicMock(),
            valid_dataset=MagicMock(),
            loss_fn=_loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            evaluator=self.evaluator,
            config_key="fusion_test_config",
            run_identifier="fusion_test_run",
        )

    # ---- Core training loop --------------------------------------------

    def test_train_smoke_test(self):
        """One epoch, three batches: optimizer steps once per batch, scheduler once per epoch."""
        trainer = self._make_trainer()
        trainer.optimizer.step = MagicMock(side_effect=trainer.optimizer.step)
        trainer.scheduler.step = MagicMock(side_effect=trainer.scheduler.step)

        trainer.train(_FakeFusionLoader(num_batches=3),
                      os.path.join(self.test_dir, "models"))

        self.assertEqual(trainer.optimizer.step.call_count, 3)
        self.assertEqual(trainer.scheduler.step.call_count, 1)

    def test_gradient_accumulation(self):
        """With accumulation=2 over 4 batches, optimizer steps only twice."""
        trainer = self._make_trainer(
            config=_make_config(gradient_accumulation_steps=2),
            scheduler_enabled=False,
        )
        trainer.optimizer.step = MagicMock(side_effect=trainer.optimizer.step)

        trainer.train(_FakeFusionLoader(num_batches=4),
                      os.path.join(self.test_dir, "models"))

        self.assertEqual(trainer.optimizer.step.call_count, 2)

    # ---- Checkpointing -------------------------------------------------

    def test_save_best_model(self):
        """``best`` is saved only when the validation metric improves."""
        scores = [0.5, 0.4, 0.6]

        def evaluator_side_effect(model, train, valid, device):
            score = scores[evaluator_side_effect.calls]
            evaluator_side_effect.calls += 1
            return score, {constants.CHECKPOINT_METRIC: score}
        evaluator_side_effect.calls = 0

        self.evaluator = MagicMock(side_effect=evaluator_side_effect)
        trainer = self._make_trainer(config=_make_config(epochs=3))
        trainer.save_model = MagicMock()

        trainer.train(_FakeFusionLoader(num_batches=1),
                      os.path.join(self.test_dir, "models"))

        prefixes = [call.args[1] for call in trainer.save_model.call_args_list]
        # 1× init + 2× best (epochs 0 and 2 improve) + 3× last (every epoch).
        self.assertEqual(prefixes.count("init"), 1)
        self.assertEqual(prefixes.count("best"), 2)
        self.assertEqual(prefixes.count("last"), 3)

    # ---- Observability -------------------------------------------------

    @patch('src.core.utils.log_training_history_csv')
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_logging_calls(self, mock_writer_cls, mock_csv_logger):
        """Per-step TB scalars and per-epoch CSV logging both fire."""
        trainer = self._make_trainer()
        mock_writer = mock_writer_cls.return_value

        # 11 batches so the global_step >= 10 tag-write triggers at least once.
        trainer.train(_FakeFusionLoader(num_batches=11),
                      os.path.join(self.test_dir, "models"))

        scalar_tags = [call.args[0] for call in mock_writer.add_scalar.call_args_list]
        self.assertIn("Loss/Train", scalar_tags)
        self.assertIn("System/LearningRate", scalar_tags)
        self.assertIn("System/GradientNorm", scalar_tags)

        # CSV logger fires once per epoch.
        self.assertEqual(mock_csv_logger.call_count, 1)

    def test_training_summary_file(self):
        """``training_summary.json`` is written and matches the expected schema."""
        trainer = self._make_trainer(config=_make_config(epochs=2))
        models_dir = os.path.join(self.test_dir, "models")
        trainer.train(_FakeFusionLoader(num_batches=3), models_dir)

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

        self.assertEqual(summary["metadata"]["config_key"], "fusion_test_config")
        self.assertEqual(summary["metadata"]["run_identifier"], "fusion_test_run")

    # ---- Fusion-specific checkpoint naming ----------------------------

    def test_checkpoint_filenames(self):
        """Fusion checkpoints use the ``<prefix>-fusion_model.pt`` naming."""
        trainer = self._make_trainer(config=_make_config(epochs=1))
        models_dir = os.path.join(self.test_dir, "models")
        trainer.train(_FakeFusionLoader(num_batches=2), models_dir)

        for prefix in ("init", "best", "last"):
            self.assertTrue(
                os.path.exists(os.path.join(models_dir, f"{prefix}-fusion_model.pt")),
                f"Expected {prefix}-fusion_model.pt to exist",
            )

    # ---- Early stopping ------------------------------------------------

    def test_early_stopping_short_circuits(self):
        """Early stopping triggers and the loop exits before all configured epochs run."""
        # Metric stays flat -> no improvement -> patience exhausted quickly.
        flat_evaluator = MagicMock(
            return_value=(0.5, {constants.CHECKPOINT_METRIC: 0.5})
        )
        config = _make_config(
            epochs=10,
            early_stopping_enabled=True,
            early_stopping_patience=2,
        )
        self.evaluator = flat_evaluator
        trainer = self._make_trainer(config=config)
        models_dir = os.path.join(self.test_dir, "models")

        trainer.train(_FakeFusionLoader(num_batches=1), models_dir)

        # 10 epochs configured but should stop after ~3 (1 baseline + 2 patience).
        with open(os.path.join(models_dir, "training_summary.json"),
                  "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertTrue(summary["training_outcome"]["early_stopped"])
        self.assertLess(summary["training_outcome"]["epochs_completed"], 10)


if __name__ == "__main__":
    unittest.main()
