
import sys
import os
# Add project root to path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from src.learning.trainer.new_trainer import (
    ContrastiveStrategy, SupervisedStrategy, metric_learning_collate
)

class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

        # Mock Model that just returns the input (identity) or shape.
        self.mock_model = MagicMock(spec=nn.Module)
        self.mock_model.return_value = torch.tensor([0.0], requires_grad=True) # Default return.

        # Mock Loss Function.
        self.mock_loss_fn = MagicMock(return_value=torch.tensor(10.0, requires_grad=True))

    def test_metric_learning_collate_single_view(self):
        """Verify metric_learning_collate handles single-view (tensor) items correctly."""
        # Simulate AugmentedDataset items: each is (image_tensor, label_tensor).
        batch = [
            (torch.randn(3, 32, 32), torch.tensor(0)),
            (torch.randn(3, 32, 32), torch.tensor(1)),
            (torch.randn(3, 32, 32), torch.tensor(0)),
        ]
        images, labels = metric_learning_collate(batch)
        self.assertEqual(images.shape, (3, 3, 32, 32))
        self.assertEqual(labels.shape, (3,))

    def test_metric_learning_collate_multi_view(self):
        """Verify metric_learning_collate handles multi-view (list) items correctly."""
        # AugmentedDataset yields ([img1, img2], label) when generating multiple
        # views: one label covers every view, and collate replicates it.
        # (ContrastiveLearningIndividualDataset is now an alias for
        # AugmentedDataset, so no dataset emits a per-view list of labels.)
        batch = [
            ([torch.randn(3, 32, 32), torch.randn(3, 32, 32)], torch.tensor(0)),
            ([torch.randn(3, 32, 32), torch.randn(3, 32, 32)], torch.tensor(1)),
        ]
        images, labels = metric_learning_collate(batch)
        self.assertEqual(images.shape, (4, 3, 32, 32))
        self.assertEqual(labels.shape, (4,))
        self.assertTrue(torch.equal(labels, torch.tensor([0, 0, 1, 1])))

    def test_contrastive_batch_processing(self):
        """
        Verify that ContrastiveStrategy processes a collated tensor batch correctly.
        Input is already collated by metric_learning_collate into (N, C, H, W) tensors.
        """
        strategy = ContrastiveStrategy(loss_fn=self.mock_loss_fn)

        # Pre-collated tensor batch (as metric_learning_collate would produce).
        batch_size = 4
        feature_dim = 10
        images = torch.randn(batch_size, feature_dim)
        labels = torch.randint(0, 2, (batch_size,))
        batch = (images, labels)

        # Configure model to return embeddings.
        self.mock_model.return_value = torch.zeros((batch_size, feature_dim), requires_grad=True)

        _, _, _ = strategy.process_batch(self.mock_model, batch, self.device)

        # Check model call args.
        args, _ = self.mock_model.call_args
        passed_input = args[0]

        # Expectation: tensor of shape (4, 10).
        self.assertEqual(passed_input.shape[0], 4)
        self.assertEqual(passed_input.shape[1], 10)

    def test_contrastive_loss_normalization(self):
        """
        Verify that ContrastiveStrategy normalizes loss by batch size for logging.
        """
        strategy = ContrastiveStrategy(loss_fn=self.mock_loss_fn)

        # Batch size of 4 with sum loss 10.0 produces an expected logged loss of 2.5.

        images = torch.randn(4, 10)
        labels = torch.randint(0, 2, (4,))
        batch = (images, labels)

        self.mock_model.return_value = images # Embeddings = Input.

        loss, _, logs = strategy.process_batch(self.mock_model, batch, self.device)

        self.assertEqual(loss.item(), 10.0) # Returned loss handles backprop (sum).
        self.assertEqual(logs["loss"], 2.5) # Logged loss is normalized.

    def test_supervised_batch_processing(self):
        """
        Verify that SupervisedStrategy processes a collated tensor batch correctly.
        """
        strategy = SupervisedStrategy()

        # Pre-collated tensor batch.
        batch_size = 4
        feature_dim = 10
        images = torch.randn(batch_size, feature_dim)
        labels = torch.randint(0, 2, (batch_size,))
        batch = (images, labels)

        # Mock model return: loss, logits, aux_logs.
        self.mock_model.return_value = (
            torch.tensor(0.5, requires_grad=True),
            torch.randn(4, 5), # Logits: 4 samples, 5 classes.
            {}
        )

        strategy.process_batch(self.mock_model, batch, self.device)

        # Check model call args.
        args, _ = self.mock_model.call_args
        passed_input = args[0]

        self.assertEqual(passed_input.shape[0], 4)

if __name__ == '__main__':
    unittest.main()
