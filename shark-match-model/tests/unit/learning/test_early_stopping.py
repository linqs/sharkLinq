import unittest
from unittest.mock import MagicMock
import torch
from src.learning.trainer.new_trainer import EarlyStopping

class TestEarlyStopping(unittest.TestCase):
    def test_early_stopping_trigger(self):
        """Test that early stopping initiates when patience is exceeded."""
        # Setup: Patience 2, Min Delta 0.1, Mode Max.
        es = EarlyStopping(patience=2, min_delta=0.1, mode="max")
        
        # Epoch 1: Score 0.5 (Best: 0.5).
        es(0.5)
        self.assertEqual(es.counter, 0)
        self.assertFalse(es.early_stop)
        
        # Epoch 2: Score 0.55 (Improvement 0.05 < 0.1) -> Counter 1.
        es(0.55)
        self.assertEqual(es.counter, 1)
        self.assertFalse(es.early_stop)
        
        # Epoch 3: Score 0.55 (No improvement) -> Counter 2 -> Stop.
        es(0.55)
        self.assertEqual(es.counter, 2)
        self.assertTrue(es.early_stop)

    def test_early_stopping_reset(self):
        """Test that counter resets upon improvement."""
        es = EarlyStopping(patience=2, min_delta=0.1, mode="max")
        
        # Epoch 1: 0.5.
        es(0.5)
        self.assertEqual(es.counter, 0)
        
        # Epoch 2: 0.52 (No imp) -> Counter 1.
        es(0.52)
        self.assertEqual(es.counter, 1)
        
        # Epoch 3: 0.7 (Improvement 0.18 > 0.1) -> Counter 0.
        es(0.7)
        self.assertEqual(es.counter, 0)
        self.assertFalse(es.early_stop)

    def test_min_mode(self):
        """Test 'min' mode (e.g. for loss)."""
        es = EarlyStopping(patience=1, min_delta=0.1, mode="min")
        
        # Epoch 1: Loss 1.0.
        es(1.0)
        
        # Epoch 2: Loss 0.95 (Imp 0.05 < 0.1) -> Counter 1 -> Stop.
        es(0.95)
        self.assertTrue(es.early_stop)

if __name__ == "__main__":
    unittest.main()
