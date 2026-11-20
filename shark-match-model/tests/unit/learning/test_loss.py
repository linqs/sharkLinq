
import unittest
import torch
import src.learning.loss.triplet as triplet_loss

class TestTripletLoss(unittest.TestCase):
    def setUp(self):
        # Create a loss function instance.
        self.loss_fn_hard = triplet_loss.create_triplet_loss(margin=0.2, mining_strategy="batch_hard")
        self.loss_fn_semi = triplet_loss.create_triplet_loss(margin=0.2, mining_strategy="semi_hard")
        
    def test_batch_hard(self):
        """Verify batch hard mining selects hardest positive and hardest negative."""
        # Embeddings with positive pair (A, B) and negative sample C.
        embedding = torch.tensor([
            [1.0, 0.0],  # A (Anchor).
            [0.6, 0.8],  # B (Pos for A).
            [0.0, 1.0]   # C (Neg for A).
        ])
        labels = torch.tensor([0, 0, 1])
        
        loss, diagnostics = self.loss_fn_hard(embedding, labels)
        
        # We check it's loosely correct (positive non-zero).
        self.assertGreater(loss.item(), 0)
        self.assertLess(loss.item(), 1.0)
        # Verify diagnostics are returned.
        self.assertIn("Mining/FractionValidTriplets", diagnostics)

    def test_semi_hard(self):
        """Verify semi-hard mining preference."""
        embedding = torch.randn(10, 128)
        # Normalize to be safe.
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        labels = torch.randint(0, 2, (10,))
        
        loss_hard, _ = self.loss_fn_hard(embedding, labels)
        loss_semi, _ = self.loss_fn_semi(embedding, labels)
        
        # Triplet loss sums over batch.
        # In general, Hard mining creates larger gradients/loss than Semi-Hard.
        if loss_hard > 0:
            self.assertGreaterEqual(loss_hard.item(), loss_semi.item())

    def test_normalization(self):
        """Verify implicit normalization."""
        # Vectors with same orientation but different lengths yield zero distance when normalized.
        embedding = torch.tensor([ [10.0, 0.0], [5.0, 0.0] ])
        labels = torch.tensor([0, 1]) # Negatives.
        
        # Hardest negative distance is 0 due to shared orientation, giving 0.2 loss per anchor without positive pairs.
        loss_fn_norm = triplet_loss.create_triplet_loss(margin=0.2, mining_strategy="batch_hard", normalize_embeddings=True)
        loss, _ = loss_fn_norm(embedding, labels)
        
        # Sums margin loss of 0.2 across both negative anchor pairs for an expected total of 0.4.
        self.assertAlmostEqual(loss.item(), 0.4, places=5)

    def test_self_distance(self):
        """Verify distance to self is ignored/handled."""
        # 2 Identical samples with DIFFERENT labels (so they are negative pair).
        # This checks that d(A, A_copy) = 0.
        embedding = torch.tensor([ [1.0, 0.0], [1.0, 0.0] ])
        labels = torch.tensor([0, 1])
        
        # Should be same as normalization test (dist 0).
        loss, _ = self.loss_fn_hard(embedding, labels)
        self.assertAlmostEqual(loss.item(), 0.4, places=5)

    def test_edge_cases(self):
        """Verify empty/singleton handling."""
        # No positives (all unique labels) -> Loss handled (pos term 0).
        emb = torch.randn(3, 4)
        lbl = torch.tensor([0, 1, 2])
        loss, _ = self.loss_fn_hard(emb, lbl)
        self.assertFalse(torch.isnan(loss))
        
        # No negatives (all same label) -> Loss handled (neg term 0 or inf?).
        lbl_same = torch.tensor([0, 0, 0])
        loss_same, _ = self.loss_fn_hard(emb, lbl_same)
        self.assertFalse(torch.isnan(loss_same))

if __name__ == '__main__':
    unittest.main()
