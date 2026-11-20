
import unittest
import torch
import math
import src.models.arcface_heads as arcface

class TestArcFace(unittest.TestCase):
    def setUp(self):
        self.in_features = 3
        self.out_features = 2
        self.s = 30.0
        self.m = 0.50
        self.arc = arcface.ArcMarginProduct(self.in_features, self.out_features, s=self.s, m=self.m, easy_margin=False)

    def test_normalization(self):
        """Verify input features and weights are normalized."""
        # Create unnormalized inputs
        # Feature: [10, 0, 0]. Norm -> [1, 0, 0].
        
        # We can just check that if we double the input magnitude, the output remains identical (before s scaling)
        # Because cosine similarity depends only on angle.
        
        input1 = torch.randn(2, 3)
        input2 = input1 * 100.0 # Same direction, larger magnitude.
        label = torch.tensor([0, 1])
        
        out1 = self.arc(input1, label)
        out2 = self.arc(input2, label)
        
        # Should be identical (within float precision)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_margin_application(self):
        """Verify margin is subtracted from the correct class angle."""
        # x aligns with class 0, so cos(theta) = 1 and the target logit is s * cos(0 + m) = s * cos(0.5).
        
        # Manually set weight for class 0 to be [1, 0, 0].
        with torch.no_grad():
            self.arc.weight[0] = torch.tensor([1.0, 0.0, 0.0])
            self.arc.weight[1] = torch.tensor([0.0, 1.0, 0.0])
        
        # Input aligned with class 0.
        x = torch.tensor([[10.0, 0.0, 0.0]])
        label = torch.tensor([0])
        
        output = self.arc(x, label)
        # Output is shape [1, 2] with class 0 equal to s * cos(m) and class 1 orthogonal (0).
        
        expected_target = self.s * math.cos(self.m)
        expected_non_target = self.s * 0.0
        
        self.assertAlmostEqual(output[0, 0].item(), expected_target, places=4)
        self.assertAlmostEqual(output[0, 1].item(), expected_non_target, places=4)

    def test_scaling(self):
        """Verify s factor behaves as expected."""
        # Use simple inputs.
        x = torch.randn(1, 3)
        lbl = torch.tensor([0])
        
        # Compute with s=1.0 by dividing result.
        out_scaled = self.arc(x, lbl)
        
        arc60 = arcface.ArcMarginProduct(self.in_features, self.out_features, s=60.0, m=self.m, easy_margin=False)
        arc60.weight.data = self.arc.weight.data.clone()
        
        out60 = arc60(x, lbl)
        
        self.assertTrue(torch.allclose(out60, out_scaled * 2.0, atol=1e-5))

    def test_easy_margin(self):
        """Verify easy_margin behavior (no exception raised, runs correctly)."""
        # Easy margin handles cases where theta + m > pi.
        # We force theta to be close to pi (input opposite to weight).
        
        with torch.no_grad():
             self.arc.weight[0] = torch.tensor([1.0, 0.0, 0.0])
             
        # Input opposite to class 0: [-1, 0, 0].
        x = torch.tensor([[-1.0, 0.0, 0.0]])
        label = torch.tensor([0])
        out = self.arc(x, label)
        self.assertFalse(torch.isnan(out).any())

if __name__ == '__main__':
    unittest.main()
