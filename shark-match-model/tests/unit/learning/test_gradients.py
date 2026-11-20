
import unittest
import torch
import torch.nn as nn
import peft

# Helper Mock for Backbone
class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3) # "Backbone layer".
        self.fc = nn.Linear(16, 10)     # "Backbone layer".
    
    def forward(self, x):
        return self.fc(self.conv(x).mean([2,3]))

class TestGradients(unittest.TestCase):
    
    def test_lora_freezing_logic(self):
        """Verify LoRA freezing logic ensures backbone parameters are frozen while adapter and head parameters train."""
        print("\n--- Starting Test B: Frozen Gradient Check (Unit) ---")
        
        # Setup model.
        backbone = MockBackbone()
        
        # Define LoRA config.
        lora_config = peft.LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["fc"], # Target the linear layer.
            lora_dropout=0.1,
            bias="none"
        )
        
        # Apply LoRA.
        model = peft.get_peft_model(backbone, lora_config)
        
        # Add a head (trainable).
        head = nn.Linear(10, 2)
        
        full_model = nn.Sequential(model, head)
        
        # Run forward/backward.
        input_tensor = torch.randn(1, 3, 32, 32)
        labels = torch.tensor([0])
        
        output = full_model(input_tensor) # (1, 2).
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        
        # Inspect gradients.
        print("Inspecting gradients...")
        print("Parameter names:", list(dict(model.named_parameters()).keys()))
        
        # Check conv (Untouched base layer).
        conv_weight = dict(model.named_parameters())["base_model.model.conv.weight"]
        self.assertIsNone(conv_weight.grad, "Base backbone layer (Conv) should have None grad (Frozen)")
        
        # Check FC base weight.
        fc_weight = dict(model.named_parameters())["base_model.model.fc.base_layer.weight"]
        self.assertIsNone(fc_weight.grad, "Targeted base layer (FC) weight should have None grad (Frozen)")
        
        # Check LoRA parameters.
        lora_a = dict(model.named_parameters())["base_model.model.fc.lora_A.default.weight"]
        self.assertIsNotNone(lora_a.grad, "LoRA Adapter A should have grad (Trainable)")
        
        lora_b = dict(model.named_parameters())["base_model.model.fc.lora_B.default.weight"]
        self.assertIsNotNone(lora_b.grad, "LoRA Adapter B should have grad (Trainable)")
        
        # Check head.
        self.assertIsNotNone(head.weight.grad, "Projection head should have grad (Trainable)")
        
        print("SUCCESS: Gradients are correctly frozen/unfrozen.")

if __name__ == "__main__":
    unittest.main()
