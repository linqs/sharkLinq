
"""Manual convergence checks verifying the trainer minimizes loss on synthetic datasets."""
import os
import shutil
import sys
import time
import unittest

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.utils.data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.learning.trainer.new_trainer import RobustTrainer, ContrastiveStrategy, SupervisedStrategy, ArcFaceModelWrapper
import src.config.config_schema
from src.core import utils

# --- Mocks and Helpers ---.
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, size=20, num_classes=2, img_size=(3, 224, 224)):
        self.size = size
        self.data = torch.randn(size, *img_size)
        # Ensure labels are 0 or 1 for 2 classes.
        self.labels = torch.randint(0, num_classes, (size,))
        self.filenames = [f"img_{i}.jpg" for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_labels(self):
        return self.labels

class MockBackbone(nn.Module):
    """
    A simple CNN backbone to stand in for ViT/MiewID to ensure speed, 
    but functionally identical for the trainer (outputs features).
    """
    def __init__(self, in_channels=3, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, output_dim)
        )
        self.output_dim = output_dim

    def forward(self, x):
        return self.net(x)
    
    def get_output_dim(self):
        return self.output_dim

class MockHead(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x, labels=None):
        return self.fc(x)

def test_overfit_batch():
    print("\n--- Starting Test A: Overfit Batch Check (Integration) ---")
    
    device = utils.get_torch_device()
    output_dir = "temp_convergence_check"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Use full-batch data across 2 classes to verify overfitting capability.
    batch_size = 20 # Full batch.
    dataset = SyntheticDataset(size=20, num_classes=2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Construct minimal training configuration for testing.
    
    # Optimizer.
    opt_config = src.config.config_schema.OptimizerConfig(
        constructor=torch.optim.AdamW,
        name="AdamW",
        learning_rate=1e-3, # High LR for fast convergence.
        weight_decay=0.0
    )
    
    # Loss.
    loss_config = src.config.config_schema.LossConfig(
        constructor=nn.CrossEntropyLoss,
        name="CrossEntropy",
        options={}
    )
    
    # Training Config.
    train_config = src.config.config_schema.TrainingConfig(
        batch_size=batch_size,
        epochs=50,
        optimizer=opt_config,
        loss=loss_config,
        scheduler=None, # No scheduler for simple overfitting.
        evaluator=None,
        min_augmentations=0, # No augs for consistence.
        max_augmentations=0,
        gradient_accumulation_steps=1
    )
    
    # Full Config Wrapper (Mocking the structure).
    class ConfigMock:
        training = train_config

    config = ConfigMock()
    
    # 3. Setup Model (ArcFace Style for simplicity in convergence check: XEnt Loss).
    import timm
    print("Initializing 'vit_tiny_patch16_224' via timm as 'actual' backbone...")
    backbone = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0) # Feature extractor.
    # Wrapper to add get_output_dim if not present (timm usually has num_features).
    if not hasattr(backbone, 'get_output_dim'):
        backbone.get_output_dim = lambda: backbone.num_features
    
    backbone = backbone.to(device)
    
    # Head.
    head = MockHead(backbone.num_features, 2).to(device)
    
    # Loss is typically inside wrapper for ArcFace, but here we use simple CrossEntropy.
    # Using SupervisedStrategy from new_trainer.
    strategy = SupervisedStrategy() 
    
    class SimpleClassifier(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.loss_fn = nn.CrossEntropyLoss()
        
        def forward(self, images, labels=None):
            feats = self.backbone(images)
            logits = self.head(feats)
            if labels is not None:
                return self.loss_fn(logits, labels), logits, {}
            return logits
            
    model = SimpleClassifier(backbone, head).to(device)
    
    # 4. Run Trainer.
    trainer = RobustTrainer(
        model=model,
        strategy=strategy,
        device=device,
        config=config,
        train_dataset=dataset,
        valid_dataset=dataset # Validate on train set.
    )
    
    print("Training...")
    trainer.train(loader, output_dir)
    
    # 5. Assertions.
    
    print("Verifying convergence...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images) # No labels -> returns logits.
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    acc = correct / total
    print(f"Final Training Accuracy: {acc*100:.2f}%")
    
    if acc < 0.95:
        print("FAILED: Model did not overfit to 100% (or close).")
        exit(1)
    
    print("SUCCESS: Model overfit the batch.")
    
    # Cleanup.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    test_overfit_batch()
