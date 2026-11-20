
import unittest
import os
import shutil
import json
import torch
from unittest.mock import patch
from PIL import Image

import src.individual_dataset
import src.retrieval.embeddings
import src.config.config
from src.core import constants
from src.core import utils

class TestImageConsistency(unittest.TestCase):
    def setUp(self):
        # Setup temporary directories.
        self.test_root = "temp_consistency_test"
        self.images_dir = os.path.join(self.test_root, "images")
        self.models_dir = os.path.join(self.test_root, "trained_models")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create a dummy image.
        self.img_name = "consistency_test_img.jpg"
        self.img_path = os.path.join(self.images_dir, self.img_name)
        # Create a specific pattern image to verify transformation (e.g., solid color).
        # 500x300 Red Image.
        img = Image.new('RGB', (500, 300), color = (255, 0, 0))
        img.save(self.img_path)
        
        # Create dummy annotations.
        self.ann_path = os.path.join(self.test_root, "annotations.tsv")
        with open(self.ann_path, 'w') as f:
            f.write(f"0\t{self.img_name}\t1\n")
            
        # Define a test configuration.
        self.model_name = "test_consistency_model"
        self.model_dir = os.path.join(self.models_dir, self.model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.config_data = {
            "data": {
                "image_size": [100, 100],
                "normalization": {
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                    "resize_mode": "squash"
                }
            }
        }
        
        # Write config.json.
        self.config_path = os.path.join(self.model_dir, "config.json")
        utils.write_json_file(self.config_path, self.config_data)

    def tearDown(self):
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root)

    def test_training_vs_inference_consistency(self):
        """
        Verifies that the image tensor loaded during training matches the one 
        loaded during inference (embeddings generation).
        """
        
        # Logic mirroring new_trainer.py.
        
        # Load config (simulated).
        train_config = self.config_data
        train_image_size = tuple(train_config["data"]["image_size"])
        train_norm = train_config["data"]["normalization"]
        
        # Initialize Dataset.
        # We patch IMAGES_DIR to point to our temp dir.
        with patch("src.core.constants.IMAGES_DIR", self.images_dir):
             train_dataset = src.individual_dataset.IndividualDataset(
                images_path=self.images_dir,
                annotations_path=self.ann_path,
                train_annotations_path=self.ann_path,
                test_annotations_path=self.ann_path,
                output_image_size=train_image_size,
                normalization=train_norm,
                device=torch.device('cpu')
            )
             
        train_tensor, _ = train_dataset[0]
        
        # Logic mirroring embeddings.py / test_embedding_identity.py.
        
        # Patch TRAINED_MODELS_DIR so _get_model_metadata finds our dummy model config.
        with patch("src.core.constants.TRAINED_MODELS_DIR", self.models_dir):
            is_finetuned, config_path, inf_output_size, inf_norm_config = \
                utils.get_model_metadata(self.model_name)
                
            # Initialize Dataset.
            with patch("src.core.constants.IMAGES_DIR", self.images_dir):
                inf_dataset = src.individual_dataset.IndividualDataset(
                    images_path=self.images_dir,
                    annotations_path=self.ann_path,
                    train_annotations_path=self.ann_path,
                    test_annotations_path=self.ann_path,
                    output_image_size=inf_output_size,
                    normalization=inf_norm_config,
                    device=torch.device('cpu')
                )
        
        inf_tensor, _ = inf_dataset[0]
                
        # Check metadata consistency.
        self.assertEqual(train_image_size, inf_output_size, "Image sizes should match")
        self.assertEqual(train_norm, inf_norm_config, "Normalization configs should match")
        
        # Check tensor shape.
        self.assertEqual(train_tensor.shape, inf_tensor.shape, "Tensor shapes should match")
        
        # Check tensor values.
        # Should be identical.
        self.assertTrue(torch.equal(train_tensor, inf_tensor), "Image tensors should be identical")
        
        print("\nSUCCESS: Training and Inference image tensors are identical.")

if __name__ == "__main__":
    unittest.main()
