
import unittest
import os
import shutil
import torch
import torchvision
from PIL import Image

import src.config.config
import src.individual_dataset
from src.core import constants

class TestConfigImageSize(unittest.TestCase):
    def setUp(self):
        self.test_root = "temp_test_config_image_size"
        self.images_dir = os.path.join(self.test_root, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Create a dummy image.
        self.img_name = "test_img.jpg"
        self.img_path = os.path.join(self.images_dir, self.img_name)
        # Create a 100x100 image.
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        img.save(self.img_path)
        
        # Create dummy annotations.
        self.ann_path = os.path.join(self.test_root, "train_annotations.tsv")
        with open(self.ann_path, 'w') as f:
            f.write(f"1\t{self.img_name}\t1\n")

    def tearDown(self):
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root)

    def test_test_config_image_size_config(self):
        """
        Verifies that test_config config specifies 100x100 image size.
        """
        config_name = "test_config"
        self.assertIn(config_name, src.config.config.CONFIGS, f"{config_name} should exist in CONFIGS")
        
        raw_config = src.config.config.CONFIGS[config_name]
        expected_size = [100, 100]
        self.assertEqual(raw_config['data']['image_size'], expected_size, 
                         f"Config image_size should be {expected_size}")

    def test_dataset_loading_size(self):
        """
        Verifies that IndividualDataset loads images at the configured size (100x100)
        when using test_config settings.
        """
        config_name = "test_config"
        raw_config = src.config.config.CONFIGS[config_name]
        
        output_image_size = tuple(raw_config['data']['image_size'])
        normalization = raw_config['data']['normalization']
        
        dataset = src.individual_dataset.IndividualDataset(
            images_path=self.images_dir,
            annotations_path=self.ann_path,
            train_annotations_path=self.ann_path,
            test_annotations_path=self.ann_path,
            channels=3,
            output_image_size=output_image_size,
            normalization=normalization,
            device=torch.device("cpu")
        )
        
        self.assertEqual(dataset.image_height(), 100)
        self.assertEqual(dataset.image_width(), 100)
        
        if len(dataset) > 0:
            img, label = dataset[0]
            # Check shape: [C, H, W].
            self.assertEqual(img.shape[1:], (100, 100), 
                             f"Loaded image tensor should have shape (3, 100, 100), got {img.shape}")
        else:
             self.fail("Dataset failed to load dummy image.")

if __name__ == "__main__":
    unittest.main()
