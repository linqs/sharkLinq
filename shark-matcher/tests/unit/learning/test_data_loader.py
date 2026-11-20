
import unittest
import torch
import torchvision.transforms as transforms
import os
import shutil
from src.core import utils
import random
import src.individual_dataset
from src.learning import augmented_dataset as cl_dataset
from unittest.mock import patch, MagicMock

class TestDataLoaderDeterminism(unittest.TestCase):
    
    def setUp(self):
        # Setup temp environment.
        self.test_root = "temp_determinism_test"
        self.images_dir = os.path.join(self.test_root, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Create dummy images (random noise).
        self.img_names = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
        self.num_classes = 2
        
        # IndividualDataset reads files with torchvision.io.read_image, so the fixture writes real images.
        import torchvision.transforms.functional as F
        from PIL import Image
        
        for i, img_name in enumerate(self.img_names):
            # Create deterministic random image (noise).
            # We use a fixed seed locally here just for image creation consistency if needed, 
            # though dataset tests rely on utils.seed_everything later.
            random.seed(i)
            # Create a localized gradient or noise to ensure geometric transforms change it.
            img_data = []
            for y in range(100):
                for x in range(100):
                    # Gradient + Noise.
                    r = (x + y + random.randint(0, 20)) % 255
                    g = (x + random.randint(0, 50)) % 255
                    b = (y + random.randint(0, 50)) % 255
                    img_data.append((r, g, b))
            
            img = Image.new('RGB', (100, 100))
            img.putdata(img_data)
            img.save(os.path.join(self.images_dir, img_name))

        # Two images per class.
        self.ann_path = os.path.join(self.test_root, "annotations.tsv")
        with open(self.ann_path, 'w') as f:
            f.write(f"0\t{self.img_names[0]}\t0\n")
            f.write(f"1\t{self.img_names[1]}\t0\n")
            f.write(f"2\t{self.img_names[2]}\t1\n")
            f.write(f"3\t{self.img_names[3]}\t1\n")

    def tearDown(self):
        # Cleanup.
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root)

    def get_individual_dataset(self):
        # Helper to get base dataset.
        return src.individual_dataset.IndividualDataset(
            images_path=self.images_dir,
            annotations_path=self.ann_path,
            train_annotations_path=self.ann_path,
            test_annotations_path=self.ann_path,
            output_image_size=(100, 100),
            normalization=None,
            device=torch.device('cpu') # Use CPU for deterministic behavior across platforms more reliably.
        )

    def test_base_dataset_determinism(self):
        """
        Verify IndividualDataset (base) determinism (file reading, processing).
        """
        print("\n--- Testing Base Dataset Determinism ---")
        seed = 42
        
        def get_item(s):
            utils.seed_everything(s)
            ds = self.get_individual_dataset()
            return ds[0][0] # Image tensor.

        t1 = get_item(seed)
        t2 = get_item(seed)
        
        self.assertTrue(torch.equal(t1, t2), "Base dataset tensors should be identical")
        print("Base Dataset: OK")

    def test_contrastive_augmentations_determinism(self):
        """
        Verify ContrastiveLearningIndividualDataset determinism (random augmentations).
        """
        print("\n--- Testing Contrastive Augmentation Determinism ---")
        seed = 123
        
        # Define random augmentations.
        augmentations = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.5)
        ]
        
        def get_augmented_views(s):
            utils.seed_everything(s)
            ds = self.get_individual_dataset()
            cl_ds = cl_dataset.AugmentedDataset(
                ds, augmentations, num_augmentations=2
            )
            # Get item 0 returns (list_of_images, list_of_labels).
            images, _ = cl_ds[0]
            # images[0] is original, images[1...] are augmented.
            return torch.stack(images)

        # Run 1.
        views1 = get_augmented_views(seed)
        
        # Run 2.
        views2 = get_augmented_views(seed)
        
        self.assertTrue(torch.equal(views1, views2), "Augmented views should be bitwise identical with same seed")
        
        # Verify that different seeds produce different augmentations.
        views_diff = get_augmented_views(seed + 1)
        # Check tensor inequality on augmented views from index 1 onwards since index 0 is the original image.
        if torch.equal(views1[1:], views_diff[1:]):
             print("Warning: Different seeds produced identical augmentations. "
                   "This might be chance, but check if random transforms are actually working.")
        else:
             print("Sanity Check: Different seeds produced different augmentations. OK.")
             
        print("Contrastive Dataset: OK")


if __name__ == "__main__":
    unittest.main()
