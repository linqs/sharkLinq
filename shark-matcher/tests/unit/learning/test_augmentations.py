import unittest
import torch
import torchvision.transforms
from unittest.mock import MagicMock, patch
import src.learning.augmented_dataset as aug_dataset
import src.config.config

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, image):
        self.image = image
    def __getitem__(self, index):
        return self.image, torch.tensor(0)
    def __len__(self):
        return 1
    def get_labels(self):
        return torch.tensor([0])
    def get_ids(self):
        return ["id0"]

class TestAugmentations(unittest.TestCase):
    def setUp(self):
        self.dummy_image = torch.randn(3, 100, 100)
        self.dataset = DummyDataset(self.dummy_image)

    def test_generate_n_views(self):
        """Verify it returns exactly num_augmentations views."""
        mock_transform = MagicMock(side_effect=lambda x: x)
        
        # We pass a dataset and just 1 transform in the list.
        aug_ds = aug_dataset.AugmentedDataset(
            self.dataset, augmentations=[mock_transform], num_augmentations=2
        )
        # Because we only have 1 mock_transform but we pass it as a list to Compose,
        # we actually need to intercept the Compose or just mock the _apply_transform_with_retry
        pass

        # Since AugmentedDataset uses Compose, let's just mock _apply_transform_with_retry
        aug_ds._apply_transform_with_retry = MagicMock(return_value=self.dummy_image)
        
        views, label = aug_ds[0]
        
        # Should return exactly N augmented views.
        self.assertEqual(len(views), 2)
        self.assertEqual(aug_ds._apply_transform_with_retry.call_count, 2)

    def test_retry_logic(self):
        """Verify it retries if a transform returns NaN."""
        call_count = 0
        def side_effect(x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return torch.tensor(float('nan'))
            return x.clone()

        # Hack: assign the side_effect function directly as the transform to bypass Compose
        aug_ds = aug_dataset.AugmentedDataset(self.dataset, num_augmentations=1)
        aug_ds.transform = MagicMock(side_effect=side_effect)
        
        # Request 1 augmentation.
        views, label = aug_ds[0]
        
        # Transform called twice (first NaN, second success)
        self.assertEqual(aug_ds.transform.call_count, 2)
        # Since num_augmentations=1 and not include_original, views is a list of 1? 
        # Wait, if num_augmentations=1, include_original=False, it returns a single tensor!
        self.assertFalse(torch.isnan(views).any())

    def test_fallback_logic(self):
        """Verify it falls back to original image if retries are exhausted."""
        aug_ds = aug_dataset.AugmentedDataset(self.dataset, num_augmentations=1, max_retries=3)
        aug_ds.transform = MagicMock(return_value=torch.tensor(float('nan')))
        
        views, label = aug_ds[0]
        
        # Should return original as the 'augmented' view.
        self.assertTrue(torch.equal(views, self.dummy_image))
        # Should have tried max_retries times.
        self.assertEqual(aug_ds.transform.call_count, 3)

    def test_individual_augmentations(self):
        """Iterate through each configured augmentation to ensure they run."""
        # Load augmentations from config.
        aug_constructors = src.config.config.AUGMENTATION_CONSTRUCTORS
        aug_configs = src.config.config.AUGMENTATIONS
        
        for name, params in aug_configs:
            with self.subTest(augmentation=name):
                if name not in aug_constructors:
                    self.fail(f"Augmentation {name} not found in constructors")
                
                # Construct the transform.
                transform = aug_constructors[name](**params)
                
                # Run it.
                try:
                    output = transform(self.dummy_image)
                    self.assertEqual(output.shape, self.dummy_image.shape)
                    self.assertFalse(torch.isnan(output).any())
                except Exception as e:
                    self.fail(f"Augmentation {name} failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
