"""Dataset wrapper that applies augmentation transforms for metric learning."""

import math

import numpy as np
import torch
import torchvision


class AugmentedDataset(torch.utils.data.Dataset):
    """Universal dataset wrapper for single-view or multi-view augmentation transforms."""
    def __init__(self, dataset, augmentations=None, max_retries=5,
                 num_augmentations=1, include_original=False,
                 rebalance_augmentations=False):
        """Initialize dataset transforms, view counts, and class rebalancing settings."""
        self.dataset = dataset
        self.max_retries = max_retries
        self.num_augmentations = num_augmentations
        self.include_original = include_original
        self.rebalance_augmentations = rebalance_augmentations

        if augmentations:
            self.transform = torchvision.transforms.Compose(augmentations)
        else:
            self.transform = None

        if self.rebalance_augmentations:
            label_counts = np.unique(self.dataset.get_labels().cpu().numpy(), return_counts=True)
            self.label_counts_dict = dict(zip(label_counts[0], label_counts[1]))
            self.max_count = max(self.label_counts_dict.values())

    def _apply_transform_with_retry(self, image):
        """Applies transform and handles rare NaN/Runtime errors."""
        if self.transform is None:
            return image

        for _ in range(self.max_retries):
            try:
                augmented = self.transform(image)
                if not torch.isnan(augmented).any():
                    return augmented
            except RuntimeError:
                pass

        # Fallback to original if all retries fail
        return image

    def __getitem__(self, index):
        """Returns a single (image, label) or ([images], label) tuple."""
        image, label = self.dataset[index]

        # Determine how many views to generate
        if self.rebalance_augmentations:
            images_for_individual = self.label_counts_dict[label.item()]
            num_augs = math.floor(self.max_count / images_for_individual) * self.num_augmentations
        else:
            num_augs = self.num_augmentations

        # Fast path 1: No augmentations requested, just original
        if num_augs == 0:
            return image, label

        # Fast path 2: Standard single-view (e.g. ArcFace config)
        if num_augs == 1 and not self.include_original and not self.rebalance_augmentations:
            return self._apply_transform_with_retry(image), label

        # Multi-view or include_original
        all_images = []
        if self.include_original:
            all_images.append(image)

        for _ in range(num_augs):
            all_images.append(self._apply_transform_with_retry(image))

        return all_images, label

    def __len__(self):
        return len(self.dataset)

    def get_labels(self):
        """Return the wrapped dataset's labels."""
        return self.dataset.get_labels()

    def get_ids(self):
        """Return the wrapped dataset's image IDs."""
        return self.dataset.get_ids()


# Class names that stored pickles may reference.
ContrastiveLearningSharkfinDataset = AugmentedDataset
ContrastiveLearningIndividualDataset = AugmentedDataset
