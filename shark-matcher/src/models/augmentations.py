"""Augmentation classes and constructors for image preprocessing."""

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.v2

AUGMENTATION_CONSTRUCTORS = {
    "RandomAffine": lambda **kwargs: torchvision.transforms.v2.RandomAffine(**kwargs),
    "RandomErasing": lambda **kwargs: torchvision.transforms.v2.RandomErasing(**kwargs),
    "RandomHorizontalFlip": lambda **kwargs: torchvision.transforms.v2.RandomHorizontalFlip(**kwargs),
    "RandomPerspective": lambda **kwargs: torchvision.transforms.v2.RandomPerspective(**kwargs),
    "RandomResizedCrop": lambda **kwargs: torchvision.transforms.v2.RandomResizedCrop(**kwargs),
    "RandomRotation": lambda **kwargs: torchvision.transforms.v2.RandomRotation(**kwargs),
    "GaussianBlur": lambda **kwargs: torchvision.transforms.v2.GaussianBlur(**kwargs),
    "ColorJitter": lambda **kwargs: torchvision.transforms.ColorJitter(**kwargs),
    "CLAHE": lambda **kwargs: CLAHE(**kwargs),
    "RandomAdjustSharpness": lambda **kwargs: torchvision.transforms.v2.RandomAdjustSharpness(**kwargs),
}


class CLAHE(object):
    """Apply Contrast Limited Adaptive Histogram Equalization to the L channel of LAB color space."""

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        """Initialize CLAHE transform with clip limit and tile grid size."""
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        """Augment input PIL Image or Tensor using CLAHE."""
        # Handle input type (Tensor or PIL).
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            # Convert [C, H, W] tensor to [H, W, C] numpy array.
            if img.shape[0] == 3:
                # RGB.
                img_np = img.permute(1, 2, 0).cpu().numpy()
                # Convert float [0, 1] arrays to uint8 [0, 255] before OpenCV processing.
                if img_np.max() <= 1.1:  # Heuristic for float.
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            else:
                # Fallback for non-RGB channel counts.
                return img
        else:
            # PIL Image.
            img_np = np.array(img)

        # Ensure RGB (PIL/Tensor) -> LAB conversion works.
        try:
            # Convert to LAB.
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

            # Split channels.
            l_chan, a_chan, b_chan = cv2.split(lab)

            # Apply CLAHE to L-channel.
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l_chan)

            # Merge channels.
            limg = cv2.merge((cl, a_chan, b_chan))

            # Convert back to RGB.
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

            if is_tensor:
                # Convert back to Tensor [C, H, W], float [0,1].
                final_tensor = torch.from_numpy(final).permute(2, 0, 1).float() / 255.0
                return final_tensor
            else:
                return PIL.Image.fromarray(final)

        except Exception:  # pylint: disable=broad-exception-caught
            # Fallback to original image on error.
            return img

    def __repr__(self):
        """Return string representation of the CLAHE transform."""
        return self.__class__.__name__ + \
            '(clip_limit={0}, tile_grid_size={1})'.format(self.clip_limit, self.tile_grid_size)
