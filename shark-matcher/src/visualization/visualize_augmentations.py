"""Visualize individual augmentation techniques applied to a single shark image."""

import argparse
import datetime
import os

import PIL.Image
import torch
import torchvision

from src.core import constants
import src.config.config


def main():
    """Save one image under each augmentation technique, next to the original, for visual inspection."""
    parser = argparse.ArgumentParser(description="Visualize individual augmentation techniques on a single image.")
    parser.add_argument("image_name", type=str, help="Name of the image file in the images directory.")
    args = parser.parse_args()

    image_path = os.path.join(constants.IMAGES_DIR, args.image_name)
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    try:
        image = PIL.Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Convert to tensor as some transforms (like RandomErasing) require it.
    # ToTensor normalizes to [0, 1] which is the standard input for these transforms.
    img_tensor = torchvision.transforms.functional.to_tensor(image)

    # Use a combined list of augmentations to ensure we cover all techniques experimented with.
    # MIEW_AUG_BOTH contains all MiewID augmentations (including blur and erasing)
    aug_configs = list(src.config.config.MIEW_AUG_BOTH)
    
    # Also add any from standard AUGMENTATIONS if they aren't already included
    miew_aug_names = [name for name, _ in aug_configs]
    for name, params in src.config.config.AUGMENTATIONS:
        if name not in miew_aug_names:
            aug_configs.append((name, params))
            miew_aug_names.append(name)

    # Set up output directory using constants.
    # VISUALIZATIONS_DIR maps to sm-data/data-out/visualizations
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(constants.VISUALIZATIONS_DIR, "vis_output", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original image
    orig_path = os.path.join(output_dir, "Original.jpg")
    torchvision.utils.save_image(img_tensor, orig_path)
    print(f"Saved original image to {orig_path}")

    # Process and save each augmentation technique
    for name, params in aug_configs:
        constructor = src.config.config.AUGMENTATION_CONSTRUCTORS.get(name)
        if not constructor:
            print(f"Warning: Augmentation {name} not found in constructors. Skipping.")
            continue
            
        try:
            # Create a copy of parameters to avoid modifying the original config.
            # Force probability to 1.0 so that the augmentation is guaranteed to apply,
            # except for RandomHorizontalFlip which we want to occasionally skip to show both states.
            vis_params = dict(params)
            if "p" in vis_params and name != "RandomHorizontalFlip":
                vis_params["p"] = 1.0

            # Increase kernel size and sigma for GaussianBlur for better visualization
            if name == "GaussianBlur":
                # kernel_size must be an odd integer
                vis_params["kernel_size"] = 25 
                # We also need to increase sigma, otherwise a large kernel with a small sigma will still look sharp
                vis_params["sigma"] = (1.0, 5.0)

            transform = constructor(**vis_params)
            
            # Deterministic transforms only need 1 image
            if name in ["CLAHE", "RandomAdjustSharpness"]:
                num_samples = 1
            else:
                num_samples = 3

            augmented_images = []
            for _ in range(num_samples):
                augmented_images.append(transform(img_tensor))
            
            grid = torchvision.utils.make_grid(augmented_images, nrow=num_samples, padding=2)
            
            out_filename = f"{name}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            
            torchvision.utils.save_image(grid, out_path)
            grid_text = "grid " if num_samples > 1 else ""
            print(f"Saved {name} augmentation {grid_text}to {out_path}")
            
        except Exception as e:
            print(f"Error applying augmentation {name}: {e}")

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
