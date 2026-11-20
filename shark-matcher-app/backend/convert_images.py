"""Image formatting utilities converting uploaded images to standardised JPEG format for the ML pipeline.

Keep conversion semantics aligned with shark-matcher/src/cli/convert_images.py.
"""

import glob
import os

import PIL.Image

# Allow loading of truncated images to prevent errors on corrupt files.
PIL.Image.LOAD_TRUNCATED_IMAGES = True


def format_image_directory(source_directory, output_directory, verbose=False):
    """Recursively convert images from source_directory to standardised JPEGs in output_directory, skipping existing outputs."""
    print(f"=== format_image_directory ===")
    print(f"Converting images from {source_directory} to {output_directory}")
    print("-" * 40)

    if not os.path.exists(source_directory):
        print(f"Warning: source directory not found at {source_directory}. Skipping.")
        return

    os.makedirs(output_directory, exist_ok=True)

    total_images = sum(
        1 for path in glob.iglob(os.path.join(source_directory, "**", "*"), recursive=True)
        if os.path.basename(path).lower().endswith((".jpg", ".png", ".jpeg"))
    )
    print(f"Total images in input directory: {total_images}")

    image_index = 0
    for path in glob.iglob(os.path.join(source_directory, "**", "*"), recursive=True):
        filename = os.path.basename(path)
        image = None
        try:
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                out_filename = filename.split(".")[0] + ".jpeg"
                out_path = os.path.join(output_directory, out_filename)

                if os.path.exists(out_path):
                    continue

                if verbose:
                    print(f"{image_index} / {total_images} Processing: {filename:<50}")

                image = PIL.Image.open(path).convert("RGB")

                # Strip ICC profile to avoid libpng warnings.
                image.info.pop("icc_profile", None)

                image.save(out_path)
                image_index += 1
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"\nSkipping {filename} due to an error: {e}")
        finally:
            if image:
                image.close()

    print(f"Loaded and converted {image_index} images.")
