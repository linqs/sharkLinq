"""Convert raw images to JPEG format with stripped ICC profiles for model training."""

import argparse
import datetime
import glob
import os
import sys

import PIL.Image

from src.core import constants
from src.core import utils

# Allow loading of truncated images to prevent errors on corrupt files.
PIL.Image.LOAD_TRUNCATED_IMAGES = True


def main():
    """Recursively scan and format raw images into a standardized JPEG output directory."""
    parser = argparse.ArgumentParser(
        description="Format a directory of raw images for use in the "
                    "Shark Matcher pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m src.cli.convert_images "
               "--input-directory my-photos "
               "--output-directory images\n")
    parser.add_argument('--input-directory', type=str, default=None,
                        help='Directory of raw images, searched recursively: '
                             'a path under the data root\'s raw-data/ '
                             'directory or an absolute path. Defaults to '
                             'raw-data/ itself.')
    parser.add_argument('--output-directory', type=str, default=None,
                        help='Version name for the output images directory '
                             'under the data root\'s images/ directory. If '
                             'not provided, a timestamp will be used.')
    args = parser.parse_args()

    # --- Resolve paths. ---
    # The input directory is natively resolved via centralized fallback logic.
    input_dir = utils.resolve_directory_path(
        os.path.join(constants.RAW_DATA_DIR, args.input_directory) if args.input_directory else constants.RAW_DATA_DIR, 
        constants.BASE_DATA_DIR
    )

    # Build the full output images directory path.
    if args.output_directory is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = os.path.join(constants.BASE_IMAGES_DIR, timestamp)
        print(f"No images version provided. Generated version name: {timestamp}")
    else:
        output_dir = os.path.join(constants.BASE_IMAGES_DIR, args.output_directory)
        print(f"Using provided images version: {args.output_directory}")

    utils.print_script_banner(__file__, args)

    format_image_directory(source_directory=input_dir,
                           output_directory=output_dir,
                           verbose=True)


def format_image_directory(source_directory, output_directory, verbose=False):
    """Read raw images from a source directory, convert to JPEG, and save to the output directory."""
    utils.print_function_banner("format_image_directory",
                                f"Converting images from {source_directory} "
                                f"to {output_directory}")

    if not os.path.exists(source_directory):
        print(f"Error: Source directory not found at {source_directory}")
        sys.exit(1)

    os.makedirs(output_directory, exist_ok=True)

    # Count images.
    total_images = 0
    for path in glob.iglob(os.path.join(source_directory, '**', '*'), recursive=True):
        filename = os.path.basename(path)
        if (filename.lower().endswith('.jpg') or
                filename.lower().endswith('.png') or
                filename.lower().endswith('.jpeg')):
            total_images += 1

    print("Total number of images in input directory:", total_images)

    # Process all images, unifying format and suffix naming.
    image_index = 0
    for path in glob.iglob(os.path.join(source_directory, '**', '*'), recursive=True):
        filename = os.path.basename(path)
        image = None  # Ensure image is None at the start of each loop
        try:
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                out_filename = filename.split('.')[0] + '.jpeg'

                if os.path.exists(os.path.join(output_directory, out_filename)):
                    continue

                if verbose:
                    print("{} / {} Processing: {}".format(
                        image_index, total_images, filename.ljust(50)))

                image = PIL.Image.open(path).convert('RGB')

                # Fixes -- libpng warning: iCCP: known incorrect sRGB profile
                if 'icc_profile' in image.info:
                    image.info.pop('icc_profile', None)

                image.save(os.path.join(output_directory, out_filename))
                image_index += 1
        except Exception as e:
            # pylint: disable=broad-exception-caught
            print(f"\nSkipping {filename} due to an error: {e}")
        finally:
            if image:
                image.close()

    print(f"Loaded and converted {image_index} images.")


if __name__ == "__main__":
    main()
