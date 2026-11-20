"""Extract dolphin names from image filenames and save them to a CSV file."""
import argparse
import os

import tqdm

from src.core import constants
from src.core import path_utils

path_utils.ensure_shark_model_in_path()

from src.core import utils


def main():
    """Extract dolphin names from image filenames in a directory."""
    parser = argparse.ArgumentParser(
        description="Extract dolphin names from image filenames in a directory."
    )
    parser.add_argument('--data-directory', type=str, required=True,
                        help='The path to the root directory of the dolphin dataset.')
    parser.add_argument(
        '--output-name', type=str, default="dolphin_annotations.csv",
        help=(
            'The filename for the output CSV. It will be saved in the dolphin '
            'annotations directory.'
        )
    )

    args = parser.parse_args()

    data_directory = utils.resolve_directory_path(
        os.path.join(constants.BASE_DATA_DIR, args.data_directory) if args.data_directory else constants.BASE_DATA_DIR, 
        constants.BASE_DATA_DIR
    )

    # Check if output_name looks like a full path; if not, prepend DATA_PROC_DIR
    if not os.path.isabs(args.output_name) and os.path.dirname(args.output_name) == "":
        output_path = os.path.join(constants.DATA_PROC_DIR, args.output_name)
    else:
        output_path = os.path.abspath(args.output_name)

    print("Starting dolphin name extraction...")
    extract_names(data_directory, output_path)
    print("Extraction complete.")


def extract_names(directory, output_csv_path):
    """Recursively find image files in a directory, extract dolphin IDs from prefixes, and write them to a CSV."""
    image_extensions = {'.jpg', '.jpeg', '.png'}
    data_rows = []

    print(f"Scanning directory: {directory}")

    # Use os.walk to recursively find all files
    for _, _, files in tqdm.tqdm(list(os.walk(directory)), desc="Processing directories"):
        for filename in files:
            # Check if the file is an image
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # Extract dolphin name (e.g., "dolphin1" from "dolphin1_1.jpg")
                name = filename.split('_')[0]
                data_rows.append([filename, name])

    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV using pandas
    import pandas as pd
    df = pd.DataFrame(data_rows, columns=['filename', 'name'])
    df.to_csv(output_csv_path, index=False)
    print(f"Annotations saved to: {output_csv_path}")


if __name__ == "__main__":
    main()
