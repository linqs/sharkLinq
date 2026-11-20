"""Analyze image and metadata files to report dataset statistics and data integrity."""

import argparse
import datetime
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt

from src.core import constants
from src.core import path_utils

path_utils.ensure_shark_model_in_path()

import src.csv_store._loader as csv_loader
import src.csv_store.media as csv_media
from src.core import utils

# Quality buckets reported in the dataset stats (mirrors csv_store's order plus an unknown bucket).
_QUALITY_BUCKETS = ['unknown', 'unmatchable', 'poor', 'medium', 'good']


def main():
    """Analyze annotation files and image directories across dataset splits."""
    parser = argparse.ArgumentParser(
        description="Analyze annotation files and image directories used for "
                    "model training and evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 src/data_scripts/analyze_data.py "
               "--annotations-version test-annotations\n"
               "  python3 src/data_scripts/analyze_data.py --plot "
               "--annotations-version test-annotations\n")
    parser.add_argument('--verbose', action='store_true',
                        help='Full verbose output.')
    parser.add_argument('--annotations-version', default=None, type=str,
                        help='Specific annotations version directory name '
                             '(e.g. "test-annotations").')
    parser.add_argument('--plot', action='store_true',
                        help='Plot results from analysis.')
    parser.add_argument('--csv-dir', type=str, default=None,
                        help='Source database-format CSV directory '
                             '(containing individuals.csv, media.csv) used to '
                             'look up per-image quality for the dataset stats. '
                             'Resolved against sm-data/database-format/ when '
                             'relative. If omitted, falls back to the '
                             '.source_csv_dir sidecar in the annotations '
                             'directory; if neither is available, the script '
                             'exits with an error.')
    args = parser.parse_args()

    # --- Set up timestamped output directory and tee stdout to a file ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = os.path.join(constants.DATA_ANALYSIS_DIR, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    out_file_path = os.path.join(out_dir, "analysis.txt")
    out_file = open(out_file_path, 'w', encoding='utf-8')  # pylint: disable=consider-using-with

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
        def flush(self):
            for s in self._streams:
                s.flush()

    sys.stdout = _Tee(sys.__stdout__, out_file)

    # --- Resolve paths. ---
    # Resolve annotations paths.
    annotations_version = args.annotations_version if args.annotations_version else "annotations"
    version_dir = os.path.join(constants.ANNOTATIONS_DIR, annotations_version)
    if not os.path.exists(version_dir):
        print(f"Error: Annotations version '{annotations_version}' not found at {version_dir}")
        sys.exit(1)

    full_path = os.path.join(version_dir, constants.FULL_ANNOTATIONS)
    train_path = os.path.join(version_dir, constants.TRAIN_ANNOTATIONS)
    valid_path = os.path.join(version_dir, constants.VALID_ANNOTATIONS)
    print(f"Using version '{annotations_version}' -> Full: {full_path}")

    # Build the full images directory path.
    images_dir = constants.IMAGES_DIR

    utils.print_script_banner(__file__, args)

    # Define the sets to analyze. Skip any whose annotation file is missing
    # or empty so callers don't have to opt into split handling.
    candidate_sets = [
        ("Full Dataset", full_path),
        ("Train Set", train_path),
        ("Valid Set", valid_path),
    ]
    analysis_sets = []
    for set_name, set_path in candidate_sets:
        if not os.path.exists(set_path):
            print(f"Skipping {set_name}: file not found at {set_path}")
            continue
        if os.path.getsize(set_path) == 0:
            print(f"Skipping {set_name}: file is empty at {set_path}")
            continue
        analysis_sets.append((set_name, set_path))

    # Pin csv_store to the CSV dir the annotations were generated from so
    # quality lookups in show_image_stats match the source. Mirrors the
    # resolver in src.retrieval.embeddings.
    csv_dir = utils.resolve_csv_dir(args)
    if csv_dir is None:
        print("ERROR: no CSV directory could be resolved. Pass --csv-dir "
              "or regenerate annotations with src.cli.convert_data so a "
              f"{constants.SOURCE_CSV_DIR_SIDECAR} sidecar is written.")
        sys.exit(1)
    csv_loader.set_db_dir(csv_dir)

    for set_name, set_path in analysis_sets:
        print(f"\n{'='*60}\nAnalyzing: {set_name}\n{'='*60}")
        show_image_data(annotations_path=set_path, images_dir=images_dir,
                        verbose=args.verbose)
        show_image_stats(annotations_path=set_path, verbose=args.verbose)
        images_per_individual_report(annotations_path=set_path,
                                     verbose=args.verbose, plot=args.plot)
        images_per_year_report(annotations_path=set_path, plot=args.plot)

    sys.stdout = sys.__stdout__
    out_file.close()
    print(f"Output written to: {out_file_path}")


def show_image_data(annotations_path=None, images_dir=constants.IMAGES_DIR, verbose=False):
    """Check image file existence, naming conventions, and orphan records against annotations."""
    utils.print_function_banner("show_image_data",
                                "Checking image files against annotations.")

    if annotations_path is not None:
        if os.path.exists(annotations_path):
            annotations = utils.load_csv_file(annotations_path, delimiter='\t')
        else:
            print(f"Error: File not found at {annotations_path}")
            return
    else:
        version = "annotations"
        path = os.path.join(constants.ANNOTATIONS_DIR, version, constants.FULL_ANNOTATIONS)
        annotations = utils.load_csv_file(path, delimiter='\t')
    if annotations is None:
        return

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return

    # Extract names from annotations
    names = []
    for _, image_name, _ in annotations:
        names.append(image_name)

    # Compare set of images and set of annotations.
    annotations_with_images = []
    annotations_without_images = []
    images_with_annotations = []
    images_without_annotations = []
    correctly_named = []
    incorrectly_named = []

    # Parse over annotations seeing if corresponding images exist.
    # Improve performance by caching listdir.
    existing_images = set(os.listdir(images_dir))

    for name in names:
        if name not in existing_images:
            annotations_without_images.append(name)
        else:
            annotations_with_images.append(name)

    # Parse over image files check if images are in annotations files and their names are correctly
    # formatted.
    for name in existing_images:
        # Ignore hidden files like .DS_Store
        if name.startswith('.'):
            continue

        if name not in names:
            images_without_annotations.append(name)
        else:
            images_with_annotations.append(name)
        if re.search(r"[A-Z]+\d\d\d\d\d\d\d\d\..*$", name) is not None:
            correctly_named.append(name)
        else:
            incorrectly_named.append(name)

    # Report results.
    utils.print_key_value_section("Image & Registry Integrity Report", {
        "Total images on disk": len(existing_images),
        "Images in metadata": len(annotations_with_images),
        "Missing from disk": len(annotations_without_images),
        "Images with metadata": len(images_with_annotations),
        "Orphan images (no metadata)": len(images_without_annotations),
        "Correctly named images": len(correctly_named),
        "Incorrectly named images": len(incorrectly_named)
    })

    if verbose:
        if annotations_without_images:
            print(f"\nAnnotations without images ({len(annotations_without_images)}):")
            for item in annotations_without_images:
                print(f"  {item}")

        if images_without_annotations:
            print(f"\nImages without annotations ({len(images_without_annotations)}):")
            for item in images_without_annotations:
                print(f"  {item}")

        if incorrectly_named:
            print(f"\nIncorrectly named images ({len(incorrectly_named)}):")
            for item in incorrectly_named:
                print(f"  {item}")


def show_image_stats(annotations_path=None, verbose=False):
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    """Check data integrity and report image counts, conventions, and duplicate annotations."""
    utils.print_function_banner("show_annotations_data",
                                "Checking data/annotations integrity and conventions.")
    annotations = []

    # Load image/metadata file if provided.
    if annotations_path is not None:
        if os.path.exists(annotations_path):
            annotations = utils.load_csv_file(annotations_path, delimiter='\t')
            print(f"Loaded formatted image metadata from {annotations_path}")
        else:
            print(f"Error: File not found at {annotations_path}")
            return
    else:
        version = "annotations"
        path = os.path.join(constants.ANNOTATIONS_DIR, version, constants.FULL_ANNOTATIONS)
        annotations = utils.load_csv_file(path, delimiter='\t')
        print(f"Found parsed images metadata file (default version '{version}') to read and check data from.")

    # No data loaded?
    if annotations is None:
        print("Error: Check data function unable to load image metadata")
        return

    # Count images per individual.
    individual_image_counts, num_individuals, num_images = _get_image_stats(annotations)

    # Count images per quality bucket using csv_store as the source of truth.
    quality_counts = _get_quality_counts(annotations)

    # Detect duplicate annotations
    # Convert each inner list to a tuple to make it hashable for set operations.
    unique_annotations_tuples = {tuple(item) for item in annotations}
    if len(unique_annotations_tuples) != len(annotations):
        print("Duplicate annotations detected.")
        seen = set()
        for item in annotations:
            item_tuple = tuple(item)
            if item_tuple in seen:
                print(f"Duplicate found: {item}")
            seen.add(item_tuple)

    # Report results.
    stats = {
        "Total individuals": num_individuals,
        "Total images mapped": num_images,
    }
    for bucket in _QUALITY_BUCKETS:
        stats[f"Images of {bucket} quality"] = quality_counts.get(bucket, 0)
    utils.print_key_value_section("Dataset Stats", stats)


def images_per_individual_report(annotations_path=None,
                                 verbose=False, plot=False):
    # pylint: disable=too-many-locals
    """Generate and print a report on the distribution of images per individual."""
    if annotations_path is None:
        version = "annotations"
        annotations_path = os.path.join(constants.ANNOTATIONS_DIR, version, constants.FULL_ANNOTATIONS)

    # Read annotations from the specified file.
    annotations = utils.load_csv_file(annotations_path, delimiter='\t')

    # Count images per individual label.
    individual_image_counts, _, num_images = _get_image_stats(annotations)

    # Count how many individuals have a certain number of images.
    image_per_individual_dict = {}
    for individual_image_count in individual_image_counts.values():
        if individual_image_count not in image_per_individual_dict:
            image_per_individual_dict[individual_image_count] = 0
        image_per_individual_dict[individual_image_count] += 1

    image_per_individual_dict = dict(sorted(image_per_individual_dict.items(), key=lambda item: item[0]))

    # Make a dictionary of the number of individuals with at least x or more images.
    individuals_with_image_count_cumulative_dict = {}
    for key in image_per_individual_dict.keys():
        # Sum counts for all image counts >= the current key.
        cumulative_count = sum(v for k, v in image_per_individual_dict.items() if k >= key)
        individuals_with_image_count_cumulative_dict[key] = cumulative_count

    utils.print_key_value_section("Images Per Individual Summary", {
        "Total individuals": len(individual_image_counts),
        "Total images in set": num_images
    })

    print("\nNumber of Individuals with Image Count:")
    rows = []
    for image_count, count in image_per_individual_dict.items():
        rows.append([image_count, count])
    utils.print_table(["Image Count", "Num Individuals"], rows)

    print("\nIndividuals with image count cumulative:")
    cumulative_rows = []
    for key, value in individuals_with_image_count_cumulative_dict.items():
        cumulative_rows.append([f">= {key}", value])
    utils.print_table(["Count Threshold", "Num Individuals"], cumulative_rows)

    if verbose:
        print("\nPer-individual image counts:")
        per_individual_rows = sorted(
            individual_image_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        utils.print_table(
            ["Individual ID", "Image Count"],
            [[label, count] for label, count in per_individual_rows],
        )

    if plot:
        # Plot the data.
        # Set the color of the bars to a custom color.
        custom_color = '#6cbbea'
        plt.bar(image_per_individual_dict.keys(), image_per_individual_dict.values(), color=custom_color)
        plt.yscale("log")
        # Add a title and labels.
        plt.title("Number of Individuals with Image Count")
        plt.xlabel("Image Count")
        plt.ylabel("Num Individuals")
        plt.show()


def images_per_year_report(annotations_path=None, plot=False):
    """Print and optionally plot the temporal distribution of images by year."""
    if annotations_path is None:
        version = "annotations"
        annotations_path = os.path.join(constants.ANNOTATIONS_DIR, version, constants.FULL_ANNOTATIONS)

    annotations = utils.load_csv_file(annotations_path, delimiter='\t')

    media_ids = []
    for row in annotations:
        try:
            media_ids.append(int(row[0]))
        except (ValueError, TypeError):
            continue

    medias = csv_media.get_medias(ids=list(set(media_ids)))
    time_by_id = {m.id: m.time for m in medias}

    year_counts = {}
    no_date_count = 0
    for mid in media_ids:
        ts = time_by_id.get(mid)
        if ts is None:
            no_date_count += 1
            continue
        year = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).year
        year_counts[year] = year_counts.get(year, 0) + 1

    year_counts = dict(sorted(year_counts.items()))

    utils.print_key_value_section("Images Per Year Summary", {
        "Total images with date": sum(year_counts.values()),
        "Images without date": no_date_count,
        "Year range": f"{min(year_counts)} – {max(year_counts)}" if year_counts else "N/A",
    })

    rows = [[year, count] for year, count in year_counts.items()]
    utils.print_table(["Year", "Num Images"], rows)

    if plot and year_counts:
        custom_color = '#6cbbea'
        plot_counts = {y: c for y, c in year_counts.items() if y >= 2000}
        years = list(plot_counts.keys())
        counts = list(plot_counts.values())
        plt.figure()
        plt.bar(years, counts, color=custom_color, width=0.8)
        plt.title("Images per Year")
        plt.xlabel("Year")
        plt.ylabel("Num Images")
        plt.xticks(years, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def _get_image_stats(annotations):
    """Calculate basic image and individual statistics from an annotations list."""
    individual_image_counts = {}
    for _, _, label in annotations:
        label = int(label)
        individual_image_counts[label] = individual_image_counts.get(label, 0) + 1

    return individual_image_counts, len(individual_image_counts), len(annotations)


def _get_quality_counts(annotations):
    """Return a mapping of quality tiers to media counts parsed from the annotation rows."""
    media_ids = []
    for row in annotations:
        try:
            media_ids.append(int(row[0]))
        except (ValueError, TypeError):
            continue

    medias = csv_media.get_medias(ids=list(set(media_ids)))
    quality_by_id = {m.id: (m.quality.lower() if m.quality else None) for m in medias}

    counts = {bucket: 0 for bucket in _QUALITY_BUCKETS}
    for mid in media_ids:
        q = quality_by_id.get(mid)
        if q in counts:
            counts[q] += 1
        else:
            counts['unknown'] += 1
    return counts


if __name__ == "__main__":
    main()
