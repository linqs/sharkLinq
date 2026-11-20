"""Convert database-format CSV files into versioned annotation TSV files for model training."""

import argparse
import csv
import datetime
import os
import random
import sys

from src.core import constants
import src.csv_store._loader as csv_loader
import src.csv_store.media as csv_media
from src.core import utils


def main():
    """Convert database-format CSV files into versioned annotation TSV files."""
    parser = argparse.ArgumentParser(
        description="Convert database-format CSV files into versioned "
                    "annotation TSV files for model training and evaluation. "
                    "By default, performs a per-image train/test split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m src.cli.convert_data "
               "--input-directory pre2019-sample-data "
               "--output-directory test-annotations --seed 42\n"
               "  python3 -m src.cli.convert_data "
               "--input-directory pre2019-sample-data "
               "--output-directory test-annotations --no-test\n")
    parser.add_argument('--no-test', dest='train_test', action='store_false',
                        help='Do not use train/test split for annotations. '
                             'Default is to use the split.')
    parser.add_argument('--input-directory', type=str, default="database-format",
                        help='Subdirectory name under database-format/ '
                             'containing the CSV files (individuals.csv, media.csv). '
                             'Default is the database-format/database-format directory.')
    parser.add_argument('--output-directory', type=str, default=None,
                        help='Version name for the output annotations directory '
                             'under sm-data/annotations/. If not provided, a '
                             'timestamp will be used.')
    parser.add_argument('--annotations-path', type=str, default=None,
                        help='Path to an annotations file to limit the data to.')
    parser.add_argument('--test-per-shark', type=int, default=1,
                        help='Number of test images to set aside per individual '
                             '(default: 1). Individuals with fewer images will '
                             'have all images in the training set.')
    parser.add_argument('--test-newest', action='store_true',
                        help='When splitting, put the newest 20%% of individuals '
                             'into the test set, using the time column of '
                             'media.csv in the input directory.')
    parser.add_argument('--full-clusters-removed', action='store_true',
                        help='Use cluster-based splitting where entire '
                             'individuals are randomly moved to the test set.')
    parser.add_argument('--min-quality', type=str, default=None,
                        choices=['unmatchable', 'poor', 'medium', 'good'],
                        help='Use quality-based splitting where media with quality below this '
                             'level are automatically put in the test set.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic splitting '
                             '(default: 42).')
    parser.set_defaults(train_test=True)

    args = parser.parse_args()

    utils.print_script_banner(__file__, args)

    # --- Resolve paths. ---
    try:
        input_dir = utils.resolve_directory_path(
            os.path.join(constants.DB_DATA_DIR, args.input_directory)
            if args.input_directory else constants.DB_DATA_DIR,
            constants.BASE_DATA_DIR
        )
    except ValueError as exc:
        print(f"Error: input directory not found. {exc}")
        sys.exit(1)

    # Build the full output annotations directory path.
    if args.output_directory is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = os.path.join(constants.ANNOTATIONS_DIR, timestamp)
        print(f"No annotations version provided. Generated version name: {timestamp}")
    else:
        output_dir = os.path.join(constants.ANNOTATIONS_DIR, args.output_directory)
        print(f"Using provided annotations version: {args.output_directory}")

    try:
        prepare_annotations(train_test=args.train_test,
                            input_dir=input_dir, annotations_file=args.annotations_path,
                            test_per_shark=args.test_per_shark,
                            full_clusters_removed=args.full_clusters_removed,
                            test_newest=args.test_newest,
                            min_quality=args.min_quality,
                            output_directory=output_dir, seed=args.seed)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def prepare_annotations(train_test=True, input_dir=None,
                        annotations_file=None, test_per_shark=1, full_clusters_removed=False,
                        test_newest=False, min_quality=None, output_directory=None, seed=None):
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-branches, too-many-statements
    """Read input database CSV files and generate formatted annotation TSV files, raising ValueError on unreadable input."""
    utils.print_function_banner("prepare_annotations",
                                f"Preparing annotations. Input: {input_dir}, "
                                f"Version: {output_directory}")
    # 1. Set Random Seed
    if seed is not None:
        print(f"Setting random seed to {seed} for deterministic splitting.")
        random.seed(seed)

    if not os.path.exists(input_dir):
        raise ValueError(f"CSV directory {input_dir} not found.")

    # 2. Set up Output Directory.
    version_dir = output_directory
    os.makedirs(version_dir, exist_ok=True)
    print(f"Saving annotations to: {version_dir}")

    # Lists to contain needed data.
    ids = []
    filenames = []
    labels = []

    # Get the annotations file if given and limit ids being prepared to only those in the file.
    if annotations_file is not None:
        focus_ids = []
        focus_annotations = utils.load_csv_file(annotations_file, delimiter='\t')
        for annotation in focus_annotations:
            focus_ids.append(int(annotation[0]))
    else:
        focus_ids = []

    shark_media_dict = {}

    csv_dir = input_dir
    print(f"Reading input data from CSV directory: {csv_dir}")

    # Load Media
    media_map = {}
    media_path = os.path.join(csv_dir, "media.csv")
    if os.path.exists(media_path):
        try:
            with open(media_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Check for header
                first_row = next(reader, None)
                if first_row and first_row[0] == "id": # Basic header check
                    pass
                elif first_row:
                    # If no header, reset cursor or handle first row
                    if first_row[0].isdigit():
                        media_map[int(first_row[0])] = first_row[2] # id -> name

                for row in reader:
                    if row:
                        media_map[int(row[0])] = row[2]
        except Exception as e:
            raise ValueError(f"Could not read media.csv: {e}") from e

    # Load Sharks/Individuals and link to Media
    sharks_path = os.path.join(csv_dir, "individuals.csv")
    if not os.path.exists(sharks_path):
        sharks_path = os.path.join(csv_dir, "sharks.csv")

    if os.path.exists(sharks_path):
        try:
            with open(sharks_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                if first_row and not first_row[0].isdigit() and "id" in first_row[0].lower():
                    pass # Header detected
                elif first_row:
                    # Process first row
                    source_id = int(first_row[5])
                    media_ref = first_row[4]
                    if media_ref and media_ref != "NULL" and media_ref != "":
                        media_id = int(media_ref)
                        if media_id in media_map:
                            if source_id not in shark_media_dict:
                                shark_media_dict[source_id] = {"ids": [],
                                                                "filenames": [],
                                                                "label": source_id}
                            shark_media_dict[source_id]["ids"].append(media_id)
                            shark_media_dict[source_id]["filenames"].append(
                                media_map[media_id] + ".jpeg")

                for row in reader:
                    if not row:
                        continue
                    source_id = int(row[5]) # Source column
                    media_ref = row[4] # Media column

                    if media_ref and media_ref != "NULL" and media_ref != "":
                        media_id = int(media_ref)
                        if media_id in media_map:
                            if source_id not in shark_media_dict:
                                shark_media_dict[source_id] = {"ids": [],
                                                                "filenames": [],
                                                                "label": source_id}

                            shark_media_dict[source_id]["ids"].append(media_id)
                            shark_media_dict[source_id]["filenames"].append(
                                media_map[media_id] + ".jpeg")

        except Exception as e:
            raise ValueError(f"Could not read individuals.csv: {e}") from e

    # Create lists of image ids, labels, and filenames.
    for shark, media_dict in shark_media_dict.items():
        for index, media_id in enumerate(media_dict["ids"]):
            if annotations_file is None or media_id in focus_ids:
                ids.append(media_id)
                filenames.append(media_dict["filenames"][index])
                labels.append(shark)

    utils.print_key_value_section("Dataset Statistics", {
        "Total Images": len(ids),
        "Unique Media IDs": len(set(ids)),
        "Unique Media Filenames": len(set(filenames)),
        "Unique Labels/Individuals": len(set(labels))
    })

    # Check ids and filenames lists are same length.
    assert len(ids) == len(filenames) == len(labels)

    full_data = []
    train_data = []
    valid_data = []
    # Optionally, make the train/test split.
    if train_test:
        train_indices, test_indices = _partition_data(ids, labels, test_per_shark,
                                                      full_clusters_removed, test_newest,
                                                      test_split_ratio=0.2, min_quality=min_quality,
                                                      input_dir=csv_dir)
        utils.print_key_value_section("Train/Test Split", {
            "Training Images": len(train_indices),
            "Test Images": len(test_indices)
        })

    for index, media_id in enumerate(ids):
        # Append all data to the full annotations file, then optionally split for train/valid.
        filename = filenames[index]
        label = labels[index]
        full_data.append([media_id, filename, label])

        if train_test:
            if index in train_indices:
                train_data.append([media_id, filename, label])
            elif index in test_indices:
                valid_data.append([media_id, filename, label])

    if train_test:
        # Check train and test add up to full.
        assert len(train_data) + len(valid_data) == len(full_data)

    # Save annotations.
    utils.write_csv_file(os.path.join(version_dir, constants.FULL_ANNOTATIONS),
                         full_data, delimiter='\t')
    utils.write_csv_file(os.path.join(version_dir, constants.TRAIN_ANNOTATIONS),
                         train_data, delimiter='\t')
    utils.write_csv_file(os.path.join(version_dir, constants.VALID_ANNOTATIONS),
                         valid_data, delimiter='\t')

    # Record the source CSV directory so downstream consumers can pin metadata reads to the same source.
    sidecar_path = os.path.join(version_dir, constants.SOURCE_CSV_DIR_SIDECAR)
    stored_csv_path = utils.portable_csv_dir(csv_dir) or os.path.abspath(csv_dir)

    with open(sidecar_path, 'w', encoding='utf-8') as f:
        f.write(stored_csv_path)


def _partition_data(ids, labels, test_per_shark=1, full_clusters_removed=False,
                    test_newest=False, test_split_ratio=0.2, min_quality=None, input_dir=None):
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-branches, too-many-statements
    """Split the dataset into training and testing index partitions."""
    train_indices = []
    test_indices = []
    label_dict = {}
    for index, _ in enumerate(ids):
        if labels[index] not in label_dict:
            label_dict[labels[index]] = []
        label_dict[labels[index]].append(index)

    if input_dir is not None:
        csv_loader.set_db_dir(input_dir)

    # --- Strategy 1: Temporal Split (Newest sharks in test set).
    if test_newest:
        print(f"Partitioning data: Newest {test_split_ratio:.0%} of sharks "
              "will be in the test set.")

        all_media = csv_media.get_medias(ids=ids)
        media_time_map = {media.id: media.time for media in all_media
                          if getattr(media, 'time', None) is not None}

        # Find the earliest media time for each shark to establish its temporal identity.
        shark_earliest_date = {}
        for shark_label, image_indices in label_dict.items():
            earliest_date = float('inf')
            for index in image_indices:
                media_id = ids[index]
                media_time = media_time_map.get(media_id)
                current_time = None
                if media_time == "":
                    current_time = 0 # Handle empty strings as Unix epoch
                elif media_time is not None:
                    current_time = int(media_time)

                if current_time is not None and current_time < earliest_date:
                    earliest_date = current_time

            if earliest_date != float('inf'):
                shark_earliest_date[shark_label] = earliest_date

        # Sort sharks by their earliest appearance date, from oldest to newest.
        sorted_sharks = sorted(shark_earliest_date.items(), key=lambda item: item[1])
        total_images = len(ids)
        test_set_target_size = int(total_images * test_split_ratio)

        # Add newest sharks to the test set until the target size is met.
        for shark_label, _ in reversed(sorted_sharks):  # Iterate from newest to oldest
            shark_indices = label_dict[shark_label]
            # Sharks with too few images always go to the training set.
            if len(shark_indices) <= test_per_shark:
                train_indices.extend(shark_indices)
            # Add newest sharks to the test set until the target size is reached.
            elif len(test_indices) < test_set_target_size:
                test_indices.extend(shark_indices)
            # Assign remaining sharks to the training set.
            else:
                train_indices.extend(shark_indices)

        # Add sharks that had no date information to the training set.
        for shark_label, indices in label_dict.items():
            if shark_label not in shark_earliest_date:
                train_indices.extend(indices)

    # --- Strategy 2: Random Cluster-Based Split (full clusters removed) ---
    elif full_clusters_removed:
        print(f"Partitioning data: Randomly assigning {test_split_ratio:.0%} "
              "of shark clusters to the test set.")
        shuffled_labels = list(label_dict.keys())
        random.shuffle(shuffled_labels)

        total_images = len(ids)
        test_set_target_size = int(total_images * test_split_ratio)

        for shark_label in shuffled_labels:
            shark_indices = label_dict[shark_label]
            if len(shark_indices) <= test_per_shark:
                train_indices.extend(shark_indices)
            elif len(test_indices) < test_set_target_size:
                test_indices.extend(shark_indices)
            else:
                train_indices.extend(shark_indices)

    # --- Strategy 3: Per-Image Split ---
    else:
        print(f"Partitioning data: Selecting {test_per_shark} image(s) per shark for the test set.")
        for shark_label, shark_indices in label_dict.items():
            # Sharks with fewer images than test_per_shark go entirely to the training set.
            if len(shark_indices) <= test_per_shark:
                train_indices.extend(shark_indices)
            else:
                # Randomly sample images for the test set from the current shark.
                test_sample = random.sample(shark_indices, test_per_shark)
                for index in shark_indices:
                    if index in test_sample:
                        test_indices.append(index)
                    else:
                        train_indices.append(index)

    # --- Post-Split Quality Filter ---
    if min_quality is not None:
        print(f"Applying quality filter: Moving train media with quality below '{min_quality}' "
              "to the test set.")

        all_media = csv_media.get_medias(ids=ids)
        media_quality_map = {media.id: media.quality for media in all_media}

        quality_order = ['unmatchable', 'poor', 'medium', 'good']
        min_idx = quality_order.index(min_quality.lower())

        new_train_indices = []
        for index in train_indices:
            media_id = ids[index]
            q = media_quality_map.get(media_id)
            if q is not None and str(q).lower() in quality_order:
                q_idx = quality_order.index(str(q).lower())
                if q_idx < min_idx:
                    test_indices.append(index)
                else:
                    new_train_indices.append(index)
            else:
                new_train_indices.append(index)

        train_indices = new_train_indices

    print(f"Total images: {len(ids)}, training images: {len(train_indices)}, "
          f"test images: {len(test_indices)}")
    assert len(ids) == (len(train_indices) + len(test_indices)), "Mismatch in partitioned data size"

    return train_indices, test_indices


if __name__ == "__main__":
    main()
