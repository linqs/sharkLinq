"""Export annotation rows and copy image files whose media quality is unknown or unrecorded in csv_store."""

import argparse
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import constants
import src.csv_store._loader as csv_loader
import src.csv_store.media as csv_media
from src.core import utils


def main():
    """Export annotation rows and copy image files with unknown quality."""
    parser = argparse.ArgumentParser(
        description="Export the list of unknown-quality images and copy them "
                    "into a sibling directory of the images dir.")
    parser.add_argument('--annotations-version', required=True, type=str,
                        help='Annotations version directory name (e.g. "thesis_new_out").')
    parser.add_argument('--annotations-file', default=constants.FULL_ANNOTATIONS, type=str,
                        help='Annotations file within the version directory '
                             f'(default: {constants.FULL_ANNOTATIONS}).')
    parser.add_argument('--csv-dir', type=str, default=None,
                        help='Source database-format CSV dir. Falls back to '
                             'the .source_csv_dir sidecar inside the annotations dir.')
    parser.add_argument('--output-txt', type=str, default=None,
                        help='Path to write the list of image names. Defaults to '
                             '<annotations-version-dir>/unknown_quality_images.txt')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to copy unknown-quality images into. '
                             'Defaults to <BASE_IMAGES_DIR>/unknown_quality_images')
    args = parser.parse_args()

    version_dir = os.path.join(constants.ANNOTATIONS_DIR, args.annotations_version)
    if not os.path.isdir(version_dir):
        print(f"Error: annotations version not found at {version_dir}")
        sys.exit(1)

    annotations_path = os.path.join(version_dir, args.annotations_file)
    if not os.path.exists(annotations_path):
        print(f"Error: annotations file not found at {annotations_path}")
        sys.exit(1)

    csv_dir = utils.resolve_csv_dir(args)
    if csv_dir is None:
        print("ERROR: no CSV directory could be resolved. Pass --csv-dir or "
              "regenerate annotations with src.cli.convert_data so a "
              f"{constants.SOURCE_CSV_DIR_SIDECAR} sidecar is written.")
        sys.exit(1)
    csv_loader.set_db_dir(csv_dir)

    annotations = utils.load_csv_file(annotations_path, delimiter='\t')
    if annotations is None:
        print(f"Error: could not load annotations from {annotations_path}")
        sys.exit(1)

    # Build {media_id: quality_or_None} from csv_store.
    media_ids = []
    for row in annotations:
        try:
            media_ids.append(int(row[0]))
        except (ValueError, TypeError):
            continue
    medias = csv_media.get_medias(ids=list(set(media_ids)))
    quality_by_id = {m.id: (m.quality.lower() if m.quality else None) for m in medias}

    # Identify unknown rows. "unknown" == media missing from csv_store OR quality is None.
    unknown_rows = []
    for row in annotations:
        try:
            mid = int(row[0])
        except (ValueError, TypeError):
            continue
        if mid not in quality_by_id or quality_by_id[mid] is None:
            unknown_rows.append(row)

    image_names = [row[1] for row in unknown_rows]
    print(f"Found {len(image_names)} annotation rows of unknown quality "
          f"({len(set(image_names))} unique image names).")

    # Resolve outputs.
    output_txt = args.output_txt or os.path.join(version_dir, "unknown_quality_images.txt")
    output_dir = args.output_dir or os.path.join(constants.BASE_IMAGES_DIR, "unknown_quality_images")

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, 'w', encoding='utf-8') as f:
        for name in image_names:
            f.write(name + "\n")
    print(f"Wrote unknown-quality image names to {output_txt}")

    # Copy unique images.
    os.makedirs(output_dir, exist_ok=True)
    images_dir = constants.IMAGES_DIR
    copied = 0
    missing = []
    skipped_existing = 0
    for name in sorted(set(image_names)):
        src = os.path.join(images_dir, name)
        dst = os.path.join(output_dir, name)
        if not os.path.exists(src):
            missing.append(name)
            continue
        if os.path.exists(dst):
            skipped_existing += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(f"Copied {copied} images to {output_dir}")
    if skipped_existing:
        print(f"Skipped {skipped_existing} images already present in destination.")
    if missing:
        print(f"WARNING: {len(missing)} unknown-quality image(s) not found in {images_dir}:")
        for name in missing[:20]:
            print(f"  {name}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
