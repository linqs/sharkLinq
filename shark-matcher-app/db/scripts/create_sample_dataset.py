"""Create an additive cutoff-year sample dataset from existing image, database, and annotations archives."""

import argparse
import csv
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Set, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from core import app_constants
from core import app_utils


def extract_year_from_name(name: str) -> Optional[int]:
    """Extract a 4-digit year from the 2-digit year prefix in an image or media name."""
    match = re.search(r"^[A-Za-z]+(\d{2})", name)
    if not match:
        match = re.search(r"(\d{2})", name)
    if match:
        y = int(match.group(1))
        return (2000 + y) if y <= 50 else (1900 + y)
    return None


def parse_date_key_from_name(name: str) -> Tuple[int, int, int]:
    """Extract a (year, month, day) tuple from a name string for chronological sorting."""
    match = re.search(r"^[A-Za-z]+(\d{2})(\d{2})?(\d{2})?", name)
    if match:
        y = int(match.group(1))
        full_year = (2000 + y) if y <= 50 else (1900 + y)
        mo_str = match.group(2)
        da_str = match.group(3)
        try:
            mo = int(mo_str) if mo_str else 0
        except ValueError:
            mo = 0
        try:
            da = int(da_str) if da_str else 0
        except ValueError:
            da = 0
        return (full_year, mo, da)
    return (0, 0, 0)


def filter_annotations_split(
    input_ann_dir: str,
    output_ann_dir: str,
    cutoff_year: int = 2018,
    source_csv_dir: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Filter an annotations split directory (full, train, valid TSVs) up to the specified cutoff year."""
    app_utils.print_function_banner(
        "filter_annotations_split",
        f"Filtering annotations split to year <= {cutoff_year} (additive, non-destructive)",
    )

    print(f"Input Annotations Dir:  {os.path.abspath(input_ann_dir)}")
    print(f"Output Annotations Dir: {os.path.abspath(output_ann_dir)}")
    print(f"Cutoff Year:            {cutoff_year}")
    print(f"Source CSV Dir:         {os.path.abspath(source_csv_dir) if source_csv_dir else 'None'}")
    print(f"Dry Run:                {dry_run}\n")

    files_to_filter = [
        ("full-annotations.tsv", app_constants.FULL_ANNOTATIONS),
        ("train-annotations.tsv", app_constants.TRAIN_ANNOTATIONS),
        ("valid-annotations.tsv", app_constants.VALID_ANNOTATIONS),
    ]

    filtered_splits: Dict[str, List[List[str]]] = {}
    stats: Dict[str, int] = {}

    for label, fname in files_to_filter:
        in_path = os.path.join(input_ann_dir, fname)
        rows = app_utils.load_csv_file(in_path, delimiter="\t")
        if rows is None:
            raise FileNotFoundError(f"Required annotations file not found: {in_path}")

        kept = [
            r for r in rows
            if r and len(r) >= 3 and (extract_year_from_name(r[1]) or 9999) <= cutoff_year
        ]
        filtered_splits[fname] = kept
        split_name = fname.replace("-annotations.tsv", "")
        stats[f"{split_name}_total"] = len(rows)
        stats[f"{split_name}_retained"] = len(kept)
        stats[f"{split_name}_excluded"] = len(rows) - len(kept)

        print(f"{fname:25}: {len(rows):5} total -> {len(kept):5} retained (<= {cutoff_year})")

    # Invariant checks:
    # 1. train + valid == full
    full_rows = filtered_splits[app_constants.FULL_ANNOTATIONS]
    train_rows = filtered_splits[app_constants.TRAIN_ANNOTATIONS]
    valid_rows = filtered_splits[app_constants.VALID_ANNOTATIONS]

    if len(train_rows) + len(valid_rows) != len(full_rows):
        raise ValueError(
            f"Split partition mismatch: train ({len(train_rows)}) + valid ({len(valid_rows)}) "
            f"!= full ({len(full_rows)})"
        )

    # 2. train and valid are disjoint
    train_set = {tuple(r) for r in train_rows}
    valid_set = {tuple(r) for r in valid_rows}
    full_set = {tuple(r) for r in full_rows}

    overlap = train_set & valid_set
    if overlap:
        raise ValueError(f"Train and valid splits overlap by {len(overlap)} rows")

    if (train_set | valid_set) != full_set:
        raise ValueError("Union of train and valid does not equal full split")

    print("\nSplit partition invariants verified:")
    print(f"  train ({len(train_rows)}) + valid ({len(valid_rows)}) == full ({len(full_rows)})")
    print(f"  Disjoint: True (overlap = 0)")

    # Shark counts
    full_sharks = {r[2] for r in full_rows}
    train_sharks = {r[2] for r in train_rows}
    valid_sharks = {r[2] for r in valid_rows}
    print(f"  Unique sharks in full:  {len(full_sharks)}")
    print(f"  Unique sharks in train: {len(train_sharks)}")
    print(f"  Unique sharks in valid: {len(valid_sharks)}")
    print(f"  Closed-set query sharks:{len(valid_sharks & train_sharks)}")
    print(f"  Open-set query sharks:  {len(valid_sharks - train_sharks)}")

    stats["unique_sharks_full"] = len(full_sharks)
    stats["unique_sharks_train"] = len(train_sharks)
    stats["unique_sharks_valid"] = len(valid_sharks)

    if dry_run:
        print("\n[DRY RUN] Completed without writing any files.")
        return stats

    os.makedirs(output_ann_dir, exist_ok=True)
    for fname, rows in filtered_splits.items():
        out_path = os.path.join(output_ann_dir, fname)
        app_utils.write_csv_file(out_path, rows, delimiter="\t", durable=True)
    print(f"\nWrote 3 annotations TSV files to: {os.path.abspath(output_ann_dir)}")

    # Write .source_csv_dir sidecar
    if source_csv_dir:
        sidecar_path = os.path.join(output_ann_dir, ".source_csv_dir")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(app_utils.portable_csv_dir(source_csv_dir) + "\n")
        print(f"Wrote .source_csv_dir pointing to: {app_utils.portable_csv_dir(source_csv_dir)}")

    return stats


def create_sample_dataset(
    input_images_dir: str,
    input_csv_dir: str,
    output_images_dir: str,
    output_csv_dir: str,
    cutoff_year: int = 2018,
    reference_policy: str = "latest_pre",
    copy_images: bool = True,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Create filtered sample dataset files and copy retained images up to the cutoff year."""
    app_utils.print_function_banner(
        "create_sample_dataset",
        f"Creating sample dataset up to year {cutoff_year} (additive, non-destructive)",
    )

    print(f"Input Images Dir: {os.path.abspath(input_images_dir)}")
    print(f"Input CSV Dir:    {os.path.abspath(input_csv_dir)}")
    print(f"Output Images Dir:{os.path.abspath(output_images_dir)}")
    print(f"Output CSV Dir:   {os.path.abspath(output_csv_dir)}")
    print(f"Cutoff Year:      {cutoff_year}")
    print(f"Reference Policy: {reference_policy}")
    print(f"Copy Images:      {copy_images}")
    print(f"Dry Run:          {dry_run}\n")

    # 1. Load Media
    media_csv_path = os.path.join(input_csv_dir, "media.csv")
    raw_media = app_utils.load_csv_file(media_csv_path)
    if raw_media is None:
        raise FileNotFoundError(f"Missing media.csv in {input_csv_dir}")

    retained_media = []
    retained_media_ids: Set[str] = set()
    retained_media_names: Set[str] = set()
    retained_media_filenames: List[str] = []
    media_id_to_name: Dict[str, str] = {}

    for row in raw_media:
        if not row:
            continue
        m_id, m_hash, m_name, m_filename, m_source, m_meta, m_loc, m_time, m_type, m_qual = row
        media_id_to_name[m_id] = m_name
        yr = extract_year_from_name(m_name)
        if yr is not None and yr <= cutoff_year:
            retained_media.append(row)
            retained_media_ids.add(m_id)
            retained_media_names.add(m_name)
            retained_media_filenames.append(m_filename)

    print(f"Media: {len(raw_media)} total -> {len(retained_media)} retained (<= {cutoff_year})")

    # 2. Load Individuals
    ind_csv_path = os.path.join(input_csv_dir, "individuals.csv")
    raw_inds = app_utils.load_csv_file(ind_csv_path)
    if raw_inds is None:
        raise FileNotFoundError(f"Missing individuals.csv in {input_csv_dir}")

    media_inds = [r for r in raw_inds if r and len(r) >= 7 and r[4] != ""]
    source_inds = [r for r in raw_inds if r and len(r) >= 7 and r[4] == ""]

    # Retained Media Individuals: those whose media was retained
    retained_media_inds = [r for r in media_inds if r[4] in retained_media_ids]
    retained_media_ind_ids = {r[0] for r in retained_media_inds}

    # Group retained media individuals by parent source ID
    source_to_retained_children: Dict[str, List[List[str]]] = {s[0]: [] for s in source_inds}
    for r in retained_media_inds:
        source_id = r[5]
        if source_id in source_to_retained_children:
            source_to_retained_children[source_id].append(r)

    retained_source_inds = []
    repaired_references_count = 0

    for s in source_inds:
        s_id = s[0]
        children = source_to_retained_children.get(s_id, [])
        if not children:
            # All observations were > cutoff_year; exclude source individual
            continue

        curr_ref = s[6]
        if curr_ref not in retained_media_ind_ids:
            # Need to repair reference to a pre-cutoff media individual
            if reference_policy == "latest_pre":
                # Sort descending by date encoded in filename, then by ID descending
                children_sorted = sorted(
                    children,
                    key=lambda c: (
                        parse_date_key_from_name(media_id_to_name.get(c[4], "")),
                        int(c[0]) if c[0].isdigit() else 0,
                    ),
                    reverse=True,
                )
            elif reference_policy == "oldest":
                children_sorted = sorted(
                    children,
                    key=lambda c: (
                        parse_date_key_from_name(media_id_to_name.get(c[4], "")),
                        int(c[0]) if c[0].isdigit() else 0,
                    ),
                )
            else:  # lowest_id
                children_sorted = sorted(
                    children,
                    key=lambda c: int(c[0]) if c[0].isdigit() else 0,
                )

            chosen_child = children_sorted[0]
            s_repaired = list(s)
            s_repaired[6] = chosen_child[0]
            retained_source_inds.append(s_repaired)
            repaired_references_count += 1
        else:
            retained_source_inds.append(s)

    retained_individuals = retained_source_inds + retained_media_inds
    # Sort individuals by integer ID
    retained_individuals.sort(key=lambda r: int(r[0]) if r[0].lstrip("-").isdigit() else 0)
    retained_all_ind_ids = {r[0] for r in retained_individuals} | {"-1"}

    print(f"Media Individuals:  {len(media_inds)} total -> {len(retained_media_inds)} retained")
    print(f"Source Individuals: {len(source_inds)} total -> {len(retained_source_inds)} retained "
          f"({repaired_references_count} references repaired)")
    print(f"Total Individuals:  {len(raw_inds)} total -> {len(retained_individuals)} retained")

    # 3. Associated Tables: AlternateNames, Tags, Labels
    alt_csv_path = os.path.join(input_csv_dir, "alternateNames.csv")
    raw_alt = app_utils.load_csv_file(alt_csv_path) or []
    retained_alt = [r for r in raw_alt if r and r[1] in retained_all_ind_ids]
    retained_alt.sort(key=lambda r: int(r[0]) if r[0].lstrip("-").isdigit() else 0)
    print(f"AlternateNames:     {len(raw_alt)} total -> {len(retained_alt)} retained")

    tags_csv_path = os.path.join(input_csv_dir, "tags.csv")
    raw_tags = app_utils.load_csv_file(tags_csv_path) or []
    retained_tags = [r for r in raw_tags if r and r[1] in retained_all_ind_ids]
    retained_tags.sort(key=lambda r: int(r[0]) if r[0].lstrip("-").isdigit() else 0)
    print(f"Tags:               {len(raw_tags)} total -> {len(retained_tags)} retained")

    labels_csv_path = os.path.join(input_csv_dir, "labels.csv")
    raw_labels = app_utils.load_csv_file(labels_csv_path) or []
    retained_labels = [r for r in raw_labels if r and r[1] in retained_all_ind_ids]
    retained_labels.sort(key=lambda r: int(r[0]) if r[0].lstrip("-").isdigit() else 0)
    print(f"Labels:             {len(raw_labels)} total -> {len(retained_labels)} retained")

    # 4. Annotations export format: individual_media.csv
    im_csv_path = os.path.join(input_csv_dir, "individual_media.csv")
    raw_im = app_utils.load_csv_file(im_csv_path) or []
    if raw_im:
        im_header = raw_im[0]
        im_data = [r for r in raw_im[1:] if r and len(r) >= 4 and r[3] in retained_media_names]
        retained_im = [im_header] + im_data
    else:
        retained_im = []
    print(f"individual_media:   {len(raw_im) - 1 if raw_im else 0} data rows -> "
          f"{len(retained_im) - 1 if retained_im else 0} retained data rows")

    # 5. Users and Tokens
    users_csv_path = os.path.join(input_csv_dir, "users.csv")
    raw_users = app_utils.load_csv_file(users_csv_path) or []
    retained_users = list(raw_users)
    print(f"Users:              {len(raw_users)} total -> {len(retained_users)} retained")

    tokens_csv_path = os.path.join(input_csv_dir, "tokens.csv")
    # Tokens table in sample dataset is kept empty
    retained_tokens: List[List[str]] = []
    print(f"Tokens:             reset to empty file (0 active tokens)")

    # 6. Actions and Proposals
    actions_csv_path = os.path.join(input_csv_dir, "actions.csv")
    raw_actions = app_utils.load_csv_file(actions_csv_path) or []
    retained_actions = []
    for r in raw_actions:
        if not r or len(r) < 5:
            continue
        obj = r[4]
        orig, fin = r[1], r[2]
        if obj == "media":
            if (orig not in ("-1", "") and orig not in retained_media_ids) or \
               (fin not in ("-1", "") and fin not in retained_media_ids):
                continue
        elif obj in ("shark", "individual"):
            if (orig not in ("-1", "") and orig not in retained_all_ind_ids) or \
               (fin not in ("-1", "") and fin not in retained_all_ind_ids):
                continue
        retained_actions.append(r)
    retained_actions.sort(key=lambda r: int(r[0]) if r[0].lstrip("-").isdigit() else 0)
    print(f"Actions:            {len(raw_actions)} total -> {len(retained_actions)} retained")

    proposals_csv_path = os.path.join(input_csv_dir, "proposals.csv")
    raw_proposals = app_utils.load_csv_file(proposals_csv_path) or []
    retained_proposals = []
    for r in raw_proposals:
        if not r or len(r) < 5:
            continue
        obj = r[4]
        orig, fin = r[1], r[2]
        if obj == "media":
            if (orig not in ("-1", "") and orig not in retained_media_ids) or \
               (fin not in ("-1", "") and fin not in retained_media_ids):
                continue
        elif obj in ("shark", "individual"):
            if (orig not in ("-1", "") and orig not in retained_all_ind_ids) or \
               (fin not in ("-1", "") and fin not in retained_all_ind_ids):
                continue
        retained_proposals.append(r)
    retained_proposals.sort(key=lambda r: int(r[0]) if r[0].lstrip("-").isdigit() else 0)
    print(f"Proposals:          {len(raw_proposals)} total -> {len(retained_proposals)} retained")

    # Sort media rows by ID
    retained_media.sort(key=lambda r: int(r[0]) if r[0].lstrip("-").isdigit() else 0)

    stats = {
        "media": len(retained_media),
        "individuals_media": len(retained_media_inds),
        "individuals_source": len(retained_source_inds),
        "individuals_total": len(retained_individuals),
        "repaired_references": repaired_references_count,
        "alternate_names": len(retained_alt),
        "tags": len(retained_tags),
        "labels": len(retained_labels),
        "individual_media": len(retained_im) - 1 if retained_im else 0,
        "users": len(retained_users),
        "tokens": len(retained_tokens),
        "actions": len(retained_actions),
        "proposals": len(retained_proposals),
        "images_copied": 0,
        "images_missing": 0,
    }

    if dry_run:
        print("\n[DRY RUN] Completed without writing any files.")
        return stats

    # 7. Write Output CSVs
    os.makedirs(output_csv_dir, exist_ok=True)
    tables_to_write = [
        ("media.csv", retained_media),
        ("individuals.csv", retained_individuals),
        ("alternateNames.csv", retained_alt),
        ("tags.csv", retained_tags),
        ("labels.csv", retained_labels),
        ("individual_media.csv", retained_im),
        ("users.csv", retained_users),
        ("tokens.csv", retained_tokens),
        ("actions.csv", retained_actions),
        ("proposals.csv", retained_proposals),
    ]

    for fname, data in tables_to_write:
        dest = os.path.join(output_csv_dir, fname)
        app_utils.write_csv_file(dest, data, durable=True)
    print(f"\nWrote 10 CSV tables to: {os.path.abspath(output_csv_dir)}")

    # 8. Copy Images
    if copy_images:
        os.makedirs(output_images_dir, exist_ok=True)
        copied = 0
        missing = 0
        total_bytes = 0

        print(f"\nCopying {len(retained_media_filenames)} cataloged images to: {os.path.abspath(output_images_dir)}")
        for filename in retained_media_filenames:
            src_path = os.path.join(input_images_dir, filename)
            dst_path = os.path.join(output_images_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                total_bytes += os.path.getsize(dst_path)
                copied += 1
            else:
                print(f"  [WARNING] Image not found on disk: {src_path}")
                missing += 1

        stats["images_copied"] = copied
        stats["images_missing"] = missing
        print(f"Image copying complete: {copied} copied ({total_bytes / (1024*1024):.2f} MB), {missing} missing.")

    return stats


def main():
    """Main CLI entry point for create_sample_dataset."""
    parser = argparse.ArgumentParser(
        description="Create an additive pre-2019 sample dataset from existing image, database, and annotations archives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-images-dir",
        default=os.path.join(app_constants.BASE_DATA_DIR, "images", "images"),
        help="Path to source directory of images.",
    )
    parser.add_argument(
        "--input-csv-dir",
        default=os.path.join(app_constants.BASE_DATA_DIR, "database-format", "thesis"),
        help="Path to source directory of database CSV files.",
    )
    parser.add_argument(
        "--output-images-dir",
        default=os.path.join(app_constants.BASE_DATA_DIR, "images", "sample-pre2019"),
        help="Path to destination directory for sample images.",
    )
    parser.add_argument(
        "--output-csv-dir",
        default=os.path.join(app_constants.BASE_DATA_DIR, "database-format", "thesis-sample-pre2019"),
        help="Path to destination directory for sample database CSV files.",
    )
    parser.add_argument(
        "--cutoff-year",
        type=int,
        default=2018,
        help="Inclusive cutoff year (records up to this year are retained).",
    )
    parser.add_argument(
        "--reference-policy",
        choices=["latest_pre", "oldest", "lowest_id"],
        default="latest_pre",
        help="Selection policy for replacement reference individuals on mixed sharks.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Skip image copying and generate CSV files only.",
    )
    parser.add_argument(
        "--input-annotations-dir",
        default=os.path.join(app_constants.ANNOTATIONS_DIR, "omega_random_out"),
        help="Path to source annotations split directory (full, train, valid TSVs).",
    )
    parser.add_argument(
        "--output-annotations-dir",
        default=os.path.join(app_constants.ANNOTATIONS_DIR, "omega_random_out_pre2019"),
        help="Path to destination annotations split directory.",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Only filter the annotations split; skip database CSV and image generation.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Skip filtering annotations split.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview counts and integrity without writing files.",
    )

    args = parser.parse_args()

    if not args.annotations_only:
        create_sample_dataset(
            input_images_dir=args.input_images_dir,
            input_csv_dir=args.input_csv_dir,
            output_images_dir=args.output_images_dir,
            output_csv_dir=args.output_csv_dir,
            cutoff_year=args.cutoff_year,
            reference_policy=args.reference_policy,
            copy_images=not args.no_copy_images,
            dry_run=args.dry_run,
        )

    if not args.no_annotations and args.input_annotations_dir and args.output_annotations_dir:
        filter_annotations_split(
            input_ann_dir=args.input_annotations_dir,
            output_ann_dir=args.output_annotations_dir,
            cutoff_year=args.cutoff_year,
            source_csv_dir=args.output_csv_dir,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
