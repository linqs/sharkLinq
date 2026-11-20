"""Dataset diff utility comparing size, properties, and identity mappings across two database snapshots."""

import argparse
from collections import defaultdict
import csv
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import constants
from src.core import utils


class _Individual:
    __slots__ = ('id', 'sex', 'length', 'media', 'source', 'reference')
    def __init__(self, id, sex, length, media, source, reference):
        self.id = id
        self.sex = sex
        self.length = length
        self.media = media
        self.source = source
        self.reference = reference


class _Media:
    __slots__ = ('id', 'hash', 'name', 'quality', 'type', 'location', 'time', 'metadata')
    def __init__(self, id, hash, name, quality, type, location, time, metadata):
        self.id = id
        self.hash = hash
        self.name = name
        self.quality = quality
        self.type = type
        self.location = location
        self.time = time
        self.metadata = metadata


def main():
    """Compare dataset size, properties, and matching differences between two database snapshots."""
    parser = argparse.ArgumentParser(
        description="Comprehensive diff between two shark-matcher dataset snapshots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n"
               "  python3 experimental/data_scripts/diff_datasets.py \\\n"
               "      --old-csv-dir sm-data/database-format/baylearn \\\n"
               "      --new-csv-dir sm-data/database-format/thesis \\\n"
               "      --old-ann-dir sm-data/annotations/baylearn \\\n"
               "      --new-ann-dir sm-data/annotations/omega_unmatch_random_out\n"
    )
    parser.add_argument('--old-csv-dir', required=True,
                        help='Path to older database-format CSV directory.')
    parser.add_argument('--new-csv-dir', required=True,
                        help='Path to newer database-format CSV directory.')
    parser.add_argument('--old-ann-dir', default=None,
                        help='Path to older annotations directory (optional; omit to skip size-by-split section).')
    parser.add_argument('--new-ann-dir', default=None,
                        help='Path to newer annotations directory (optional; omit to skip size-by-split section).')
    args = parser.parse_args()

    # --- Set up timestamped output directory and tee stdout to a file ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = os.path.join(constants.DATA_COMPARISONS_DIR, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    out_file_path = os.path.join(out_dir, "diff.txt")
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

    utils.print_script_banner(__file__, args)
    print(f"Output directory: {out_dir}")

    # --- Load raw data ---
    print("Loading old dataset...")
    old_ind = _load_individuals(args.old_csv_dir)
    old_med = _load_media(args.old_csv_dir)

    print("Loading new dataset...")
    new_ind = _load_individuals(args.new_csv_dir)
    new_med = _load_media(args.new_csv_dir)

    # --- Build mappings (hash-based; fall back to name-based if no common hashes) ---
    (old_source_by_id, old_media_by_key,
     old_key_to_source, old_source_to_keys,
     old_key_to_ind_props) = _build_mappings(old_ind, old_med)

    (new_source_by_id, new_media_by_key,
     new_key_to_source, new_source_to_keys,
     new_key_to_ind_props) = _build_mappings(new_ind, new_med)

    common_keys = set(old_key_to_source) & set(new_key_to_source)
    if not common_keys and old_key_to_source and new_key_to_source:
        print(
            "\nNOTICE: No common media hashes found between datasets.\n"
            "  Falling back to name-based matching.\n"
            "  (Expected when comparing datasets with different hash schemes,\n"
            "   e.g. biology-format exports vs. live-database exports.)\n"
        )
        key_label = 'name'
        (old_source_by_id, old_media_by_key,
         old_key_to_source, old_source_to_keys,
         old_key_to_ind_props) = _build_mappings(old_ind, old_med, use_name=True)
        (new_source_by_id, new_media_by_key,
         new_key_to_source, new_source_to_keys,
         new_key_to_ind_props) = _build_mappings(new_ind, new_med, use_name=True)
    else:
        key_label = 'hash'

    print(f"Old: {len(old_source_by_id)} source sharks, {len(old_key_to_source)} media {key_label}s")
    print(f"New: {len(new_source_by_id)} source sharks, {len(new_key_to_source)} media {key_label}s")

    # --- Cross-dataset source matching ---
    classification, old_to_new, new_src_contributors = _match_sources(
        old_source_to_keys, new_key_to_source
    )

    # --- Load annotations (optional) ---
    if args.old_ann_dir and args.new_ann_dir:
        old_ann = _load_annotations(args.old_ann_dir)
        new_ann = _load_annotations(args.new_ann_dir)
        has_annotations = True
    else:
        old_ann = new_ann = None
        has_annotations = False

    # --- Run sections ---
    _section_completeness(old_source_by_id, new_source_by_id,
                          old_media_by_key, new_media_by_key,
                          old_key_to_ind_props, new_key_to_ind_props)

    if has_annotations:
        _section_size(old_ann, new_ann, old_med, new_med)
    else:
        print("\n" + "=" * 60)
        print("SECTION 2: DATASET SIZE COMPARISON")
        print("=" * 60)
        print("  (skipped — --old-ann-dir / --new-ann-dir not provided)")

    _section_properties(old_source_by_id, new_source_by_id,
                        old_media_by_key, new_media_by_key,
                        old_to_new, classification)

    _section_matching(old_source_to_keys, new_source_to_keys,
                      old_key_to_source, new_key_to_source,
                      classification, old_to_new, new_src_contributors,
                      key_label=key_label)

    print("\n" + "=" * 60)
    print("Diff complete.")
    print("=" * 60)

    sys.stdout = sys.__stdout__
    out_file.close()
    print(f"Output written to: {out_file_path}")


def _s(v):
    """Return string value or None if empty."""
    return v if v not in (None, '') else None


def _i(v):
    """Return integer value or None if empty."""
    return int(v) if v not in (None, '') else None


def _f(v):
    """Return float value or None if empty."""
    return float(v) if v not in (None, '') else None


def _load_individuals(csv_dir):
    """Return {id: _Individual} parsed from individuals.csv."""
    path = os.path.join(csv_dir, 'individuals.csv')
    result = {}
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.reader(fh):
            if len(row) < 7:
                continue
            ind_id = _i(row[0])
            if ind_id is None:
                continue
            result[ind_id] = _Individual(
                id=ind_id,
                sex=_s(row[2]),
                length=_f(row[3]),
                media=_i(row[4]),
                source=_i(row[5]),
                reference=_i(row[6]),
            )
    return result


def _load_media(csv_dir):
    """Return {id: _Media} parsed from media.csv."""
    path = os.path.join(csv_dir, 'media.csv')
    result = {}
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.reader(fh):
            if len(row) < 10:
                continue
            media_id = _i(row[0])
            if media_id is None:
                continue
            result[media_id] = _Media(
                id=media_id,
                hash=_s(row[1]),
                name=_s(row[2]),
                quality=_s(row[9]),
                type=_s(row[8]),
                location=_s(row[6]),
                time=_i(row[7]),
                metadata=_s(row[5]),
            )
    return result


def _load_annotations(ann_dir):
    """Return {split: [(media_id, filename, label)]} for full/train/valid splits."""
    splits = {}
    for split_name, fname in [('full', 'full-annotations.tsv'),
                               ('train', 'train-annotations.tsv'),
                               ('test',  'valid-annotations.tsv')]:
        path = os.path.join(ann_dir, fname)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            splits[split_name] = []
            continue
        rows = []
        with open(path, newline='', encoding='utf-8') as fh:
            for line in fh:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 3:
                    rows.append((int(parts[0]), parts[1], int(parts[2])))
        splits[split_name] = rows
    return splits


def _build_mappings(ind_by_id, media_by_id, use_name=False):
    """Construct index lookup mappings from individual and media records."""
    source_by_id = {
        ind.id: ind
        for ind in ind_by_id.values()
        if ind.source == ind.id and ind.media is None
    }

    get_key = (lambda m: m.name) if use_name else (lambda m: m.hash)

    media_by_key = {
        get_key(m): m
        for m in media_by_id.values()
        if get_key(m) is not None
    }

    key_to_source = {}
    key_to_ind_props = {}
    source_to_keys = defaultdict(set)
    for ind in ind_by_id.values():
        if ind.source == ind.id or ind.media is None:
            continue  # skip source/archive individuals
        media = media_by_id.get(ind.media)
        if media is None:
            continue
        k = get_key(media)
        if k is None:
            continue
        key_to_source[k] = ind.source
        key_to_ind_props[k] = (ind.sex, ind.length)
        source_to_keys[ind.source].add(k)

    return source_by_id, media_by_key, key_to_source, dict(source_to_keys), key_to_ind_props


def _match_sources(old_source_to_hashes, new_hash_to_source):
    """Classify each old source shark and identify the best matching new source."""
    # First pass: for each old source find dest counts.
    dest_counts = {}  # old_src_id -> {new_src_id: hash_count}
    for old_id, hashes in old_source_to_hashes.items():
        counts = defaultdict(int)
        for h in hashes:
            new_src = new_hash_to_source.get(h)
            if new_src is not None:
                counts[new_src] += 1
        dest_counts[old_id] = dict(counts)

    # Build inverse: which old sources fed each new source.
    new_src_contributors = defaultdict(set)
    for old_id, counts in dest_counts.items():
        for new_id in counts:
            new_src_contributors[new_id].add(old_id)
    new_src_contributors = dict(new_src_contributors)

    classification = {}
    old_to_new = {}

    for old_id, counts in dest_counts.items():
        if not counts:
            classification[old_id] = 'disappeared'
            old_to_new[old_id] = None
            continue

        # Majority new source (most hashes received).
        majority_new = max(counts, key=counts.__getitem__)
        old_to_new[old_id] = majority_new

        if len(counts) >= 2:
            classification[old_id] = 'fragmented'
        else:
            # Exactly one new destination.
            new_id = majority_new
            contributors = new_src_contributors.get(new_id, set())
            if len(contributors) >= 2:
                classification[old_id] = 'merged_in'
            else:
                classification[old_id] = 'renamed'

    return classification, old_to_new, new_src_contributors


def _pct(null_count, total):
    """Format null count as a percentage string."""
    if total == 0:
        return "N/A"
    return f"{null_count / total * 100:.1f}%"


def _section_completeness(old_source_by_id, new_source_by_id,
                          old_media_by_hash, new_media_by_hash,
                          old_hash_to_ind_props, new_hash_to_ind_props):
    """Print the metadata completeness comparison section."""
    print("\n" + "=" * 60)
    print("SECTION 1: METADATA COMPLETENESS")
    print("=" * 60)

    # --- Source shark properties ---
    old_sharks = list(old_source_by_id.values())
    new_sharks = list(new_source_by_id.values())
    n_old_s, n_new_s = len(old_sharks), len(new_sharks)

    print(f"\n[Source Shark Properties]  (old: {n_old_s} sharks, new: {n_new_s} sharks)")

    shark_props = [
        ('sex',    lambda s: s.sex),
        ('length', lambda s: s.length),
    ]
    rows = []
    for label, getter in shark_props:
        old_null = sum(1 for s in old_sharks if getter(s) is None)
        new_null = sum(1 for s in new_sharks if getter(s) is None)
        delta = new_null - old_null
        rows.append([
            label,
            old_null, _pct(old_null, n_old_s),
            new_null, _pct(new_null, n_new_s),
            f"{delta:+d}",
        ])
    utils.print_table(
        ['Property', 'Old null', 'Old null%', 'New null', 'New null%', 'Delta'],
        rows,
    )

    # --- Image properties ---
    old_hashes = list(old_media_by_hash.keys())
    new_hashes = list(new_media_by_hash.keys())
    n_old_m, n_new_m = len(old_hashes), len(new_hashes)

    print(f"\n[Image Properties]  (old: {n_old_m} images, new: {n_new_m} images)")

    # Getters: (label, fn_for_old_hash, fn_for_new_hash)
    img_props = [
        ('sex',      lambda h: old_hash_to_ind_props.get(h, (None, None))[0],
                     lambda h: new_hash_to_ind_props.get(h, (None, None))[0]),
        ('length',   lambda h: old_hash_to_ind_props.get(h, (None, None))[1],
                     lambda h: new_hash_to_ind_props.get(h, (None, None))[1]),
        ('quality',  lambda h: old_media_by_hash[h].quality,
                     lambda h: new_media_by_hash[h].quality),
        ('type',     lambda h: old_media_by_hash[h].type,
                     lambda h: new_media_by_hash[h].type),
    ]
    rows = []
    for label, old_getter, new_getter in img_props:
        old_null = sum(1 for h in old_hashes if old_getter(h) is None)
        new_null = sum(1 for h in new_hashes if new_getter(h) is None)
        delta = new_null - old_null
        rows.append([
            label,
            old_null, _pct(old_null, n_old_m),
            new_null, _pct(new_null, n_new_m),
            f"{delta:+d}",
        ])
    utils.print_table(
        ['Property', 'Old null', 'Old null%', 'New null', 'New null%', 'Delta'],
        rows,
    )


def _quality_bucket(media_id, med_by_id):
    """Return 'poor' for Unmatchable/Poor quality, 'good' for Medium/Good, None if unknown."""
    m = med_by_id.get(media_id)
    if m is None or m.quality is None:
        return None
    q = m.quality.lower()
    if q in ('unmatchable', 'poor'):
        return 'poor'
    if q in ('medium', 'good'):
        return 'good'
    return None


def _split_stats(ann_rows, med_by_id, full_rows):
    """Compute size-comparison metrics for one split's annotation rows."""
    img_count = len(ann_rows)
    split_sharks = {r[2] for r in ann_rows}
    shark_count = len(split_sharks)

    full_shark_img_counts = defaultdict(int)
    for r in full_rows:
        full_shark_img_counts[r[2]] += 1

    b2   = sum(1 for r in ann_rows if full_shark_img_counts.get(r[2], 0) == 2)
    b3_5 = sum(1 for r in ann_rows if 3 <= full_shark_img_counts.get(r[2], 0) <= 5)
    b6p  = sum(1 for r in ann_rows if full_shark_img_counts.get(r[2], 0) >= 6)

    poor = sum(1 for r in ann_rows if _quality_bucket(r[0], med_by_id) == 'poor')
    good = sum(1 for r in ann_rows if _quality_bucket(r[0], med_by_id) == 'good')
    unknown = img_count - poor - good
    return img_count, shark_count, b2, b3_5, b6p, poor, good, unknown


def _section_size(old_ann, new_ann, old_med, new_med):
    """Print dataset size comparison table across splits."""
    print("\n" + "=" * 60)
    print("SECTION 2: DATASET SIZE COMPARISON")
    print("=" * 60)

    def delta(a, b):
        d = b - a
        return f"+{d}" if d >= 0 else str(d)

    old_full = old_ann.get('full', [])
    new_full = new_ann.get('full', [])

    for split in ('full', 'train', 'test'):
        old_rows = old_ann.get(split, [])
        new_rows = new_ann.get(split, [])

        (o_img, o_sh, o_b2, o_b35, o_b6p, o_poor, o_good, o_unk) = _split_stats(old_rows, old_med, old_full)
        (n_img, n_sh, n_b2, n_b35, n_b6p, n_poor, n_good, n_unk) = _split_stats(new_rows, new_med, new_full)

        print(f"\n[{split.capitalize()} Split]")
        utils.print_table(
            ['Metric', 'Old', 'New', 'Delta'],
            [
                ['Image Count',                   o_img,  n_img,  delta(o_img,  n_img)],
                ['Shark Count',                   o_sh,   n_sh,   delta(o_sh,   n_sh)],
                ['Imgs from 2-img Sharks (full)',   o_b2,   n_b2,   delta(o_b2,   n_b2)],
                ['Imgs from 3-5-img Sharks (full)', o_b35,  n_b35,  delta(o_b35,  n_b35)],
                ['Imgs from 6+-img Sharks (full)',  o_b6p,  n_b6p,  delta(o_b6p,  n_b6p)],
                ['Unmatchable/Poor Quality Imgs', o_poor, n_poor, delta(o_poor, n_poor)],
                ['Medium/Good Quality Imgs',      o_good, n_good, delta(o_good, n_good)],
                ['Unknown/Null Quality Imgs',     o_unk,  n_unk,  delta(o_unk,  n_unk)],
            ],
        )


def _null_display(val):
    """Format None as 'null' and other values as strings."""
    return 'null' if val is None else str(val)


def _print_transition_table(title, transitions, total_compared):
    """Print a collapsed 3-row transition table: null->known, known->different, known->null."""
    print(f"\n--- {title} ---")
    unchanged = transitions.get(('_same_', '_same_'), 0)
    null_to_known   = sum(v for (o, n), v in transitions.items() if o is None  and n is not None)
    known_to_known  = sum(v for (o, n), v in transitions.items() if o is not None and n is not None and o != n)
    known_to_null   = sum(v for (o, n), v in transitions.items() if o is not None and n is None)
    total_changed = null_to_known + known_to_known + known_to_null
    print(f"  Compared: {total_compared}  |  Unchanged: {unchanged}  |  Changed: {total_changed}")
    utils.print_table(
        ['Transition', 'Count'],
        [
            ['null -> known',          null_to_known],
            ['known -> different known', known_to_known],
            ['known -> null',          known_to_null],
        ]
    )


def _section_properties(old_source_by_id, new_source_by_id,
                        old_media_by_hash, new_media_by_hash,
                        old_to_new, classification):
    """Print property changes across matched source sharks and common media."""
    print("\n" + "=" * 60)
    print("SECTION 3: PROPERTY CHANGES")
    print("=" * 60)

    # --- 2a. Source shark properties ---
    print("\n[Source Shark Properties]")
    sex_trans = defaultdict(int)
    len_trans = defaultdict(int)
    compared_sharks = 0

    for old_id, new_id in old_to_new.items():
        if new_id is None:
            continue
        old_s = old_source_by_id.get(old_id)
        new_s = new_source_by_id.get(new_id)
        if old_s is None or new_s is None:
            continue
        compared_sharks += 1

        # sex
        key = (old_s.sex, new_s.sex)
        sex_trans[key if key[0] != key[1] else ('_same_', '_same_')] += 1

        # length — bucket into changed/unchanged
        ol, nl = old_s.length, new_s.length
        if ol == nl:
            len_trans[('_same_', '_same_')] += 1
        else:
            len_trans[(ol, nl)] += 1

    _print_transition_table(f"Source shark: sex (matched pairs: {compared_sharks})",
                             sex_trans, compared_sharks)
    _print_transition_table(f"Source shark: length (matched pairs: {compared_sharks})",
                             len_trans, compared_sharks)

    # --- 2b. Media properties ---
    print("\n[Media Properties]")
    common_hashes = set(old_media_by_hash) & set(new_media_by_hash)
    print(f"  Common media records compared: {len(common_hashes)}")

    media_props = ['quality', 'type', 'location', 'time', 'metadata']
    prop_trans = {p: defaultdict(int) for p in media_props}

    for h in common_hashes:
        om = old_media_by_hash[h]
        nm = new_media_by_hash[h]
        for prop in media_props:
            ov = getattr(om, prop)
            nv = getattr(nm, prop)
            key = (ov, nv) if ov != nv else ('_same_', '_same_')
            prop_trans[prop][key] += 1

    for prop in media_props:
        _print_transition_table(f"Media: {prop}", prop_trans[prop], len(common_hashes))


def _section_matching(old_source_to_hashes, new_source_to_hashes,
                      old_hash_to_source, new_hash_to_source,
                      classification, old_to_new, new_src_contributors,
                      key_label='hash'):
    """Print source shark identity matching, merge, and fragmentation details."""
    print("\n" + "=" * 60)
    print("SECTION 4: MATCHING DIFF")
    print("=" * 60)

    old_hashes = set(old_hash_to_source)
    new_hashes = set(new_hash_to_source)
    common = old_hashes & new_hashes
    only_old = old_hashes - new_hashes
    only_new = new_hashes - old_hashes

    utils.print_key_value_section(f"Media {key_label} overview", {
        f"Old dataset ({key_label}s)": len(old_hashes),
        f"New dataset ({key_label}s)": len(new_hashes),
        "Common":                      len(common),
        "Only in old (removed)":       len(only_old),
        "Only in new (added)":         len(only_new),
    })

    # --- Old source shark classification counts ---
    buckets = defaultdict(int)
    for b in classification.values():
        buckets[b] += 1

    utils.print_key_value_section("Old source shark classifications", {
        "Total old source sharks":  len(classification),
        "Intact, no merge": buckets['renamed'],
        "Merged into another shark":  buckets['merged_in'],
        "Fragmented (split across new sources)": buckets['fragmented'],
        "Disappeared (all media removed)":       buckets['disappeared'],
    })

    # --- New source shark classification ---
    new_bucket_counts = defaultdict(int)
    for new_id, contributors in new_src_contributors.items():
        if len(contributors) >= 2:
            new_bucket_counts['merge_result'] += 1
        else:
            new_bucket_counts['renamed'] += 1

    entirely_new_count = len(new_source_to_hashes) - len(new_src_contributors)
    utils.print_key_value_section("New source shark classifications", {
        "Total new source sharks":          len(new_source_to_hashes),
        "Intact, no merge":                 (
            f"{new_bucket_counts['renamed']}"
            + (f"  # +{new_bucket_counts['renamed'] - buckets['renamed']} vs old-side:"
               f" fragmented old sources can feed new sources with only one contributor"
               if new_bucket_counts['renamed'] != buckets['renamed'] else "")
        ),
        "Merger result (2+ old sources)":   new_bucket_counts['merge_result'],
        "Entirely new (no old hashes)":     entirely_new_count,
    })

    # --- Merge detail ---
    merge_results = {
        new_id: contribs
        for new_id, contribs in new_src_contributors.items()
        if len(contribs) >= 2
    }
    if merge_results:
        print(f"\n--- Merge detail ({len(merge_results)} new source sharks from merges) ---")
        rows = []
        for new_id, contribs in sorted(merge_results.items(),
                                        key=lambda x: -len(x[1])):
            new_hashes_for_src = new_source_to_hashes.get(new_id, set())
            contrib_parts = []
            for old_id in sorted(contribs):
                old_hashes_in_new = old_source_to_hashes.get(old_id, set()) & new_hashes_for_src
                contrib_parts.append(f"old_src_{old_id}({len(old_hashes_in_new)}img)")
            added_new = len(new_hashes_for_src - old_hashes)
            added_str = f" + {added_new} new" if added_new else ""
            rows.append([f"new_src_{new_id}",
                         len(new_hashes_for_src),
                         " + ".join(contrib_parts) + added_str])
        utils.print_table(['New source', 'Total images', 'Composed of'], rows)

    # --- Fragmentation / unmatched detail ---
    fragmented_ids = [sid for sid, b in classification.items() if b == 'fragmented']
    if fragmented_ids:
        print(f"\n--- Fragmentation detail ({len(fragmented_ids)} old sources split) ---")

        # dest_counts per fragmented old source
        unmatched_total = 0
        rows = []
        for old_id in sorted(fragmented_ids):
            hashes = old_source_to_hashes.get(old_id, set())
            dest = defaultdict(int)
            for h in hashes:
                ns = new_hash_to_source.get(h)
                if ns is not None:
                    dest[ns] += 1
            majority_new = max(dest, key=dest.__getitem__)
            majority_count = dest[majority_new]
            minority_count = sum(v for k, v in dest.items() if k != majority_new)
            unmatched_total += minority_count

            dest_str = ", ".join(
                f"new_src_{nid}:{cnt}" + (" (majority)" if nid == majority_new else "")
                for nid, cnt in sorted(dest.items(), key=lambda x: -x[1])
            )
            rows.append([f"old_src_{old_id}", len(hashes), dest_str])

        utils.print_table(['Old source', 'Orig images', 'Split across'], rows)

        print(f"\n  Unmatched media sharks (minority remaps from fragmented sources): {unmatched_total}")
    else:
        unmatched_total = 0
        print("\n  No fragmented source sharks.")
        print("  Unmatched media sharks: 0")

    # --- Summary ---
    utils.print_key_value_section("Matching diff summary", {
        "Media sharks merged (old sources in clean whole-group moves)": buckets['merged_in'],
        "Unmatched media sharks (individual remaps from splits)":        unmatched_total,
    })


if __name__ == "__main__":
    main()
