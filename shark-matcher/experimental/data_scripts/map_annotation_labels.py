#!/usr/bin/env python3
"""Map annotation labels from full-annotations.tsv onto valid-annotations.tsv by matching first-column IDs."""

import pathlib
import sys


def main():
    """Map annotation labels from full-annotations.tsv onto valid-annotations.tsv."""
    # Default file paths
    script_dir = pathlib.Path(__file__).parent
    full_file = script_dir / 'data' / 'annotations' / 'bayLearn' / 'full-annotations.tsv'
    valid_file = script_dir / 'data' / 'annotations' / 'full-annotations.tsv'
    output_file = script_dir / 'data' / 'annotations' / 'full-annotations2.tsv'

    # Allow command-line arguments to override defaults
    if len(sys.argv) >= 2:
        full_file = pathlib.Path(sys.argv[1])
    if len(sys.argv) >= 3:
        valid_file = pathlib.Path(sys.argv[2])
    if len(sys.argv) >= 4:
        output_file = pathlib.Path(sys.argv[3])

    print(f"Loading full annotations from: {full_file}")
    mapping = load_full_annotations(full_file)
    print(f"Loaded {len(mapping)} entries from full-annotations.tsv")

    print(f"Updating valid annotations from: {valid_file}")
    updated_count, not_found_count = update_valid_annotations(valid_file, mapping, output_file)

    print(f"\nResults:")
    print(f"  Updated {updated_count} rows")
    print(f"  Not found: {not_found_count} rows")
    print(f"  Output written to: {output_file}")


def load_full_annotations(full_file):
    """Load full annotations and create a mapping from first_col to third_col."""
    mapping = {}
    with open(full_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                first_col = parts[0]
                third_col = parts[2]
                mapping[first_col] = third_col
    return mapping


def update_valid_annotations(valid_file, mapping, output_file):
    """Update valid annotations with correct third column values."""
    updated_count = 0
    not_found_count = 0

    with open(valid_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.rstrip('\n')
            if not line:  # Preserve empty lines
                f_out.write('\n')
                continue

            parts = line.split('\t')
            if len(parts) >= 3:
                first_col = parts[0]
                filename = parts[1]
                old_third_col = parts[2]

                if first_col in mapping:
                    new_third_col = mapping[first_col]
                    f_out.write(f'{first_col}\t{filename}\t{new_third_col}\n')
                    if old_third_col != new_third_col:
                        updated_count += 1
                else:
                    # Keep original if not found in mapping
                    f_out.write(line + '\n')
                    not_found_count += 1
                    print(f"Warning: First column value '{first_col}' not found in full-annotations.tsv", file=sys.stderr)
            else:
                # Keep line as-is if it doesn't have 3 columns
                f_out.write(line + '\n')

    return updated_count, not_found_count


if __name__ == '__main__':
    main()
