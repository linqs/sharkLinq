# Experimental Data Scripts

The `experimental/data_scripts` folder holds one-off dataset comparison, export, and label-mapping scripts that are outside the supported workflow.

## Modules

| File | Responsibility |
| --- | --- |
| `diff_datasets.py` | Compares two database-format snapshots (size, properties, identity matching) and optionally their annotation splits; writes `diff.txt` into a timestamped folder under `DATA_COMPARISONS_DIR` in `src/core/constants.py` |
| `export_unknown_quality.py` | Lists annotation rows whose media quality is unknown or missing and copies those images into a sibling folder |
| `build_dolphin_dataset.py` | Extracts dolphin names from image filenames in a directory and writes them to a CSV file in the data root's `data-proc/` folder |
| `map_annotation_labels.py` | Copies the third column of a full annotations TSV onto another annotations TSV by matching first-column IDs |

## Entry Points

Run from the repository root. The first three scripts use argparse, so `--help` is safe.

```bash
python3 -m experimental.data_scripts.diff_datasets --help
python3 -m experimental.data_scripts.export_unknown_quality --help
python3 -m experimental.data_scripts.build_dolphin_dataset --help
```

`map_annotation_labels.py` has no `--help`. It takes up to three positional paths (full annotations, annotations to update, output file) and otherwise falls back to fixed defaults beside the script, so read its `main()` before running it.

```bash
python3 experimental/data_scripts/map_annotation_labels.py FULL_TSV TARGET_TSV OUTPUT_TSV
```

## Boundaries and Dependencies

- The scripts depend on `src/core/constants.py` for data paths; the first three also use `src/csv_store` or `src/core/utils.py`.
- Nothing under `src/` imports this folder, and it is excluded from the installed package.
- `export_unknown_quality.py` copies image files and `map_annotation_labels.py` writes a new TSV, so check the destination first.
- The supported dataset utilities live in `src/data_scripts`.
