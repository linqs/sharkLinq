# Data Scripts

The `src/data_scripts` package holds the supported dataset utilities: one analyzes prepared annotations, the other converts a biology-format CSV into database-format CSV files.

## Modules

| File | Responsibility |
| --- | --- |
| `analyze_data.py` | Reports image counts, per-individual and per-year distributions, and data-integrity checks for an annotations version; `--plot` adds figures |
| `biology_to_db_format.py` | Converts a biology-format annotations CSV into a database-format directory (`individuals.csv`, `media.csv`, `alternateNames.csv`, `tags.csv`, and empty placeholder tables) |

## Entry Points

Run from the repository root. Both scripts use argparse, and both also run by path.

```bash
python3 -m src.data_scripts.analyze_data --help
python3 -m src.data_scripts.biology_to_db_format --help
```

- `analyze_data` echoes its output to a timestamped `analysis.txt` under the data root's `data-out/data-analysis/` folder.
- `biology_to_db_format` writes into a new directory under the data root's `database-format/` folder by default; `--output-root` and `--output-name` change the destination.
- The root `README.md` walkthrough step "Validate Data (Optional)" shows a worked example.

## Boundaries and Dependencies

- Paths come from `src/core/constants.py`; `analyze_data` reads per-image quality through `src/csv_store`.
- `analyze_data` needs the source CSV directory, given by `--csv-dir` or the `.source_csv_dir` file that `src.cli.convert_data` writes beside the annotations.
- `src/cli/run_pipeline.py` runs `analyze_data.py` as its validation step.
- One-off research scripts live in `experimental/data_scripts`, outside the supported workflow.
- Tests live in `tests/unit/data_scripts`.
