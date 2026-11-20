# Command-Line Steps

The `src/cli` package holds the command-line steps that prepare data and drive the pipeline, plus a TensorBoard launcher.

## Modules

| File | Responsibility |
| --- | --- |
| `convert_images.py` | Recursively reformats raw images into JPEG files with stripped ICC profiles under the data root's `images/` directory |
| `convert_data.py` | Converts a database-format CSV directory into versioned annotation TSV files with a train/test split, and records the source CSV directory beside them |
| `run_pipeline.py` | Runs the six pipeline steps in order by launching the other commands as child processes |
| `run_tensorboard.py` | Starts TensorBoard on the logs of one or more trained models |

## Entry Points

Run every command from the repository root. Use `--help` on each for the current options.

```bash
python3 -m src.cli.convert_images --help
python3 -m src.cli.convert_data --help
python3 -m src.cli.run_pipeline --help
python3 -m src.cli.run_tensorboard --help
```

### Pipeline Steps

`run_pipeline` runs these steps in order; `--skip-train` skips step 4.

1. Format images (`src.cli.convert_images`).
2. Prepare annotations (`src.cli.convert_data`).
3. Validate data (`src/data_scripts/analyze_data.py`, run by path).
4. Train (`src.learning.trainer.new_trainer`).
5. Generate embeddings (`src.retrieval.embeddings`); `--stacked` adds multimodal and PSL artifacts and needs the PSL setup in `src/psl/README.md`.
6. Evaluate embeddings (`src.retrieval.evaluation`).

## Boundaries and Dependencies

- Data locations come from `src/core/constants.py`, and relative inputs are resolved with `src/core/utils.py`.
- `convert_data.py` reads records through `src/csv_store` and is also imported by `src/retrieval/embeddings.py`.
- `run_pipeline.py` and `run_tensorboard.py` add the repository root to `sys.path` through `src/core/path_utils.py`.
- Completion email recipients for `run_pipeline` come from `SHARK_NOTIFICATION_EMAILS`; `--no-email` suppresses them.
