# Unit Tests

The `tests/unit` folder holds the deterministic unit tests for `src/`; its subfolders mirror the `src/` packages.

## Modules

| Folder | Covers | Test files |
| --- | --- | --- |
| `api/` | `src/api` | `test_server.py`: endpoints, authentication, generation locking, error responses |
| `cli/` | `src/cli` | `test_convert_data.py`: split strategies and the source-CSV sidecar; `test_run_pipeline.py`: child-command construction |
| `csv_store/` | `src/csv_store` | `test_snapshot_csv_resolution.py`: snapshot CSV directory resolution |
| `data_scripts/` | `src/data_scripts` | `test_analyze_data.py`, `test_biology_to_db_format.py` |
| `learning/` | `src/learning`, `src/config`, `src/models` | Arcface heads, augmentations, config validation and image size, data loader determinism, early stopping, evaluator, gradients and LoRA freezing, image consistency, triplet loss, trainer, PK sampler, strategies |
| `models/` | `src/models` | `test_miewid_backbone.py` |
| `multimodal/` | `src/multimodal` | `test_fusion_dataset.py`, `test_metadata_features.py` |
| `retrieval/` | `src/retrieval` | Embedding generation and publication, `Retriever`, fine evaluation, evaluation guards and embedding-set resolution, inference snapshots |

### Shared Fixtures

`conftest.py` provides three fixtures that point the CSV store at the committed directories in `tests/fixtures/`: `csv_convert_data`, `csv_fine_evaluation_full`, and `csv_fine_evaluation_partial`. Each restores the previous data directory and clears the cache afterward.

## Entry Points

Run from the repository root with the repository runner, which resolves paths relative to `tests/`.

```bash
python3 -m tests.tools.run_tests --run unit
python3 -m tests.tools.run_tests --run unit/learning/test_sampler.py
python3 -m tests.tools.run_tests --help
```

The standard run (`python3 -m tests.tools.run_tests`) covers this folder and `tests/contracts`.

## Boundaries and Dependencies

- Tests use only committed fixtures and test-local data; they need no `sm-data` directory, trained model, or database.
- Tests for `src/config` and most of `src/models` sit under `learning/`.
- `src/psl`, `src/visualization`, and `src/core` have no dedicated test folder here.
