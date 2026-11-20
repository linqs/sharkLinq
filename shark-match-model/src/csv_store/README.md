# CSV Store

The `src/csv_store` package serves individual and media records from `individuals.csv` and `media.csv`, so the matcher needs no database.

## Modules

| File | Responsibility |
| --- | --- |
| `_loader.py` | Reads and caches both CSV files on first use; `set_db_dir()` redirects the source directory and `clear_cache()` discards the cache |
| `models.py` | The `Individual` and `Media` record classes |
| `individual.py` | Individual queries: `get_individuals`, `get_matching_individual_ids`, `is_media_individual`, `is_source_individual` |
| `media.py` | Media queries: `get_medias`, `get_matching_media_ids`, `get_individual_ids_by_media_ids`, `load_medias_with_individuals` |

## Entry Points

None. This package is imported, never run.

## Boundaries and Dependencies

- The root `README.md` section "CSV Store" owns the design rationale, cache behavior, and snapshot handoff.
- `_loader.py` skips rows with too few columns: individuals need 7 and media need 10.
- Without `set_db_dir()`, the loader reads from `DB_DATA_DIR` in `src/core/constants.py`.
- Relative directories are resolved with `resolve_relative_csv_dir` in `src/core/utils.py`.
- Query functions return copies, so callers cannot change the cached records.
- Importers are `src/cli`, `src/data_scripts`, `src/retrieval`, `src/psl`, and `src/multimodal`.
- Code that switches datasets in one process reuses `set_db_dir()` and `clear_cache()` instead of adding another cache.
- `tests/parity` compares this package with the SQLite database.
