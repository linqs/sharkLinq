# Core

The `src/core` package holds the constants, data-root paths, and shared helpers that the rest of the matcher imports.

## Modules

| File | Responsibility |
| --- | --- |
| `constants.py` | Scalar constants, default model list, determinism toggles, and every path into the data root |
| `path_utils.py` | Adds the repository root and the optional sibling `pairx` checkout to `sys.path` |
| `utils.py` | Shared helpers, grouped below |

### Helper Groups in `utils.py`

| Group | Examples |
| --- | --- |
| Path and CSV-directory resolution | `resolve_directory_path`, `resolve_csv_dir`, `portable_csv_dir` |
| Model and embedding loading | `load_model`, `load_embedding_model`, `load_model_for_resume`, `resolve_model_dirs`, `resolve_embedding_dirs` |
| Annotation and media metadata | `get_annotation_paths`, `parse_media_name`, `get_key_metadata` |
| Pickle compatibility | `PICKLE_MODULE_ALIASES`, which maps earlier module names in stored pickles to current modules |
| Reproducibility | `seed_everything` |
| Notifications | `send_email`, `notify_completion` |
| Console output | `print_script_banner`, `print_table` |
| File input and output | `load_csv_file`, `write_json_file`, `load_psl_file` |

## Entry Points

None. This package is imported, never run.

## Environment Settings

`constants.py` reads these variables once at import time.

| Variable | Effect |
| --- | --- |
| `SHARK_DATA_ROOT` | Absolute data-root location; defaults to the sibling `sm-data` directory |
| `SHARK_INFERENCE_SNAPSHOT_ROOT` | Root for app-owned inference snapshots; defaults to `inference-snapshots/` under the data root |
| `PAIRX_ROOT` | Location of the optional `pairx` checkout; defaults to a sibling `pairx` directory |
| `SHARK_INFERENCE_API_KEY` | Shared Bearer key read by the inference service |
| `SHARK_INFERENCE_HOST`, `SHARK_INFERENCE_PORT` | Bind address of the inference service, resolved by `resolve_inference_bind_address` |
| `SHARK_DETERMINISTIC_DATALOADER`, `SHARK_STRICT_DETERMINISM`, `SHARK_RESEED_BEFORE_TRAIN` | Opt-in determinism toggles, all off by default |

`send_email` in `utils.py` reads `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, and
`GMAIL_REFRESH_TOKEN` when it sends. All three are needed before
`SHARK_NOTIFICATION_EMAILS` can deliver mail.

## Boundaries and Dependencies

- `constants.py` is the only authority for data-root paths; other code imports them instead of rebuilding them.
- `utils.py` imports `src/config/config_schema.py` and the backbones in `src/models/` at load time, and `src/config/config.py` lazily when it reads notification recipients.
- `src/config`, `src/models`, `src/learning`, `src/retrieval`, `src/csv_store`, `src/cli`, and `src/api` all import from here.
- `utils.py` carries heavy third-party imports (torch, transformers, peft, timm, Google API clients), so importing it is not cheap.
