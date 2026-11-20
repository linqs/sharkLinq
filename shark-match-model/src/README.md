# Source Package

The `src` package holds the Shark Matcher implementation: configuration, training, embedding generation, retrieval, and the inference API.

## Modules

| Folder or file | Responsibility | README |
| --- | --- | --- |
| `__init__.py` | Package marker; defines `__version__` | none |
| `individual_dataset.py` | `IndividualDataset`, the PyTorch dataset that loads images, IDs, and labels from annotation TSVs | none |
| `core/` | Constants, data-root paths, and shared helpers | `src/core/README.md` |
| `cli/` | Command-line steps: format images, prepare annotations, run the pipeline, launch TensorBoard | `src/cli/README.md` |
| `config/` | Training configuration registry and its validation schema | `src/config/README.md` |
| `api/` | FastAPI inference service | `src/api/README.md` |
| `retrieval/` | Embedding generation, generation storage, neighbors, evaluation, suggestions | `src/retrieval/README.md` |
| `models/` | Backbones, projection and ArcFace heads, augmentations | `src/models/README.md` |
| `csv_store/` | CSV-backed individual and media records | `src/csv_store/README.md` |
| `contracts/` | Strict version-1 data models for snapshots, generations, and model status | `src/contracts/README.md` |
| `learning/` | Trainer, loss, evaluator, sampler, augmented dataset | `src/learning/README.md` |
| `data_scripts/` | Dataset analysis and conversion scripts | `src/data_scripts/README.md` |
| `multimodal/` | Metadata late fusion for the multimodal and stacked modes | `src/multimodal/README.md` |
| `psl/` | Probabilistic Soft Logic reranking | `src/psl/README.md` |
| `visualization/` | Embedding and pipeline inspection tools | `src/visualization/README.md` |

## Running Code

Run every command from the `shark-match-model/` repository root with `python3 -m src.PACKAGE.MODULE`. The entry points of each package are listed in that package's README.

## Boundaries

- Nothing here imports `shark-matcher-app` or a database driver.
- Data lives outside the repository in the `sm-data` directory (or the `SHARK_DATA_ROOT` location); `src/core/constants.py` owns every path into it.
