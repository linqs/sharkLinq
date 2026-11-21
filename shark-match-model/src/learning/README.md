# Learning

The `src/learning` package trains re-identification models: the trainer, its loss functions, the evaluator that scores checkpoints, and the dataset and sampler that feed batches.

## Modules

| File | Responsibility |
| --- | --- |
| `augmented_dataset.py` | `AugmentedDataset`: wraps a base dataset to produce augmented views, with retry and fallback to the original image |
| `pk_sampler.py` | `PKSampler`: yields batches of P identities with K images each, oversampling small identities |
| `trainer/trainer.py` | `RobustTrainer`, the training strategies, early stopping, per-configuration orchestration, and the training command |
| `loss/triplet.py` | `create_triplet_loss`: builds a triplet loss with `batch_hard` or `semi_hard` mining |
| `evaluator/evaluators.py` | `compute_embeddings` and `hits_at_k`, which score a checkpoint on the validation split |

### Subfolders

| Folder | Holds |
| --- | --- |
| `trainer/` | The training loop, strategies (`ContrastiveStrategy`, `SupervisedStrategy`), `ArcFaceModelWrapper`, and the `main()` entry point |
| `loss/` | Loss construction and its helpers |
| `evaluator/` | Retrieval metrics used for checkpoint selection and early stopping |

## Entry Points

Run from the repository root; `--help` lists every option.

```bash
python3 -m src.learning.trainer.trainer --help
```

A typical run names an output model, an annotations version, and one or more configuration keys from `src/config/config.py`. Add `--dry-run` to validate the setup without training.

```bash
python3 -m src.learning.trainer.trainer \
    --model-name MODEL_NAME \
    --annotations-version ANNOTATIONS_VERSION \
    --configs CONFIG_KEY
```

Checkpoints, the saved configuration, and training logs go to `trained-models/MODEL_NAME/CONFIG_KEY` in the data root. `src.cli.run_tensorboard` reads those logs.

## Boundaries and Dependencies

- Configurations come from `src/config/config.py` and are validated by `src/config/config_schema.py`; the trainer does not define its own.
- Backbones, heads, and augmentations come from `src/models`.
- `hits_at_k` and `create_triplet_loss` are referenced from the configuration registry, so renaming them breaks configurations.
- Datasets come from `src/individual_dataset.py`; the trainer also uses `src/core/constants.py` and `src/core/utils.py`.
- `src/retrieval/evaluation.py` imports `K_THRESHOLDS` from the evaluator, `src/api/server.py` and `src/retrieval/retrieval.py` import `AugmentedDataset`, and `src/visualization` imports `AugmentedDataset` too.
- `evaluators.py` has an empty `main()` placeholder; it is not an entry point.
- Completion email recipients come from `SHARK_NOTIFICATION_EMAILS`; `--no-email` suppresses them.
- Tests live in `tests/unit/learning`; `tests/manual` holds trainer parity and determinism checks.
