# Training Configuration

The `src/config` package holds the registry of named training configurations and the schema that validates them.

## Modules

| File | Responsibility |
| --- | --- |
| `config.py` | The `CONFIGS` registry of named training configurations, augmentation lists, the projection-head constructor table, the early-stopping default, and `NOTIFICATION_EMAILS` |
| `config_schema.py` | Pydantic models that validate a configuration: `FullConfig` composes `TrainingConfig`, `ModelConfig`, `LoRAConfig`, and `DataConfig` |

## Entry Points

None. The trainer reads this package; a configuration key is passed to it with `--configs`.

List the current keys without running training:

```bash
python3 -c "import src.config.config as c; print(*c.CONFIGS)"
```

## Adding or Changing a Configuration

1. Add or edit an entry in `CONFIGS` in `config.py`; every value must satisfy `FullConfig`.
2. Validate it with `python3 -m src.learning.trainer.trainer --configs CONFIG_KEY --dry-run`, which checks the configuration and data paths without training.
3. Keep names and values in `config.py` only; other documents refer to the registry instead of copying it.

## Boundaries and Dependencies

- `config.py` imports the evaluator and triplet loss from `src/learning`, and the heads, augmentations, and backbones from `src/models`.
- `config_schema.py` has no matcher imports; `src/models/vit_backbone.py` and `src/core/utils.py` import it.
- `NOTIFICATION_EMAILS` is read from the comma-separated `SHARK_NOTIFICATION_EMAILS` variable; an empty list disables notifications.
- `NUM_WORKERS` is read from `SHARK_NUM_WORKERS`; the root `README.md` prerequisites say when to lower it.
