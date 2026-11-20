# Trained-Fusion Research Baselines

> **Status:** Research notes; not part of the supported Shark Matcher workflow.

This directory contains the trained-head baselines used for multimodal
ablation experiments. They are separate from the integrated late-fusion path:
they reuse `FusionDataset` and the shared metadata feature implementation but
train their own fusion model and publish no application-facing embedding
generation.

## Historical Rationale

The recorded experiments compared a direct concatenation head with a residual
head that begins at the image-only representation. The concatenation approach
historically overfit and underperformed the image-only baseline; the residual
approach improved on that baseline but did not match the calibrated late-fusion
results. These observations explain why both remain as research ablations and
why neither replaced the integrated pipeline.

The exact registered methods, model dimensions, optimizer settings, epoch
counts, and stopping policy are maintained in [`configs.py`](configs.py).
This guide does not duplicate them as current defaults.

## Workflow

Training, generation, and evaluation remain separate commands:

```bash
python3 -m experimental.multimodal.trained_fusion.train_multimodal \
    --image-embeddings-name <image-embeddings-name> \
    --configs <config-key> \
    --model-name <fusion-run-name>

python3 -m experimental.multimodal.trained_fusion.generate_fused_embeddings \
    --fusion-model-name <fusion-run-name>/<config-key>

python3 -m src.retrieval.evaluation \
    --embeddings-name <fusion-run-name>/<config-key>
```

Use the commands' `--help` output and the `TRAINED_FUSION_CONFIGS` registry in
`configs.py` for current options, evaluation modes, and valid configuration
keys. The training and generation CLIs expose the maintained compatibility
choice for reproducing an older whole-corpus metadata-statistics fit.

## Implementation Owners

- [`configs.py`](configs.py) owns the trained-fusion registry and training
  configuration.
- [`fusion_model.py`](fusion_model.py) owns the supported fusion-head
  implementations.
- [`train_multimodal.py`](train_multimodal.py) owns dataset setup and training.
- [`generate_fused_embeddings.py`](generate_fused_embeddings.py) loads a saved
  run and generates its retrieval artifacts.
- [`multimodal_evaluator.py`](multimodal_evaluator.py) adapts the shared
  evaluator for these models.
- [`src/multimodal/fusion_dataset.py`](../../../src/multimodal/fusion_dataset.py) and
  [`src/multimodal/metadata_features.py`](../../../src/multimodal/metadata_features.py) remain the shared data
  and feature implementations; this directory does not maintain copies.

## Historical Compute Note

On the thesis dataset used for the recorded ablation, the original
configuration took roughly 5–15 minutes per head for a 50-epoch MPS run over
4,193 samples. This is preserved as historical experiment context, not a
runtime or resource guarantee for the current registry, hardware, or dataset.
