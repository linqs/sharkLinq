# Multimodal Late-Fusion Pipeline

Score-level combination of frozen image embeddings and engineered metadata for
individual re-identification. The image model remains frozen; the pipeline
optimizes fusion weights and generates a separate multimodal retrieval view.

## Integrated Workflow

The supported generation path is the existing embeddings pipeline. It reuses
the shared dataset and image embeddings, resolves the current metadata feature
specification, optimizes fusion weights, creates image and centroid results,
and publishes them through the same atomic generation mechanism as the base
artifacts.

```bash
python3 -m src.retrieval.embeddings \
    --model-name TRAINED_MODEL_NAME \
    --annotations-version ANNOTATIONS_VERSION \
    --use-multimodal

python3 -m src.retrieval.evaluation \
    --embeddings-name EMBEDDINGS_NAME \
    --use-multimodal
```

The current generation order, optimization policy, and artifact names belong
to [`src/retrieval/embeddings.py`](../retrieval/embeddings.py). The active
metadata features belong to [`multimodal_config.py`](multimodal_config.py),
while [`metadata_features.py`](metadata_features.py) owns the representations,
normalization statistics, and saved-spec resolution. They are intentionally
not copied into this guide.

### Stacked Multimodal and PSL

The integrated pipeline can combine multimodal fusion with PSL reranking. The
stacked mode enables multimodal generation, learns PSL weights from the base
neighbors, and reuses those weights for the base and multimodal reranking
passes.

```bash
JAVA_HOME="..." python3 -m src.retrieval.embeddings \
    --model-name TRAINED_MODEL_NAME \
    --annotations-version ANNOTATIONS_VERSION \
    --stacked
```

Use `python3 -m src.retrieval.embeddings --help` and
`python3 -m src.retrieval.evaluation --help` for the maintained flags and
output behavior.

## Code Owners

The repA, repB, and repC experiments recorded in
`experimental/multimodal/README.md` are research evidence; their optimizer
flags are not the current orchestrator interface. For current behavior:

- [`multimodal_config.py`](multimodal_config.py) owns the current default
  feature specification.
- [`metadata_features.py`](metadata_features.py) owns feature specifications,
  fitted statistics, encoding, normalization, and feature-group boundaries.
- [`fusion_dataset.py`](fusion_dataset.py) composes frozen image embeddings
  with features produced by the shared metadata encoder.
- [`optimize_late_fusion.py`](optimize_late_fusion.py) owns fusion-weight
  random search and coordinate refinement.
- [`src/retrieval/embeddings.py`](../retrieval/embeddings.py) composes these
  for integrated generation and owns the data flow and artifact set.

The post-hoc fusion utility, trained-head baselines, and the historical
benchmark record live in `experimental/multimodal/`.
