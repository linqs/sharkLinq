# Visualization

Tools for inspecting the shark-matcher feature space, embedding quality, and
training-data transformations. Run commands from the **`shark-matcher/`
repository root**.

The command-line interfaces and output names are owned by the scripts in this
directory. Use each command's `--help` output for its current arguments rather
than treating this guide as a second option reference.

## Scripts

### `visualization.py` — Embedding Dimensionality Reduction

Loads cached image or centroid embeddings and generates 3D PCA, t-SNE, and NCA
views. Points are colored by individual label, and low-frequency individuals
can be filtered before plotting. Plots are interactive unless an output
directory is requested.

```bash
python3 -m src.visualization.visualization \
    --model-name EMBEDDINGS_NAME

python3 -m src.visualization.visualization \
    --model-name EMBEDDINGS_NAME --output-dir OUTPUT_DIRECTORY
```

See [`visualization.py`](visualization.py) for the artifact-loading and plot
generation behavior.

### `visualize_model.py` — Model Feature-Space Analysis

Generates a two-dimensional embedding projection and positive/negative pair
distance distributions from an existing embeddings set. For ArcFace
configurations it also attempts angular analysis by loading the corresponding
trained model; a model-load failure is reported without hiding the earlier
plots.

```bash
python3 -m src.visualization.visualize_model \
    --model-name EMBEDDINGS_NAME
```

See [`visualize_model.py`](visualize_model.py) for the current analysis steps,
artifact requirements, and output handling.

### `visualize_pipeline.py` — Training-Data Transformations

Loads a trained model configuration and writes combined and isolated
augmentation views using the configured transformations, falling back to the
shared augmentation registry when the model has no saved augmentation list.

Batch visualization is **not currently supported**. The old
`PositiveMatchSampler` path was removed, and `visualize_batch()` deliberately
raises `NotImplementedError` until it is migrated to the shared
[`PKSampler`](../learning/pk_sampler.py). Consequently, the command may write
the augmentation figures before stopping at the batch step; it should not be
treated as a successful end-to-end batch-visualization workflow.

```bash
python3 -m src.visualization.visualize_pipeline \
    --model-name TRAINED_MODEL_NAME
```

The limitation and current execution order are authoritative in
[`visualize_pipeline.py`](visualize_pipeline.py).

### `visualize_augmentations.py` — Individual Augmentations

Applies each registered augmentation to one image and saves examples beneath
the configured visualization output root. This utility reads the shared
augmentation registries rather than maintaining a local list.

```bash
python3 -m src.visualization.visualize_augmentations IMAGE_FILENAME
```

See [`visualize_augmentations.py`](visualize_augmentations.py) and
[`src/config/config.py`](../config/config.py) for current behavior and augmentation
configuration.

## Dependencies and Data

Visualization dependencies are maintained in the repository requirement files
and package metadata, particularly
[`requirements/requirements-visualization.txt`](../../requirements/requirements-visualization.txt). The
scripts reuse the paths defined in [`src/core/constants.py`](../core/constants.py).

The embedding-analysis commands expect artifacts produced by
[`src.retrieval.embeddings`](../retrieval/embeddings.py). The pipeline and
individual-augmentation commands use the configured images, annotations, and
trained-model configuration. Refer to the implementations above for the exact
artifacts each command reads; those names are intentionally not duplicated
here.

## Detailed Script Help

```bash
python3 -m src.visualization.visualization --help
python3 -m src.visualization.visualize_model --help
python3 -m src.visualization.visualize_pipeline --help
python3 -m src.visualization.visualize_augmentations --help
```
