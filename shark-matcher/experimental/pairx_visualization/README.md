# PAIR-X Visualization for Shark-Matcher

> **Status:** Research notes; not part of the supported Shark Matcher workflow.

These research scripts use
[PAIR-X](https://github.com/pairx-explains/pairx) to explain pairwise visual
similarity between two shark images. They extract intermediate feature maps,
backpropagate similarity relevance, match spatial descriptors, and render the
most relevant corresponding regions across the image pair.

## Entry Point

[`pairx_vis.py`](pairx_vis.py) is the current entry point. It loads trained
shark-matcher models through the shared model loader, uses
`IndividualDataset` for the corresponding preprocessing, can list model-layer
keys, and passes selected layers to PAIR-X.

## External Dependency

PAIR-X is an optional external repository, not vendored into shark-matcher.
Follow its upstream installation guidance for its own Python dependencies. The
shark-matcher dependency declarations and package metadata remain authoritative
for dependencies owned by this repository.

By default, [`src/core/constants.py`](../../src/core/constants.py) discovers a `pairx`
checkout beside `shark-matcher`. A checkout elsewhere can be selected with the
portable `PAIRX_ROOT` environment variable; no machine-specific absolute path
is required in source or documentation.

```bash
# Run from the folder that holds shark-matcher, so pairx becomes its sibling.
git clone https://github.com/pairx-explains/pairx.git pairx
```

## Workflow

Run commands from the **`shark-matcher/` repository root**.

### List Layers

Inspect the available layer keys for the selected model:

```bash
python3 -m experimental.pairx_visualization.pairx_vis \
    --model-name <trained-model-name> \
    --list-layers
```

### Generate an Explanation

Generate an explanation for two dataset images and one or more layer keys:

```bash
python3 -m experimental.pairx_visualization.pairx_vis \
    --model-name <trained-model-name> \
    --visualize <layer-key> \
    --image-filenames <first-image> <second-image> \
    --output-directory <run-name>
```

### Defaults and Options

- When no shark-matcher model is supplied, the script asks the installed PAIR-X
  package for its configured default model.
- Model resolution, preprocessing, output paths, similarity handling, and
  rendering options are owned by [`pairx_vis.py`](pairx_vis.py) and
  [`src/core/constants.py`](../../src/core/constants.py).
- The supported similarity behavior and handling of an unsupported metric name
  are also owned by `pairx_vis.py`; this guide is not a second metric or default
  registry.

Use the command help for current options:

```bash
python3 -m experimental.pairx_visualization.pairx_vis --help
```

## Choosing Layers

Later layers generally expose model features closer to the final similarity
decision, while earlier layers show lower-level structure. Layer names depend
on the loaded model, so use `--list-layers` rather than relying on layer numbers
recorded for a different model revision. Trying several layers and comparing
their explanations remains a research judgment, not a stable configuration
constant.
