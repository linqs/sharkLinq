# Probabilistic Soft Logic (PSL) Reranking Pipeline

Post-hoc reranking of neural nearest-neighbor results using biological
constraints formulated as a Hinge-Loss Markov Random Field (HL-MRF). PSL uses
precomputed image neighbors and their associated metadata; it does not retrain
the image model.

## Prerequisites

Install the repository's PSL dependency group and a supported Java runtime.
The Python dependency constraints and the special `pslpython` installation
note are maintained in [`requirements/requirements-psl.txt`](../../requirements/requirements-psl.txt),
not duplicated here.

`JAVA_HOME` must be configured before PSL imports initialize the JVM. Use the
JDK discovery mechanism appropriate for your platform and verify the current
dependency guidance in the requirement file; no machine-specific JDK path is
stored in this repository.

## Integrated Workflow

The supported generation path is the existing embeddings pipeline. It creates
the base embeddings and neighbors, learns enabled biological-rule weights from
the training split, applies PSL reranking, and publishes the complete result
through the same atomic generation mechanism as the other embedding artifacts.

```bash
python3 -m src.retrieval.embeddings \
    --model-name TRAINED_MODEL_NAME \
    --annotations-version ANNOTATIONS_VERSION \
    --use-psl

python3 -m src.retrieval.evaluation \
    --embeddings-name EMBEDDINGS_NAME \
    --use-psl
```

Use `python3 -m src.retrieval.embeddings --help` and
`python3 -m src.retrieval.evaluation --help` for current arguments. The
generation order and artifact set are authoritative in
[`src/retrieval/embeddings.py`](../retrieval/embeddings.py),
[`src/retrieval/neighbors.py`](../retrieval/neighbors.py), and the shared
generation store.

## Rules and Metadata

- [`psl_rules.py`](psl_rules.py) is the single source of truth for active rules,
  rule templates, starting or fixed weights, predicate encodings, and length
  thresholds.
- Neural-prior and fixed biological rules stay outside MLE weight learning;
  enabled learnable biological rules are calibrated from the training split.
- Unknown metadata is encoded so unsupported biological constraints stay silent
  rather than imposing a match penalty.
- [`psl_engine.py`](psl_engine.py) owns CSV metadata loading, neural-distance
  conversion, inference, and neighbor reordering.

This guide does not copy the metadata dictionary, rule inventory, numeric
thresholds, or top-neighbor defaults.

### One-Off Rule Learning

For a one-off rule-learning experiment, use the existing training CLI:

```bash
python3 -m src.psl.train_psl \
    --embeddings-name EMBEDDINGS_NAME
```

The CLI can select an approved subset of rules for an ablation. Consult
`python3 -m src.psl.train_psl --help` and the registries in
`psl_rules.py` for current names and behavior rather than editing a copied
configuration from this guide.

## How It Works

1. `load_metadata_from_csvs()` resolves per-image biological metadata from the
   CSV snapshot pinned to the stored dataset.
2. `learn_weights()` builds training-split truth labels and calibrates the
   enabled learnable rules while preserving fixed-rule weights.
3. `PSLReranker` converts neural distances into PSL observations, performs
   inference over the selected neighbor set, and returns a reordered result.
4. The integrated pipeline stages the learned weights and reranked neighbors
   with the other selected embedding artifacts before publication.

## Implementation Owners

- [`psl_rules.py`](psl_rules.py) owns predicates, rule configuration,
  encodings, and weight persistence.
- [`psl_engine.py`](psl_engine.py) owns metadata loading and inference.
- [`train_psl.py`](train_psl.py) owns weight learning and its research CLI.

Run each entry point with `--help` for its maintained command interface.
