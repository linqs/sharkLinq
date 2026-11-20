# PSL Research

> **Status:** Research notes; not part of the supported Shark Matcher workflow.

Research utilities for PSL reranking. The supported, integrated workflow,
rules, and engine are documented in `src/psl/README.md` and implemented in
`src/psl/`.

## Post-Hoc Reranking Utility

Existing embeddings can be reranked without regenerating the image embeddings:

```bash
python3 -m experimental.psl.psl_rerank \
    --embeddings-name <embeddings-name>
```

[`psl_rerank.py`](psl_rerank.py) learns weights and writes PSL neighbors into
the selected existing embeddings directory. Unlike the integrated generation
path, this convenience utility updates that directory in place and does not
publish a complete generation through the atomic staging workflow. Prefer the
integrated path for a new release artifact.

## Research Records

- `docs/psl-rule-experimentation.md` is a historical record
  of rule-design experiments, not a live defaults reference.
