# Experimental Research Code

> **Status:** Research notes; not part of the supported Shark Matcher workflow.

This directory holds research code that the supported Shark Matcher workflow
does not use.

- Nothing under `src/` imports it.
- It is excluded from the installed package.
- The default test, lint, and CI gates skip it.

## Contents

| Directory | Contents |
| --- | --- |
| `multimodal/` | Post-hoc late-fusion utility, trained-head fusion baselines, and benchmark records |
| `psl/` | Post-hoc PSL reranking utility and rule-design records |
| `pairx_visualization/` | PAIR-X explanation visualizations |
| `data_scripts/` | One-off dataset comparison, export, and label-mapping scripts |
| `thesis/` (`thesis_eval.py`, `thesis_eval_bucketed.py`) | Thesis evaluation sweeps |
| `data_scripts/build_dolphin_dataset.py` | Dolphin dataset construction |

## Running Research Code

Research code depends on `src`, never the reverse. Run from the repository root:

- Modules: `python3 -m experimental.thesis.thesis_eval --help`.
- Data scripts, by path: `python3 experimental/data_scripts/diff_datasets.py --help`.

## Research Tests

The research tests run only on request:

```bash
python3 -m tests.tools.run_tests --run tests/experimental
```
