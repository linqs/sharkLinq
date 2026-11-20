# Thesis Evaluation Sweeps

The `experimental/thesis` folder holds the scripts that ran the thesis evaluation sweeps over stored embedding sets; they are outside the supported workflow.

## Modules

| File | Responsibility |
| --- | --- |
| `thesis_eval.py` | Runs the baseline evaluation, plus optional PSL, multimodal, stacked, and k-reciprocal overlays, under three scenarios: standard, minimum quality `poor`, and minimum 5 instances per individual |
| `thesis_eval_bucketed.py` | Runs the same variants over query buckets by images per individual, image quality, and media type |

## Entry Points

Run from the repository root. Both scripts use argparse, so `--help` is safe.

```bash
python3 -m experimental.thesis.thesis_eval --help
python3 -m experimental.thesis.thesis_eval_bucketed --help
```

- `--embeddings-name` takes one or more embedding sets under the data root's `embeddings/` folder.
- `--centroid` switches the base retrieval method from image-image to leave-one-out centroid.
- `thesis_eval_bucketed.py` adds bucket selectors (`--four-quality-buckets`, `--media-type-only`, `--media-type-buckets`, `--stacked-bucketing`) and `--output-dir`.

## Outputs

Each run writes one text file per embedding set. The file name carries the active overlay flags and a timestamp.

- `thesis_eval.py` writes to `THESIS_RESULTS_DIR` in `src/core/constants.py`.
- `thesis_eval_bucketed.py` writes there too unless `--output-dir` names another folder.

## Boundaries and Dependencies

- Both scripts call `run_single_evaluation` in `src/retrieval/evaluation.py` and capture its printed report.
- Embedding sets must already hold the artifacts each overlay needs; generate them with `src.retrieval.embeddings` first.
- Nothing under `src/` imports this folder, and it is excluded from the installed package.
- The root `experimental/README.md` describes the research code as a whole.
