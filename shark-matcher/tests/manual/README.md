# Manual Tests

Tests here are **not** collected by the default suite. `tests/pytest.ini` sets
`testpaths = unit contracts`, so `python3 -m pytest` never picks these up. Run them by
hand when you need them.

## `trainer_parity.py`

Proves that a change to `src/learning/trainer/new_trainer.py` did not alter what
the training pipeline actually does.

### Workflow Around a Refactor

```bash
# 1. Record what the pipeline does today, BEFORE editing anything.
PARITY_STATE="$(mktemp)"
python3 tests/manual/trainer_parity.py --save "$PARITY_STATE"

# 2. Make the change.

# 3. Prove the behaviour is identical. Exits non-zero if it is not.
python3 tests/manual/trainer_parity.py --compare "$PARITY_STATE"
rm "$PARITY_STATE"
```

### Stored Baseline

A baseline of the current pipeline is kept in the data workspace beside the
fixtures (`constants.GOLDEN_PATH`). From the repository root:

```bash
python3 tests/manual/trainer_parity.py --compare \
    "$(python3 -c 'from src.core import constants; print(constants.GOLDEN_PATH)')/parity_baseline_v3.pt"
```

It must exit 0 unless a change is meant to alter training; re-baseline with
`--save` only after reviewing such a change.

### Self-Check

```bash
python3 tests/manual/trainer_parity.py --check
```

Runs the whole thing twice in one process and asserts the two runs are bitwise
identical. **Run this first if a `--compare` fails.** If `--check` also fails,
the harness has become non-deterministic and no comparison it produces means
anything — fix that before trusting any result.

### Scenarios

Ten scenarios, each fingerprinted at six stages. Each scenario exists to reach
code the others cannot:

| Scenario | Reaches |
| --- | --- |
| `triplet` / `arcface` | the two production paths — multi-view, P×K vs shuffle |
| `triplet_singleview` | the single-view branch of `metric_learning_collate` |
| `triplet_earlystop` | both `EarlyStopping` branches and the early-stop break path |
| `triplet_nometric` | the warn-and-fall-back paths when the checkpoint metric is absent |
| `build_triplet` / `build_arcface` | the real `build_model_and_strategy`, incl. label encoding and dynamic margins |
| `build_pk_mismatch` | the P×K batch-size warn-and-override branch |
| `build_no_pk` | the shuffle fallback when P/K are unset |
| `triplet_partial_accum` | a **trailing partial accumulation group** (4 batches at `accumulation_steps=3` -> groups `[3, 1]`) |

### Stages

The six stages, reported in order:

| Stage | Catches |
| --- | --- |
| `post_build` | a change in the number or order of RNG draws during model construction |
| `pipeline` | a change in sampling, augmentation, or collation |
| `losses` | a change in the forward pass or loss |
| `grad_norms` | a change in accumulation, clipping, or the optimiser step |
| `post_train` | any drift in the final weights |
| `probe_logits` | what the model would actually output in production |

- The report names the **first** stage that diverged.
- That ordering is the point: a divergence at `post_build` shifts the seed stream
  and makes every later stage differ for reasons unrelated to the loop, so it
  must be resolved first.
- Comparison is bitwise (`torch.equal`), not tolerance-based.

### Scope

- **Real:** `RobustTrainer.train` in full — gradient accumulation, grad-norm
  logging, clipping, the scaler, checkpoint selection, early stopping, epoch
  bookkeeping, summary writing — plus both strategies, `ArcFaceModelWrapper`,
  the real triplet loss, real ArcFace heads, `AugmentedDataset`, `PKSampler`,
  `metric_learning_collate`, `setup_augmentations`, the real `hits_at_k`
  evaluator, and the real `build_model_and_strategy`.
- **Pseudo:** the images (seeded RNG, never read from disk) and the backbone (a
  tiny CNN rather than MiewID/ViT, so a run takes seconds and needs no network).
- **Not covered:** `run_single_config` (filesystem layout, resume,
  notifications), the LoRA branch of `build_model_and_strategy` (`peft` targets
  module names the stub backbone lacks), and real backbone construction. Changes
  there need separate verification.

### Coverage

Measured line coverage of `new_trainer.py` is **~62%**; most of the remainder is
`main`/argparse, `print_config_summary`, `perform_dry_run_checks` and
`run_single_config`. Re-measure with:

```bash
COVERAGE_STATE="$(mktemp)"
python3 -m coverage run --branch --source=src.learning.trainer.new_trainer \
    tests/manual/trainer_parity.py --save "$COVERAGE_STATE"
python3 -m coverage report -m
rm "$COVERAGE_STATE"
```

### Caveats

> [!NOTE]
> **Dead code.** Genuinely dead code is not executed here, so a clean `--compare`
> after deleting it is *expected* and proves nothing about whether it was truly
> unreachable. Justify those deletions structurally (as with `flatten_batch`,
> where `metric_learning_collate` always returns a stacked 4-D tensor), not by
> parity alone.

> [!NOTE]
> **Why `triplet_partial_accum` exists.**
>
> - The original nine ran 4 or 6 batches at `accumulation_steps=2`, so none had a partial final group.
> - The branch that flushes when the loader runs out was unguarded, and B4 (the under-weighted trailing group) was invisible to `--compare`.
> - Measure batch counts before trusting a green run to cover an accumulation change.

### Determinism

Pinned to CPU, `num_workers=0`, no AMP, explicit DataLoader generators,
`torch.set_num_threads(1)`, and the seed re-applied before each scenario.
A baseline is device-specific: MPS repeats its own results exactly but differs
numerically from CPU, so compare MPS runs only against an MPS baseline.

## `embedding_identity.py`

Diagnostic (not a test, and not collected — it needs real trained models and
real images, neither of which a checkout has). It answers two questions:

```bash
# Do two models produce the same embeddings for the same images?
python3 tests/manual/embedding_identity.py \
    --use-stored --models model_a model_b --images AN00XXXX01.jpeg

# Stored set named differently from the model (as with --embeddings-name)?
python3 tests/manual/embedding_identity.py \
    --use-stored --models model_a --embeddings-name my-set --images AN00XXXX01.jpeg

# Drift check: do the embeddings on disk still match what today's code makes?
python3 tests/manual/embedding_identity.py \
    --compare-generated-to-stored --models model_a --images AN00XXXX01.jpeg

# Preprocessing only, to tell "they resize differently" apart from
# "they embed differently".
python3 tests/manual/embedding_identity.py \
    --compare-images --models model_a model_b --images AN00XXXX01.jpeg
```

"Not identical" is information, not failure — different models are supposed to
differ. It exits non-zero only when it could not perform the comparison
(missing image, model that will not load, image absent from stored
embeddings).

> It delegates to `EMBEDDING_BATCH_SIZE` and `_compute_embeddings` rather than
> embedding images itself, so it always measures the production path. Keep it delegating.

## `determinism_checks.py`

Acceptance checks for the determinism toggles in `src/core/constants.py`
(`SHARK_DETERMINISTIC_DATALOADER`, `SHARK_STRICT_DETERMINISM`,
`SHARK_RESEED_BEFORE_TRAIN`) and for MPS run-to-run reproducibility, which is
skipped without MPS. It exits non-zero if any check fails:

```bash
python3 tests/manual/determinism_checks.py
```

## `convergence_check.py`

Checks that the trainer overfits a small synthetic batch, a different guarantee
from parity, which only proves training did not change.

- It needs no data workspace.
- It writes a scratch `temp_convergence_check/` directory in the current directory
  and removes it on success.

```bash
python3 tests/manual/convergence_check.py
```

## Test Data

- Automated tests use the small fixtures committed under `tests/fixtures/` and
  need no data workspace.
- Manual checks needing real images use the sample dataset,
  `pre2019-sample-data`, in the data workspace.
- The parity baseline lives at `constants.GOLDEN_PATH`.
