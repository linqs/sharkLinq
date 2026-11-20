# Repo cleanup — running issue list

> **Status:** Internal working log of the repository cleanup, kept for its
> history; not setup or usage documentation. Current instructions live in
> `README.md` and `tests/manual/README.md`.

Working list of everything identified, so items can be taken one at a time
without losing any.

- Add to it as new things surface.
- Mark items Done rather than deleting them, so the history stays readable.

**Status key:** Done · Open · Needs your decision · Held, revisit later

## Progress at a glance

| § | Area | Done | Open | Decide | Held |
| --- | --- | :---: | :---: | :---: | :---: |
| 1 | Trainer — dead code and duplication | **8 / 8** | 0 | 0 | 0 |
| 2 | Trainer — behaviour questions | **7 / 7** | 0 | 0 | 0 |
| 3 | Determinism gaps | **4 / 4** | 0 | 0 | 0 |
| 4 | Test suite and lint health | **7 / 7** | 0 | 0 | 0 |
| 5 | Packaging and environment | **4 / 4** | 0 | 0 | 0 |
| 6 | Data layout | **3 / 3** | 0 | 0 | 0 |
| 7 | Removed — restore if wanted | **7 / 7** | 0 | 0 | 0 |
| | **Total** | **40 / 40** | **0** | **0** | **0** |

### Current State

- **Suite: 192 passed, 0 failed, 0 errors** (was 121 / 39 / 19).
- `tests/src/learning`: 51 passed.
- Parity: exit 0 against `parity_baseline_v3.pt`.
- Determinism: 4/4.
- Not collected, run by hand: `test_convergence.py` passes; `embedding_identity.py` verified on real models.

**Nothing open.**

- B4 was the one deliberate behaviour change; the baseline moved to `parity_baseline_v3.pt`.
- The remaining gap is coverage, not a task: nothing in `tests/src` exercises `run_pipeline`'s staging/atomicity (see §4 caveats).

---

## How to verify a change

**Rule for anything under §1–§3:** it touches the training pipeline, so verify
it with the parity harness.

- The current baseline is `parity_baseline_v3.pt` (10 scenarios).
- It supersedes `v2`, which is kept for cross-reference (see B4).

### Running the Harness

Run it from the parent of the repo, not from the repo root: `sm-data/` is a
sibling of `shark-matcher/`, so the path below does not resolve from inside the
repo. Starting from the `shark-matcher/` root:

```bash
cd ..
python3 shark-matcher/tests/manual/trainer_parity.py \
    --compare sm-data/test-data/model-tests/golden_states/parity_baseline_v3.pt
```

- It must exit 0. See `tests/manual/README.md`.
- Items marked **[behaviour]** will intentionally fail. Decide whether the change is wanted, then re-baseline with `--save`.

### Guard Status

Measured with `coverage`, not assumed:

- **GUARDED** — the harness executes that code *and* a deliberate regression
  there was shown to be caught.
- **UNGUARDED** — parity will pass whatever you do. Review by reading, not by
  running.

> **Dead-code caveat.** Genuinely dead code is by definition not executed, so a
> clean `--compare` after deleting it is *expected* and proves nothing. Justify
> those deletions structurally.

---

## 1. Trainer — dead code and duplication complete

`new_trainer.py` net **−65 lines** (+52 / −117). Parity clean on all 9 scenarios
× 6 stages; pylint 10.00/10; suite unchanged. Committed in `cf56d7b`.

| | ID | Guard | Item and what changed | Lines |
| :---: | --- | --- | --- | ---: |
| Done | T1 | UNGUARDED<br>(dead by design) | **`flatten_batch` was almost entirely unreachable.** `metric_learning_collate` always returns `torch.stack(...)`, i.e. a 4-D tensor, so the list-of-lists, flat-list and `dim()==3` branches could never be taken.<br>**Done:** deleted the function; its one call site (the step-0 TensorBoard grid) now slices the batch directly. Also dropped collate's `isinstance(labels, list)` branch, which `AugmentedDataset` can never trigger. Equivalence checked explicitly for batches smaller than, equal to and larger than the 8-image grid. | **−26**<br><sub>+8 / −34</sub> |
| Done | T2 | GUARDED | **`ArcFaceModelWrapper` had an empty `else:` branch** — four lines of stale self-questioning comments over a bare `pass`.<br>**Done:** removed, and replaced with a statement of the actual contract: heads return either `(loss, logits)` or bare logits. | **−7**<br><sub>+4 / −11</sub> |
| Done | T3 | GUARDED | **`SupervisedStrategy.__init__` was empty** — a comment and `pass`.<br>**Done:** dropped; the explanation moved to the class docstring. | **0**<br><sub>+4 / −4</sub> |
| Done | T4 | GUARDED | **`get_validation_metric` was byte-identical in both strategies.**<br>**Done:** hoisted into `TrainingStrategy` (no longer abstract); both copies deleted. | **−8**<br><sub>+7 / −15</sub> |
| Done | T5 | GUARDED | **`AugmentedDataset` was constructed twice with identical arguments** in the two branches of `build_model_and_strategy`.<br>**Done:** extracted `_build_train_dataset`. Deliberately a helper, **not** a hoist — see caveat below. | **+1**<br><sub>+14 / −13</sub> |
| Done | T6 | n/a | **Duplicate `import utils`** at lines 29 and 35.<br>**Done:** removed. | **−1** |
| Done | T7 | GUARDED | **The early-stop break path duplicated the epoch bookkeeping** (avg loss, initial/final loss, timing, CSV, save `last`) — a prime spot for the two paths to drift.<br>**Done:** early stopping now sets a flag and falls through to the shared bookkeeping, so there is one code path. Also removed an unreachable `if initial_loss is None` guard inside the old break path. | **−5**<br><sub>+8 / −13</sub> |
| Done | T8 | GUARDED | **`CHECKPOINT_METRIC` and `EARLY_STOPPING_METRIC` were the same constant**, under a 20-line comment block explaining how to set them independently.<br>**Done:** collapsed to one; early stopping reads `CHECKPOINT_METRIC` directly, exactly as before. See caveat below. | **−19**<br><sub>+7 / −26</sub> |

### Caveats carried forward from §1

- **T5 used a helper, not a hoist.** In the ArcFace branch the dataset is built
  *after* label encoding rewrites `train_base_ds.labels`, and
  `AugmentedDataset.__init__` reads those labels when `rebalance_augmentations`
  is on. Lifting the call above the branch would have changed behaviour for that
  combination, so `_build_train_dataset` is called from both original positions.
- **T8 collapsed rather than split.** The documented independence never existed
  at the time. It was subsequently implemented for real in **B7**, which added a
  second constant plus per-config overrides.
- **T7 was verified beyond parity.** Parity does not fingerprint
  `training_summary.json`, so the early-stop run's summary, CSV row count and
  checkpoint set were diffed across the change: byte-identical.
- **One test was corrected.** `test_metric_learning_collate_multi_view`
  simulated `([img1, img2], [lbl, lbl])`. `ContrastiveLearningIndividualDataset`
  is now an alias for `AugmentedDataset` (`src/retrieval/retrieval.py:47`), whose
  `__getitem__` returns a single label, so nothing can emit a per-view label
  list. The test now asserts the shape actually produced, plus label values.

---

## 2. Trainer — behaviour questions

Six of seven addressed, all **parity-clean** — every new knob defaults to the
existing behaviour, so no re-baseline was needed. Committed in `8ef1b5e`.

| | ID | Item and what changed | Lines |
| :---: | --- | --- | ---: |
| Done | B1 | **Loss logging was asymmetric between strategies** — `ContrastiveStrategy` divided by batch size, `SupervisedStrategy` did not.<br>**Done:** added `TrainingStrategy._base_logs`, which both strategies now call, with a `LOSS_IS_SUM_REDUCED` class flag encoding the real difference (the triplet loss sums over anchors; `CrossEntropyLoss` mean-reduces). `logs["loss"]` is now a documented per-sample mean for every strategy, and the duplicated `Data/UniqueClasses` computation is gone.<br>**Correction:** the original finding overstated this. Measured before changing anything: triplet `sum/N` and CE `mean` were *already* the same quantity, so the asymmetry was structural, not semantic. **Logged values are unchanged** — verified by diffing `training_summary.json` losses across the change (arcface `[19.7, 17.63]`, triplet `[0.531, 0.519]` both sides). Historical comparability is therefore preserved. | **+20**<br><sub>+30 / −10</sub> |
| Done | B2 | **`print_config_summary` ran before the P×K batch-size override**, so the summary reported a batch size the run never used.<br>**Done:** extracted `resolve_pk_batch_size(config)` and call it in `run_single_config` *before* the summary. It is idempotent and still called from `build_model_and_strategy`, so callers that bypass `run_single_config` (including the parity harness) are unaffected. Verified: with `batch_size=7` and P×K=4, the summary now prints 4. | **+17**<br><sub>+28 / −11</sub> |
| Done | B3 | **Gradient clipping was hardcoded to 1.0.**<br>**Done:** now `config.training.grad_clip_norm`, default `1.0`. Deliberately still clips *all* parameters, not just trainable ones, as requested — only the threshold moved to config. Verified: post-clip norms track the configured value (`1.0 -> 1.0`, `0.25 -> 0.25`). | **+6**<br><sub>+7 / −1</sub> |
| Done | B4 **[behaviour]** | **Last partial accumulation group was under-weighted.** The loop flushes on `(step+1) == len(train_loader)`. When the final group held `k < accumulation_steps` batches, the gradient was still divided by `accumulation_steps`, so that step contributed `k/accumulation_steps` of a normal update — the last slice of *every* epoch was systematically under-weighted.<br>**Done:** the loss is now divided by the size of the group the batch actually belongs to (`group_size`), which equals `accumulation_steps` for every full group and `k` for a trailing partial one. **This is the only item in this file that deliberately changes training behaviour**, and the only one that required a re-baseline.<br>**First finding: the harness could not see this at all.** Measured before changing anything — all nine scenarios ran 4 or 6 batches at `accumulation_steps=2`, so every group divided evenly and no partial group existed anywhere. A clean `--compare` after the fix would have been an artifact, exactly like the D3 blind spot. Added scenario **`triplet_partial_accum`** (4 batches, `accumulation_steps=3` -> groups `[3, 1]`) to close it. Verified as a control that adding it left all nine originals bit-identical to `v2`.<br>**Isolation evidence** — see the caveat below. | +18 |
| Done | B5 | **`writer.add_scalars` opened a nested `SummaryWriter` per tag**, leaking file handles.<br>**Done:** replaced with a loop of `add_scalar(f"Validation/Metrics/{key}", ...)`. Verified concretely: the old code created **12 nested writer subdirectories** per run, the new code creates **zero**. | **+3**<br><sub>+4 / −1</sub> |
| Done | B6 | **`send_completion_notice` fired a real email on every run**, hardcoded to one address.<br>**Done:** recipients now live in `src.config.NOTIFICATION_EMAILS` (a list, so several people can subscribe), read through a shared `utils.notify_completion` used by **both** the trainer and `run_pipeline.py`. `--no-email` suppresses a single run and is passed through to the training subprocess, so a pipeline run sends at most one notice rather than one per config. An empty list disables notifications globally, and a failing recipient is warned about rather than taking down a run whose model is already saved. All paths verified. | **+58**<br><sub>+61 / −3</sub> |
| Done | B7 | **Checkpoint and early-stopping metrics were the same hardwired constant.**<br>**Done:** `constants.py` now defines **two** real constants, `CHECKPOINT_METRIC` and `EARLY_STOPPING_METRIC`, each with its own documented purpose, and `config.training.checkpoint_metric` / `early_stopping_metric` override either per config. Both constants default to `average_valid_hit_at_10`, so out of the box behaviour is unchanged. Verified at all three levels: defaulted, split at the constants level with no config change, and per-config override winning over the constant. The trainer announces the split at startup. | **+33**<br><sub>+38 / −5</sub> |

**Net +111 lines** across `new_trainer.py`, `config_schema.py` and `config.py`.

### How B4 was isolated

The requirement was to show that *only* the intended change altered behaviour.

### Evidence

Four independent pieces, all measured:

1. **Nine of ten scenarios are bit-identical** across the fix, all six stages OK.
   They are exactly the scenarios with no trailing partial group, so the change
   is provably inert where it should be.
2. **In `triplet_partial_accum`, `post_build` and `pipeline` stay clean.**
   - No change to model construction, RNG draw count, sampling, augmentation or collation.
   - Divergence begins at `losses`: the change is confined to the gradient/optimiser path.
3. **Every epoch-1 loss is bit-identical.**
   - The first difference appears at epoch 2 step 0, precisely after the first partial flush applied a differently-weighted update.
   - A forward-pass or data change would have diverged at step 0.
4. **Grad norms separate full from partial flushes.**
   - Full groups: ratio **1.000** (unchanged).
   - Partial groups: **0.818 -> 1.000** and **0.870 -> 1.000**.
   - The under-weighted gradients had sat *below* the 1.0 clipping threshold and were never clipped; correctly weighted, they reach it. This is the under-weighting made visible.

### Re-Baseline

- Re-baselined to `parity_baseline_v3.pt` (10 scenarios), verified stable with a follow-up `--compare` at exit 0.
- `v2` is retained for cross-reference.
- The nine shared scenarios are identical between the two files, so the only difference in `v3` is the new scenario plus the corrected partial-group weighting.

### Permanent Unit Guard

Three tests in `tests/src/learning/test_new_trainer.py` observe the divisor the
real loop applies (by wrapping the strategy and `scaler.scale`) rather than
reimplementing the arithmetic.

- `[3, 3, 3, 1]` for the partial case, plus controls for the evenly-divisible and `accumulation_steps=1` cases.
- Confirmed they *catch* the bug: reverting the fix fails only the partial-group assertion (`3.0 != 1`) and leaves both controls green.

### Not Measured

What this does to real training outcomes. The fix makes the last step of each
epoch count for more, which is correct.

- Whether it moves final accuracy on a real run is a separate question.
- Worth watching on the next full training run, especially where `len(train_loader) % accumulation_steps` is small relative to `accumulation_steps` (the under-weighting was largest there).

### Notes from §2

- **Nothing here required a re-baseline.** Every added knob defaults to the
  current behaviour, so `--compare` stayed green on all 9 scenarios × 6 stages.
  B4 is the only §2 item that would break parity, and it is held.
- **B5 changes TensorBoard tag layout.** Old runs wrote tag
  `Validation/Metrics` into 12 per-metric subdirectories; new runs write 12
  distinct tags `Validation/Metrics/<name>` into the main event file. Historical
  TB runs are still readable but will not line up side-by-side with new ones.
- **Notifications are now shared.** `utils.notify_completion` and
  `utils.get_notification_recipients` are the single implementation; the trainer
  and the pipeline both call them, so there is one subscriber list and one
  opt-out. `src.config` is imported lazily inside those helpers so `utils` does
  not take a hard dependency on the training config (which pulls in torch).
- **Pipeline notices are deduplicated.** `run_pipeline.py --no-email` passes the
  flag to the training subprocess and sends a single `pipeline`-prefixed notice
  at the end, instead of one email per trained config.

---

## 3. Determinism gaps complete

Not bugs today, but they set how much the parity harness can guarantee, and they
are what makes a refactor risky.

### Toggles

**Each item ships behind an OFF-by-default toggle in `constants.py`**, so runs
stay bit-identical to the baseline of their time (`parity_baseline_v2.pt`, now
superseded by `v3`).

- Every toggle also honours an environment variable, so a setting can be A/B'd for one run without editing the file.
- Acceptance tests live in `tests/manual/determinism_checks.py`.
- A toggle is only meaningful if the OFF row fails the check and the ON row passes.

Agreed order: **D2 -> D1 -> D3**, one at a time.

| | ID | Item |
| :---: | --- | --- |
| Done | D1 | **`utils.seed_everything` (`utils.py:1335`) does not pin everything.** No `PYTHONHASHSEED`, no `torch.use_deterministic_algorithms`, no `cudnn.deterministic`. The parity harness sets these itself; production runs do not. |
| Done | D2 | **DataLoaders got no `generator=` and no `worker_init_fn`**, and `NUM_WORKERS` defaults to 4 (`src/config.py:83`). Worker seeding derived from the global RNG, so it was reproducible *only* while the number of RNG draws before iteration was unchanged.<br>**Done:** `constants.DETERMINISTIC_DATALOADER` (env `SHARK_DETERMINISTIC_DATALOADER`), default OFF, applied through `_build_dataloader_kwargs` (`new_trainer.py:203`). Committed in `b19dc3c`.<br>**Turning it ON legitimately shifts the RNG stream** — see the note below on why it shifts even at `num_workers=0`. |
| Done | D3 | **The seed was applied once**, in `run_single_config`, before dataset construction, LoRA wrapping and head init — all of which consume RNG. Any change that adds or reorders a draw during construction shifted every augmentation downstream: the run diverged completely while being functionally identical.<br>**Done:** `constants.RESEED_BEFORE_TRAIN` (env `SHARK_RESEED_BEFORE_TRAIN`), default OFF. When ON, `RobustTrainer.train()` re-seeds at entry. Weights still depend on construction; the data stream depends only on `RANDOM_SEED`. Pure RNG-state reset — no control flow changes.<br>**Verified:** all-OFF -> baseline exit 0. ON -> `--check` passes, alone and with D1+D2. Acceptance test reproduces the production ordering and shows OFF is *not* invariant to a construction change while ON *is*.<br>**Warning: The parity harness is blind to this toggle** — see note below. | +47 |
| Done | D4 | **Filed as "MPS is not bitwise reproducible, parity must stay on CPU". Measured, and that was wrong.** MPS reproduces itself exactly: identical probe logits across repeated training runs, both in production configuration (autocast bfloat16) and with AMP off, and identical across separate processes for matmul, reductions and the `cdist` CPU fallback.<br>What is true is that **MPS and CPU differ numerically** (max abs 8.2e-07 on a probe) because they run different kernels — so a baseline is device-specific and an MPS run must be compared against an MPS baseline. That is a labelling requirement, not a barrier.<br>**Real finding:** `PYTORCH_ENABLE_MPS_FALLBACK` must be set *before* `torch` is imported — PyTorch reads it while registering the fallback. `new_trainer.py` gets this right; `trainer_parity.py` and `determinism_checks.py` did not, and would have died with `NotImplementedError: aten::_cdist_backward` on MPS. Both fixed. Any new entry point that imports torch before `new_trainer` inherits this trap.<br>**Verified** by `check_d4_mps_reproducibility` in `determinism_checks.py`. | +64 |

---

### Notes from §3

- **D4 reversed the original finding.**
  - It was filed as documentation-only on the assumption that MPS reductions are unordered.
  - Testing showed MPS is reproducible run-to-run and across processes, so parity work is *not* confined to CPU; it is confined to one device per baseline.
  - The genuine MPS hazard is the fallback env var's import-order sensitivity, which is a crash rather than a numerical issue.
- **`trainer_parity.py` cannot see D3.**
  - `run_scenario` re-seeds immediately before calling `train()`, so the harness already has D3's semantics baked in and `--compare` reports no change with the toggle either way. That `exit 0` is an artifact, not evidence.
  - `run_single_config` seeds once *before* dataset and model construction, so D3 **does** shift the stream in production, measured directly against a production-ordered flow.
  - Use `determinism_checks.py`, which reproduces that ordering, for anything involving D3.
  - A comment at the re-seed site records this so the result is not misread later.
- **Turning D2 ON legitimately shifts the RNG stream**, so `--compare` against the committed baseline fails with the toggle ON.
  - That is expected, not a regression.
  - The divergence is confined to the four `build_*` scenarios (the only ones routed through `_build_dataloader_kwargs`) and starts at `pipeline` with `post_build` clean: the signature of a data-stream change rather than a model-construction one.
  - Running with any determinism toggle ON needs its own baseline.
- **D1's honest scope.**
  - `warn_only=True` is deliberate: a hard error would abort a long run on the first op without a deterministic kernel, whereas a warning names it and lets the run finish.
  - The scan found none on the CPU path, so on this machine D1 buys no observable change. Its payoff is on CUDA, where kernel selection and cuDNN autotuning are genuine nondeterminism sources.
  - `PYTHONHASHSEED` cannot affect the *current* interpreter (CPython reads it at startup); it is set so spawned DataLoader workers inherit it, and `seed_everything` prints a note if the parent was not launched with it.
- **Why D2 shifts even at `num_workers=0`:** with no generator, DataLoader's base-seed draw consumes from the *global* RNG.
  - Supplying a generator moves that draw elsewhere, so every later main-process augmentation shifts by one draw.
  - The toggle therefore changes behaviour at any worker count, not only with workers.

## 4. Test suite and lint health complete

Was **121 passed, 39 failed, 19 errors**. Now **179 passed, 0 failed, 0 errors**.
The `tests/src/learning` subset stayed at 48 passed throughout, and parity
stayed at exit 0 after every item.

Everything here was **stale tests lagging behind source refactors**, confirmed
case by case, not inferred from error messages. **No source bug was found.**

- In each case the source change was located first and shown to be deliberate (commit message, updated docstring, or an explicit rationale comment) before any assertion was touched.
- Where a removed behaviour still had a live counterpart, the test was retargeted rather than deleted.
- Where the behaviour no longer exists anywhere, the test was dropped and the reason recorded inline.

| | ID | Item and what changed |
| :---: | --- | --- |
| Done | S1 | **`test_embeddings.py` — 29 failures -> 0.** Seven distinct causes. 6 failures were just a missing `import utils`. The focus/`train_test` concept (`_compute_focus_labels`, `_compute_affected_labels(train_test=)`, `UpdateResult.old_focus_labels`, `update_centroid_*(train_test=)`) was removed in `179e46d`, "Removing legacy train-test embeddings infrastructure" — the incremental path runs `prepare_annotations(train_test=False)`, so there was only ever one focus; split-focus tests dropped. `_compute_centroids` now L2-renormalizes (documented rationale: unnormalized centroids bias distances toward high-spread individuals); test now asserts the normalized values plus unit norm. `store_embeddings` split into `store_image_embeddings` / `store_centroid_embeddings`; `store_transformed_embeddings` (NCA) removed in `5396b31` with no successor writer, so its test was dropped. `update_embeddings` no longer stages internally — see caveat. |
| Done | S2 | **`test_server.py` — 19 errors -> 0.** All 19 were one autouse fixture clearing `server._generating`, so the file failed at setup and masked three further drifts. The per-model `_generating` set became a global `threading.Lock`; the endpoint is synchronous, returns `{"status": "ok"}` and 503 rather than `"started"` / `"already_running"`. The guard is deliberately global (shared retriever state + on-disk files), so `test_different_model_not_blocked_by_running_model` was **inverted** rather than deleted. The lock-release-on-failure invariant — what the old cleanup tests protected — was kept and retargeted. Masked drifts: `/health` also reports loaded/failed models; `/match` returns `{ids, distances, labels}` and no longer 404s on empty (docstring: always 200, because the client treats non-200 as transport failure). |
| Done | S3 | **`test_retrieval.py` — 4 failures -> 0.** Retrieval method 2 changed from `get_centroid_neighbors` (dict lookup) to `get_loo_centroid_neighbors`, which recomputes the query individual's centroid with the query removed. Tests rewritten against that contract, including one pinning the leave-one-out property itself (0.51764, not the 0.26105 a query-inclusive centroid gives) and one showing precomputed centroid files are no longer needed. Fixture embeddings changed from zeros to unit vectors so the math is actually exercised. |
| Done | S4 | **`test_fine_evaluation.py` — 3 failures -> 0.** `_get_filtered_test_instances` now pins csv_store to the dataset's `csv_dir` so evaluation cannot read CSVs that disagree with the embeddings. A bare `MagicMock` retriever yielded a Mock path. Fixed by giving the mock the fixture's real directory. Also found and fixed a **test-isolation bug** — see caveat. |
| Done | S5 | **`test_fusion_dataset.py` 2 + `test_multimodal_config.py` 1 -> 0.** `FUSION_CONFIGS` was deleted in `baf01fa`; the guarantee moved to `trained_fusion.configs.TRAINED_FUSION_CONFIGS`, which `run_single_config` indexes, so the test was repointed. `FusionDataset` gained `fit_split`, defaulting to `'train'` — a deliberate leakage fix; the test asserted the whole-corpus fit. Now asserts the train-only default plus a case for the opt-in legacy `fit_split='all'`. `get_medias` is now reached via `load_medias_with_individuals`, which forwards ids by keyword. |
| Done | S6 | **CI was failing; it should now pass.** The workflow runs `python -m tests.run_tests --run tests/src`, which exits non-zero on any failure — so with 39 failed / 19 errors it was red, not skipped. That command now exits 0 with 179 passed. Verified in a clone with **no sibling `sm-data/`** (CI checks out only the repo): still 179 passed, so the suite has no `sm-data` dependency. `pip install -r requirements.txt` also resolves cleanly on 3.12 (dry run exit 0, `JPype1 1.7.1`, no `pslpython`), confirming E2's fix holds. **Not verified** — see caveats. |
| Done | S7 | **Lint noise cleared.** Removed unused `List`/`Union` from `src/config_schema.py` and put the stdlib import first (7.06 -> 7.41). `src/config.py` deliberately did **not** get `new_trainer.py`'s file-level `import-error` disable: the 7 `E0401`s were an invocation artifact, not a real path problem — `import src.config` works at runtime and pointing pylint at the repo root clears all 7 (0.00 -> 5.93). Added a minimal `.pylintrc` with `source-roots=.` so `src.*` resolves from any entry point. Path configuration only, no rule changes. |

### Caveats carried forward from §4

- **`update_embeddings` no longer stages internally, and these tests no longer
  cover atomicity.**
  - It used to `mkdtemp` / `shutil.move` / `rmtree`.
  - `run_pipeline` now wraps the *whole* pipeline (embeddings, centroids, neighbors) in one outer staging directory and swaps at the end.
  - Atomicity therefore still exists and covers more than before, but it lives in `run_pipeline` and **nothing in `tests/src` exercises it**.
  - Verified that `run_pipeline` really does create and swap that directory rather than assuming it. Worth a test.
- **A test-isolation bug was fixed in `tests/src/conftest.py`.**
  `_loader._custom_db_dir` is a module global that takes precedence over
  `constants.DB_DATA_DIR`, and the csv fixture never reset it. Once any test
  triggered csv-store pinning, that directory leaked into every test that ran
  afterwards. The fixture now saves and restores it, and yields its path.
- **`new_trainer.py` now reports `C0302` (1113 lines > 1000).** Not a
  regression — with imports resolving, pylint simply sees more. Left alone as
  out of scope; the file-level disable there was not touched.
- **S6 is a local determination, not a CI observation.**
  - The `gh` CLI is not installed and there is no GitHub access from here, so the **actual Actions run status was never read**; the conclusion is inferred from running the workflow's own command locally.
  - The dependency dry run resolved **macOS arm64** wheels, not `ubuntu-latest` x86_64.
  - The workflow still pins `actions/checkout@v2` and `actions/setup-python@v2`, both long deprecated, which can fail a run for reasons unrelated to the tests. Bumping those is an open question, not done.

---

## 5. Packaging and environment

| | ID | Item |
| :---: | --- | --- |
| Done | E1 | **`httpx` added to `requirements.txt`.** `starlette.testclient` needs it; without it `test_server.py` failed at *collection*, masking the whole suite. |
| Done | E2 | **PSL install fixed.** `pslpython==2.4.0` pins `JPype1==1.4.0` exactly, which has no Python 3.12 wheel and cannot compile (touches `_ts.frame`, removed in CPython 3.11) — so `pip install -r requirements.txt` failed outright. The pin is stale: pslpython runs fine on `JPype1>=1.5`. `requirements.txt` now pins JPype1 and documents `pip install --no-deps pslpython==2.4.0`. Fresh-env resolution verified. |
| Done | E3 | **`requirements.txt` is now authoritative; `setup.py` derives from it.** `install_requires` was a hand-maintained set of 9 unpinned packages, missing fastapi, uvicorn, pydantic, httpx, tensorboard, pandas, opencv-python, sentence-transformers, matplotlib and the google-auth stack — so `pip install -e .` (how `shark-matcher-app` consumes this repo) produced a materially different environment from `pip install -r requirements.txt`. `setup.py` now parses `requirements.txt`, skipping comments and blanks, so the two cannot drift again. Verified: 25 requirements parsed, all valid PEP 508 specifiers, `python setup.py --version` exits 0. |
| Done | E4 | **Decided: stays optional.** **`pslpython` is no longer installed by `requirements.txt`** (it needs the documented `--no-deps` step). Acceptable since PSL is experiments-only, but if it should be part of the default install, this needs a different approach. Confirmed as the intended state: PSL is experiments-only (`src/experiments/psl/`), nothing in `tests/src` imports it, so CI is unaffected. Note `setup.py` now derives from `requirements.txt` (E3), so pslpython is absent from `pip install -e .` too — consistent by construction. |

---

## 6. Data layout

| | ID | Item |
| :---: | --- | --- |
| Done | L1 | **Symlinked rather than removed.** `sm-data/images/test-data/` now contains `images -> ../images` (relative), so `SHARK_TEST_MODE=true` resolves `IMAGES_DIR` to the real corpus instead of a nonexistent path, and `new_trainer.main()`'s startup check passes. Verified: all **21/21** fixture images resolve through `IMAGES_DIR` under `SHARK_TEST_MODE`, and `ANNOTATIONS_DIR` resolves too. Symlinked in preference to removing because removing leaves the mode equally broken, just less visibly. **`TEST_IMAGES_DIR` was deliberately left pointing at the real directory**, not through the symlink: those fixture constants are not gated on `SHARK_TEST_MODE` and should not depend on a path that exists only to serve it. **Caveat: `sm-data` is not version-controlled**, so this symlink is part of the local data layout, not something a checkout provides — recreation command recorded in `tests/manual/README.md`. Also noted: `sm-data/trained-models/test-data/` is missing entirely, so `SHARK_TEST_MODE` is still only partly provisioned; it is not needed for the startup check. |
| Done | L2 | **`golden_states/` repurposed** as the parity-baseline location; now holds `parity_baseline_v2.pt` and the current `parity_baseline_v3.pt`. `constants.GOLDEN_PATH` points at it and has a consumer again, though the harness takes the path as an argument rather than reading the constant. |
| Done | L3 | **`src/learning/trainer/lora_trainer.py` deletion committed** in `7d894f8`. |

---

## 7. Removed — restore if wanted

Deleted during this cleanup. All recoverable from git.

| | ID | Item |
| :---: | --- | --- |
| Done | R1 | **Restored.** Verifies the trainer actually *converges* on synthetic data — a different guarantee from parity, which only proves training didn't change. Passes: exit 0, final training accuracy 100% ("model overfit the batch"), ~20s. Lives in `tests/manual/`, so it is **not** in the 189 suite count; run it by hand.<br>**It needed a bootstrap to run at all.** It imports `torch` at the top and `new_trainer` below, so it inherited the D4 trap: `PYTORCH_ENABLE_MPS_FALLBACK` must be set *before* torch is imported or it dies with `NotImplementedError: aten::_cdist_backward` on MPS. It also had no `sys.path` insert, so `src.*` did not resolve when run as a script. Both added, matching `trainer_parity.py`. |
| Done | R2 | **Restored as `tests/manual/embedding_identity.py` after a full audit — not restored as-was.** Renamed: it has no assertions and is a diagnostic, so the `test_` prefix was misleading and invited collection.<br>**It was only partly functional.** Audited part by part against current APIs:<br>• **The stored-embeddings path was dead** — it read `retriever.embeddings` / `.filenames`, neither of which exists; a `hasattr` guard turned that into a silent "could not load" instead of an error. Rewritten against `Retriever._get_model_data`.<br>• **It reimplemented the embedding pipeline** (own batch loop, dim-squeezing, L2 normalize, own batch-size rule) and that rule had **drifted from production**: it used 16 only for fine-tuned models and `mega`, while `_get_batch_size` uses 16 for everything but `miew`. For `vit` it measured batch size 1 against production's 16 — so it was not measuring production, the one thing it exists for. Now delegates to `_get_batch_size` and `_compute_embeddings`.<br>• **It wrote scratch into the live `EMBEDDINGS_DIR`** and rmtree'd it; now `tempfile.TemporaryDirectory`.<br>• **A missing image was a warning**, and the unresolved name was appended anyway, silently misaligning every later index. Now fatal.<br>• **No exit code**, so nothing could act on it. Now returns 1 on operational failure.<br>• Inherited the same **D4 / `sys.path` bootstrap** gap as R1; both added.<br>**New capability:** `--compare-generated-to-stored`, the actual drift check — does what is on disk still match what today's code generates?<br>**Verified on real data**, all four paths: missing image -> exit 1; `--compare-images` (miew 440² vs vit 384², shape mismatch reported rather than silently skipped); `--use-stored` pairwise on two real models (max abs 3.9e-02, "different"); absent-from-stored -> exit 1; and the drift check on a real LoRA/ArcFace model — stored vs regenerated agree to **1.1e-07**, i.e. no drift. pylint 10.00/10. |
| Done | R3 | **Restored, and it was not dead** — `fusion_model.py` was *moved* to `src/experiments/multimodal/trained_fusion/`, not deleted, and is imported by `generate_fused_embeddings.py` and `train_multimodal.py`.<br>**Correction: the one-line import fix this file predicted was not enough.** With the import fixed, 16 of 18 tests still failed. The module had also been narrowed to the two methods that survive in `TRAINED_FUSION_CONFIGS` — `concatenate` and `residual` — and `num_heads` was dropped. The old file exercised `gate`, `cross-attention`, `temp_scaling` and `attention_pooling`, none of which exist. Rewritten against the current contract: 10 passed / 12 subtests, including the residual model's zero-init identity property (an untrained residual model reproduces the image-only baseline exactly) and a check that every `fusion_method` in the registry constructs. |
| Done | R4 | `verify_trainer_parity.py` — imported the deleted `lora_trainer`; did not even import. |
| Done | R5 | `test_training_regression.py` — both scenarios hardcoded `"run": False`; compared against a `shark-matcher-old/` checkout that does not exist. |
| Done | R6 | `generate_golden_logits.py` — cross-repo `sys.path` surgery to load a trainer from another checkout; obsolete for same-repo comparison. |
| Done | R7 | `compare_pipeline_parity.py` — compared the old vs new dataset pipeline; the "old" side (`new_trainer_lg.py`) is gone. |

---

## Infrastructure built along the way

Not issues — the tooling that makes the rest of this list safe to work through.

- **Environment.** pyenv virtualenv `shark` (3.12.13), set as the local default
  for the workspace folder via `.python-version`. Both repos' requirements installed.
- **Test fixture constants** in `constants.py`
  (`TEST_ANNOTATIONS_DIR`, `TEST_TRAIN_ANNOTATIONS`, `TEST_VALID_ANNOTATIONS`,
  `TEST_FULL_ANNOTATIONS`, `TEST_IMAGES_DIR`). Deliberately not gated on
  `SHARK_TEST_MODE` so a test's inputs don't change with an env var. All 21
  referenced images verified present.
- **Parity harness** `tests/manual/trainer_parity.py` + `README.md`. Validated
  three ways: `--check` bitwise identical; an injected regression
  (`clip_grad_norm_` 1.0 -> 0.9) caught and localised to `grad_norms` with exit 1;
  clean `--compare` exit 0 after revert.
- **Harness coverage audit and expansion.** Measured against `new_trainer.py`
  with `coverage`.
  - The original two scenarios left the most substantive §1 items unguarded: `build_model_and_strategy` was 1/57 statements covered (so T5 had *no* signal), the early-stop break path 1/14 (T7 likewise), and T8's fall-back branches zero.
  - Added seven scenarios (`triplet_singleview`, `triplet_earlystop`, `triplet_nometric`, `build_triplet`, `build_arcface`, `build_pk_mismatch`, `build_no_pk`), taking coverage **45% -> 62%** and turning T2, T4, T5, T7, T8 into GUARDED.
  - Confirmed by injecting a botched T5 dedup: caught by all three `build_*` triplet scenarios, while the two original scenarios stayed green. The old baseline would have missed it entirely.
- **Parity baseline** committed at
  `sm-data/test-data/model-tests/golden_states/parity_baseline_v3.pt`
  (`v2` retained for cross-reference).

## Commit history for this work

| Commit | Branch | What |
| --- | --- | --- |
| `7d894f8` | `main` | Replaced dead trainer parity tests with a working parity harness; test fixture constants; `httpx` + PSL requirements; removed `test_fusion_model.py`; committed pending `lora_trainer.py` deletion. |
| `3f39a82` | `training_cleanup` | Expanded parity harness coverage 45% -> 62%; per-item guard status. |
| `cf56d7b` | `training_cleanup` | §1 complete: T1–T8, net −65 lines in `new_trainer.py`. |
| `1254976` | `training_cleanup` | Restructured this file with status checkmarks; fixed broken restore commands. |
| `8ef1b5e` | `training_cleanup` | §2: B1–B3, B5–B7 addressed, B4 held. Net +111 lines. |
| `ea78a28` | `training_cleanup` | Shared notification helper used by trainer + pipeline; `EARLY_STOPPING_METRIC` added to `constants.py`. |
| `b19dc3c` | `training_cleanup` | D2 behind a toggle + determinism acceptance tests; clarified per-config emails. |
| `e5aa03b` | `training_cleanup` | D1 behind a toggle: strict determinism flags in `seed_everything`. |
| `6807fab` | `training_cleanup` | D3 behind a toggle: re-seed before `train()`; documented the harness's blind spot. |
| `05b3e56` | `training_cleanup` | D4: measured MPS reproducibility (original claim was wrong); fixed MPS fallback import order in both manual test entry points. |
| `9b3bc60` | `training_cleanup` | S7: unused typing imports dropped; `.pylintrc` `source-roots=.` fixes `src.*` resolution at the root instead of per-file disables. |
| `3f43a8e` | `training_cleanup` | S5, S4, S3: stale tests retargeted at current APIs; csv fixture isolation bug fixed. 121 -> 135 passed. |
| `8175efb` | `training_cleanup` | S2: `test_server` retargeted at the lock-based concurrency guard and current endpoint contracts. 19 errors -> 0. |
| `6abedaf` | `training_cleanup` | S1: `test_embeddings` retargeted at the post-cleanup API. Suite fully green at 179 passed. |
| `c9513cc` | `training_cleanup` | S6 CI determination; E3 `setup.py` derives from `requirements.txt`; L1 `SHARK_TEST_MODE` images symlink. |
