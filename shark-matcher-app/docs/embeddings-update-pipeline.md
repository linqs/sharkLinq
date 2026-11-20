# Embeddings Update Pipeline

> **Status:** Part 1 is an as-recorded historical repair log, not current
> implementation guidance. Its snippets, module relationships, model examples,
> and intermediate fixes describe earlier revisions and may have since been
> replaced. For current behavior, start with Part 2 and follow its links to the
> app and matcher source authorities.

## Part 1 — Bugs Fixed

### Bug 1: Return tuple slots swapped in `frontend/project.py`

- **Error:** `'str' object does not support item assignment`
- **Cause:** `update_embeddings()` returned error responses with the message in the headers slot and `None` in the payload slot.

**Fix:**

```python
# Before (wrong)
return None, http.HTTPStatus.INTERNAL_SERVER_ERROR, "Could not reach inference service..."

# After (correct)
return "Could not reach inference service...", http.HTTPStatus.INTERNAL_SERVER_ERROR, None
```

Why it crashed:

- `_call_api_handler` in `common.py` expects `(payload, status_code, headers_dict)`.
- When a string landed in the headers slot, it tried `headers['qgg-body'] = True` on a string.

---

### Bug 2: `No module named 'db'` — path not set before import in `server.py`

- **Error:** `ModuleNotFoundError: No module named 'db'`
- **Cause:** inside `_run_embedding_pipeline`, the call to `constants.ensure_shark_app_in_path()` came *after* the `import db.scripts.db_dump` statement, so the import ran before `shark-matcher-app` was on `sys.path`.

```python
# Before (wrong order)
import db.scripts.db_dump          # crashes — shark-matcher-app not in path yet
constants.ensure_shark_app_in_path()

# After (correct order)
constants.ensure_shark_app_in_path()
import db.scripts.db_dump
```

---

### Bug 3: `No module named 'backend.db'` — legacy pickle compat shadowed the real package

- **Error:** `ModuleNotFoundError: No module named 'backend.db'`
- **Cause:** `server.py` runs at startup:

```python
sys.modules['backend'] = src  # legacy pickle compat
```

- This makes `import backend.X` resolve to `src.X`.
- When `db_dump` then does `import backend.db.individual`, Python looks for
  `src.db.individual`, which doesn't exist; the `db/` package lives in
  `shark-matcher-app`, not `shark-matcher/src`.

- **Fix:** temporarily pop the `backend` alias around the import, then restore it.

```python
_backend_compat = sys.modules.pop('backend', None)
try:
    import db.scripts.db_dump
finally:
    if _backend_compat is not None:
        sys.modules['backend'] = _backend_compat
```

Once imported, the `db` modules are cached in `sys.modules`, so the compat alias no longer interferes.

---

### Bug 4: Wrong model name passed to `store_image_embeddings` — `constants.py` + `server.py` + `retrieval.py`

- **Error:** `Could not find a config file for fine-tuned model 'miew_arcface'`

**Cause:**

- `DEFAULT_MODELS` entries only had a `dir` field (the embeddings subdirectory name, e.g. `miew_arcface`).
- That value was passed as `model_name` to `store_image_embeddings`, which then looked for a trained model directory called `miew_arcface`; the actual one is `initial_miew_arcface_new`.
- The *embeddings directory* (`dir`) and *trained model* (`model`) are distinct but were conflated.

- **Fix:** added a `model` field to `DEFAULT_MODELS` entries in `constants.py`, propagated through `retrieval.py`'s `_model_configs` parsing and `server.py`'s `_parse_models_env` and `embeddings_generate` endpoint.

```python
DEFAULT_MODELS = [
    {"display_name": "MiewID Multimodal", "dir": "miew_arcface",
     "model": "miew_arcface", ...},
]
```

---

### Bug 5: Full re-generation instead of incremental update — `server.py`

- **Symptom:** Every trigger re-embedded all 4000+ images instead of only new ones.
- **Cause:** `_run_embedding_pipeline` called `store_image_embeddings` unconditionally, which regenerates every embedding from scratch; `update_embeddings` (incremental) was never called from the server.

- **Fix:** restructure `_run_embedding_pipeline` to branch on whether `annotations_version` is supplied:

```python
if annotations_version is None:
    # UI-triggered incremental path
    _embeddings.update_embeddings(model_name=model_name, model_dir=model_dir)
else:
    # Full regeneration (pre-built annotations supplied by caller)
    _embeddings.store_image_embeddings(...)
```

---

### Bug 6: Empty multimodal vocabulary from `fit_split="train"` — `embeddings.py`

- **Error:** Degenerate alpha weights written to `late_fusion_config.json` (all near-zero).

**Cause:**

- `_run_multimodal_pipeline` built `FusionDataset` with `fit_split="train"`.
- `prepare_annotations(train_test=False)` writes *empty* `train-annotations.tsv` and `valid-annotations.tsv` files.
- With an empty train split, `fit_norm_stats` had zero examples to fit, producing empty vocabulary groups.
- The alpha optimiser then saw only zero-similarity matrices for every metadata group, resulting in garbage weights.

- **Fix:** use `fit_split="all"` so vocabularies and numeric ranges are fit on the full corpus.

---

### Bug 7: Centroid label/embedding row-count mismatch — `embeddings.py`

- **Error:** Potential `IndexError` when `FusionDataset` filters IDs not found in the CSV store.

**Cause:**

- `_compute_centroids` was called with `stored_dataset.get_labels()` (all N labels) but `fused` embeddings only had `len(fusion_ds)` rows.
- If any media ID was in `stored_dataset.pt` but absent from the CSV (e.g. a record added to the DB but not yet in a CSV dump), `len(fusion_ds) < len(stored_dataset)` and the label tensor had the wrong length.

- **Fix:** use `fusion_ds.get_labels()` instead of `stored_dataset.get_labels()`, so both tensors align to the same filtered set.

---

### Bug 8: `torch.topk` crash on empty query tensor — `optimize_late_fusion.py` + `embeddings.py`

- **Error:** `The size of tensor a (100) must match the size of tensor b (4194) at non-singleton dimension 1`

#### Root Cause

- With `train_test=False`, `stored_dataset` has 0 train IDs.
- `_make_kfold_seed_data` therefore produces `query_indices` tensors of shape `(0,)`, and all similarity matrices have shape `(0, 4194)`.
- Inside `_score`, `torch.topk(fused, 100, dim=1)` on a `(0, 4194)` tensor triggers a PyTorch internal broadcast error.

#### Fix A: Defensive Guard in `_score`

In `optimize_late_fusion.py`:

```python
Q = query_indices.shape[0]
if Q == 0:
    return {"hits_at_1": 0.0, ..., "avg_k": float("inf")}
```

#### Fix B: Skip Re-Optimisation Without Train IDs

In `embeddings.py`:

- When `fusion_ds.get_train_ids()` is empty (the normal incremental UI path), reuse the existing `late_fusion_config.json` rather than running 1000 optimizer trials that would all score nan.
- Falls back to image-only alphas if no config exists yet.

```python
if not has_train_ids:
    if os.path.exists(config_path):
        # reuse existing alphas — they were fit on a proper train split
        alphas = existing_cfg.get("alphas", {"alpha_image": 1.0})
    else:
        alphas = {"alpha_image": 1.0}
else:
    # full K-fold optimisation path (train_test=True runs)
    ...
```

---

## Part 2 — Current Embeddings Update Workflow

The current design keeps database ownership in `shark-matcher-app` and ML
generation in `shark-matcher`. The repositories exchange an immutable public
snapshot identifier over the authenticated inference-service API; the matcher
does not import the app database layer or receive a trusted backup.

```text
browser
  -> shark-matcher-app update handler
       -> publish one public inference snapshot
       -> authenticated HTTP generation request
            -> shark-matcher server adapter
                 -> shared embedding pipeline
                      -> atomic generation publisher
                 -> retriever reload
```

### Source Authorities

| Responsibility | Authoritative source |
|---|---|
| Browser-facing update coordination | [`frontend/project.py`](../frontend/project.py) |
| Inference HTTP transport, authentication, and timeout | [`frontend/inference_client.py`](../frontend/inference_client.py) |
| Public database snapshot publication | [`backend/inference_snapshot.py`](../backend/inference_snapshot.py) and [`backend/inference_snapshot_contract.py`](../backend/inference_snapshot_contract.py) |
| Protected generation endpoint and live-mode selection | companion matcher `src/api/server.py` |
| Snapshot validation | companion matcher `src/retrieval/inference_snapshot.py` |
| Full and incremental artifact computation | companion matcher `src/retrieval/embeddings.py` and `src/retrieval/neighbors.py` |
| Staging, validation, manifests, retention, rollback, and active pointer | companion matcher `src/retrieval/generation_store.py` |

Those files own changing route details, configuration fields, artifact names,
retention rules, and error messages. They are intentionally not duplicated
here.

### Request and Snapshot Boundary

**App side:**

- `frontend.project.update_embeddings()` asks the inference service which model
  views are loaded, publishes one snapshot, and supplies the same snapshot UUID
  to each requested generation.
- The snapshot publisher reuses the public export path for the inference tables
  inside one database transaction.
- Authentication and transport stay centralized in `inference_client._request()`;
  the update handler does not implement a second service client.

**Matcher side:**

- It validates the snapshot identifier, manifest, files, hashes, row counts, CSV
  shape, and containment before computation.
- Snapshot input is read-only.
- Annotation preparation and every generated artifact are owned by the matcher
  after that boundary.
- Sensitive authentication, session, and operational tables are not part of the
  public inference snapshot.

### Generation and Publication

#### Endpoint

- The protected matcher endpoint serializes live generation with its existing
  non-blocking generation lock.
- It resolves one supported artifact mode for all configured views sharing an
  embedding directory and rejects live layouts that require the offline PSL workflow.
- `_run_embedding_pipeline()` adapts the validated snapshot to the existing
  `run_pipeline()` coordinator; it does not maintain a second embedding algorithm.

#### Publication

`run_pipeline()` delegates publication to `publish_generation()`. The publisher:

1. Seeds an isolated staging generation from the prior active state.
2. Runs the existing full or incremental builders.
3. Validates the selected artifact contract.
4. Writes matching metadata and manifest provenance.
5. Replaces the active pointer.

Failed builds remove their staging state and leave the prior active generation selected.

#### Computation

The computation layer already reuses `UpdateResult` when possible:

- New image embeddings are computed while unchanged vectors are carried over.
- Existing incremental centroid and neighbor functions are selected when the
  prior artifacts needed by those functions are present.
- A bootstrap update or missing prior centroid artifacts selects the existing
  full recomputation functions instead of a parallel recovery path.
- The configured multimodal view and cached review suggestions are rebuilt by
  their established shared functions before validation and publication.

#### Reload

After successful publication, the server reloads every configured view backed
by the updated embedding directory. Required-model reload failures are reported
instead of silently claiming readiness.

### Failure and Concurrency Semantics

- Snapshot publication failure stops before an inference request is sent.
- An overlapping live generation is rejected immediately by the server lock.
- Build or validation failure leaves the previous active generation intact and
  releases the lock.
- A reload failure occurs after publication; the endpoint reports failure, but
  the already validated generation remains published for later recovery.
- The app updates loaded models sequentially. If a later model fails, earlier
  successful model publications are not rolled back as a group.
- Completed input snapshots have a lifecycle separate from embedding
  publication and can remain after a later inference failure.

Generation remains synchronous: the app client holds one authenticated HTTP
connection open until each requested model finishes. Dataset-stat caching is
invalidated only after all requested updates succeed.

## Part 3 — Current Limitations and Deferred Work

### Completed Foundations

- Atomic generation staging, validation, provenance, active-pointer
  replacement, retention, and rollback are centralized in
  `generation_store.py`.
- Incremental image, centroid, image-neighbor, and centroid-neighbor paths are
  selected by the shared embedding coordinator when their prerequisites exist.
- Live server generation derives one baseline or multimodal layout for all
  views sharing an embedding directory; PSL and stacked generation remain
  explicit offline workflows.

### Alpha Re-optimization Cadence

- The ordinary UI snapshot has no training split, so live multimodal generation
  reuses an existing fusion configuration when available and otherwise uses the
  implementation's fallback.
- Significant retuning remains an offline research or maintenance operation.
- The current decision and fallback belong to matcher `src/retrieval/embeddings.py`.

### Error Reporting and Logging

- The matcher endpoint reports generation and required-reload failures.
- The app client currently reduces non-success service responses to a failed request.
- More detailed API error propagation and the broader logging overhaul remain V2
  work; this guide does not prescribe a second exception or logging path.

### Synchronous Request Lifetime

- Live generation can outlast ordinary browser or proxy timeouts even though the
  app client permits a long request.
- A future asynchronous job/progress protocol would require a coordinated API
  design across both repositories and is not part of the V1 workflow.
