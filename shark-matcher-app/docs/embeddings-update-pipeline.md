# Embeddings Update Pipeline

`shark-matcher-app` owns the database and `shark-match-model` owns embedding
generation. The services exchange an immutable public snapshot identifier over
the authenticated inference API; the matcher does not import the app database
layer or receive a trusted backup.

```text
browser
  -> shark-matcher-app update handler
       -> publish one public inference snapshot
       -> authenticated HTTP generation request
            -> shark-match-model server adapter
                 -> shared embedding pipeline
                      -> atomic generation publisher
                 -> retriever reload
```

## Source Authorities

| Responsibility | Authoritative source |
|---|---|
| Browser-facing update coordination | [`frontend/project.py`](../frontend/project.py) |
| Inference HTTP transport, authentication, and timeout | [`frontend/inference_client.py`](../frontend/inference_client.py) |
| Public database snapshot publication | [`backend/inference_snapshot.py`](../backend/inference_snapshot.py) and [`backend/inference_snapshot_contract.py`](../backend/inference_snapshot_contract.py) |
| Protected generation endpoint and live-mode selection | companion matcher `src/api/server.py` |
| Snapshot validation | companion matcher `src/retrieval/inference_snapshot.py` |
| Full and incremental artifact computation | companion matcher `src/retrieval/embeddings.py` and `src/retrieval/neighbors.py` |
| Staging, validation, manifests, retention, rollback, and active pointer | companion matcher `src/retrieval/generation_store.py` |

Route details, configuration fields, artifact names, retention rules, and error
messages live in those files rather than here.

## Request and Snapshot Boundary

- **App side:** `frontend.project.update_embeddings()` queries loaded model
  views, publishes one public snapshot inside a single database transaction, and
  sends that snapshot UUID to each generation request through
  `inference_client._request()`.
- **Matcher side:** the matcher treats the snapshot as read-only and validates
  its UUID, manifest, files, hashes, row counts, CSV shape, and path containment
  before computing anything.
- **Data isolation:** authentication, session, and operational tables are
  excluded from the public snapshot.

## Generation, Publication, and Reload

1. The protected matcher endpoint acquires its non-blocking generation lock and
   resolves one supported artifact mode (baseline or multimodal) per embedding
   directory. Live layouts requiring the offline PSL workflow are rejected.
2. `_run_embedding_pipeline()` passes the validated snapshot to `run_pipeline()`,
   which delegates publication to `publish_generation()`.
3. `publish_generation()` seeds an isolated staging directory from the active
   generation, runs the incremental or full builders, validates the artifact
   contract, writes metadata and manifest provenance, and atomically swaps the
   active pointer.
4. Incremental computation carries over unchanged image embeddings and reuses
   incremental centroid and neighbor updates when prior artifacts exist;
   otherwise it runs the full builders. Multimodal views and cached review
   suggestions are rebuilt before publication.
5. After publication succeeds, the server reloads every configured view backed
   by the updated embedding directory.

## Failure and Concurrency Behavior

- Snapshot publication failure aborts before any inference request is sent.
- Overlapping live generation requests are rejected immediately by the server
  lock.
- Build or validation failure deletes the staging directory, keeps the previous
  active generation active, and releases the lock.
- A required-model reload failure is reported as an error, though the already
  validated generation remains published.
- Loaded models are updated sequentially without cross-model rollback, and
  dataset-stat caching is invalidated only after all requested updates succeed.
- Completed input snapshots have their own lifecycle and may remain after a
  later inference failure.

## Current Limitations

- **Synchronous request lifetime:** the app holds one authenticated HTTP
  connection open until each requested model finishes, which can outlast
  ordinary browser or proxy timeouts.
- **Fusion weight tuning:** live snapshots have no training split, so live
  multimodal updates reuse the existing fusion configuration (or its fallback);
  retuning weights remains an offline operation.
- **Error reporting:** the app client reduces non-success matcher responses to a
  generic failed request.
