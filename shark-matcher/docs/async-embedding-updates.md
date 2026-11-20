# Async Embedding Updates — Design Notes

> **Status:** Design note; a simplified Option A is implemented.
> `POST /embeddings/generate` with `"background": true` starts the job and returns its id,
> and `GET /embeddings/status` (no path parameter) reports it. `src/api/server.py` owns the behavior.

Design notes for making `/embeddings/generate` non-blocking. The sections below
record the synchronous behavior and options weighed before background mode existed.

## Synchronous Behaviour Before Background Mode

`POST /embeddings/generate` runs the full pipeline on the request thread
and holds the HTTP connection open until it finishes:

1. Image embeddings
2. Centroid embeddings
3. Image-to-image neighbour index
4. Centroid neighbour index
5. In-process `Retriever._load_model_data(model_name)` to hot-reload

A process-wide `threading.Lock` prevents concurrent runs; a second caller
gets `503`. The client (`shark-matcher-app/frontend/inference_client.py`)
sets a 1-hour socket timeout and blocks until completion.

### Why this is okay today

- Only one user triggers regeneration (the admin flow in
  `frontend/project.py:update_embeddings`).
- Failures surface as HTTP 500 and the frontend shows an error.
- No frontend progress UI exists, so there's nothing to starve.

### Why we want async eventually

- A minutes-to-an-hour blocking request is fragile to any middleware that
  enforces read/idle timeouts (reverse proxies, load balancers, browsers
  if ever invoked directly).
- The frontend can't show progress — the user watches a spinner with no
  signal whether it's alive, stalled, or crashed.
- If the app process restarts while waiting, the client loses the
  result even though the server-side pipeline may have completed.

## Options

### Option A — start + poll

`POST /embeddings/generate` returns immediately with a `job_id`.
`GET /embeddings/status/{job_id}` returns current phase, progress, and
terminal state (`running`, `succeeded`, `failed`).

**Pros:** small, obvious contract; frontend can render a progress bar;
survives client disconnects; easy to test.
**Cons:** need a job registry with eviction; need to decide persistence
(see "State" below).

### Option B — server-sent events

`POST /embeddings/generate` opens an SSE stream; server emits progress
events until the final `done` event.

- **Pros:** no polling; real-time progress.
- **Cons:** still a long-lived connection (same proxy-timeout risk as sync);
  harder to test; reconnection semantics are fiddly; harder for the app's
  existing `http.client` wrapper to consume.

### Option C — webhook callback

`POST /embeddings/generate` takes a callback URL; server pings it on
completion.

- **Pros:** fully decoupled; no polling; fits a queue-based architecture.
- **Cons:** two-way networking requirement between services; overkill for
  the current single-machine deployment; hardest to secure.

### Recommendation

Option A. It's the smallest change that gives the frontend something to render,
and it degrades gracefully if the client goes away.

## Edge cases to design through (the reason this is parked)

1. **Job registry lifetime.** Where do jobs live?
   - *In-process dict* — simplest. Loses state on restart; fine if
     restart is rare and the pipeline is idempotent.
   - *On-disk JSON file* — survives restart but needs locking.
   - *Database row* — robust, but pulls in a dep the inference service
     doesn't currently have.

2. **Pipeline survival across server restart.** The pipeline can run for
   an hour. If the server restarts mid-run, does the job resurrect, or
   is it marked `failed` and the user retries?
   - Simplest: mark as `failed` on startup sweep; force retry.
   - Harder: checkpoint per phase (image -> centroid -> neighbours -> centroid neighbours) so a restart can resume. Phases already produce separate `.pt` files, so resume-by-phase is tractable but adds dedup logic.

3. **Concurrent generation.** Today the `threading.Lock` serialises
   globally. With a job registry, is it still global, or per-model?
   - Per-model would let `miew` and `vit_triplet` regenerate in parallel,
     but they share CPU/GPU — probably no throughput win and more risk
     of OOM.
   - Keep global until there's evidence for per-model.

4. **Retriever reload race.** The sync version reloads `Retriever` only
   after the pipeline succeeds, so `/match` calls during the run still
   hit the old embeddings. With async, make sure:
   - `/match` never sees a half-written state (hold the old model data
     until the new one loads successfully, then swap atomically).
   - `/stats` reports which model is mid-update so the frontend can
     disable the refresh button or show "updating…".

5. **Failure reporting.** Currently an exception becomes HTTP 500 with
   the exception message. With async, the failure lives in the job
   record — decide on a shape:
   ```json
   {"state": "failed", "error": "message", "traceback": "...", "failed_phase": "centroid_embeddings"}
   ```
   Don't leak traceback to the frontend in production; keep it in logs.

6. **Progress granularity.** The pipeline has four phases of wildly
   different durations. Useful progress probably means:
   - Emit phase name on entry (`image_embeddings`, `centroid_embeddings`, `image_neighbors`, `centroid_neighbors`).
   - Within `store_embeddings`, the inner loop iterates over images —
     percent-within-phase is cheap to surface.
   - Whole-pipeline percent is misleading (phases aren't equal weight);
     prefer phase + per-phase percent.

7. **Job eviction.** How long do completed/failed jobs stick around for
   the frontend to poll? Suggestion: 24 hours, then drop.

8. **Client contract for `update_embeddings` in the app.** Today the app
   blocks on `generate_embeddings(...)` and returns updated stats
   afterward. With async, the HTTP handler has to decide:
   - *Option A1*: app handler blocks on its own poll loop, preserving
     the existing request/response shape — simpler on the frontend, but
     reintroduces the blocking connection we wanted to remove.
   - *Option A2*: app exposes its own job status endpoint, the UI polls
     it, and the refresh/stats update happens when the job completes —
     proper fix, more UI work.

9. **Authorization.** Only admins trigger this today. If we add polling
   endpoints, make sure `/embeddings/status/{job_id}` doesn't leak job
   existence to unauth'd callers. Probably already covered by whatever
   gates the admin route — confirm at implementation time.

10. **Idempotency / dedup.** If the frontend double-submits (fat-finger
    or flaky network), do we create two jobs for the same model or
    return the existing running job's id? Returning the existing id is
    friendlier and maps nicely to the current `503` behaviour.

## Minimum viable async (if we pick Option A)

Endpoints:
- `POST /embeddings/generate` -> `{"job_id": "...", "state": "running"}`. If a job for `model_name` is already running, return its id with `200` instead of creating a new one (addresses edge case 10).
- `GET /embeddings/status/{job_id}` -> `{"job_id", "model_name", "state", "phase", "phase_percent", "started_at", "finished_at", "error"}`.

Job state and pipeline:
- State is stored in an in-process dict guarded by a lock; wiped on restart
  (accept edge case 2 as "retry on restart"); evicted after 24h.
- The pipeline runs in a `threading.Thread` (not `BackgroundTasks`, which must
  survive the request returning).
- Retriever reload happens on the worker thread after the last phase; emit a
  final `state: reloading` phase so the frontend knows stats are about to change.

Frontend change (app side):
- `inference_client.generate_embeddings()` becomes two calls: `start()`
  returning `job_id`, and `poll(job_id)` returning status.
- `update_embeddings` in `frontend/project.py` either polls internally
  (A1, minimal change) or exposes job state to the UI (A2, proper fix).

## When to revisit

Triggers that should prompt picking this up:
- Adding a second admin who might trigger generation concurrently.
- Putting the inference service behind a proxy that enforces an idle
  timeout shorter than the pipeline duration.
- First actual frontend progress UI request.
- Deploying the services on separate machines where a stable
  long-lived HTTP connection is no longer a given.
