# shark-matcher — Container

> **Status:** Deferred design placeholder, not a supported deployment path for
> the source release. The [Dockerfile](Dockerfile), [environment example](.env.example),
> and companion app Compose file are unfinished scaffolding, not for production
> use. Container implementation and verification follow the code-cleanup
> releases.

## Purpose

This directory describes the intended **ML inference service** container. The
companion app container guide (`shark-matcher-app/containerization/README.md`)
describes the small web/database tier.

The repositories already interact through an HTTP client and immutable
inference snapshots in the non-container workflow. The container work must
preserve that boundary rather than package one repo inside the other.

| Container | Intended responsibility |
|---|---|
| `shark-matcher` | Model loading, inference, embedding generation, retrieval, and validation of app-published snapshots |
| `shark-matcher-app` | Browser/API service, SQLite, image management, and publication of public inference snapshots |

### Why This Split

- The split replaces cross-repository Python imports with an HTTP API boundary.
- A "training versus everything else" split was rejected: retrieval would still
  need the database models.
- A "server and model versus web UI" split was rejected: it forces a frontend
  rearchitecture and does not remove the circular imports.
- Only model artifacts, embedding generations, and app-published inference
  snapshots cross the shared filesystem; the SQLite database stays private to
  the app.

## HTTP and Security Boundary

### Service and Client

- The matcher owns its FastAPI service in [`src/api/server.py`](../src/api/server.py).
- The app calls it through
  `frontend/inference_client.py` in the `shark-matcher-app` repository.
- Neither container should import Python modules from the other repository.

### Credential and Health Checks

- The inference service requires the same high-entropy `SHARK_INFERENCE_API_KEY`
  in both processes.
- Protected operations use Bearer authentication; the app's client supplies the header.
- The public `/health` probe reports process liveness **only**.
- The authenticated `/ready` endpoint checks required-model readiness and
  reports load failures.
- Container orchestration must distinguish those checks and must not assume a
  successful health check means inference is ready.
- The interactive API documentation routes are disabled in the current server.

### Compose and Network

The future Compose/network setup should:

- Keep the matcher accessible to the app on an internal network.
- Avoid publishing its inference port to the host by default.
- Provide credentials through a secure runtime configuration mechanism, not a
  secret committed to a Dockerfile or example file.

The exact exposure, key provisioning, and health/readiness checks require their
own implementation and security tests.

## Data Ownership and Embedding Updates

### Snapshot Flow

1. The app owns the live SQLite database and publishes a validated public
   inference snapshot using its existing snapshot publisher.
2. The matcher receives only a snapshot identifier over authenticated HTTP.
3. It reads the completed snapshot from an explicitly configured shared root and
   validates it with
   [`src/retrieval/inference_snapshot.py`](../src/retrieval/inference_snapshot.py).
4. It uses [`run_pipeline()`](../src/retrieval/embeddings.py) with
   [`publish_generation()`](../src/retrieval/generation_store.py) for atomic
   embedding updates.

The matcher also needs access to the corresponding source images and model
checkpoints. No database connection, full trusted backup, or app-side ML
implementation crosses this boundary.

### Mounts

The eventual mounts should follow those responsibilities:

- App-private SQLite storage.
- App-write/matcher-read access to completed inference snapshots.
- Matcher-read access to the images and trained models it needs.
- Matcher-write access to embedding generations.

Decide and test mount paths, ownership, permissions, publication visibility, and
snapshot retention during the later container implementation. Do not grant both
containers a blanket read/write mount of the entire data workspace.

### Paths

- The path authorities are [`src/core/constants.py`](../src/core/constants.py) and the app's
  `core/app_constants.py` in the `shark-matcher-app` repository.
- Both accept `SHARK_INFERENCE_SNAPSHOT_ROOT` for the cross-service snapshot boundary.
- Other data locations currently resolve relative to each checkout; mapping them
  into separate images needs a deliberate portability audit, not an assumption
  that a sibling checkout or a particular host path exists.
- The `/sm-data` path shown in the present Dockerfile is a proposed container
  mount convention, **not** a required host directory.

## Existing Scaffolding

[`Dockerfile`](Dockerfile) and the app's `containerization/docker-compose.yml`
capture an early full-workspace-volume experiment. They have not been
reconciled with the current authentication, snapshot, readiness, path, and
artifact contracts.

- [`.env.example`](.env.example) is similarly historical: its model and
  optional-tool settings are not a current configuration reference.
- Use `src/core/constants.py` and `src/api/server.py` for live configuration.
- Use [`requirements.txt`](../requirements/requirements.txt) and [`setup.py`](../setup.py)
  for package dependencies.

## Later Acceptance

The deferred implementation must:

- Build and start both images.
- Prove the app has no ML packages or matcher checkout.
- Verify a valid shared key and rejection of missing or incorrect keys.
- Validate liveness separately from required-model readiness.
- Exercise an app-published snapshot through an authenticated embedding update
  and retriever reload.
- Test that the matcher cannot access the app database and that both containers
  can use their own data roots without developer-specific absolute paths.

Until those checks pass, use the repository-supported non-container workflows
in the root READMEs.
