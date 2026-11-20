# shark-matcher-app — Container

> **Status:** Deferred design placeholder, not a supported deployment path for
> the source release. The [Dockerfile](Dockerfile),
> [Compose file](docker-compose.yml), and [environment example](.env.example)
> are unfinished scaffolding, not for production use. Container implementation
> and verification follow the code-cleanup releases.

## Purpose

This directory describes the intended **lightweight web application**
container. The companion matcher container guide
(`shark-match-model/containerization/README.md`) describes the ML inference tier.

The repositories already communicate via HTTP and an immutable inference
snapshot in the non-container workflow; later container work should use those
same owners.

| Container | Intended responsibility |
|---|---|
| `shark-matcher-app` | Browser/API service, SQLite, image management, and publication of public inference snapshots |
| `shark-match-model` | Model loading, inference, embedding generation, retrieval, and validation of app-published snapshots |

### Why This Split

The section "Why This Split" in `shark-match-model/containerization/README.md` gives the reasons.

## Lightweight Application Image

The final app image contains only the app runtime code and dependencies: its
web server, browser assets, database layer, and data-management commands.

- Do **not** copy in the matcher repository, model weights, embeddings, training
  code, or ML dependency stack.
- The app's [`setup.py`](../setup.py) and [`requirements.txt`](../requirements/requirements.txt)
  separate runtime package requirements from development tooling; the future
  image build must honor that distinction and exclude tests, caches, and local
  artifacts.
- The current [Dockerfile](Dockerfile) installs the full contributor
  requirements and copies the entire build context, so it does not yet
  establish the desired image size or dependency boundary.

## Inference Client and Credential

- The authoritative app-to-matcher client is
  [`frontend/inference_client.py`](../frontend/inference_client.py).
- It uses [`core/app_constants.py`](../core/app_constants.py) for the service URL and
  shared `SHARK_INFERENCE_API_KEY`, and sends a Bearer credential on protected
  operations.
- The matcher validates that same key with its existing FastAPI dependency.
- The app must not implement model selection or a second inference client to
  accommodate containers; it queries loaded models through the service.
- Provision the credential securely to both processes at runtime; never bake it
  into images, Compose defaults, or an example file.

## Data and Network Boundary

### Snapshot Flow

1. The app keeps exclusive access to its SQLite database and owns browser-facing images.
2. [`backend/inference_snapshot.py`](../backend/inference_snapshot.py) publishes
   only approved public inference data from a single database transaction.
3. [`frontend/project.py`](../frontend/project.py) sends the same immutable
   snapshot identifier to the matcher over authenticated HTTP.
4. The matcher reads and validates the completed snapshot from a shared root
   before its existing atomic generation pipeline runs.

No full database, credentials, sessions, or trusted backup is exposed to the
inference tier.

### Mounts

- App-private database storage is separate from app-write/matcher-read completed snapshots.
- The matcher also needs read access to the source images and trained models
  used in inference, and write access to its embedding generations.
- The app needs no model or embedding mounts for its HTTP requests.
- Image visibility, snapshot publication permissions, file ownership,
  retention, and failure recovery need a tested cross-container contract.
- Avoid the present Compose file's blanket read/write mount of the entire data
  workspace into both containers.

### Paths

- The app's [`core/app_constants.py`](../core/app_constants.py) and the matcher's
  `src/core/constants.py` (in the `shark-match-model` repository) own path resolution.
- Both support `SHARK_INFERENCE_SNAPSHOT_ROOT` for the shared snapshot location.
- For other data roots, future container work must reconcile the existing
  checkout-relative defaults with the separate images and provide explicit,
  portable configuration where required.
- The `/sm-data` path in the current Dockerfile and Compose file is a proposed
  **container** mount convention, not an instruction to place data at an
  absolute path on the host.

### Network

- The intended service-to-service URL is internal to the container network.
- Only the browser-facing app port should be published by default.
- The current [`docker-compose.yml`](docker-compose.yml) publishes the inference
  port and waits for `/health`, which checks process liveness but not
  required-model readiness. The matcher exposes a separate authenticated
  `/ready` check.
- The Compose service does not yet supply the required shared credential, so its
  dependency ordering and security do not prove a usable or protected deployment.

## Existing Scaffolding

[`docker-compose.yml`](docker-compose.yml), [`Dockerfile`](Dockerfile), and
[`.env.example`](.env.example) record an earlier design, not executable
installation instructions.

- The environment example includes an app-side model setting that no longer owns
  model selection.
- Consult `core/app_constants.py`, the matcher server/configuration, and the package
  metadata for current behavior instead of treating the example as live defaults.
- Do not run the database initialization, standalone `docker run`, or
  `docker compose up` examples in the old scaffolding as a supported release workflow.

## Later Acceptance

The later container checkpoint must:

- Build a demonstrably small app image with no matcher/ML code.
- Provision the shared key securely and keep the matcher internal.
- Use distinct liveness and readiness checks.
- Verify real app-to-matcher matching and snapshot-driven embedding updates.
- Test that the app cannot read matcher-only artifacts, the matcher cannot read
  SQLite or trusted exports, and the host paths are portable.

Until this work passes those checks, use the repository-supported
non-container workflows in the root READMEs.
