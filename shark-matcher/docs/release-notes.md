# Shark Matcher v1.0.0

This is the first public source release of the Shark Matcher training and
inference repository. It is intended for use with the synchronized v1.0.0
release of `shark-matcher-app`.

## Highlights

- The matcher is independently installable and no longer imports the app or
  accesses its SQLite database.
- App-owned inference snapshots provide validated, bounded embedding inputs.
- Embedding updates use versioned staging, validation, atomic publication,
  retention, and rollback.
- The inference boundary requires a shared Bearer credential, while public
  health remains separate from authenticated readiness and model diagnostics.
- Model-load failures are reported through stable status contracts.
- Repository-owned deterministic, packaging, artifact, and portability gates
  cover the supported source release.

## Compatibility and Limitations

- Use the matching v1.0.0 app release for the coordinated HTTP and snapshot
  contracts.
- Existing legacy flat embeddings remain readable until a successful update
  publishes a versioned generation.
- Embedding generation is synchronous by default. The optional `background`
  request flag runs one job at a time and returns a job identifier.
- Typed API errors, structured logging, and broader lifecycle policy are
  deferred work.
- CPU on macOS and Linux is the V1 release gate; CUDA and MPS are best-effort.
- Container files remain deferred design scaffolding, not a supported
  deployment workflow.
- This source release is licensed under the MIT License; see `LICENSE`.

## Validation

The approved candidate passed the repository-owned deterministic and Python
lint gates, clean artifact builds and isolated installation from both wheel and
source archive, and tracked-source and artifact portability checks.
