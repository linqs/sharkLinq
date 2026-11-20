# Shark Matcher App v1.0.0

This is the first public source release of the Shark Matcher browser and data
application. It is intended for use with the synchronized v1.0.0 release of
`shark-match-model`.

## Highlights

- The app owns SQLite data and publishes immutable public inference snapshots
  without exposing account or session records to the matcher.
- All protected matcher requests use the shared authenticated inference client.
- Browser-accessible exports are separated from complete trusted server-side
  backups.
- Account creation, password verification, session issuance and expiry,
  throttling, and schema migration have explicit security contracts.
- The shared disposable demo composes the existing application operations while
  blocking unsupported or persistent writes.
- The application package excludes matcher ML code and dependencies.
- The committed immutable CI fixture drives the deterministic test suite.

## Compatibility and Limitations

- Use the matching v1.0.0 matcher release for inference and embedding updates.
- Existing sessions are intentionally invalidated by the V1 authentication
  schema migration; users must sign in again.
- Browser sessions remain in the existing JavaScript-readable transport for
  V1, and cookie/CSRF migration is deferred work. Rendered values are
  HTML-escaped.
- One server-side media-pagination path and broader large-query work remain
  deferred.
- Trusted server-side backups intentionally contain credentials and session
  data and must never be exposed through client-accessible routes.
- Container files remain deferred design scaffolding, not a supported
  deployment workflow.
- This source release is licensed under the MIT License; see the root `LICENSE` file.

## Validation

The approved candidate passed the repository-owned deterministic, Python lint,
and JavaScript lint gates with one documented V2 pagination skip, clean artifact
builds and isolated installation from both wheel and source archive,
immutable-fixture verification, and tracked-source and artifact portability
checks.
