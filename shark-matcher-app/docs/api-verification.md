# Phase 1 API verification record

> **Status:** Internal verification record; not usage documentation. Current
> usage lives in `docs/direct-api.md`.

## Compatibility target

Existing authenticated web workflows retain their successful response shapes,
filtering, rankings, account permissions, and database effects. Intentional
changes are authenticated image access, exact route/method handling,
server-assigned acting identities, enforcement of existing account/proposal
permissions, bounded input sizes, safe filesystem access, and safe error
responses. HTTPS and deployment configuration belong to phase 2.

## Baseline and checks

Before server changes:

- The application suite passed **412 tests with one skip**.
- Five role-based Chromium workflows passed against the original server.
- The initial logout check needed its URL expectation corrected for the existing
  `/static/index.html#login` redirect.

### Automated Checks

| Check | Result |
| --- | --- |
| Application deterministic suite | 430 passed, 1 existing skip |
| Application Python / JavaScript lint | Passed |
| Matcher deterministic suite | 337 passed |
| Matcher error-level Python lint | Passed |
| Chromium / Firefox / WebKit | 45 passed (15 cases per engine; main matrix plus gallery and password reruns) |
| Patch whitespace checks | Passed |

### API Security Tests

`tests/frontend/test_api_security.py` uses real HTTP requests against production
route handlers and a disposable copy of the committed fixture. It checks:

- GET/POST result parity for search filters, pagination and both matching methods.
- Short image names, missing images, all five account roles, and exact routes.
- Unauthenticated rejection across protected routes and GET rejection for writes.
- Conflicting credentials, cookie scope, logout revocation, private image caching,
  image-only serving, account binding and server-assigned action authors.
- Proposal author/admin rules, URL traversal, descendant symlinks, absolute paths,
  and malicious filenames already stored in the database before delete, rename,
  or export.
- Invalid queries/bodies/images, size/count/pixel limits, rollback, and sanitized
  errors. Inference transport tests distinguish failure statuses, malformed
  results and timeouts from successful empty rankings.

### Browser Workflows

Each browser test uses temporary data. The workflows cover:

- login failure and logout; five-role navigation;
- filtered search, matching, loaded images and reloads;
- profile and password updates; account creation and deletion;
- file upload and duplicate rejection; filtered export with images;
- individual edits; demo randomize and clear; expired sessions;
- proposal creation, archive and author deletion; merge; embedding-update initiation;
- gallery reference, unmatch and delete controls.

The browser startup check exposed a spinner regression: `user/auth` did not
start the loading indicator but its response decremented it. The existing
`sendRequest()` now stops the indicator only for requests that started it.

### Deterministic Suite

The existing suite continues covering detailed filtering, pagination, upload
transactions, merge/unmatch/reference behavior, authentication, demo isolation,
export privacy, packaging, and inference snapshots. Browser coverage complements
these by exercising forms, navigation, images, cookies and downloads.

## Reuse and path ownership

### Reuse

- GET extends the existing `build_api_route()` / server dispatcher and calls
  `filter_individuals()` and `_get_neighbors()`.
- Session resolution is shared by the existing username/token authentication and
  new Bearer/image-cookie callers.
- `save_media()` now owns image conversion for standalone and batch uploads.
- `resolve_served_path()` lives in the existing `core/app_utils.py` and is reused by
  served assets, image operations and export copying.
- The existing proposal, user, demo, inference transport and test runners were
  extended in place.
- The browser harness was added after finding no automated browser suite. It
  composes the existing HTTP test harness and committed database fixture rather
  than creating another application implementation.

### Paths

- Runtime paths continue through the existing constants/configuration authorities.
- Test paths use temporary-directory fixtures.
- No developer-specific rooted path or new rooted operating-system path was
  added to source, tests or examples.

## Evidence limits and rollout

### What Was Run

- Automated tests ran locally on macOS with Python 3.12.
- The browser tests use generated valid image files and a deterministic matcher
  stub over real HTTP.
- The matcher suite exercises its production server/retrieval contracts with fixtures.
- This verifies application behavior and integration contracts, not production
  model quality, real dataset performance, or an actual deployment.
- The new Linux browser CI job is configured but has not been run on the remote
  CI service in this workspace.

### Before Announcing Availability

- Run the documented curl login/search/match/logout flow against the intended
  private deployment with a real existing image and ready configured model.
- Confirm that deployment-specific image sizes fit the configured limits.
- The one existing skipped media-pagination baseline remains outside this change.
- Automated coverage is evidence for the workflows listed above, not proof about
  every possible UI interaction or dataset.

### Transport

- The app's network binding and plaintext HTTP transport are unchanged.
- Browser cookies deliberately omit `Secure` until the HTTPS phase.
- Session authentication and filesystem containment do not encrypt network traffic.
