# Shark Matcher App — Browser Cache & Reference-Data Audit

> **Status:** Historical as-found audit with a current disposition summary.

- The Implementation Status table records the V1 outcome.
- The detailed request counts, source locations, examples, and recommendations
  below are investigation evidence, not a live reference manual.
- When they differ, the status table and current source take precedence.

## Scope

Audit of how reference data (location options, model lists, match methods, user
names, selector options) travels between the server and the browser, how it is
cached in `localStorage`, and how often it is re-sent.

Companion to `performance-audit.md`, which covers the database layer: that
document is about SQL round trips, this one is about client<->server traffic and
browser storage.

## Implementation Status

| ID | Description | Status | Notes |
|---|---|---|---|
| C1 | `/api/data/stats` recomputed from scratch on every call | **Done** | Server-side TTL cache in `frontend/project.py`; unused fields dropped from the payload |
| C2 | Model/method selectors refetch stats unconditionally | **Done** | `util.js` selectors now route through the shared cache |
| C3 | Logout clears only credentials | **Done** | `clearAllLocal()` in `core.js`, sweeping by key prefix; also wired into the session-expiry path |
| C4 | Stored filters override URL params and leak across tabs | **Done** | URL is now the source of truth (`Util.hasFilterParams`); filters stored per tab under `SHARKMATCHER.FILTERS.<page>` |
| C5 | Unguarded `JSON.parse` / `.length` with no recovery path | **Done** | `readCacheEntry` drops corrupt entries instead of throwing; `getFromLocal` deleted |
| C6 | Five inconsistent cache patterns | **Done** | All reference data now goes through `Core.getCached(key, ttl, fetcher)` |
| C7 | `users` cache never refreshes after login | **Done** | 5-minute TTL, refreshed by `handlerHome` on every navigation |
| C8 | No server-side name validation; no HTML escaping | **Done** | `app_utils.valid_name()` on all four write endpoints; `Util.escapeHtml()` on every render site |
| C9 | Full staff directory (incl. emails) in every viewer's browser | **Done** | `User.to_list_dict()` never returns an email; username/role go to admins only |
| C10 | No cache headers on `/static/` | **Done** | `no-cache` + `Last-Modified`/304 revalidation, shared with the image handler |
| C11 | `frontend/project.py` uses `backend.db.media` without importing it | **Done** | Explicit import added |
| C12 | Dead storage and dead files | **Done** | `MATCH_PARAMS`, `clearCredentials`, `sjclOLD.js` all removed |

---

## Table of Contents

1. [Five Caching Patterns, No Two Alike](#1-five-caching-patterns-no-two-alike)
2. [Redundant Traffic](#2-redundant-traffic)
3. [Bugs](#3-bugs)
4. [Safety](#4-safety)
5. [Recommended Order](#5-recommended-order)
6. [Request Counts After Items 1–5](#6-request-counts-after-items-15)

---

## 1. Five Caching Patterns, No Two Alike

**Resolved (C6).** As found:

| Key | Written by | TTL | Refresh trigger | Cleared on logout? |
|---|---|---|---|---|
| `label_classes` / `location_classes` / `tag_classes` | `util.js:690 setSelectorOptions` | 5 min via `last_selector_update` | every nav + `forceUpdate` on edit | No |
| `users` | `util.js:743 setLocalUsers` | **none** | **login only** | No |
| `unembedded_items` | `util.js:770` + `home.js:772` | 5 min via `last_unembedded_update` | every nav | No |
| `loaded_models` / `match_methods` | `util.js:1325,1388` + `home.js:779,783` | **none** | **unconditional refetch on every render** | No |
| `SHARKMATCHER.CREDENTIALS` / `FILTERS` / `MATCH_PARAMS` / `THEME` | `core.js` | n/a | n/a | credentials only |

Further inconsistencies, all since removed:

- **Naming.** `SHARKMATCHER.*` for some keys, bare names for others.
- **Storage shape.** `setToLocal` wrapped values in a redundant envelope
  (`localStorage['label_classes'] = '{"label_classes":[…]}'`) while
  `loaded_models`, `match_methods`, and `unembedded_items` were stored as bare
  arrays.
- **Two theme systems.** `SHARKMATCHER.THEME` is read by an inline script in
  `index.html:17`; `highContrastMode` is read by `home.js:30`. They remain
  independent — both are device preferences, neither holds user data.

### As It Stands Now

#### The Cache Primitive

All reference data goes through one primitive in `core.js`:

```js
getCached(key, ttlMs, fetcher)   // async: cache when fresh, fetch otherwise
getCachedValue(key)              // sync read for render paths
clearCache(key)                  // targeted invalidation
```

It owns, for every cache:

- the storage key (`cache.<name>`) and the envelope (`{time, value}`);
- expiry and in-flight collapsing;
- corrupt-entry recovery.

#### The Registry

On top of it, `util.js` holds a registry. The only thing that varies per cache is
its name and its fetcher:

```js
const CACHE_FETCHERS = {
    'selectors':     () => fetchSelectorOptions(),
    'users':         () => fetchUsers(),
    'dataset_stats': () => SharkMatcher.get_dataset_stats(),
};
```

#### Derived Operations

Three operations are derived from it, so every cache supports all three
identically and adding a cache is one line:

| Operation | Behaviour on fetch failure | Use |
|---|---|---|
| `loadCache(name, force?)` | **rejects** | Foreground read; caller surfaces the error |
| `refreshCache(name, ctx, force?)` | logs, **swallows** | Background warm on navigation |
| `clearCache(name)` | n/a | Mark stale, let the next reader refetch |

#### Reject Versus Swallow

The reject-vs-swallow split used to be a property of each *cache*:

- Stats had a bespoke `getDatasetStats` that rejected.
- Selectors and users had `refreshX` wrappers that swallowed.

It is really a property of the *call site*: a background warm must never abort a
render, and a foreground read must not fail silently. Naming the two operations
instead of the two caches made all three even. `handlerHome` now reads as three
identical lines.

#### The Three Caches

An unknown cache name throws rather than silently caching under a key nothing
reads.

| Cache key | Fetcher | TTL | Refreshed by | Sync readers |
|---|---|---|---|---|
| `cache.selectors` | utility individual (id −1) | 5 min | `handlerHome`, forced on selector edit | `getSelectorOptions` |
| `cache.users` | `user/filter` | 5 min | `handlerHome`, login | `getLocalUsers` |
| `cache.dataset_stats` | `data/stats` | 5 min | `handlerHome`, both selectors, home overview | `getUnembeddedImagesCache`, `getCachedLoadedModels`, `getCachedMatchMethods` |

Search filters live under `SHARKMATCHER.FILTERS.<page>`, one entry per tab.
`clearAllLocal()` sweeps both prefixes, so a new cache or a new filtered tab is
covered without touching the logout path.

---

## 2. Redundant Traffic

### `/api/data/stats` Is the Hot Spot

`frontend/project.py:83 _get_dataset_stats()` does, per call:

- 5 count queries against `Individuals` and the associated-object tables
- a **full `Media` table scan** (`backend.db.media.get_medias(get_full=False)`)
- a **cross-service HTTP call** to the inference service returning every cached
  filename

Nothing cached it server-side.

#### Cost per Tab

Every tab handler calls `Home.handlerHome()` first (`search.js:14`,
`match.js:16`, `settings.js:12`, `upload.js:15`, `history.js:15`,
`proposal.js:15`, `individual.js:13`, `merge.js:21`), which fires
`setSelectorOptions` + `setUnembeddedImagesCache`.

| Tab | Warm cache (<5 min) | Cold cache |
|---|---|---|
| Search / Upload / History / Proposals | 0 | 1 stats + 1 utility-individual |
| Home (`path === ''`) | **1 stats** | 2 stats + 1 utility-individual |
| **Match** | **2 stats** | **3 stats** + 1 utility-individual |
| Settings | 1 utility-individual + 1 filter_users | 2 stats + 2 utility-individual + 1 filter_users |

The Match tab is the worst case: `renderModelSelector` (`util.js:1386`) and
`renderMethodSelector` (`util.js:1323`) each call `get_dataset_stats()`
unconditionally, ignoring the cache entirely, to read two small fields.

### Most of the Payload Is Never Read

These fields have **zero** readers in the frontend:

- `embedded_image_names` (one entry per embedded image) and `embedded_image_ids` (always `[]`);
- `unembedded_images` and `media_without_embedding_count`;
- `archive_individuals`, `total_individuals`, and `update_time`;
- the `AlternateNames` / `Tags` / `Labels` counts.

Relatedly, `base.js:46 get_dataset_stats()` sends a payload of flags
(`'total_individuals': true`, …) that `project.py:63` ignores completely — the
field-selection contract was stubbed and never wired up.

### The Utility-Individual Fetch Has the Same Shape of Waste

`get_individuals(-1, …, get_full=true)` returns full objects (`id`,
`individual`, `name`, `type`, `time`) when `setToLocal` keeps only `name`.
`Individual.to_dict(compact=True)` already exists for exactly this and is only
ever used for action logging (`frontend/action.py:216`).

### Auth Amplifies Every Redundant Call

Every API request runs `authenticate_user` before the target function, which
argon2-hashes the token (`backend/db/user.py:440`).

- Measured at **22.8 ms and 64 MiB per call** on an M-series Mac; likely 50–100 ms on a server VM.
- `frontend/user.py:277` then re-runs `get_users()` for the same user that
  `backend/db/user.py:432` already fetched.

So each redundant stats call costs an argon2 hash, a duplicate user lookup, a
full media scan, and an inference round-trip.

---

## 3. Bugs

### C4 — Stored filters override URL params and leak across tabs — Fixed

`search.js:249-252` as found:

```js
const filters = SharkMatcher.getFilters();
if (filters !== undefined) { params = filters; }
```

`renderSearchFilters` is shared by search, match, and review, and all three
write the *same* `SHARKMATCHER.FILTERS` key (`search.js:104`, `match.js:121`,
`review.js:126`).

#### Consequences

Two consequences:

1. Filters set on the match tab appear on the search tab.
2. Because `executeSearch` still uses the *original* `params` from the URL, the
   filter popover displays different values than the ones that produced the
   results. A bookmarked or shared search link renders with the wrong filters
   shown.

#### Fix

- The URL is now the source of truth. Stored filters are consulted only when the
  path carries none, via `Util.hasFilterParams(params)`.
- Each tab stores its own set under `SHARKMATCHER.FILTERS.<page>`, so search,
  match, and review no longer overwrite one another.
- `setFilters(page, filters)` also replaced a 15-positional-argument signature
  with an object.

### C5 — Unguarded parse/access with no recovery path — Fixed

As found:

- `setSelectorOptions` at `util.js:699` did `localLabels.length > 0`, where
  `getFromLocal` returns `undefined` for a missing key.
- `core.js:74 JSON.parse(data)` was unwrapped.
- Both throw *synchronously*, and `home.js:43` calls it fire-and-forget with no
  `try`/`catch`. A throw aborted `handlerHome` and the page rendered blank,
  unrecoverable except by manually clearing storage.

Note the inconsistency: `getUnembeddedImagesCache` (`util.js:762`),
`match.js:23`, and `util.js:1302` all *did* wrap `JSON.parse` in `try`/`catch`.
Only this path did not.

**Fix.**

- `readCacheEntry` is the single parse site. It catches, drops the bad entry,
  and returns null, so a corrupt value degrades to a cache miss instead of
  aborting a render.
- `getFromLocal`/`setToLocal` are gone.
- `getFilters` got the same treatment.

### C11 — `frontend/project.py` uses `backend.db.media` without importing it — Fixed

Line 118 calls `backend.db.media.get_medias(...)`; the imports at lines 16–23 did
not include it. It worked only because `backend.db.individual` (imported at line
17) imports `backend.db.media` at `backend/db/individual.py:9`, binding the
attribute on the package. Removing that transitive import would have broken
`/api/data/stats` at runtime, so an explicit import was added.

### C7 — The users cache never refreshes — Fixed

`setLocalUsers` was called only at `login.js:141`, with no TTL and no
`forceUpdate`. A user created via Settings would not appear in the author
dropdowns (`proposal.js:334`, `history.js:260,268`) for anyone until they logged
out and back in.

**Fix.** The users cache is an ordinary `getCached` entry with a 5-minute TTL,
refreshed by `handlerHome` on every navigation. `fetchUsers` also narrows each
record to `{id, name}` — the only fields any reader uses — so emails and roles
are no longer persisted in the browser (partial C9).

### C12 — Dead storage — Fixed

Three dead items:

- `SHARKMATCHER.MATCH_PARAMS` was write-only. `setMatchParams` was called at
  `match.js:129`; `getMatchParams` had no readers, and `match.js:20` explicitly
  says stored params are not used because they go stale when the server is
  reconfigured. So every match wrote to `localStorage` and nothing ever read it.
- `clearCredentials` (`core.js:193`) fell off the end and returned `undefined`
  on its default `sendDelete=true` path, a vestige of a server-side delete that
  was never built. It had **no callers at all**: logout goes through
  `clearAllLocal()`, which clears `CREDENTIALS_KEY` via `SESSION_KEYS`.
- `sjclOLD.js` (28 KB) sat next to `sjcl.min.js` and was never imported.

**Fix.** All three removed, along with the now-orphaned `MATCH_PARAMS_KEY`,
`METHOD_KEY`, and `MODEL_KEY` constants and their re-exports in `base.js`.

---

## 4. Safety

### C9 — Every viewer's browser holds the full staff directory, including emails — Fixed

#### Problem

- `/api/user/filter` is `min_role_required='viewer'` (`server.py:62`), and the
  users fetch runs at login for *every* role.
- `to_secure_dict` correctly strips password and salt, but includes `email` and `role`.
- `settings.js:106` only *displays* the user list to admins (`userRoleValue >= 2`),
  so the UI implies viewers should not have this, but they already do, persisted
  indefinitely.
- Neither `email` nor `role` is ever read from that cache (only `id` and `name`,
  via `getLocalUsersFormatted` and `getAuthorUsername`).

#### Fix, Client Side

`fetchUsers` discards everything but `id` and `name` before caching, so emails no
longer sit in browser storage.

#### Fix, Server Side

`User.to_list_dict()` is a second serializer used only by `filter_users`.

- It never returns an email.
- It returns `username`/`role` only to admins, matching `settings.js:106`, which
  already gates the user-list UI at role >= 2.
- Viewers still get `id` and `name`, which is what author attribution on the
  history and proposal tabs needs.

`to_secure_dict` was deliberately **not** narrowed:

- It is shared by `get_user`, `post_user` and `login_user`, all of which return a
  single user to themselves or an admin and legitimately need the email.
- `login_user` in particular feeds the stored credentials that render the
  self-profile's email row (`settings.js:186`) and are posted back on a
  self-update (`settings.js:156`). Narrowing it in place would have broken both
  silently.

#### Residual, by Design

A viewer can still enumerate every user's `id` and display `name`. That is
inherent to showing "proposed by <name>" on pages viewers can see; removing it
means dropping author attribution for viewers, which is a product decision rather
than a security fix.

The route was left at `viewer` rather than raised to `admin` for the same reason.
`getAuthorUsername` and `getLocalUsersFormatted` power the author column and
author filter on History and Proposals, both viewer-accessible, and cutting
viewers off would render raw numeric author IDs there.

### C3 — Nothing is cleared on logout — Fixed

`login.js:78 executeLogout` removed only `SHARKMATCHER.CREDENTIALS` and
`userMode`. The user directory, all location/label/tag options, unembedded image
names, models, and saved filters survived logout — so on a shared workstation the
next person inherited the previous user's cached data.

**Fix.** `clearAllLocal()` clears the three explicit session keys and then sweeps
every key under the `cache.` and `SHARKMATCHER.FILTERS.` prefixes.

- It runs on explicit logout and on session expiry (`routes.js:124`).
- Theme and high-contrast are intentionally preserved: they hold no user data,
  and wiping an accessibility setting on logout would be hostile.

### C8 — Stored-XSS -> token theft chain — Fixed

#### Problem

- The auth token lives in `localStorage` (readable by any script).
- There was **no HTML-escaping helper anywhere** in the codebase.
- Reference data was interpolated straight into `innerHTML` in ~75 places
  (`util.js:616` `${item.name}`, `home.js:804`
  `media_without_embedding_names.join(', ')`, and throughout).
- `validName` (`util.js:810`) enforced `^[a-zA-Z0-9-_ ]*$` **on the client
  only**. `frontend/individual.py:811 post_object` did no name validation at
  all, so a direct API call stored whatever it was given, and it then executed
  in every user's browser.

#### Fix, Server Side

`app_utils.valid_name()` is the enforcing check:

- It allows letters from any script, digits, spaces, and the punctuation real
  names need (`- _ ' . , ( ) /`), and caps length at 128.
- It refuses the characters that make injection work: `< > " & ; = ` `` ` ``
  `` $ { }`` and control characters.
- It runs in `post_individual`, `post_object`, `post_media`, and `post_user`
  before any write, returning `400` and leaving the stored record untouched.

The rule was **widened** from the original client regex specifically so that no
existing record is rejected:

- All 12,349 names across `Individuals`, `AlternateNames`, `Labels`, `Tags`,
  `Media`, and `Users` pass.
- That includes `AlternateNames` id 74, `I'M JUST KEN`, the only row using
  punctuation outside the original set.
- `validName` in `util.js` was updated to the same rule so client and server
  agree; it remains advisory, for instant feedback while typing.

#### Fix, Client Side

`Util.escapeHtml()` escapes `& < > " '`. It is applied at
every site that interpolates user-derived text into markup:

- Two shared helpers escape their own output, covering all their call sites:
  `displayValue()` (17 sites) and `getAuthorUsername()` (2 sites).
- `listToLinks()` escapes once, covering all six `*Links` fragments in `home.js`
  and `individual.js`.
- 42 direct sites across `home.js`, `settings.js`, `search.js`, `individual.js`,
  `review.js`, `match.js`, `history.js`, `proposal.js`, and `util.js` — names,
  filenames, usernames, emails, selector options, and the `value="…"` attributes
  built from URL params (`params.keywords`, `params.tags`, `params.name`,
  `params.alternate_names`), which were a reflected-XSS vector via crafted links.

Deliberately **not** escaped:

- `alert()`, `confirm()`, and `console.debug()` arguments (`settings.js:635`,
  `util.js:629,633`, `upload.js:167`, `routes.js:66`). Those are text contexts,
  and escaping there would show users a literal `&#39;`.
- Link hrefs: `Core.formHashPath` builds them through `URL.searchParams`, which
  percent-encodes.

#### Out of Scope, by Decision

The token stays in `localStorage`; it is **not**
being moved to an `HttpOnly` cookie.

- That change would sever the exfiltration half of the chain outright, since
  script could not read the token at all.
- It touches login, logout, session expiry, and every authenticated request, so
  it is a separate piece of work.
- Until then, escaping and validation are what stand between a stored name and a
  stolen session.

### C10 — No cache headers on `/static/` — Fixed

#### Problem

`handlers.py:134 _serve_file` set only `Content-Type`: no `Cache-Control`,
`Last-Modified`, or `ETag`. `frontend_images` did it properly with
`Cache-Control` + `Last-Modified` + 304 handling.

- **428 KB** of JS/CSS (387 KB JS, 40 KB CSS, 1 KB HTML) was re-fetched on every
  full page load.
- There are **16 `location.reload(true)` calls** (after every object
  save/delete, mode change, embeddings update). The `true` argument is
  deprecated and ignored by all modern browsers.

#### Fix

Static assets are served with `Cache-Control: no-cache` plus `Last-Modified`, and
`_serve_file` answers a matching `If-Modified-Since` with a bodyless `304`.
`no-cache` does not mean "do not store"; it means "store, but revalidate before
every use". The 428 KB stops moving while a deploy can never be served stale.

#### Rejected: Plain `max-age`

- Assets are referenced by unversioned paths (`index.html:15` ->
  `/static/js/main.js`, no build hash).
- Modern browsers revalidate only the top-level document on a normal reload, not
  subresources.
- Under `max-age=3600` a user could run pre-deploy JS against a post-deploy
  server for an hour, and `location.reload()` would not fix it.
- Hashed filenames plus `immutable` would be faster still, but need a build step
  this app does not have.

#### Not Affected: Reload After Save

Those 16 reloads exist to refresh *data*, and all data moves over `POST /api/…`
(`sharkMatcher/core.js:317`), which browsers do not cache. Static asset caching
and data freshness do not intersect.

#### Shared With the Image Handler

`_file_mtime()` and `_client_copy_is_current()` are now used by both
`_serve_file` and `frontend_images`, so the conditional-request logic exists
once. The two differ only in policy, stated as named constants:

- `STATIC_CACHE_CONTROL = 'no-cache'`: unversioned, must revalidate.
- `IMAGE_CACHE_CONTROL = 'public, max-age=3600'`: large, rarely changed,
  addressed by content-bearing filename.

The image policy's tradeoff is that replacing an image under an existing filename
leaves viewers on the old one for up to an hour.

---

## 5. Recommended Order

| # | Fix | Addresses | Breaking? | Status |
|---|---|---|---|---|
| 1 | Server-side TTL memoization for `_get_dataset_stats()`; drop unread fields from the payload | C1 | No | Done |
| 2 | Route the model/method selectors through the shared TTL cache instead of unconditional refetch | C2 | No | Done |
| 3 | `clearAllLocal()` driven by a single key registry; call it in `executeLogout` | C3 | No | Done |
| 4 | Prefer URL params over stored filters; namespace stored filters per tab | C4 | No | Done |
| 5 | Consolidate the five cache patterns into one `getCached(key, ttl, fetcher)` helper | C5, C6, C7 | No | Done |
| 6 | Server-side name validation in `post_object`; `escapeHtml` helper on interpolated names | C8 | No | Done |
| 7 | `Cache-Control` + `Last-Modified`/304 in `_serve_file`, matching `frontend_images` | C10 | No | Done |
| 8 | Narrow `/api/user/filter`'s response for non-admins, or raise the route above `viewer` | C9 | No | Done |
| 9 | Delete `MATCH_PARAMS` and `sjclOLD.js`; unify the cache operations | C12 | No | Done |

All items are complete. Moving the token out of `localStorage` and into an
`HttpOnly` cookie would break the C8 exfiltration chain entirely, but has been
**explicitly deferred**; `localStorage` remains the token store for now.

---

## 6. Request Counts After Items 1–5

Reference-data requests per navigation, warm cache in parentheses:

| Tab | Before | After |
|---|---|---|
| Search / Upload / History / Proposals | 2 (0) | 3 (0) |
| Home | 3 (1) | 3 (0) |
| Match | 4 (2) | 3 (0) |
| Settings | 4 (2) | 4 (1) |

### Client Side

- Cold-cache counts barely move: the work is genuinely needed on a first load,
  and the users fetch moved from login into the navigation path (C7), adding one.
- What changed is the warm path. Every tab now costs **zero** reference-data
  requests inside the TTL, where the match tab previously spent two on every
  visit and home one.
- Concurrent callers on a cold load also collapse into a single request each, so
  the match tab issues one `data/stats` rather than three.

### Server Side

- A `data/stats` rebuild that does get through now costs 5 fewer SQL queries
  (the three associated-object counts plus a smaller payload).
- Repeat calls within 30 seconds cost nothing beyond auth.
- `authenticate_user` now stashes the verified `User` on the per-request handler,
  and `filter_users`, `post_user` and `delete_user` read it instead of
  refetching the same row, saving one query on each.

### Remaining Duplicate

`backend.db.user.authenticate_user` fetches the user, then
`frontend.user.authenticate_user` fetches it again. This is **not** addressed:
collapsing it would mean changing the DB function's boolean return contract,
which four existing test assertions depend on, so it was left for a separate
change.
