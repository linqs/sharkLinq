# Direct Search and Matching API

Phase 1 exposes the existing search and match results to authenticated terminal
clients. Use the **application server's** address. The matcher service and its
service credential remain an internal application dependency.

## Login and Call the API

### Accounts and Sessions

- Existing accounts and roles apply.
- Login creates the same seven-day, revocable session used by the web app.
- There is no separate API key or account type.
- The login contract accepts the hexadecimal SHA-256 derivative of the
  password, exactly as the browser does. Treat that derivative as a credential.

### Log In

The example uses Python 3 for the password prompt/JSON and curl for HTTP. Set
the application address and your username first:

```bash
export SHARK_URL='http://server-host:12345'
export SHARK_USER='your-username'

SHARK_TOKEN=$(
  python3 -c 'import getpass, hashlib, json, os; print(json.dumps({"username": os.environ["SHARK_USER"], "password": hashlib.sha256(getpass.getpass("Password: ").encode()).hexdigest()}))' |
  curl --silent --show-error --fail-with-body \
    -H 'Content-Type: application/json' \
    --data-binary @- "$SHARK_URL/api/user/login" |
  python3 -c 'import json, sys; result = json.load(sys.stdin); content = result.get("content"); sys.exit("Login failed") if not isinstance(content, dict) or not content.get("token") else print(content["token"])'
)
```

Check that login succeeded before continuing:

- An incorrect password keeps the browser's existing `200` response with
  `content: false`; no token is issued.
- Repeated failed authentication attempts return `429` and `Retry-After`; a successful login clears the token failures recorded from your address.
- Do not put credentials in URLs or save tokens in shared scripts or logs.

### Call the API

```bash
# Search/filter sharks. --get is required: curl otherwise uses POST with --data.
curl --silent --show-error --fail-with-body --get \
  -H "Authorization: Bearer $SHARK_TOKEN" \
  --data-urlencode 'sex=F' \
  --data-urlencode 'min_length=10' \
  --data-urlencode 'paginate=true' \
  --data-urlencode 'page=1' \
  --data-urlencode 'page_size=50' \
  "$SHARK_URL/api/individual/filter"

# Match an image already in the database, then filter the ranked results.
curl --silent --show-error --fail-with-body --get \
  -H "Authorization: Bearer $SHARK_TOKEN" \
  --data-urlencode 'image_name=existing-image-name' \
  --data-urlencode 'sex=F' \
  "$SHARK_URL/api/match/filter"

# Revoke this session when finished.
curl --silent --show-error --fail-with-body \
  -H "Authorization: Bearer $SHARK_TOKEN" \
  -H 'Content-Type: application/json' --data '{}' \
  "$SHARK_URL/api/user/logout"
unset SHARK_TOKEN
```

- `image_name` is the database image name (`Media.name`), not a filesystem
  path, shark name, or uploaded image.
- The API looks it up explicitly, regardless of name length.
- Missing names return `404`; ambiguous names return `409`.

### Transport

- HTTPS, proxy configuration and network binding are unchanged in this phase.
- HTTP does not encrypt passwords, derivatives, tokens, or results in transit;
  use the intended private network.
- Transport encryption is phase 2.

## Query Parameters

Both GET endpoints accept these existing filters.

- Omit unused parameters rather than sending empty values.
- Use `--data-urlencode` for spaces and reserved URL characters.
- Unknown parameters, repeated scalar parameters, invalid numbers, invalid
  booleans, and reversed ranges return `400`.

| Parameter | Value |
| --- | --- |
| `sex` | `F`, `M`, `Unknown` (legacy `U` also accepted) |
| `allow_unknown` | `true` or `false`; existing filter defaults apply when omitted |
| `min_length`, `max_length` | Numeric length bounds in the dataset's units |
| `min_time`, `max_time` | Numeric Unix timestamp bounds, in seconds |
| `min_associated`, `max_associated` | Nonnegative integer associated-image bounds |
| `locations`, `qualities`, `max_qualities` | Existing location/quality filter values |
| `alternate_names`, `tags`, `labels` | Existing nickname, tag, label filter values |
| `ids`, `names`, `keywords` | Existing shark IDs, names, or keyword filters |

List parameters accept a comma-separated value or repeated parameters
containing one value each, e.g. `locations=AN&locations=FAR`.

- Do not combine repeated values with comma-separated values.
- Filtering, unknown-value handling, ranking and response formatting are shared
  with the web app.

### Search Parameters

Search additionally supports:

| Parameter | Value |
| --- | --- |
| `paginate` | `true` enables pagination; omitted/`false` returns the original full list |
| `page` | Positive integer, default `1` |
| `page_size` | Positive integer, default `50`, up to the `SHARK_MAX_PAGE_SIZE` limit under "Limits" |
| `attribute` | `name` (default), `id`, `associated_medias` or `associated images`; existing fallback to `name` applies |
| `order` | `ascending` (default) or `descending` |

### Matching Parameters

Matching additionally supports:

| Parameter | Value |
| --- | --- |
| `image_name` | Required exact database image name |
| `method` | `nearest neighbor` (default) or `nearest centroid` |
| `model` | Configured matcher model name/alias; omitted uses the matcher's default |

- Matching retains the full ranked result format; it does not introduce pagination.
- GET requests are read-only and do not add images or regenerate embeddings.

## Responses and Errors

The existing envelope and its `content` are preserved:

```json
{
  "success": true,
  "message": null,
  "endpoint": "individual/filter*",
  "content": [],
  "server-version": "..."
}
```

- The literal `*` in legacy endpoint labels is retained for compatibility; it is
  not part of the URL.
- Search `content` is the existing list, or the existing pagination object with
  `results`, `total_count`, `page`, `page_size`, `has_more`.
- Match `content` retains the existing query and ranked individual objects.
- Check the HTTP status and the envelope; a valid empty result remains successful.

### Status Codes

| Status | Meaning |
| --- | --- |
| `400` | Invalid query/body (a missing, unsupported or wrongly typed field, named in the message), unknown matching option, unsafe filename |
| `401` | Missing, invalid, expired or conflicting session credentials; checked before any field |
| `403` | Authenticated account lacks permission, including managing another account |
| `404` | Unknown route, query image, or record identifier |
| `405` | Wrong HTTP method; `Allow` lists permitted methods |
| `409` | Ambiguous image name or existing operation conflict |
| `413`, `415` | Body/image/batch limit exceeded, or unsupported content type |
| `429` | Authentication attempt limit, or an embeddings update within an hour of the last app-started update; observe `Retry-After` |
| `500` | Unexpected application failure, with a generic message |
| `502`, `503`, `504` | Invalid/failed matcher response, unavailable matcher/model, or timeout |

## Shared Endpoint Protections and Browser Compatibility

### Routes and Methods

- All existing data routes still require account authentication.
- The public exceptions are login, runtime UI configuration, logout's idempotent
  empty call, and packaged browser assets. `user/auth` verifies credentials itself.
- Existing write operations use POST with JSON.
- GET is supported only for search, filtered matching, embeddings update
  status, runtime configuration, and static reads.
- Each endpoint accepts only its own fields plus the session fields; any other
  field is refused with `400` instead of being ignored.
- Route names match exactly; accidental suffixes no longer reach handlers.

### Authorization

- Authorization headers work alongside the browser's existing POST credentials.
- Conflicting header/body identity is rejected.
- Acting user IDs and usernames come from the verified session, so a submitted
  ID cannot impersonate another author.
- The server checks self-account updates, role permissions, proposal status
  changes (admin or above), and proposal deletion (author only).
- Demo permissions and isolation continue through the existing demo dispatcher.

### Image Cookie and Caching

- Login and startup authentication provision an image-only, host-only cookie:
  `HttpOnly; SameSite=Lax; Path=/static/frontend-images/`.
- It expires with the session and is cleared/revoked on logout.
- API handlers do not accept this cookie as authentication; image reads also
  accept a Bearer token.
- Protected images and API responses use `Cache-Control: no-store`; packaged
  assets retain revalidation.
- Image URLs carry a new cache version so browsers do not reuse old public images.
- The cookie's `Secure` flag is deferred with HTTPS.

### Filesystem Safety

- Operations reuse the configured image and export roots, which remain the path
  authorities; no new directory layout is required.
- Traversal, absolute paths, backslashes, encoded traversal and descendant
  symlinks are rejected for served files.
- Image writes require a single safe filename.
- Public packaged assets are limited to browser file types.
- API errors do not expose raw exceptions or server filesystem paths.

### Limits

Limits are configured through the existing constants/environment boundary:

| Environment variable | Default |
| --- | --- |
| `SHARK_MAX_JSON_BYTES` | 4 MiB |
| `SHARK_MAX_UPLOAD_BYTES` | 256 MiB for `media/save` and `media/upload_batch` |
| `SHARK_MAX_IMAGE_BYTES` | 25 MiB per decoded image |
| `SHARK_MAX_IMAGE_PIXELS` | 50,000,000 |
| `SHARK_MAX_UPLOAD_ITEMS` | 100 |
| `SHARK_MAX_PAGE_SIZE` | 1000 for individual search |

- Set positive integer values before startup if the dataset requires different limits.
- Batch validation happens before writes and reuses the existing atomic upload
  processing and image conversion.
- This phase adds no database schema migration; existing session-schema
  migration requirements still apply.

## Verification

The existing deterministic test runner remains the default. Browser workflows
have a separate CI job. It reuses:

- the production HTTP test harness;
- a disposable copy of the committed database fixture;
- generated test images;
- deterministic inference responses over real HTTP.

```bash
python3 -m tests.tools.run_tests
python3 -m tests.tools.run_tests --py-style --js-style
python3 -m playwright install chromium firefox webkit
python3 -m tests.tools.run_tests --browser
```

- Install development requirements before running browser tests.
- On Linux, use `python3 -m playwright install --with-deps chromium firefox webkit`
  to install browser system dependencies as CI does.
- Failure traces/screenshots go to `test-results/` and are uploaded by the browser CI job.
- Run app test jobs in separate checkouts or sequentially: the test database
  initializer owns one disposable database per checkout.

See `docs/api-verification.md` for executed checks and limits of the evidence.
Live-model and deployment validation remain separate from deterministic
browser regression testing.
