# Frontend

The HTTP layer: the web server, request routing, authentication, API handlers, and the browser interface it serves.

## Modules

| Path | Responsibility |
| --- | --- |
| `server.py` | Defines the `ROUTES` table and the request handler class, exits at startup if the database is missing, uninitialised, or on the legacy session schema, and runs the threaded HTTP server. |
| `common.py` | Route building, request dispatch, response formatting, and `APIError`. |
| `handlers.py` | Static-asset serving, redirects, and authenticated image serving from the images directory. |
| `user.py` | Login, logout, session authentication, role checks, user administration, and the authentication-attempt limiter. |
| `individual.py` | Individual and associated-object endpoints, including merge, unmatch, and reference updates. |
| `media.py` | Media endpoints, file storage, and batch uploads. |
| `match.py` | Candidate matching and match filtering. |
| `proposal.py` | Proposing, reviewing, and deleting database changes. |
| `action.py` | Reading the audit log and recording actions. |
| `review.py` | Unmatch and merge suggestions from the inference service; its route is currently commented out in `server.py`. |
| `project.py` | Dataset statistics, embedding updates, and data export. |
| `inference_client.py` | HTTP client for the `shark-match-model` inference service. |
| `cli/` | Command-line server runner; see `frontend/cli/README.md`. |
| `static/` | Browser interface assets; see `frontend/static/README.md`. |
| `package.json`, `package-lock.json`, `eslint.config.js` | Node tooling for the JavaScript lint gate only. |
| `__init__.py` | Marks the package. |

## Running

Start the server through the orchestrator or the module runner. Both are documented in `scripts/README.md` and `frontend/cli/README.md`.

```bash
python3 -m scripts.shark_identification --run --host 127.0.0.1
```

The JavaScript lint gate needs Node.js and the dependencies pinned in `package-lock.json`:

```bash
cd frontend
npm ci
cd ..
python3 -m tests.tools.run_tests --js-style
```

## Request Flow

1. `server.py` matches the URL against `ROUTES`.
2. `common.py` authenticates the caller with `user.py` and checks the route's minimum role.
3. A handler module reads or writes through `backend/db/` and returns a status and JSON body.
4. Embedding updates and matching call the inference service through `inference_client.py`.

Roles, lowest to highest: `viewer`, `collaborator`, `admin`, `dev`, `owner`. The role each route requires is set in `ROUTES`.

## Boundaries

- Handlers call `backend/db/` for data and never run SQL files directly.
- Naming pairing: `frontend/` modules are HTTP handlers and `backend/db/` modules are data access. See the Layout and Naming Pairing section of `db/README.md`.
- `server.py` also mounts the optional demo routes from `demo/`.
- `cli/serve.py` starts the weekly backup process from `db/scripts/backup_runner.py`.
- `project.py` uses `backend/inference_snapshot.py` and the export code in `db/scripts/db_dump.py`.
- Tests live in `tests/frontend/`; browser workflows live in `tests/browser/`.
