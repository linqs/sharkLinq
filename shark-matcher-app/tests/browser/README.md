# Browser Tests

Real-browser workflow tests that drive the web interface with Playwright against a disposable server.

## Modules

| File | Responsibility |
| --- | --- |
| `conftest.py` | Fixtures: a temporary copy of the CI test database, generated placeholder images, a fixed-response inference stub on a local port, the app server, and a `login` helper. |
| `test_workflows.py` | Workflows for every role: navigation, search and image access, login failure and logout, profile and user management, upload and export, editing and the Demo round, expired sessions, proposals, merge and embedding update, and image actions. |
| `__init__.py` | Marks the package. |

## Running

These tests are outside the standard run. Install the Playwright browsers once, then run the browser matrix with the repository runner:

```bash
python3 -m pip install -r requirements/requirements-dev.txt
python3 -m playwright install chromium firefox webkit
python3 -m tests.tools.run_tests --browser
```

To run one browser, call pytest directly:

```bash
python3 -m pytest tests/browser --browser chromium
```

- `--browser` in the runner runs Chromium, Firefox, and WebKit, keeping a trace and a screenshot for each failure.
- Playwright and `pytest-playwright` come from `requirements/requirements-dev.txt`.
- The CI job named Browser workflows runs the same runner command.

## Isolation

- `tests/conftest.py` points `SHARK_DATA_ROOT` at a temporary directory and sets `SHARK_TEST_MODE`, so no real data is touched.
- The browser blocks requests to Google Fonts so page loads never wait on the network.
- The inference service is a local stub; no `shark-match-model` process is needed.
- The committed CI fixture in `tests/ci-test-data/` is only read, never written.

## Boundaries

- Reuses the server harness from `tests/frontend/test_server.py`.
- Covers the interface in `frontend/static/` and the handlers in `frontend/`.
- Deterministic, non-browser tests are in the other `tests/` folders.
