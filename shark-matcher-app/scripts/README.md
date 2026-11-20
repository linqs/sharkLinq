# Scripts

Operator entry points that start the app and schedule its weekly database snapshot.

## Modules

| File | Responsibility |
| --- | --- |
| `shark_identification.py` | Orchestrator: starts the web server, runs the server against the disposable test database, or takes a database snapshot. |
| `setup_cron.sh` | Interactive installer for a weekly cron job that runs the snapshot. |
| `__init__.py` | Marks the package. |

## Orchestrator

Run it from the repository root as a module. Running with no options prints the help text.

```bash
python3 -m scripts.shark_identification --help
```

| Option | Effect |
| --- | --- |
| `-r`, `--run` | Start the web server (`python3 -m frontend.cli.serve . --no-browser`). |
| `--host HOST` | Listen only on this interface address; default is every interface. |
| `--port PORT` | Serve on this port; default is `DEFAULT_PORT` in `core/app_constants.py`. |
| `--demo` | Offer the shared Demo mode. |
| `-t`, `--test` | Run the server with `SHARK_TEST_MODE=true`, which selects the test database. |
| `--repopulate` | With `--test`, first run `db.database --test init --force` and `populate --all` from the committed CI fixture. |
| `-s`, `--snapshot` | Run `python3 -m db.database dump`, a trusted dump of the live database. |

Examples:

```bash
python3 -m scripts.shark_identification --run --host 127.0.0.1
python3 -m scripts.shark_identification --test --repopulate
```

- Each run appends a line to `shark_identification.log` under `data-out/logs/` in the data root.
- A failing step exits with that step's status code.
- Ctrl-C and `SIGTERM` stop the server and the backup runner; `SIGKILL` cannot be caught and leaves both running, so stop them by hand.
- The server needs `SHARK_INFERENCE_API_KEY` set for matching; see the root `README.md` Getting Started section.

## Cron Installer

`setup_cron.sh` is interactive and changes the user's crontab only after a `y` answer.

```bash
./scripts/setup_cron.sh
```

It does the following:

1. Writes `scripts/cron_wrapper.sh`, a generated and git-ignored wrapper that runs `python3 -m scripts.shark_identification -s`.
2. Creates `data-out/logs/cron_output.txt` inside the checkout for the wrapper's log.
3. Shows the proposed crontab, with a weekly Sunday 00:00 entry in a marked block, and asks for confirmation.
4. Records `SHARK_PYTHON` and, when set, `SHARK_DATA_ROOT` in that block, because cron has a minimal environment.

Full setup, monitoring, and removal steps are in `docs/cron-setup.md`.

## Boundaries

- `shark_identification.py` imports only `core` and starts everything else as subprocesses.
- The server it starts is `frontend/cli/serve.py`; the dump it runs is `db/database.py`.
- Running the file by path (`python3 scripts/shark_identification.py`) fails with `No module named 'core'` unless the app is installed or `PYTHONPATH` includes the repository root.
- Ctrl-C reaches the running step; if only the orchestrator receives it, the step gets SIGINT after a 10-second grace period.
- Tests live in `tests/frontend/test_server_lifecycle.py` and `tests/frontend/test_v1_inference_contract.py`.
