# Frontend CLI

The command-line runner that starts the web server and its weekly backup process.

## Modules

| File | Responsibility |
| --- | --- |
| `serve.py` | Parses the command line, starts the backup process, optionally opens a browser, and runs `frontend.server`. |
| `__init__.py` | Marks the package. |

## Running

`serve.py` takes one required argument, the project directory, which must exist.

```bash
python3 -m frontend.cli.serve --help
python3 -m frontend.cli.serve . --no-browser --host 127.0.0.1
```

| Option | Effect |
| --- | --- |
| `PROJECT_DIR` | Required directory the server is started for; the orchestrator passes `.`. |
| `--port PORT` | Port to serve on; default is `DEFAULT_PORT` in `core/app_constants.py`. |
| `--host HOST` | Interface address to listen on; default is every interface. |
| `--no-browser` | Do not open a browser window. |
| `--demo` | Enable the shared Demo mode. |

Most operators use `python3 -m scripts.shark_identification --run`, which calls this module with `--no-browser`.

## Behavior

- Starts `db/scripts/backup_runner.py` as a background process before serving.
- On Ctrl-C or `SIGTERM`, stops the server and terminates the backup process.
- Prints the server address on start.
- The server exits if the database is missing or outdated, but only warns if the inference service is unreachable.

## Boundaries

- Imports `core`, `frontend.server`, and `db.scripts.backup_runner`.
- Holds no request handling; see `frontend/README.md`.
- Tests live in `tests/frontend/test_server_lifecycle.py`.
