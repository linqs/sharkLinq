# Test Tools

The `tests/tools` folder holds the repository-owned test runner and the Python lint gate that CI and contributors use.

## Modules

| File | Responsibility |
| --- | --- |
| `run_tests.py` | Unified runner: runs pytest on chosen targets (default `tests/unit` and `tests/contracts`) and optionally the lint gate |
| `python_style_tests.py` | Runs Pylint with `config/pylintrc` in errors-only mode over `src`, `setup.py`, and `tests`, or over the paths you pass |

## Entry Points

Run from the repository root. Both commands accept `--help`.

```bash
python3 -m tests.tools.run_tests
python3 -m tests.tools.run_tests --py-style
python3 -m tests.tools.run_tests --run unit/cli/test_convert_data.py
python3 -m tests.tools.python_style_tests --help
```

- Without options, `run_tests` runs the standard deterministic suite.
- `--py-style` alone runs only the lint gate, not the tests.
- `--run` paths resolve relative to `tests/`, so `--run parity` selects the opt-in parity tests.
- `python_style_tests` takes optional repository-relative paths to lint.

## Boundaries and Dependencies

- `run_tests.py` imports `src/core` for path setup and the script banner.
- CI calls `python -m tests.tools.run_tests` and `python -m tests.tools.run_tests --py-style`; a contract test pins both commands.
- `python_style_tests.py` expands directories itself because Pylint skips modules outside packages when given a directory.
- Pylint comes from the development requirements in `requirements/requirements-dev.txt`.
