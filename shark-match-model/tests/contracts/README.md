# Contract Tests

The `tests/contracts` folder holds cross-cutting tests that pin the matcher's public boundaries: wire shapes, generation storage, pickle compatibility, packaging, and documentation rules.

## Modules

| File | Responsibility |
| --- | --- |
| `test_contracts.py` | Version-1 snapshot, generation, pointer, and model-status shapes; data-root, snapshot-root, `pairx`, and bind-address defaults; legacy `ModelSpec` fields; checkpoint validation |
| `test_v1_generation_contract.py` | Atomic generation publication: pointer switching, staging cleanup, retention, rollback, legacy flat layout, concurrent readers and publishers, snapshot provenance |
| `test_v1_architecture_contract.py` | API request and response shapes and the readiness route, no app or database dependency, no use of experimental code, wheel and source-archive contents, version sync, CI workflow, ignore rules, path portability, documentation rules |
| `test_pickle_compatibility.py` | Stored pickles that name earlier module paths still load the current classes |
| `test_python_style_gate.py` | The lint gate in `tests/tools` reaches modules outside packages |

### Documentation Rules Enforced

These checks run over every tracked or new Markdown file, except files that open with a `Status` banner, so new documentation is held to them too.

- Relative links must resolve to an existing file and heading inside the repository, and never into `experimental/`.
- Files are named by repository path, not by GitHub branch links.
- Headings use Title Case, code fences carry a language tag, placeholders are `UPPER_CASE`, and horizontal rules are not used.
- Documented `SHARK_*` names must appear in the matcher source.
- Documented service ports and bind hosts must match the server defaults.
- Documented `MODE` and `RERANK` values must match the `SHARK_MODELS` parser.
- No service key literal and no developer-specific rooted path may appear.

## Entry Points

Run from the repository root.

```bash
python3 -m tests.tools.run_tests --run contracts
python3 -m pytest tests/contracts -q -p no:cacheprovider
```

The standard runner (`python3 -m tests.tools.run_tests`) includes this folder.

## Boundaries and Dependencies

- Tests need no `sm-data` directory, trained model, or database; they build their data in temporary directories.
- Packaging tests copy the repository into a temporary directory and build a wheel and a source archive, so they are slower than unit tests.
- These tests read `src/`, `setup.py`, `requirements/`, `.github/workflows/python-tests.yml`, and the Markdown files, so edits there can fail them.
