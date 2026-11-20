# Shark Matcher Tests

The standard matcher suite covers training components, data conversion,
retrieval and embedding publication, and inference contracts. It needs no live
database or external sample workspace.

## Running Tests

Use the repository-owned runner from the `shark-match-model/` root:

```bash
# Run the standard deterministic suite (tests/unit and tests/contracts).
python3 -m tests.tools.run_tests

# Run one test module or directory beneath tests/.
python3 -m tests.tools.run_tests --run unit/learning/test_sampler.py

# Run the Python style gate.
python3 -m tests.tools.run_tests --py-style
```

The runner, `tests/tools/run_tests.py`, owns target selection and options. Use
its `--help` output rather than copying the interface here.

## Layout

| Folder | Contents | In standard run |
| --- | --- | :---: |
| `unit/` | Unit tests; subfolders mirror the `src/` packages (`api`, `cli`, `csv_store`, `data_scripts`, `retrieval`, `learning`, `models`, `multimodal`) | yes |
| `contracts/` | Cross-cutting checks: architecture, generation and snapshot contracts, pickle compatibility, style gate | yes |
| `parity/` | SQLite versus CSV store parity; needs matching external data | on request |
| `manual/` | Trainer parity, determinism, convergence, embedding identity | on request |
| `fixtures/` | Committed CSV fixtures used by the unit tests | n/a |
| `tools/` | The test runner and the Python style gate | n/a |

## Test Data Boundaries

The standard suite uses committed fixtures in `tests/fixtures/` and test-local
generated data. It does not require the sibling `sm-data` workspace, trained
models, or a populated application database.

External sample data belongs to walkthroughs and manual validation, not to the
deterministic suite.

## Manual and Live-Data Checks

- The manual workflows are described in the `tests/manual` README. They are
  excluded from the standard suite because some need special timing, real
  images, trained models, or before-and-after state.
- SQLite/CSV parity is opt-in because it needs matching external data. The root
  README documents that workflow.
- Visualization commands are application utilities, not test helpers.

## PK Sampling

`PKSampler` in `src/learning/pk_sampler.py` is the current metric-learning
sampler. It is exercised by `tests/unit/learning/test_sampler.py` and by
trainer-level tests.
