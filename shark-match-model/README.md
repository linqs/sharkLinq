# Shark Matcher (`shark-match-model`)

The **Shark Matcher** ML inference service is the computer vision backbone of
the Shark Matcher system: a re-identification pipeline built around triplet-loss
and ArcFace training on ViT and MiewID model architectures.

- It handles model training, embedding generation, and nearest-neighbour retrieval.
- It exposes all of this over a REST API for the companion application service
  (`shark-matcher-app`).

New to Shark Matcher? The *Getting Started* section of the companion
repository's `README.md` (`shark-matcher-app/README.md`) takes you from two
clones and a data directory to a first match.

## Prerequisites & Installation

### Required Sister Directory (`sm-data`)

> [!IMPORTANT]
> Runtime and walkthrough data live in a separately downloaded directory named
> `sm-data` (distributed as `sm-data.zip` in the `linqs/sharkLinq` GitHub
> Releases; see the root `README.md`). It must sit at `sharkLinq/sm-data/` as a
> sibling of `shark-match-model/` unless `SHARK_DATA_ROOT` names another location.
> It is not bundled in Git or with the committed unit-test fixtures.

The expected host layout is:

```text
PARENT_DIRECTORY/
├── sm-data/               # separately provided sample or working data
├── shark-match-model/         # this repository
└── shark-matcher-app/     # companion application, when needed
```

To keep the data elsewhere, set `SHARK_DATA_ROOT` to the directory's absolute
path. Use the same value for the matcher, `shark-matcher-app`, and any cron job,
because each derives its data and snapshot locations from it.

#### Directory Ownership

The top-level directories that running Shark Matcher uses are:

| Directory | Written by | Holds |
| --- | --- | --- |
| `raw-data/` | you | raw images for `src.cli.convert_images` |
| `images/` | matcher and app | the formatted image corpus in `images/images/`, including app uploads |
| `annotations/` | matcher | versioned annotation TSVs from `src.cli.convert_data` |
| `database-format/` | you | table-CSV exports, one directory per dataset |
| `trained-models/` | matcher | one directory per checkpoint, each with a `config.json` |
| `embeddings/` | matcher | one directory per embedding set |
| `db/` | app | the SQLite database |
| `inference-snapshots/` | app | snapshots the matcher reads during embedding updates |
| `data-out/` | matcher and app | logs, backups, and analysis output |
| `test-data/` | maintainer | the parity baseline; not part of the sample dataset |

Other directories serve research workflows. [`src/core/constants.py`](src/core/constants.py) and
the app's `core/app_constants.py` own the exact paths; consult them rather than
reproducing deeper paths or filenames in documentation.

#### Keeping the Data Consistent

Matches go missing or point at the wrong shark unless three pairings hold:

1. **App database and embedding set.** Populate the app database from the same
   `database-format/` dataset that the served embedding set was built from.
   With both services running, the **Embeddings** panel on the app's Home page
   lists *Images without Embeddings* and *Embeddings without Images*; both
   should be empty apart from images added since the last update.
2. **Embedding set and checkpoint.** Serve each embedding set with the
   checkpoint that produced it. Embedding updates embed new images with the
   configured checkpoint, so a mismatch mixes two embedding spaces.
3. **Images and media records.** Every media record's file must exist in
   `images/images/`. The app's `python3 -m db.database print` reports
   records whose image is missing.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/linqs/sharkLinq.git
   cd sharkLinq/shark-match-model
   ```

2. **Set up a virtual environment**:
   Create and activate an isolated Python 3.12 virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   To exit the virtual environment when finished:
   ```bash
   deactivate
   ```

3. **Install the package**:
   With the virtual environment active, install repository dependencies:
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements/requirements.txt
   ```

   Package-only alternative: `python -m pip install ".[training,visualization]"`.

   - A plain `pip install .` omits TensorBoard, which the trainer imports, and
     matplotlib, which `analyze_data.py` imports.

   - An installed package cannot find the sibling `sm-data` directory, so also
     set `SHARK_DATA_ROOT` as described above.
   - [`setup.py`](setup.py) and the requirement files are the authorities for
     runtime and optional dependencies.

### Tested Platforms

| Platform | Verified |
| --- | --- |
| macOS, Apple silicon (CPU and MPS) | Install, both test suites, training, embeddings, evaluation, both servers, and the web interface |
| Linux aarch64, Debian-based Python 3.12 container (CPU) | Install, both test suites, serving, search and matching, one-epoch training, and PSL reranking with OpenJDK 21 |

#### Linux Notes

- PSL needs a Java runtime. It was verified on the demo data, and failed with an
  unclear Java error on a 12-shark, 95-image slice, so use a realistic dataset.

## CSV Store (`src/csv_store/`)

The inference and retrieval metadata paths read individual and media records
from annotation CSVs rather than querying the SQLite database owned by
`shark-matcher-app`.

- Incremental embedding updates consume explicit CSV input through the snapshot handoff.
- The matcher has no runtime import or database-path dependency on the app repository.
- The CSV-backed access layer preserves the record interface matcher callers expect.

Authorities for column positions and relationship semantics:

- [`src/csv_store/_loader.py`](src/csv_store/_loader.py): accepted CSV layouts.
- [`src/csv_store/models.py`](src/csv_store/models.py): record attributes.

## Example Walkthrough: Sample Dataset

> [!NOTE]
> **Not required for the app demo.** The sample dataset (`sm-data.zip` from
> `linqs/sharkLinq` GitHub Releases) already includes formatted images, a
> trained model (`omega_miew_arcface_random`), and precomputed embeddings
> (`pre2019-sample-data`). Skip this section and follow *Getting Started* in
> `shark-matcher-app/README.md` if you only want to run the app demo; use the
> steps below when preparing your own data, training a new model, or
> regenerating embeddings.

Run all commands from the `shark-match-model/` repository root.

- The walkthrough uses the sample dataset, `pre2019-sample-data`, in the
  sibling `sm-data` workspace, not the small committed test fixtures.
- Relative input arguments resolve beneath the data roots owned by
  [`src/core/constants.py`](src/core/constants.py), so do not prefix them with `sm-data/`.
- The sample workspace, a compatible Python environment, and any model downloads
  must be available before the full training workflow.

### 1. Format Images

The sample's images are already formatted in `images/images/`; skip this step
unless you add your own photos. To format raw photos placed in
`raw-data/RAW_IMAGES/`:

```bash
python3 -m src.cli.convert_images \
    --input-directory RAW_IMAGES \
    --output-directory images
```

Use `python3 -m src.cli.convert_images --help` for the current path and overwrite
behavior. The implementation and `src/core/constants.py` own the resolved locations.

### 2. Prepare Annotations

Generate versioned annotation TSV files from the database-format CSV files.

```bash
python3 -m src.cli.convert_data \
    --input-directory pre2019-sample-data \
    --output-directory walkthrough-annotations
```

The command records its source CSV directory with the generated annotations so
later embedding generation can reuse the same metadata source. Consult
`python3 -m src.cli.convert_data --help` for current split controls and output
behavior rather than copying their defaults here.

- The bundle's own `annotations/pre2019-sample-data/` is the training split of
  the shipped checkpoint, cut from the original dataset before the sample was
  made, so no `convert_data` setting reproduces it.
- A regenerated split trains a new, different model. Do not evaluate the shipped
  checkpoint on it: its valid images include images that checkpoint trained on.

### 3. Validate Data (Optional)

Analyze the dataset to check integrity and view distribution statistics.

```bash
# Validate annotations with full verbose output.
python3 src/data_scripts/analyze_data.py --verbose \
    --annotations-version walkthrough-annotations

# Generate an analysis report with a plot.
python3 src/data_scripts/analyze_data.py --plot \
    --annotations-version walkthrough-annotations
```

### 4. Train Model

Choose a current training configuration key from
[`src/config/config.py`](src/config/config.py), replace `CONFIG_KEY` below, and train it
against the generated annotations. Configuration names and values intentionally
live in that source registry rather than in this README.

```bash
python3 -m src.learning.trainer.trainer \
    --model-name walkthrough-model \
    --annotations-version walkthrough-annotations \
    --configs CONFIG_KEY
```

- Training may download the selected pretrained backbone; the ViT configs fetch `google/vit-large-patch16-384`, about 1.2 GB.
- Completion emails go to the comma-separated addresses in
  `SHARK_NOTIFICATION_EMAILS`; none are sent when it is unset.
- `--epochs N` runs a different number of epochs without editing the config; a
  scheduler `T_max` equal to the configured epochs is rescaled to match.
- Training stops if the output folder already holds trained weights; pass
  `--overwrite` to replace them.
- The trainer's `--help` output covers validation-only and notification controls.

### 5. Generate Embeddings

Generate and cache vector embeddings for the trained model to enable retrieval.

```bash
python3 -m src.retrieval.embeddings \
    --model-name walkthrough-model/CONFIG_KEY \
    --annotations-version walkthrough-annotations
```

The script publishes each completed build as an immutable generation. Readers
switch to it only after its artifacts, metadata, and manifest validate; if
construction or validation fails, the prior complete generation stays active.

| Flag | Effect |
| --- | --- |
| `--update-embeddings` | Incremental update; reuses unchanged artifacts inside isolated staging |
| `--dataset-only`, `--suggestions-only` | Publish maintenance generations through the same path |
| `--rollback` | Repoint the embedding set to its retained prior valid generation |

If `current.json` names a generation that no longer exists, every command on that
embedding set stops with an error. Restore the generation directory, or delete
`current.json` and regenerate the embeddings.

Owners (use `--help` for the current command reference):

- [`src/retrieval/generation_store.py`](src/retrieval/generation_store.py):
  storage layout, validation rules, active-generation resolution, and retention.
- [`src/retrieval/embeddings.py`](src/retrieval/embeddings.py): the CLI and
  artifact-building operations.

### 6. Evaluate Embeddings

Evaluate the retrieval performance using Hits@K metrics.

```bash
python3 -m src.retrieval.evaluation \
    --embeddings-name walkthrough-model/CONFIG_KEY
```

Evaluation methods and split behavior are defined by the command's `--help`
output and [`src/retrieval/evaluation.py`](src/retrieval/evaluation.py).

### 7. Visualize Results (Utility)

Visualize the generated embedding feature space.

```bash
# Visualize the embedding feature space.
python3 -m src.visualization.visualize_model \
    --model-name walkthrough-model/CONFIG_KEY
```

The plots go to a timestamped folder under `data-out/visualizations/` in the
data root; the command exits non-zero when the embeddings are missing.

The other visualization utilities and their current limitations are described
in `src/visualization/README.md`.

## Running the Inference Server

The service requires a shared key of at least 32 characters in
`SHARK_INFERENCE_API_KEY`. Generate one, keep the printed value for the
companion app, and start the service bound to this machine only:

```bash
source .venv/bin/activate
export SHARK_INFERENCE_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
echo "$SHARK_INFERENCE_API_KEY"
SHARK_INFERENCE_HOST=127.0.0.1 python3 -m src.api.server
```

- Never commit the key or reuse a published one.
- Without `SHARK_INFERENCE_HOST` the service listens on `0.0.0.0`, which accepts connections from other machines.
- `SHARK_INFERENCE_PORT` changes the port from `8000`.
- The app reaches the service through its `INFERENCE_SERVICE_URL` setting.

### Health and Authentication

| Endpoint | Auth | Use |
| --- | --- | --- |
| `GET /health` | Public | Process liveness only; success does not mean the configured models loaded |
| `GET /ready` | Bearer | Traffic admission and model-availability monitoring; succeeds only when every required model is ready |
| `GET /stats` | Bearer | Per-model status and active source identity, for diagnosis |

Matching, statistics, readiness, suggestions, and embedding-generation requests
all require the shared Bearer credential.

Owners:

- [`src/api/server.py`](src/api/server.py): response contracts and authentication behavior.
- [`src/retrieval/retrieval.py`](src/retrieval/retrieval.py): model lifecycle.
- `src/contracts`: shared wire models.
- [`src/core/constants.py`](src/core/constants.py): authoritative configuration definitions.

### Configuring Models and Retrieval

`SHARK_MODELS` lists the models the server loads, as comma-separated entries:

```bash
export SHARK_MODELS="DISPLAY_NAME:MODE:RERANK:EMBEDDING_SET:CHECKPOINT"
```

- `DISPLAY_NAME` is the name the app shows for the model.
- `MODE` is `baseline`, `psl`, `multimodal`, or `stacked`. Each mode reads its
  own artifact files from the embedding set, listed in
  [`src/retrieval/generation_store.py`](src/retrieval/generation_store.py);
  `baseline` needs the fewest.
- `RERANK` is `none` or `kreciprocal`.
- `EMBEDDING_SET` names a directory under `embeddings/`.
- `CHECKPOINT` names a directory under `trained-models/`, or one of the
  pre-canned backbones in `PRECANNED_MODELS`. Matching needs only the embedding
  set; embedding updates also need the checkpoint.

- `_parse_models_env()` in [`src/api/server.py`](src/api/server.py) defines how
  omitted trailing fields are filled and also accepts a JSON array of model objects.
- Without `SHARK_MODELS`, the server loads `DEFAULT_MODELS` from
  [`src/core/constants.py`](src/core/constants.py): the demo bundle's
  `omega_miew_arcface_random` checkpoint on the `pre2019-sample-data` embedding set.
- A model that cannot load is reported by `/ready` and `/stats`; the startup log
  names the command that would generate its embeddings.
- [`src/retrieval/retrieval.py`](src/retrieval/retrieval.py) owns artifact
  selection and loading.

The files under `containerization/` are deferred design scaffolding, not a
supported production deployment.

## Testing

### Standard Tests

The repository-owned runner executes the standard deterministic suite without
requiring `sm-data`, a live database, or downloaded models:

```bash
python3 -m tests.tools.run_tests
```

Run its existing style gate separately when preparing a change:

```bash
python3 -m tests.tools.run_tests --py-style
```

- The suite uses committed files under `tests/fixtures/` plus test-local
  generated data, independent of the sample workspace.
- [`tests/tools/run_tests.py`](tests/tools/run_tests.py) owns target selection and optional
  integration-data arguments; use its `--help` output for the current interface.

### Parity Tests

Parity tests validate that the CSV store produces identical results to the live
SQLite database for the same dataset. They require a populated database and the
corresponding CSV directory:

```bash
export PARITY_DB_PATH=../sm-data/db/sharkMatcher.db
export PARITY_CSV_DIR=../sm-data/database-format/DATASET
python3 -m pytest tests/parity/ -v
```

- Replace `DATASET` with the sample directory being compared.
- These tests are excluded from the standard runner; run them as an opt-in
  validation when deploying a new dataset.
- Manual training and embedding diagnostics are also outside the standard suite;
  see `tests/manual/README.md`.

## Detailed Script Help

Each script includes comprehensive help documentation and more examples. Run any
script with the `--help` flag for details:

```bash
python3 -m src.cli.convert_data --help
```

## License

Released under the MIT License by the LINQS Lab at UC Santa Cruz; see the root
`LICENSE` file. Credits for the sample dataset and the MiewID model are in the
*License and Credits* section of the root `README.md`.

## Repository Structure

```text
shark-match-model/
├── src/                      # the matcher implementation
│   ├── core/                 # constants, data-root paths, shared helpers
│   ├── cli/                  # convert_images, convert_data, run_pipeline, run_tensorboard
│   ├── config/               # training configuration registry and schema
│   ├── api/                  # FastAPI inference service
│   ├── retrieval/            # embeddings, generation store, neighbors, evaluation, suggestions
│   ├── learning/             # trainer, loss, evaluator, sampler, augmented dataset
│   ├── models/               # backbones, heads, augmentations
│   ├── csv_store/            # CSV-backed individual and media records
│   ├── contracts/            # strict version-1 wire models
│   ├── data_scripts/         # dataset analysis and conversion scripts
│   ├── multimodal/           # metadata late fusion
│   ├── psl/                  # Probabilistic Soft Logic reranking
│   └── visualization/        # embedding and pipeline inspection tools
├── tests/                    # standard, opt-in parity, and manual verification
│   ├── unit/                 # unit tests, mirroring src/
│   ├── contracts/            # boundary, packaging, and documentation contracts
│   ├── parity/               # CSV store versus SQLite, opt-in
│   ├── manual/               # trainer parity, determinism, and convergence checks
│   ├── fixtures/             # committed CSV fixtures
│   └── tools/                # test runner and Python lint gate
├── containerization/         # deferred deployment scaffolding
├── docs/                     # release notes and architecture notes
├── config/                   # tool configuration (pylintrc)
├── requirements/             # requirement sets used by setup.py, CI and Docker
├── setup.py                  # package and optional-dependency metadata
└── MANIFEST.in               # source-archive file list
```

- The source tree is organized by ownership boundary; each folder under `src/`
  and each `tests/` folder except `fixtures/` has a `README.md` listing its
  modules and entry points.
