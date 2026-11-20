# Shark Matcher App (`shark-matcher-app`)

The **Shark Matcher App** is the web interface and database tooling for the
Shark Matcher system. Researchers and conservationists use it to manage shark
data, view computer vision matches, and curate sightings. It talks to the
models in the sibling directory (`shark-match-model`).

New to Shark Matcher? The [Getting Started](#getting-started) section takes you
from cloning `sharkLinq` and downloading the release data to a first match.

## Prerequisites & Installation

### Required Sister Directory (`sm-data`)
> [!IMPORTANT]
> The application and inference service share one data directory, `sm-data`.
> It must sit beside this repository, or both services must receive its
> absolute path in `SHARK_DATA_ROOT`.

- The app owns its SQLite database, images, and inference snapshots.
- The matcher owns model and embedding-generation artifacts.
- Authoritative paths live in each repository's constants file.
- The *Required Sister Directory* section of the matcher's `README.md`
  (`../shark-match-model/README.md` from this checkout) describes the directory.

### Prerequisites

- Python 3.12 for both repositories.
- The `sqlite3` command-line tool, which the database scripts call.
- A platform listed under *Tested Platforms* in `shark-match-model/README.md`.
- Node.js only for the JavaScript lint gate (`npm ci` in `frontend/`).

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/linqs/sharkLinq.git
   ```

2. **Set up a virtual environment**:
   Create and activate an isolated Python 3.12 virtual environment:
   ```bash
   cd sharkLinq/shark-matcher-app
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   To exit the virtual environment when finished:
   ```bash
   deactivate
   ```

3. **Install `shark-matcher-app` requirements**:
   With the virtual environment active, install application dependencies:
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements/requirements.txt
   ```
   Run `npm ci` in `frontend/` as well if you will use the JavaScript lint gate.

4. **Install `shark-match-model` requirements**: Follow the "Installation"
   section of the shark-match-model README, in that repository's own virtual
   environment.

   Package-only alternative for this app: `python -m pip install .`; the
   matcher's package install needs extras that its "Installation" section names.

   - Installed packages cannot find the sibling `sm-data` directory, so set
     `SHARK_DATA_ROOT`.
   - Run the database scripts from a checkout, because `populate` runs the
     integrity suite under `tests/`.
   - [`setup.py`](setup.py), the matcher's `setup.py`, and the requirement files
     are the authorities for dependency versions.

## Getting Started

This walkthrough takes you from a fresh `sharkLinq` clone and the release data
directory to matching sharks in the browser. Each stage says how to confirm it
worked. Start every terminal in the `sharkLinq` root directory.

> [!NOTE]
> **Ready to demo out of the box.** The sample dataset (`sm-data.zip` from
> `linqs/sharkLinq` GitHub Releases) already supplies a trained model
> (`omega_miew_arcface_random`), precomputed embeddings (`pre2019-sample-data`),
> formatted images, and database CSVs. You do **not** need to run data cleaning,
> model training, or embedding generation to run the app demo with the sample
> data; those steps in `shark-match-model/README.md` are only needed when
> bringing your own dataset or training a new model.

Any `sm-data` directory used with the app holds four components, all pre-built
in the sample download:

- an embedding set (`pre2019-sample-data`);
- the `database-format/` dataset it was built from (`pre2019-sample-data`);
- its formatted images;
- the trained checkpoint that produced the set (`omega_miew_arcface_random`),
  used when updating embeddings with new photos.

### 1. Install Both Repositories

Follow [Prerequisites & Installation](#prerequisites--installation) above to
clone and set up both repositories. The matcher's requirements include PyTorch,
so its first install can take several minutes.

**Confirm:** both installs finish without errors.

### 2. Download and Place the Data

Download `sm-data.zip` from the `linqs/sharkLinq` GitHub Releases (see the root
`README.md`) and extract it into `sharkLinq/sm-data/` so it sits beside
`shark-match-model/` and `shark-matcher-app/`, or export `SHARK_DATA_ROOT` with its
absolute path in every terminal used below.

**Confirm:** from the `sharkLinq` root directory, `ls sm-data` lists
`embeddings`, `database-format`, `images`, and, for stage 7, `trained-models`.

### 3. Start the Matcher

In a first terminal, generate the key both services share, choose the model to
serve, and start the matcher on this machine only:

```bash
cd shark-match-model
source .venv/bin/activate
export SHARK_INFERENCE_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
echo "$SHARK_INFERENCE_API_KEY"
export SHARK_MODELS="Shark Matcher:baseline:none:EMBEDDING_SET:CHECKPOINT"
SHARK_INFERENCE_HOST=127.0.0.1 python3 -m src.api.server
```

- Replace `EMBEDDING_SET` with a directory under `embeddings/` (for the sample data, `pre2019-sample-data`).
- Replace `CHECKPOINT` with the directory under `trained-models/` that produced it (for the sample data, `omega_miew_arcface_random`).
- You can also omit `SHARK_MODELS` entirely to load the sample bundle defaults (`DEFAULT_MODELS`).
- Keep the printed key for the next stages.
- The *Configuring Models and Retrieval* section of `shark-match-model/README.md`
  explains every field.

**Confirm:** in another terminal, replacing `SHARED_KEY` with the printed key:

```bash
curl http://127.0.0.1:8000/health
curl -i -H "Authorization: Bearer SHARED_KEY" http://127.0.0.1:8000/ready
```

- The first start can take a minute or more; until then `curl` cannot connect, so retry.
- `/health` answers once the process is up.
- `/ready` returns `200` once every model has loaded; otherwise the matcher's
  startup log names the missing files.

### 4. Prepare the App Database

In a second terminal, create the database and load the dataset the embedding
set was built from, including its test user:

```bash
cd shark-matcher-app
source .venv/bin/activate
python3 -m db.database init
python3 -m db.database populate --all --data-path DATASET
```

- Replace `DATASET` with that directory's name under `database-format/` (for the sample data, `pre2019-sample-data`).
- `init` refuses to wipe a database that already holds data; to start over,
  back it up and rerun `init` with `--force`.
- The dataset's `users.csv` supplies the test administrator, `admin` with
  password `shark`. Change that password before exposing the app beyond this
  machine.

**Confirm:** `python3 -m db.database print` shows:

- non-zero `Media` and `Individuals` counts;
- `Images in DB but NOT in folder (missing): 0`;
- `Users` at `1` (the test administrator), and `Tokens`, `Actions`, and `Proposals` empty;
- a non-zero `orphaned` count (images the database does not use) is harmless.

Details: `db/README.md`.

### 5. Start the App

In the same terminal, start the app on this machine only:

```bash
export SHARK_INFERENCE_API_KEY='SHARED_KEY'
python3 -m scripts.shark_identification --run --host 127.0.0.1
```

Open `http://127.0.0.1:12345` and sign in as `admin` with password `shark`.

**Confirm:** the **Embeddings** panel on the Home page shows an embedding
count, and *Images without Embeddings* and *Embeddings without Images* are both
`0`. Non-zero counts mean the database and embedding set come from different
datasets. Details: the
[Application Usage & Orchestration](#application-usage--orchestration)
section below.

### 6. Run a First Match

1. Open the **Search** tab and press Enter in the empty search box.
   Cards show sharks as *Shark 4163*; the number is the shark's name.
2. Open the **Match** tab, type that number (or an image name shown under a
   photo), and press Enter.

The method and model sit behind the filter icon at the end of the box; the
defaults work.

**Confirm:** the query shark appears on the left, beside a ranked list of
candidate sharks.

### 7. Add New Photos

#### Upload

1. In the **Upload** tab, choose the new photos.
2. Complete the form for each photo. Name, location, and date are filled from
   the filename where possible, and the type defaults to *underwater*.
3. Press **Upload Files**.

Each photo becomes a new shark labelled `fresh`:

- The app opens the **Search** tab with `fresh` in the search box, listing the
  new sharks first, then existing sharks that already carry the label.
- The Home page counts the new photos under *Images without Embeddings*.

#### Update Embeddings

1. On the Home page, with an administrator or owner account, press
   **Update Embeddings** and confirm the dialog.
2. Wait a few minutes. The button reads *Updating Embeddings...*.
3. An *Embeddings update complete.* alert appears and Home reloads.

The app sends the matcher a snapshot of the database, and the matcher embeds
the new images with the configured checkpoint.

- After an update started from the app, the button stays disabled for an hour;
  hover over it to see the time remaining.
- Embeddings built with the matcher's command line do not start this wait.
- The server enforces the same limit, answering `429` with a `Retry-After`
  header; an owner can override it by sending `"force": true` to
  `/api/embeddings/update`.

**Confirm:** after the update finishes, *Images without Embeddings* returns to
`0`. Design details: `docs/embeddings-update-pipeline.md`.

### 8. Stop

Press Ctrl-C in the app terminal, then in the matcher terminal. Stopping the app
during an embedding update interrupts that update.

### Next Steps

- Shared matching demo: `demo/README.md`.
- Scheduled backups: `docs/cron-setup.md`.
- Tests: `python3 -m tests.tools.run_tests` in either repository.

## Application Usage & Orchestration

Application server startup and database snapshots are orchestrated through
`python3 -m scripts.shark_identification`, run from the repository root.

### Inference Service Dependency

> [!IMPORTANT]
> The web server relies on the `shark-match-model` inference service for all
> retrieval and embedding operations. The app can start without it, but match
> requests and inference-backed dataset statistics fail until it is ready.

- Start the service as in stage 3 of *Getting Started*; the *Running the
  Inference Server* section of the matcher's `README.md` covers the health
  endpoints and authentication.
- Both services must share the same `SHARK_INFERENCE_API_KEY`.
- App settings: [`core/app_constants.py`](core/app_constants.py).
- Matcher settings: its `constants.py` and `src/api/server.py` files.

### Prepare the Application Database

Stage 4 of *Getting Started* initializes the database and populates it.
Two cases need more:

- If no users were loaded, provision the first owner.
- Installations created before expiring sessions were introduced must also run
  the authentication migration before starting the server.

The exact commands and trusted restore boundary are in `db/README.md`; database
paths and related values remain authoritative in
[`core/app_constants.py`](core/app_constants.py).

### Running the Web Server

Stage 5 of *Getting Started* starts the app. The web interface is then at
`http://127.0.0.1:12345` (or `http://localhost:12345`).

#### Server Behavior

- The server address defaults to port `12345`, configured via `DEFAULT_PORT` in
  [`core/app_constants.py`](core/app_constants.py); `--port` starts it on another port.
- The server listens on every network interface unless `--host` names one;
  `--host 127.0.0.1` keeps it on this machine.
- The app reaches the matcher at `INFERENCE_SERVICE_URL`, which defaults to
  `http://localhost:8000`.
- Matcher model selection remains owned by the inference service; the app does
  not load models.
- On startup the app checks the configured inference service for liveness. If it
  cannot connect, the server logs a warning and match requests fail until the
  service is available.
- The startup check does not replace operational readiness monitoring.
- While it runs, the server writes a complete trusted backup each Sunday between
  midnight and 1 AM.
- The optional cron job in `docs/cron-setup.md` takes the same backup whether or not
  the server is running, so with both enabled each week produces two copies.
- Stop the server with Ctrl-C or `SIGTERM`. Stopping it during an embedding
  update interrupts that update and can leave a `.staging-*` directory under
  the inference snapshot root, which is safe to delete.

#### Shared Matching Demo

Start the server with demo support:

```bash
python3 -m scripts.shark_identification --run --demo
```

The `--demo` flag adds a shared, disposable matching exercise to the mode
selector without altering production data. See `demo/README.md` for
randomization, permissions, and reset behavior.

#### Disposable Test Database

To try the app against a disposable test database rebuilt from the committed CI
fixture:

```bash
python3 -m scripts.shark_identification --test --repopulate
```

### Other Useful Commands

To see all options of the script:

```bash
python3 -m scripts.shark_identification --help
```

- **Generate embeddings**: authorized administrators use the Update Embeddings
  control in the application. The server creates an immutable public snapshot
  and asks the configured inference service to publish updated vector
  embeddings; it does not import matcher code.

## Workflows & Database Management

CSV database population, snapshots, and data inspection are covered in
`db/README.md`.

## Testing

The repository-owned runner uses the committed, immutable CI fixture and does
not depend on the separate `sm-data` sample workspace:

```bash
python3 -m tests.tools.run_tests
```

Run the existing language-specific style gates when preparing a change:

```bash
python3 -m tests.tools.run_tests --py-style
python3 -m tests.tools.run_tests --js-style
```

[`tests/tools/run_tests.py`](tests/tools/run_tests.py) owns test selection. The fixture
policy in `tests/ci-test-data/README.md` is
strict: the committed CSV and TSV inputs must never be edited, regenerated, or
written by tests. New cases should use temporary inputs or mocks.

## License

Released under the MIT License by the LINQS Lab at UC Santa Cruz; see the root
`LICENSE` file. Credits for the sample dataset and the MiewID model are in the
*License and Credits* section of the root `README.md`.

## Repository Structure

```text
shark-matcher-app/
├── core/                     # shared constants (app_constants.py) and helpers (app_utils.py)
├── scripts/                  # shark_identification.py (server, test mode, snapshots), setup_cron.sh
├── backend/                  # application-domain logic and inference-snapshot publication
│   └── db/                   # data access, one module per table
├── frontend/                 # HTTP handlers and the web server
│   ├── cli/                  # command-line runner
│   └── static/               # browser interface
├── db/                       # database administration
│   ├── sql/                  # schema and indexes
│   └── scripts/              # export, backup, sample data, one-off migrations
├── demo/                     # optional shared matching demo
├── tests/                    # deterministic, opt-in browser, and manual verification
│   ├── core/ backend/ db/ frontend/ demo/ packaging/ scripts/ style/   # deterministic
│   ├── browser/              # Playwright, opt-in
│   ├── manual/               # live-database audit
│   ├── ci-test-data/         # immutable fixture
│   └── tools/                # test runner and style gates
├── docs/                     # workflow and architecture notes, cron guide, release notes
├── containerization/         # deferred container design placeholder
├── config/                   # tool configuration (pylintrc)
├── requirements/             # requirement sets used by setup.py, CI and Docker
├── .github/                  # CI workflow definitions
├── setup.py                  # package and optional-dependency metadata
└── MANIFEST.in               # source-archive file list
```

- The source tree is organized by ownership boundary; the main folders, and
  `docs/`, have a `README.md` listing their modules and how to run them.
- `frontend/` also holds the Node tooling: `package.json`, its lockfile, and
  `eslint.config.js`.
- `frontend/` modules are HTTP handlers, `backend/db/` modules are data access,
  and top-level `db/` holds SQL, the administration CLI (`db/database.py`, run
  as `python3 -m db.database`), and runnable scripts.
