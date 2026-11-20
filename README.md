# sharkLinq

`sharkLinq` bundles the Shark Matcher inference service, the Shark Matcher web application, and setup instructions for the sample dataset.

## Repository Layout

```text
sharkLinq/
├── sm-data/               # sample or working data (downloaded from GitHub Releases)
├── shark-matcher/         # ML inference service (training, embeddings, retrieval API)
└── shark-matcher-app/     # web application and SQLite database tooling
```

Both `shark-matcher` and `shark-matcher-app` automatically look for `sm-data/` as a sibling directory. To store `sm-data/` elsewhere, set `SHARK_DATA_ROOT` to its absolute path in every terminal running either service.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/linqs/sharkLinq.git
cd sharkLinq
```

### 2. Download and Position the Sample Data

The sample dataset (`sm-data.zip`) is distributed as a GitHub Release asset because its model checkpoints and embedding files exceed GitHub's 100 MB per-file Git limit.

1. Download `sm-data.zip` from the latest release on `https://github.com/linqs/sharkLinq/releases` and extract it in the `sharkLinq/` root directory:

   ```bash
   curl -L -o sm-data.zip https://github.com/linqs/sharkLinq/releases/latest/download/sm-data.zip
   unzip sm-data.zip
   rm sm-data.zip
   ```

   Or with the GitHub CLI:

   ```bash
   gh release download --repo linqs/sharkLinq --pattern "sm-data.zip"
   unzip sm-data.zip
   rm sm-data.zip
   ```

2. Confirm `sm-data/` sits beside `shark-matcher/` and `shark-matcher-app/`:

   ```bash
   ls sm-data
   ```

   Expected directories include `annotations`, `database-format`, `embeddings`, `images`, and `trained-models`.

### 3. Install and Run

Follow the **Getting Started** section in `shark-matcher-app/README.md` to create the Python 3.12 virtual environments, install dependencies, populate the database, and start both services.

- For inference service details, training, and embedding generation, see `shark-matcher/README.md`.
- For the interactive matching demo (`--demo`), see `shark-matcher-app/demo/README.md`.
