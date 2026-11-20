# sharkLinq

`sharkLinq` bundles the Shark Matcher inference service, the Shark Matcher web application, and setup instructions for the sample dataset.

## Repository Layout

```text
sharkLinq/
├── LICENSE                # MIT License for the repository
├── sm-data/               # sample or working data (downloaded from GitHub Releases)
├── shark-match-model/     # ML inference service (training, embeddings, retrieval API)
└── shark-matcher-app/     # web application and SQLite database tooling
```

Both `shark-match-model` and `shark-matcher-app` automatically look for `sm-data/` as a sibling directory. To store `sm-data/` elsewhere, set `SHARK_DATA_ROOT` to its absolute path in every terminal running either service.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/linqs/sharkLinq.git
cd sharkLinq
```

### 2. Download and Position the Sample Data

The sample dataset (`sm-data.zip`) is distributed as a GitHub Release asset because its model checkpoints and embedding files exceed GitHub's 100 MB per-file Git limit.

> [!NOTE]
> The sample download already includes formatted images, database CSVs, precomputed embeddings (`pre2019-sample-data`), and a trained model checkpoint (`omega_miew_arcface_random`). You do not need to clean data, train a model, or generate embeddings to run the app demo with the sample data.

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

2. Confirm `sm-data/` sits beside `shark-match-model/` and `shark-matcher-app/`:

   ```bash
   ls sm-data
   ```

   Expected directories include `annotations`, `database-format`, `embeddings`, `images`, and `trained-models`.

### 3. Install and Run

Follow the **Getting Started** section in `shark-matcher-app/README.md` to create the Python 3.12 virtual environments, install dependencies, populate the database, and start both services.

- For inference service details—or optional data cleaning, model training, and embedding generation on your own data—see `shark-match-model/README.md`.
- For the interactive matching demo (`--demo`), see `shark-matcher-app/demo/README.md`.

## License and Credits

### Software

`sharkLinq` (`shark-match-model` and `shark-matcher-app`) is released under the MIT License by the LINQS Lab at UC Santa Cruz; see the `LICENSE` file.

### Sample Dataset

The images and annotations in the demo bundle come from a public dataset licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). Credit the authors whenever you reuse them:

- Paul E. Kanive, Salvador J. Jorgensen, Jay J. Rotella, Barbara A. Block, Taylor K. Chapple, Scot D. Anderson, & Timothy White. (2020). Data from: Estimation of regional annual abundance and evidence for increasing numbers of white sharks off central California [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.4022914

### MiewID Model

The matcher downloads the pretrained MiewID model (`conservationxlabs/miewid-msv2`) on first use, and the demo checkpoint is fine-tuned from it. Cite MiewID as its authors ask:

```bibtex
@misc{WildMe2023,
  author = {Otarashvili, Lasha},
  title = {MiewID},
  year = {2023},
  url = {https://github.com/WildMeOrg/wbia-plugin-miew-id},
  doi = {10.5281/zenodo.13647526},
}
```
