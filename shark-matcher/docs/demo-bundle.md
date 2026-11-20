# Demo Data Bundle

> **Status:** Maintainer-only checklist for building and publishing the demo
> data bundle; not setup or usage documentation.

A demo bundle is one zip whose root is a ready `sm-data/` directory holding
only what serving and updating one model needs.

## What a Bundle Contains

| Path | Contents | Leave out |
| --- | --- | --- |
| `database-format/DATASET/` | The five public tables: `individuals.csv`, `media.csv`, `tags.csv`, `labels.csv`, and an empty `alternateNames.csv` (shark nicknames are removed); `users.csv` holding only the test admin (`admin`, password `shark`); empty `tokens.csv`, `actions.csv`, `proposals.csv` | Real accounts, password and session hashes, user-linked history, shark nicknames |
| `images/images/` | Exactly the files that `media.csv` names | Everything else |
| `annotations/DATASET/` | The annotation TSVs the embedding set was built from, kept as the checkpoint's own training split (see *Prepare Annotations* in `README.md`); their `.source_csv_dir` file names `DATASET` | Regenerated splits |
| `embeddings/EMBEDDING_SET/` | Generated from `DATASET` with this repository's code; the artifacts of each mode the demo serves, listed in `src/retrieval/generation_store.py` | |
| `CREDITS.txt` (bundle root) | The CC BY 4.0 dataset credit and citation; it must travel with the images and annotations | |
| `trained-models/CHECKPOINT/` | `config.json` and the `best-*` files that embedding updates load | `init-*` and `last-*` files, `logs/` |
| Everywhere | | `test-data/` folders, `.DS_Store` files, and old `data-out/` output |

## Before Publishing

1. Review `tags.csv`, `labels.csv`, and the media names for personal notes,
   and check `alternateNames.csv` is still empty.
2. Keep the zip under GitHub's 2 GiB per-asset limit. Neighbour files grow with
   the square of the image count, so a few hundred sharks keeps it small.
3. Unzip it into an empty directory and set `SHARK_DATA_ROOT` to the `sm-data`
   folder inside.
4. Follow the *Getting Started* section of `shark-matcher-app/README.md` and
   check:
   - `/ready` returns `200`.
   - `print` reports no missing or orphaned images.
   - The Home page shows no images without embeddings.
5. Publish the zip's SHA-256 checksum next to its download link.
