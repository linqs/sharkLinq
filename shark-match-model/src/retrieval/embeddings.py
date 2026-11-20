# pylint: disable=no-member, too-many-arguments, too-many-locals, too-many-statements, too-many-instance-attributes, broad-exception-caught, import-error, duplicate-code, wrong-import-position, line-too-long, no-name-in-module, import-outside-toplevel, ungrouped-imports, too-many-nested-blocks, too-many-branches, unused-variable, bare-except, unused-import, not-callable, unnecessary-pass, too-few-public-methods, too-many-positional-arguments, protected-access
"""Functions for building and storing vector embeddings in cached .pt files in the embeddings directory."""

import argparse
import collections
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Subset
import tqdm

from src.core import constants
import src.cli.convert_data
import src.individual_dataset
import src.retrieval.generation_store
import src.retrieval.inference_snapshot
import src.retrieval.neighbors
import src.retrieval.rerank
from src.core import utils

# Controls which saved backbone checkpoint prefix ("best", "last", or "init") is loaded.
BACKBONE_CHECKPOINT = "best"


# Batched and single-image MiewID embeddings agree to 1e-7 with identical neighbor order.
EMBEDDING_BATCH_SIZE = 16

# Named tuple capturing newly embedded filenames and labels requiring centroid updates.
UpdateResult = collections.namedtuple(
    "UpdateResult", ["new_filenames", "affected_labels"])


def main():
    """Generate and cache vector embeddings for a trained model."""
    parser = argparse.ArgumentParser(
        description="Generate and cache vector embeddings for a trained "
                    "Shark Matcher model. By default, regenerates the "
                    "dataset and all embeddings; neighbor files need "
                    "--store-neighbors, --use-psl or --stacked.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m src.retrieval.embeddings "
               "--model-name test-model/vit_triplet_1 "
               "--annotations-version test-annotations\n"
               "  python3 -m src.retrieval.embeddings "
               "--model-name cross_prod_tests "
               "--annotations-version test-annotations\n"
               "  python3 -m src.retrieval.embeddings "
               "--model-name test-model/vit_triplet_1 "
               "--annotations-version test-annotations --dry-run\n")
    parser.add_argument('--print', dest="print", action="store_true",
                        help="Print information on the stored dataset.")
    parser.add_argument('--model-name', type=str, default='lora',
                        help='Name of the trained model directory (relative to '
                             'sm-data/trained-models/). If this points to a '
                             'parent directory with config subdirectories, all '
                             'are processed. Default: "lora".')
    parser.add_argument('--embeddings-name', type=str, default=None,
                        help='Name of the embeddings output directory '
                             '(relative to sm-data/embeddings/). Defaults to '
                             '--model-name if not specified.')
    parser.add_argument('--update-embeddings', action='store_true',
                        help='Update existing embeddings instead of '
                             'regenerating all of them.')
    parser.add_argument('--annotations-version', type=str, default=None,
                        help='Annotations version directory name under '
                             'sm-data/annotations/ (e.g. "test-annotations"). '
                             'Must match the version used during training.')
    parser.add_argument('--csv-dir', type=str, default=None,
                        help='Source database-format CSV directory '
                             '(containing individuals.csv, media.csv) used to '
                             'pin metadata reads on the generated dataset. '
                             'Resolved against sm-data/database-format/ when '
                             'relative. If omitted, falls back to the '
                             '.source_csv_dir sidecar in the annotations '
                             'directory; if neither is available, the script '
                             'exits with an error. Incremental updates use the '
                             'same explicit or sidecar-resolved CSV input.')
    parser.add_argument('--snapshot-id', type=str, default=None,
                        help='Validated inference snapshot UUID under the '
                             'inference snapshot root (default '
                             'sm-data/inference-snapshots/) used as the '
                             'immutable CSV source for live updates.')
    parser.add_argument('--store-neighbors', action='store_true',
                        help='Also compute and store image and centroid '
                             'neighbor dictionaries (stored_image_neighbors*.pt '
                             'and stored_centroid_neighbors*.pt) for offline '
                             'evaluation. Automatically enabled when --use-psl '
                             'or --stacked is set. Default: disabled.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate paths and configuration without '
                             'generating embeddings. Prints a summary of '
                             'resolved settings and exits.')
    parser.add_argument('--suggestions-only', action='store_true',
                        help='Skip the embedding/neighbor pipeline and just '
                             'regenerate the cached review suggestions '
                             '(unmatch + merge) for each resolved model. '
                             'Publishes a new generation copied from the '
                             'active generation after validation.')
    parser.add_argument('--dataset-only', action='store_true',
                        help='Skip the embedding/neighbor pipeline and just '
                             'regenerate the cached dataset for each resolved '
                             'model in a copy of the active generation. Useful '
                             'for older caches '
                             'whose dataset is missing attributes added later '
                             '(e.g. csv_dir). Aborts if the new dataset would '
                             'desync from the stored embeddings; otherwise '
                             'publishes the validated copy as a new generation.')
    parser.add_argument('--rollback', action='store_true',
                        help='Atomically repoint --embeddings-name to its '
                             'retained prior valid generation without '
                             'regenerating or copying artifacts.')
    parser.add_argument('--use-psl', action='store_true',
                        help='Enable PSL (Probabilistic Soft Logic) reranking '
                             'of image neighbors using biological constraints '
                             '(sex matching, length disagreement). Rule weights '
                             'are learned automatically from the training split '
                             'and saved as psl_rules.json alongside the embeddings. '
                             'Requires pslpython, OpenJDK 17+, and JAVA_HOME to '
                             'be set. Default: disabled.')
    parser.add_argument('--psl-seed', type=int, default=42,
                        help='Random seed for deterministic PSL weight learning '
                             'and inference (default: 42).')
    parser.add_argument('--use-multimodal', action='store_true',
                        help='Run the late-fusion multimodal pipeline after base '
                             'embeddings. Optimises per-group alpha weights on a '
                             'validation slice of the training set, generates '
                             'fused embeddings, and stores image and centroid '
                             'neighbours with the _multimodal suffix.')
    parser.add_argument('--stacked', action='store_true',
                        help='Run the full stacked pipeline: multimodal late '
                             'fusion (implies --use-multimodal) plus PSL '
                             'reranking on both the base image neighbours '
                             '(stored with a _psl suffix) and the multimodal neighbours '
                             '(stored with a _multimodal_psl suffix). PSL weights are learned '
                             'once from the base image neighbours and reused '
                             'for both reranking passes.')
    args = parser.parse_args()

    if args.rollback and any((
            args.print,
            args.update_embeddings,
            args.dry_run,
            args.suggestions_only,
            args.dataset_only,
            args.use_psl,
            args.use_multimodal,
            args.stacked,
            args.store_neighbors,
            args.snapshot_id,
    )):
        parser.error("--rollback cannot be combined with generation, "
                     "maintenance, print, or dry-run modes")

    utils.print_script_banner(__file__, args)
    print(f"  BACKBONE_CHECKPOINT: {BACKBONE_CHECKPOINT}")

    if args.rollback:
        embeddings_name = args.embeddings_name or args.model_name
        rolled_back = src.retrieval.generation_store.rollback_generation(
            os.path.join(constants.EMBEDDINGS_DIR, embeddings_name)
        )
        print("Rollback complete. Active generation: "
              f"{rolled_back.generation_id}")
        return

    if args.use_psl or args.stacked:
        _require_psl_runtime()

    source_snapshot = (
        src.retrieval.inference_snapshot.load_inference_snapshot(args.snapshot_id)
        if getattr(args, "snapshot_id", None) else None
    )
    resolved_model_dirs = utils.resolve_model_dirs(args.model_name)

    for model_name in resolved_model_dirs:
        embeddings_name = args.embeddings_name if args.embeddings_name else model_name
        start_time = time.time()

        if args.print:
            src.individual_dataset.IndividualDataset.print_stored_info(embeddings_name)
            continue

        if args.suggestions_only:
            run_pipeline(
                model_name,
                embeddings_name,
                args,
                start_time,
                maintenance_mode="suggestions",
            )
            continue

        if args.dataset_only:
            run_pipeline(
                model_name,
                embeddings_name,
                args,
                start_time,
                maintenance_mode="dataset",
            )
            continue

        if source_snapshot is None:
            valid = _validate_pipeline_inputs(model_name, embeddings_name, args)
            if args.dry_run:
                print("Dry run complete.")
                continue
            if not valid:
                sys.exit(1)

        run_pipeline(
            model_name,
            embeddings_name,
            args,
            start_time,
            source_snapshot=source_snapshot,
        )


def run_pipeline(model_name: str, embeddings_name: str,
                 args: argparse.Namespace, start_time: float,
                 maintenance_mode=None, source_snapshot=None):
    """Build and publish one complete immutable embedding generation."""
    if maintenance_mode not in {None, "dataset", "suggestions"}:
        raise ValueError(f"Unsupported maintenance mode: {maintenance_mode}")

    embedding_set_root = os.path.join(
        constants.EMBEDDINGS_DIR, embeddings_name
    )
    if maintenance_mode is not None and not os.path.isdir(embedding_set_root):
        raise RuntimeError(
            f"Existing embeddings directory not found: {embedding_set_root}"
        )

    if source_snapshot is None:
        initial_csv_dir = None
        publication_source = "offline_annotations"
        publication_snapshot_id = None
    else:
        initial_csv_dir = os.fspath(source_snapshot.directory)
        publication_source = "snapshot"
        publication_snapshot_id = source_snapshot.snapshot_id
    requires_annotation_csv = (
        maintenance_mode == "dataset" or maintenance_mode is None
    )
    if requires_annotation_csv and initial_csv_dir is None:
        initial_csv_dir = utils.resolve_csv_dir(args)
        if initial_csv_dir is None:
            raise RuntimeError(
                "No CSV directory could be resolved. Pass --csv-dir or "
                "regenerate annotations with src.cli.convert_data so a source "
                "CSV sidecar is written."
            )

    retrieval_mode = src.retrieval.generation_store.resolve_retrieval_mode(
        use_psl=getattr(args, "use_psl", False),
        use_multimodal=getattr(args, "use_multimodal", False),
        use_stacked=getattr(args, "stacked", False),
    )

    def build(staging_dir, _previous_dir):
        """Build artifacts within the publisher's isolated directory."""
        if maintenance_mode == "dataset":
            return regenerate_dataset_only(
                model_name,
                embeddings_name,
                args,
                staging_dir,
                csv_dir=initial_csv_dir,
            )
        if maintenance_mode == "suggestions":
            num_images = _stored_dataset_image_count(staging_dir)
            generate_and_store_suggestions(
                embeddings_name,
                model_dir=staging_dir,
                retrieval_mode=retrieval_mode,
                raise_on_error=True,
            )
            return num_images
        return _build_generation_artifacts(
            model_name,
            embeddings_name,
            args,
            staging_dir,
            initial_csv_dir,
            retrieval_mode,
        )

    seed_neighbors = (
        _should_store_neighbors(args)
        if maintenance_mode != "suggestions"
        else (retrieval_mode in {"psl", "stacked"}
              or _should_store_neighbors(args))
    )
    return src.retrieval.generation_store.publish_generation(
        embedding_set_root=embedding_set_root,
        embedding_set=embeddings_name,
        model_name=model_name,
        source=publication_source,
        snapshot_id=publication_snapshot_id,
        start_time=start_time,
        retrieval_mode=retrieval_mode,
        build=build,
        seed_previous=(maintenance_mode is not None
                       or args.update_embeddings),
        seed_neighbors=seed_neighbors,
        inherit_previous_identity=maintenance_mode is not None,
    )


def store_image_embeddings(model_name, model_dir, annotations_version=None, channels=3, csv_dir=None):
    """Create the dataset and generate image embeddings for a model, storing them to model_dir."""
    utils.print_function_banner(
        "store_image_embeddings",
        f"Storing image embeddings for model:'{model_name}'.")

    device = utils.get_torch_device()
    is_finetuned_model, model_config_path, output_image_size, normalization_config = (
        utils.get_model_metadata(model_name))

    src.individual_dataset.IndividualDataset.create_and_store(
        model_name=model_name,
        channels=channels,
        annotations_version=annotations_version,
        output_image_size=output_image_size,
        normalization=normalization_config,
        output_dir=model_dir,
        csv_dir=csv_dir,
    )

    dataset = torch.load(os.path.join(model_dir, "stored_dataset.pt"),
                         map_location="cpu", weights_only=False)
    model = utils.load_embedding_model(
        model_name, is_finetuned_model, model_config_path, device,
        backbone_prefix=BACKBONE_CHECKPOINT)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=EMBEDDING_BATCH_SIZE, shuffle=False)
    embeddings = _compute_embeddings(model, loader, device)

    os.makedirs(model_dir, exist_ok=True)
    torch.save(embeddings, os.path.join(model_dir, "stored_embeddings.pt"))


def store_centroid_embeddings(model_dir, suffix=""):
    """Compute and store centroid embeddings and labels from existing image embeddings in model_dir."""
    utils.print_function_banner(
        "store_centroid_embeddings",
        f"Storing centroid embeddings in dir:'{model_dir}'.")

    dataset = torch.load(os.path.join(model_dir, "stored_dataset.pt"),
                         map_location="cpu", weights_only=False)
    embeddings_np = torch.load(
        os.path.join(model_dir, f"stored_embeddings{suffix}.pt"),
        map_location="cpu", weights_only=False).cpu().detach().numpy()

    labels_np = dataset.get_labels().cpu().detach().numpy()
    centroid_embeddings, centroid_labels = _compute_centroids(embeddings_np, labels_np)

    centroid_embeddings = torch.as_tensor(centroid_embeddings)
    centroid_labels = torch.as_tensor(centroid_labels, dtype=torch.long)

    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        centroid_embeddings,
        os.path.join(model_dir, f"stored_centroid_embeddings{suffix}.pt"),
    )
    torch.save(
        centroid_labels,
        os.path.join(model_dir, f"stored_centroid_labels{suffix}.pt"),
    )


def update_embeddings(model_name, model_dir, csv_dir):
    """Incrementally update vector embeddings for new images while retaining stored vectors for unchanged images."""
    utils.print_function_banner("update_embeddings", f"Updating embeddings for model: {model_name}.")
    csv_dir = os.fspath(csv_dir)

    device = utils.get_torch_device()
    is_finetuned_model, model_config_path, output_image_size, normalization_config = (
        utils.get_model_metadata(model_name))

    stored_dataset_path = os.path.join(model_dir, "stored_dataset.pt")
    stored_embeddings_path = os.path.join(model_dir, "stored_embeddings.pt")

    with tempfile.TemporaryDirectory(
            prefix="shark-match-model-annotations-") as annotations_dir:
        src.cli.convert_data.prepare_annotations(
            train_test=False,
            input_dir=csv_dir,
            output_directory=annotations_dir,
        )

        if (not os.path.exists(stored_dataset_path)
                or not os.path.exists(stored_embeddings_path)):
            print("Stored dataset or embeddings not found. Running full "
                  "generation instead of update.")
            store_image_embeddings(
                model_name=model_name,
                model_dir=model_dir,
                annotations_version=annotations_dir,
                csv_dir=csv_dir,
            )
            return None

        # Load stored data into memory before overwriting stored_dataset.pt below.
        stored_dataset = torch.load(
            stored_dataset_path, weights_only=False, map_location='cpu')
        stored_filenames = stored_dataset.get_filenames()
        stored_embeddings = torch.load(
            stored_embeddings_path, weights_only=False, map_location='cpu')

        # Rebuild the dataset in model_dir. Safe to overwrite because stored
        # data is already in memory.
        print("Rebuilding dataset to check for changes.")
        src.individual_dataset.IndividualDataset.create_and_store(
            model_name,
            annotations_version=annotations_dir,
            output_image_size=output_image_size,
            normalization=normalization_config,
            output_dir=model_dir,
            csv_dir=csv_dir,
        )
        current_dataset = torch.load(
            stored_dataset_path, weights_only=False, map_location='cpu')
        current_filenames = current_dataset.get_filenames()

    stored_filenames_set = set(stored_filenames)
    new_filenames = [fn for fn in current_filenames if fn not in stored_filenames_set]
    updated = 0

    if not new_filenames:
        print("No new images to add.")
        if len(stored_filenames) != len(current_filenames) or stored_filenames != current_filenames:
            print("Dataset has changed (deletions/reordering). Re-assembling embeddings.")
            filename_to_embedding = dict(zip(stored_filenames, stored_embeddings))
            final_embeddings_list = [filename_to_embedding[fn] for fn in current_filenames
                                     if fn in filename_to_embedding]
            final_embeddings = torch.stack(final_embeddings_list) if final_embeddings_list else torch.empty(0)
        else:
            print("No changes to dataset images. Skipping image embedding update.")
            final_embeddings = stored_embeddings
    else:
        print(f"Found {len(new_filenames)} new images to process.")
        new_image_indices = [current_filenames.index(fn) for fn in new_filenames]
        new_images_subset = Subset(current_dataset, new_image_indices)

        model = utils.load_embedding_model(
            model_name, is_finetuned_model, model_config_path, device,
            backbone_prefix=BACKBONE_CHECKPOINT)
        new_loader = torch.utils.data.DataLoader(
            new_images_subset, batch_size=EMBEDDING_BATCH_SIZE,
            shuffle=False)
        new_embeddings = _compute_embeddings(model, new_loader, device)
        updated = len(new_embeddings)

        filename_to_stored_embedding = dict(zip(stored_filenames, stored_embeddings))
        filename_to_new_embedding = dict(zip(new_filenames, new_embeddings))
        final_embeddings_list = []
        for fn in current_filenames:
            if fn in filename_to_new_embedding:
                final_embeddings_list.append(filename_to_new_embedding[fn])
            elif fn in filename_to_stored_embedding:
                final_embeddings_list.append(filename_to_stored_embedding[fn])
        final_embeddings = torch.stack(final_embeddings_list)

    carried_over = len(final_embeddings) - updated
    assert carried_over + updated == len(current_dataset)
    print(f"Embeddings carried over: {carried_over}, updated: {updated}. "
          f"Previously: {len(stored_embeddings)}, now: {len(final_embeddings)}.")

    affected_labels = _compute_affected_labels(stored_dataset, current_dataset)
    torch.save(final_embeddings, stored_embeddings_path)

    return UpdateResult(new_filenames=new_filenames, affected_labels=affected_labels)


def update_centroid_embeddings(model_dir, update_result):
    """Incrementally update centroid embeddings by recomputing only labels whose member images changed."""
    utils.print_function_banner(
        "update_centroid_embeddings",
        f"Incrementally updating centroids in dir: {model_dir}. "
        f"Affected: {update_result.affected_labels}.")

    current_embeddings = torch.load(
        os.path.join(model_dir, "stored_embeddings.pt"),
        map_location="cpu", weights_only=False).cpu().detach().numpy()
    current_dataset = torch.load(
        os.path.join(model_dir, "stored_dataset.pt"),
        map_location="cpu", weights_only=False)
    current_labels_np = current_dataset.get_labels().cpu().detach().numpy()

    old_centroids_np = torch.load(
        os.path.join(model_dir, "stored_centroid_embeddings.pt"),
        map_location="cpu", weights_only=False).cpu().detach().numpy()
    old_centroid_labels_np = torch.load(
        os.path.join(model_dir, "stored_centroid_labels.pt"),
        map_location="cpu", weights_only=False).cpu().detach().numpy()
    old_map = {int(lbl): old_centroids_np[i]
               for i, lbl in enumerate(old_centroid_labels_np)}

    affected_set = set(update_result.affected_labels)

    # Identify labels present in the current dataset and decide which need fresh
    # computation: any label affected by the update or newly present.
    unique_labels = sorted({int(l) for l in current_labels_np})
    fresh_labels_set = {
        lbl for lbl in unique_labels
        if lbl in affected_set or lbl not in old_map
    }

    # Compute fresh centroids in a single batch through the same code path as
    # the full recomputation, ensuring bit-identical floating-point behavior.
    fresh_map = {}
    if fresh_labels_set:
        mask = np.isin(current_labels_np, list(fresh_labels_set))
        fresh_centroids, fresh_labels_list = _compute_centroids(
            current_embeddings[mask], current_labels_np[mask])
        fresh_map = {int(lbl): vec for lbl, vec in zip(fresh_labels_list, fresh_centroids)}

    final_centroid_vectors = []
    final_centroid_labels = []
    reused_count = 0
    recomputed_count = 0
    for lbl in unique_labels:
        if lbl in fresh_map:
            final_centroid_vectors.append(list(fresh_map[lbl]))
            recomputed_count += 1
        else:
            final_centroid_vectors.append(list(old_map[lbl]))
            reused_count += 1
        final_centroid_labels.append(lbl)

    print(f"Centroids reused: {reused_count}, recomputed: {recomputed_count}.")

    centroid_embeddings_tensor = torch.as_tensor(final_centroid_vectors)
    centroid_labels_tensor = torch.as_tensor(final_centroid_labels, dtype=torch.long)

    print("Saving incrementally updated centroid embeddings.")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(centroid_embeddings_tensor,
               os.path.join(model_dir, "stored_centroid_embeddings.pt"))
    torch.save(centroid_labels_tensor,
               os.path.join(model_dir, "stored_centroid_labels.pt"))


def regenerate_dataset_only(model_name: str, embeddings_name: str,
                            args: argparse.Namespace, model_dir,
                            csv_dir=None):
    """Regenerate stored_dataset.pt inside publisher staging while preserving existing embeddings."""
    csv_dir = csv_dir or utils.resolve_csv_dir(args)
    if csv_dir is None:
        raise RuntimeError(
            "No CSV directory could be resolved. Pass --csv-dir or "
            "regenerate annotations with src.cli.convert_data so a "
            f"{constants.SOURCE_CSV_DIR_SIDECAR} sidecar is written."
        )

    _, _, output_image_size, normalization_config = (
        utils.get_model_metadata(model_name))
    channels = 3

    dataset_path = os.path.join(model_dir, "stored_dataset.pt")
    previous_filenames = None
    if os.path.exists(dataset_path):
        previous_filenames = list(torch.load(
            dataset_path, weights_only=False, map_location='cpu').get_filenames())

    src.individual_dataset.IndividualDataset.create_and_store(
        model_name=model_name,
        channels=channels,
        annotations_version=args.annotations_version,
        output_dir=model_dir,
        csv_dir=csv_dir,
        output_image_size=output_image_size,
        normalization=normalization_config,
    )

    new_dataset = torch.load(
        dataset_path, weights_only=False, map_location='cpu')
    new_filenames = list(new_dataset.get_filenames())
    new_count = len(new_filenames)
    if previous_filenames is not None and new_filenames != previous_filenames:
        raise RuntimeError(
            "Regenerated dataset filenames differ in content or order from "
            "the stored dataset, so its rows would not match "
            "stored_embeddings.pt. Aborting to keep indices aligned. Run a "
            "full embeddings regeneration instead."
        )
    embeddings_path = os.path.join(model_dir, "stored_embeddings.pt")
    if os.path.exists(embeddings_path):
        stored_embeddings = torch.load(
            embeddings_path, weights_only=False, map_location='cpu')
        if stored_embeddings.shape[0] != new_count:
            raise RuntimeError(
                f"Regenerated dataset has {new_count} filenames but "
                f"stored_embeddings.pt has {stored_embeddings.shape[0]} "
                "rows. Aborting to keep indices aligned. Run a full "
                "embeddings regeneration instead."
            )

    print("Regenerated stored_dataset.pt in staged generation for "
          f"{embeddings_name} (csv_dir={csv_dir}).")
    return new_count


def remove_image_from_embeddings(model_dir, filename):
    """Remove a specific image's embeddings and dataset records directly from cached files in model_dir."""
    utils.print_function_banner("remove_image_from_embeddings", f"Attempting to remove {filename} from {model_dir}")

    dataset_path = os.path.join(model_dir, "stored_dataset.pt")
    if not os.path.exists(dataset_path):
        return

    try:
        dataset = torch.load(dataset_path, weights_only=False, map_location="cpu")
        filenames = dataset.get_filenames()
        if filename not in filenames:
            return

        idx = filenames.index(filename)
        print(f"Removing '{filename}' at index {idx} from dataset and embeddings.")

        dataset.filenames.pop(idx)
        if hasattr(dataset, "texts") and dataset.texts:
            dataset.texts.pop(idx)
        dataset.labels = torch.cat([dataset.labels[:idx], dataset.labels[idx+1:]])
        if hasattr(dataset, "ids") and getattr(dataset, "ids", None):
            if isinstance(dataset.ids, list):
                dataset.ids.pop(idx)
            else:
                dataset.ids = torch.cat([dataset.ids[:idx], dataset.ids[idx+1:]])
        torch.save(dataset, dataset_path)

        embeddings_path = os.path.join(model_dir, "stored_embeddings.pt")
        if os.path.exists(embeddings_path):
            embeddings = torch.load(embeddings_path, weights_only=False, map_location="cpu")
            embeddings = torch.cat([embeddings[:idx], embeddings[idx+1:]])
            torch.save(embeddings, embeddings_path)

        neighbors_path = os.path.join(model_dir, "stored_image_neighbors.pt")
        if os.path.exists(neighbors_path):
            neighbors = torch.load(neighbors_path, weights_only=False, map_location="cpu")
            if filename in neighbors:
                del neighbors[filename]
            torch.save(neighbors, neighbors_path)

        metadata_path = os.path.join(model_dir, "embeddings_metadata.json")
        if os.path.exists(metadata_path):
            metadata = utils.load_json_file(metadata_path)
            if metadata:
                metadata["update_time"] = time.time()
                utils.save_json_file(metadata, metadata_path)

    except Exception as e:
        print(f"Error while removing image from embeddings in {model_dir}: {e}")


def generate_and_store_suggestions(embeddings_name: str, model_dir,
                                   retrieval_mode="baseline",
                                   raise_on_error=False):
    """Compute and persist review suggestions in model_dir using the standard retrieval path."""
    try:
        # Lazy imports avoid a circular dependency with retrieval and suggestions modules.
        import src.retrieval.retrieval  # pylint: disable=import-outside-toplevel
        import src.retrieval.suggestions  # pylint: disable=import-outside-toplevel
        target_dir = os.path.normpath(os.fspath(model_dir))
        model_spec = {
            "display_name": embeddings_name,
            "dir": os.path.basename(target_dir),
            "model": embeddings_name,
            "retrieval_mode": retrieval_mode,
            "rerank": "none",
        }
        suggestions_retriever = src.retrieval.retrieval.Retriever(
            preload_models=[model_spec],
            embeddings_root=os.path.dirname(target_dir),
        )
        src.retrieval.suggestions.store_suggestions(
            suggestions_retriever, target_dir)
    except Exception as suggestions_err:  # pylint: disable=broad-exception-caught
        print(f"WARNING: Review suggestion generation failed for "
              f"'{embeddings_name}': {suggestions_err}. "
              "suggestions.pt was not written.")
        if raise_on_error:
            raise RuntimeError(
                f"Review suggestion generation failed for '{embeddings_name}'"
            ) from suggestions_err


def _should_store_neighbors(args) -> bool:
    """Return whether neighbor dictionary artifacts should be computed and stored."""
    return bool(
        getattr(args, "store_neighbors", False)
        or getattr(args, "use_psl", False)
        or getattr(args, "stacked", False)
    )


def _compute_embeddings(model, loader, device):
    """Compute L2-normalized vector embeddings for all images in the dataloader."""
    embeddings_list = []

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Computing embeddings"):
            if isinstance(batch, list) or isinstance(batch, tuple):
                input_images = batch[0]
            else:
                input_images = batch

            input_images = input_images.to(device)
            output = model(input_images)

            # Squeeze batch dimension if present to ensure 2D tensor for batch=1 cases if model output 3D.
            if output.dim() > 2 and output.size(0) == 1:
                output = output.squeeze(0)

            # For some models (like direct timm or HF), output might need reshaping if it's [B, 1, D] -> [B, D].
            if output.dim() == 3 and output.size(1) == 1:
                output = output.squeeze(1)

            embeddings_list.append(output.cpu())

    if len(embeddings_list) > 0:
        embeddings = torch.cat(embeddings_list, dim=0)
    else:
        return torch.tensor([])

    if embeddings.dim() == 3 and embeddings.size(1) == 1:
        embeddings = embeddings.squeeze(1)
    elif embeddings.dim() != 2:
        embeddings = embeddings.view(embeddings.size(0), -1)

    normalized_np = src.retrieval.rerank.l2_normalize(
        embeddings.detach().cpu().numpy(), axis=1, eps=1e-12
    )
    return torch.from_numpy(normalized_np)


def _compute_centroids(embeddings, labels):
    """Compute one L2-renormalized centroid vector per unique individual label."""
    centroids_np, unique_labels_np = src.retrieval.rerank.compute_centroids(
        embeddings, labels
    )
    centroid_embeddings = [list(row) for row in centroids_np]
    centroid_labels = [int(lbl) for lbl in unique_labels_np]
    return centroid_embeddings, centroid_labels


def _compute_affected_labels(old_dataset, new_dataset):
    """Identify individual labels whose image membership changed between old and new dataset versions."""
    def build_filename_map(dataset):
        """Map each dataset filename to its integer identity label."""
        filenames = dataset.get_filenames()
        labels = dataset.get_labels().cpu().detach().numpy()
        return {fn: int(labels[i]) for i, fn in enumerate(filenames)}

    old_map = build_filename_map(old_dataset)
    new_map = build_filename_map(new_dataset)

    affected = set()
    for fn in set(old_map) | set(new_map):
        old_label = old_map.get(fn)
        new_label = new_map.get(fn)
        if old_label == new_label:
            continue
        if old_label is not None:
            affected.add(old_label)
        if new_label is not None:
            affected.add(new_label)
    return affected


def _stored_dataset_image_count(model_dir) -> int:
    """Return the filename count from stored_dataset.pt in model_dir."""
    dataset_path = os.path.join(model_dir, "stored_dataset.pt")
    try:
        stored_dataset = torch.load(
            dataset_path, weights_only=False, map_location='cpu'
        )
        return len(stored_dataset.get_filenames())
    except Exception as read_error:  # pylint: disable=broad-exception-caught
        raise RuntimeError(
            f"Unable to read stored dataset image count from {dataset_path}"
        ) from read_error


def _run_multimodal_pipeline(
        staging_dir: str, seed: int = 42, store_neighbors: bool = False):
    """Run late-fusion multimodal pipeline, optimizing alphas and generating fused artifacts in staging_dir."""
    import json  # pylint: disable=import-outside-toplevel
    from src.multimodal import optimize_late_fusion as optimizer  # pylint: disable=import-outside-toplevel
    from src.multimodal import fusion_dataset as fusion_dataset_mod  # pylint: disable=import-outside-toplevel
    import src.multimodal.metadata_features as metadata_features  # pylint: disable=import-outside-toplevel

    utils.print_function_banner("_run_multimodal_pipeline",
                                f"Generating multimodal embeddings in: {staging_dir}")

    device = utils.get_torch_device()

    stored_dataset = torch.load(
        os.path.join(staging_dir, "stored_dataset.pt"),
        map_location="cpu", weights_only=False)
    image_embeddings = torch.load(
        os.path.join(staging_dir, "stored_embeddings.pt"),
        map_location="cpu", weights_only=False)

    # Build FusionDataset with default metadata spec (all six features).
    spec = metadata_features._resolve_metadata_spec(
        None, representation_b=False, representation_c=False)
    # fit_split="all" so vocabularies and numeric ranges are computed from the
    # full corpus.  fit_split="train" would produce empty stats when the
    # stored_dataset has no train/test split (train_test=False annotation prep),
    # causing zero-dim vocabulary groups and a degenerate alpha optimisation.
    fusion_ds = fusion_dataset_mod.FusionDataset(
        stored_dataset, image_embeddings, spec, split=None, fit_split="all")
    slices = metadata_features._group_slices(spec, fusion_ds.norm_stats)
    group_order = tuple(slices.keys())

    config_path = os.path.join(staging_dir, "late_fusion_config.json")

    # Reuse existing alphas or fall back to image-only when lacking train IDs to avoid empty evaluation queries.
    has_train_ids = len(fusion_ds.get_train_ids()) > 0

    if not has_train_ids:
        if os.path.exists(config_path):
            print("[Multimodal] No train/test split — reusing existing late_fusion_config.json.")
            with open(config_path, encoding="utf-8") as f:
                existing_cfg = json.load(f)
            alphas = existing_cfg.get("alphas", {"alpha_image": 1.0})
            print(f"[Multimodal] Reused alphas: {alphas}")
        else:
            print("[Multimodal] No train/test split and no existing config — using image-only alpha.")
            alphas = {"alpha_image": 1.0}
    else:
        # Prepare K-fold cross-validation data for alpha search.
        full_ids = list(fusion_ds.get_ids())
        id_to_index = {mid: i for i, mid in enumerate(full_ids)}
        labels = fusion_ds.get_labels().to(device)

        print("[Multimodal] Building 5-fold cross-validation data...")
        seed_data = optimizer._make_kfold_seed_data(
            fusion_ds, k=5, seed=seed, slices=slices, device=device,
            id_to_index=id_to_index, labels=labels, min_instances=2,
        )

        # Random search then coordinate refinement.
        (_, best_w, best_sc), _ = optimizer._random_search(
            seed_data, "hits_at_10", n_trials=1000, low=0.001, high=1.0,
            seed=seed, group_order=group_order, tune_image=False, tied_metadata=False)
        (_, best_w, best_sc), _ = optimizer._coordinate_refine(
            best_w, seed_data, "hits_at_10",
            group_order=group_order, tune_image=False, tied_metadata=False)

        alphas = optimizer._weights_to_alphas(best_w)
        print(f"[Multimodal] Optimised alphas: {alphas}")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "config_name": "multimodal_tuned",
                "alphas": alphas,
                "weights": best_w,
                "score": best_sc,
                "metadata_spec": spec.to_dict(),
            }, f, indent=2)
        print(f"[Multimodal] Saved late fusion config to {config_path}")

    # Build fused embeddings in pure NumPy: weighted concatenation of L2-normalised streams.
    img_np = fusion_ds.image_embeddings.detach().cpu().numpy().astype(np.float32)
    meta_np = fusion_ds.metadata.detach().cpu().numpy().astype(np.float32)
    img_l2 = src.retrieval.rerank.l2_normalize(img_np, axis=1, eps=1e-8)
    group_l2 = {
        g: src.retrieval.rerank.l2_normalize(meta_np[:, a:b], axis=1, eps=1e-8)
        for g, (a, b) in slices.items()
    }

    streams = []
    a_img = float(alphas.get("alpha_image", 0.0))
    if a_img > 0.0:
        streams.append(a_img * img_l2)
    for group in group_order:
        if group not in group_l2:
            continue
        a = float(alphas.get(f"alpha_{group}", 0.0))
        if a > 0.0:
            streams.append(a * group_l2[group])

    fused_np = np.concatenate(streams, axis=1)
    fused_np = src.retrieval.rerank.l2_normalize(fused_np, axis=1, eps=1e-8)
    fused = torch.from_numpy(fused_np)

    torch.save(fused, os.path.join(staging_dir, "stored_embeddings_multimodal.pt"))
    print(f"[Multimodal] Saved fused embeddings (shape {tuple(fused.shape)})")

    if store_neighbors:
        src.retrieval.neighbors.store_neighbors(
            staging_dir, centroid=False, transformed=False, suffix="_multimodal")

    # Compute centroids using fusion_ds labels to prevent out-of-bounds access if filtered IDs make len(fusion_ds) < len(stored_dataset).
    centroid_embeddings, centroid_labels = _compute_centroids(
        fused_np,
        fusion_ds.get_labels().cpu().detach().numpy())
    centroid_embeddings_tensor = torch.as_tensor(centroid_embeddings)
    centroid_labels_tensor = torch.as_tensor(
        centroid_labels, dtype=torch.long)
    torch.save(centroid_embeddings_tensor,
               os.path.join(staging_dir, "stored_centroid_embeddings_multimodal.pt"))
    torch.save(centroid_labels_tensor,
               os.path.join(staging_dir, "stored_centroid_labels_multimodal.pt"))

    if store_neighbors:
        src.retrieval.neighbors.store_neighbors(
            staging_dir, centroid=True, transformed=False, suffix="_multimodal")

    print("[Multimodal] Pipeline complete.")


def _require_psl_runtime():
    """Exits with install guidance unless pslpython and a discoverable Java runtime are available."""
    # pylint: disable=import-outside-toplevel, import-error, unused-import
    try:
        import pslpython.runtime
    except ImportError as error:
        sys.exit(f"PSL reranking needs the pslpython package ({error}). "
                 "See src/psl/README.md for the install steps.")
    import jpype
    try:
        jpype.getDefaultJVMPath()
    # JPype raises CalledProcessError, not JVMNotFoundException, when macOS has no JVM.
    except (jpype.JVMNotFoundException, subprocess.CalledProcessError, OSError):
        sys.exit("PSL reranking needs a Java 17+ runtime that JPype can find "
                 "(set JAVA_HOME). See src/psl/README.md.")


def _build_generation_artifacts(model_name: str, embeddings_name: str,
                                args: argparse.Namespace, staging_dir,
                                initial_csv_dir, retrieval_mode: str) -> int:
    """Build all selected computation artifacts inside a publisher-owned staging directory."""
    channels = 3
    use_psl = getattr(args, 'use_psl', False)
    psl_seed = getattr(args, 'psl_seed', 42)
    use_multimodal = getattr(args, 'use_multimodal', False)
    use_stacked = getattr(args, 'stacked', False)
    if use_stacked:
        use_multimodal = True
    store_neighbors = _should_store_neighbors(args)
    psl_weights = None

    if not store_neighbors and os.path.isdir(staging_dir):
        for entry in os.listdir(staging_dir):
            if entry.startswith((
                    "stored_image_neighbors",
                    "stored_centroid_neighbors",
            )):
                try:
                    os.remove(os.path.join(staging_dir, entry))
                except OSError:
                    pass

    update_result = None
    if args.update_embeddings:
        print("Re-caching dataset, updating embeddings as needed and "
              "re-computing centroids for "
              f"model: {model_name} embeddings: {embeddings_name}.")
        update_result = update_embeddings(
            model_name, staging_dir, csv_dir=initial_csv_dir)
    else:
        print("Caching dataset, full embeddings generation, and centroid "
              f"computation for model: {model_name} "
              f"embeddings: {embeddings_name}.")
        store_image_embeddings(
            model_name=model_name,
            model_dir=staging_dir,
            annotations_version=args.annotations_version,
            channels=channels,
            csv_dir=initial_csv_dir,
        )

    old_centroid_files_present = (
        os.path.exists(os.path.join(staging_dir, "stored_centroid_embeddings.pt")) and
        os.path.exists(os.path.join(staging_dir, "stored_centroid_labels.pt"))
    )
    if update_result is not None and old_centroid_files_present:
        update_centroid_embeddings(
            model_dir=staging_dir, update_result=update_result)
    else:
        store_centroid_embeddings(model_dir=staging_dir)

    if store_neighbors:
        if update_result is not None:
            src.retrieval.neighbors.update_image_neighbors(
                model_dir=staging_dir,
                new_filenames=update_result.new_filenames,
            )
        else:
            src.retrieval.neighbors.store_neighbors(
                model_dir=staging_dir, centroid=False)

        old_centroid_neighbors_present = os.path.exists(
            os.path.join(staging_dir, "stored_centroid_neighbors.pt")
        )
        if (update_result is not None and old_centroid_files_present
                and old_centroid_neighbors_present):
            src.retrieval.neighbors.update_centroid_neighbors(
                model_dir=staging_dir, update_result=update_result)
        else:
            src.retrieval.neighbors.store_neighbors(
                model_dir=staging_dir, centroid=True)

    if use_multimodal:
        _run_multimodal_pipeline(
            staging_dir, seed=psl_seed, store_neighbors=store_neighbors
        )

    if use_psl or use_stacked:
        from src.psl.psl_engine import load_metadata_from_csvs
        from src.psl.train_psl import learn_weights as _psl_learn_weights
        from src.psl.psl_rules import save_weights as _psl_save_weights
        staged_dataset_for_psl = torch.load(
            os.path.join(staging_dir, "stored_dataset.pt"),
            weights_only=False,
            map_location='cpu',
        )
        psl_csv_dir = getattr(staged_dataset_for_psl, 'csv_dir', None)
        if not psl_csv_dir:
            raise RuntimeError(
                "--use-psl is enabled but the staged dataset has no csv_dir "
                "attribute. Regenerate the dataset with a current build of "
                "src.retrieval.embeddings."
            )
        psl_csv_dir = utils.resolve_relative_csv_dir(psl_csv_dir)
        metadata = load_metadata_from_csvs(psl_csv_dir)

        staged_neighbors = torch.load(
            os.path.join(staging_dir, "stored_image_neighbors.pt"),
            map_location='cpu',
            weights_only=False,
        )
        print("[PSL] Learning weights from training split...")
        psl_weights = _psl_learn_weights(
            staged_dataset_for_psl,
            staged_neighbors,
            metadata,
            psl_top_k=100,
            seed=psl_seed,
        )
        _psl_save_weights(psl_weights, staging_dir)

        src.retrieval.neighbors.store_psl_neighbors(
            model_dir=staging_dir,
            metadata=metadata,
            psl_seed=psl_seed,
            psl_weights=psl_weights,
            psl_top_k=100,
        )
        if use_stacked:
            src.retrieval.neighbors.store_psl_neighbors(
                model_dir=staging_dir,
                metadata=metadata,
                psl_seed=psl_seed,
                psl_weights=psl_weights,
                psl_top_k=100,
                suffix="_multimodal",
            )

    generate_and_store_suggestions(
        embeddings_name,
        model_dir=staging_dir,
        retrieval_mode=retrieval_mode,
    )
    return _stored_dataset_image_count(staging_dir)


def _validate_pipeline_inputs(model_name: str, embeddings_name: str,
                               args: argparse.Namespace) -> bool:
    """Validate all required paths and configuration before running the pipeline."""
    ok = True
    path_checks = {}

    # Images directory.
    images_ok = os.path.exists(constants.IMAGES_DIR)
    path_checks["Images Directory"] = f"{constants.IMAGES_DIR} ({'OK' if images_ok else 'MISSING'})"
    if not images_ok:
        ok = False

    # Annotations.
    if args.annotations_version:
        full_ann = os.path.join(
            constants.ANNOTATIONS_DIR, args.annotations_version, constants.FULL_ANNOTATIONS)
        full_ann_ok = os.path.exists(full_ann)
        path_checks["Full Annotations"] = f"{full_ann} ({'OK' if full_ann_ok else 'MISSING'})"
        if not full_ann_ok:
            ok = False
        try:
            train_ann, valid_ann = utils.get_annotation_paths(args.annotations_version)
            for label, path in [("Train Annotations", train_ann), ("Valid Annotations", valid_ann)]:
                exists = os.path.exists(path)
                path_checks[label] = f"{path} ({'OK' if exists else 'MISSING'})"
                if not exists:
                    ok = False
        except FileNotFoundError as e:
            path_checks["Annotations Split"] = f"ERROR: {e}"
            ok = False
    else:
        ann_dir_ok = os.path.exists(constants.ANNOTATIONS_DIR)
        path_checks["Annotations Directory"] = (
            f"{constants.ANNOTATIONS_DIR} ({'OK' if ann_dir_ok else 'MISSING'})")
        if not ann_dir_ok:
            ok = False

    # Trained model paths — only applicable to fine-tuned models.
    if model_name.strip().lower() not in constants.PRECANNED_MODELS:
        trained_model_dir = os.path.join(constants.TRAINED_MODELS_DIR, model_name)
        config_path = os.path.join(trained_model_dir, "config.json")
        for label, path in [("Trained Model Dir", trained_model_dir), ("Model Config", config_path)]:
            exists = os.path.exists(path)
            path_checks[label] = f"{path} ({'OK' if exists else 'MISSING'})"
            if not exists:
                ok = False

    # Embeddings output directory (informational — created on demand).
    path_checks["Embeddings Output Dir"] = os.path.join(constants.EMBEDDINGS_DIR, embeddings_name)

    # CSV directory (required by full and incremental generation).
    resolved_csv = utils.resolve_csv_dir(args)
    if resolved_csv:
        path_checks["CSV Dir"] = f"{resolved_csv} (OK)"
    else:
        path_checks["CSV Dir"] = (
            "MISSING (--csv-dir not given and no sidecar in annotations dir)"
        )
        ok = False

    settings = {
        "Model Name": model_name,
        "Embeddings Name": embeddings_name,
        "Update Mode": args.update_embeddings,
    }
    utils.print_key_value_section("Resolved Settings", settings)
    utils.print_key_value_section("Path Verification", path_checks)
    if not ok:
        print("Validation FAILED: one or more required paths are missing (see above).")
    return ok


if __name__ == "__main__":
    main()
