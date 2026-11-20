# Import order ensures MPS fallback is configured before torch initialization and helpers load.
# pylint: disable=wrong-import-position, protected-access
"""Diagnostic tool to compare embeddings across models and detect drift against stored embeddings on disk."""
import argparse
import os
import sys
import tempfile

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import constants
import src.individual_dataset
import src.retrieval.embeddings as embeddings_mod
import src.retrieval.retrieval
from src.core import utils

VALID_EXTENSIONS = (".jpeg", ".jpg", ".png")


def resolve_image(name):
    """Return the on-disk filename for `name`, trying known extensions.

    Raises rather than warning: the original appended an unresolved name and
    carried on, which silently misaligned every later index.
    """
    if os.path.exists(os.path.join(constants.IMAGES_DIR, name)):
        return name
    for ext in VALID_EXTENSIONS:
        candidate = f"{name}{ext}"
        if os.path.exists(os.path.join(constants.IMAGES_DIR, candidate)):
            return candidate
    raise FileNotFoundError(
        f"Image '{name}' not found in {constants.IMAGES_DIR} "
        f"(tried bare name and {', '.join(VALID_EXTENSIONS)}).")


def write_annotations(image_names, path):
    """Write the headerless `id<TAB>filename<TAB>label` TSV the dataset reads."""
    with open(path, "w", encoding="utf-8") as fh:
        for index, name in enumerate(image_names):
            fh.write(f"{index}\t{name}\t0\n")


def build_dataset(model_name, annotations_path, device):
    """Construct an IndividualDataset using the model's own preprocessing."""
    _, _, output_image_size, normalization = utils.get_model_metadata(model_name)
    return src.individual_dataset.IndividualDataset(
        images_path=constants.IMAGES_DIR,
        annotations_path=annotations_path,
        output_image_size=output_image_size,
        normalization=normalization,
        device=device,
    )


def generate_embeddings(model_name, annotations_path, device):
    """Embed the images through the production path.

    Deliberately delegates to `EMBEDDING_BATCH_SIZE` and `_compute_embeddings`
    rather than repeating them: a local copy is exactly how this tool drifted
    away from production in the first place.
    """
    is_finetuned, config_path, _, _ = utils.get_model_metadata(model_name)
    dataset = build_dataset(model_name, annotations_path, device)
    model = utils.load_embedding_model(model_name, is_finetuned, config_path, device)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=embeddings_mod.EMBEDDING_BATCH_SIZE,
        shuffle=False)
    return embeddings_mod._compute_embeddings(model, loader, device).cpu()


def load_stored_embeddings(model_name, image_names, embeddings_name=None):
    """Pull the stored embedding rows for `image_names` off disk, from the set `embeddings_name` when given."""
    spec = model_name
    if embeddings_name:
        spec = {"display_name": model_name, "dir": embeddings_name,
                "model": model_name, "retrieval_mode": "baseline",
                "rerank": "none"}
    retriever = src.retrieval.retrieval.Retriever(preload_models=[spec])
    # Deliberate protected access: _get_model_data is where filenames and
    # embeddings live, and there is no public accessor for them. The retrieval
    # unit tests reach for it the same way.
    data = retriever._get_model_data(model_name)  # pylint: disable=protected-access
    if not data:
        raise RuntimeError(
            f"No stored data loaded for '{model_name}'. Generate embeddings "
            f"first, or check it is a known model.")

    stored_filenames = list(data["filenames"])
    index_of = {name: i for i, name in enumerate(stored_filenames)}
    missing = [n for n in image_names if n not in index_of]
    if missing:
        raise RuntimeError(
            f"{len(missing)} image(s) absent from stored embeddings for "
            f"'{model_name}': {missing[:5]}")

    rows = data["embeddings"]
    if not torch.is_tensor(rows):
        rows = torch.as_tensor(rows)
    return torch.stack([rows[index_of[n]] for n in image_names]).cpu()


def get_processed_images(model_name, annotations_path, device):
    """Return the preprocessed image tensors, before any model runs.

    Separates "the two models preprocess differently" from "the two models
    embed differently", which the embedding comparison alone cannot do.
    """
    dataset = build_dataset(model_name, annotations_path, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    batches = [(b[0] if isinstance(b, (list, tuple)) else b) for b in loader]
    return torch.cat(batches, dim=0).cpu()


def report(tensors, label):
    """Print every pairwise comparison. Returns True if any pair is identical."""
    names = list(tensors)
    print(f"\n--- {label}: {len(names)} source(s) ---")
    for name in names:
        print(f"  {name:38} shape={tuple(tensors[name].shape)}")

    if len(names) < 2:
        print("\nNothing to compare -- need at least two sources.")
        return False

    print(f"\n{'pair':<52} {'max abs diff':>14}  verdict")
    any_identical = False
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a, b = tensors[first], tensors[second]
            pair = f"{first} vs {second}"
            if a.shape != b.shape:
                print(f"{pair:<52} {'-':>14}  DIFFERENT SHAPE")
                continue
            diff = (a - b).abs().max().item()
            if torch.equal(a, b):
                verdict = "IDENTICAL (bitwise)"
                any_identical = True
            elif torch.allclose(a, b, atol=1e-5):
                verdict = "identical within 1e-5"
                any_identical = True
            else:
                verdict = "different"
            print(f"{pair:<52} {diff:14.3e}  {verdict}")
    return any_identical


def main():
    """Parse arguments, gather the requested tensors, and report. Returns an exit code."""
    parser = argparse.ArgumentParser(
        description="Compare embeddings across models, or against what is "
                    "stored on disk.")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names to compare.")
    parser.add_argument("--images", nargs="+", required=True,
                        help="Image filenames, resolved against IMAGES_DIR.")
    parser.add_argument("--embeddings-name", default=None,
                        help="Stored embedding set to read when it is not "
                             "named after the model (as with the embeddings "
                             "command's --embeddings-name).")
    parser.add_argument("--use-stored", "--use_stored", action="store_true",
                        dest="use_stored",
                        help="Read embeddings off disk instead of generating "
                             "them.")
    parser.add_argument("--compare-generated-to-stored", action="store_true",
                        help="For each model, compare freshly generated "
                             "embeddings against the stored ones. This is the "
                             "drift check.")
    parser.add_argument("--compare-images", "--compare_images",
                        action="store_true", dest="compare_images",
                        help="Compare preprocessed image tensors instead of "
                             "embeddings, to tell preprocessing differences "
                             "apart from model differences.")
    args = parser.parse_args()

    try:
        images = [resolve_image(name) for name in args.images]
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Models: {args.models}")
    print(f"Images: {images}")

    device = utils.get_torch_device()
    tensors = {}

    with tempfile.TemporaryDirectory(prefix="embedding_identity_") as workdir:
        annotations_path = os.path.join(workdir, "annotations.tsv")
        write_annotations(images, annotations_path)

        try:
            for position, model_name in enumerate(args.models, start=1):
                # Repeated names get a position suffix so each load is its own source.
                label = (model_name if args.models.count(model_name) == 1
                         else f"{model_name} #{position}")
                if args.compare_images:
                    tensors[label] = get_processed_images(
                        model_name, annotations_path, device)
                elif args.compare_generated_to_stored:
                    tensors[f"{label} [generated]"] = generate_embeddings(
                        model_name, annotations_path, device)
                    tensors[f"{label} [stored]"] = load_stored_embeddings(
                        model_name, images, args.embeddings_name)
                elif args.use_stored:
                    tensors[label] = load_stored_embeddings(
                        model_name, images, args.embeddings_name)
                else:
                    tensors[label] = generate_embeddings(
                        model_name, annotations_path, device)
        except (RuntimeError, FileNotFoundError, KeyError) as exc:
            print(f"ERROR: {exc}")
            return 1

    label = "Preprocessed images" if args.compare_images else "Embeddings"
    report(tensors, label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
