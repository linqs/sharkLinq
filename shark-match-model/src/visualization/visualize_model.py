# pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
"""Visualize model embeddings, distance distributions, and angular characteristics."""

import argparse
import datetime
import json
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not found. Embedding projection will use simple slicing.")
    # Gracefully degrade when sklearn is unavailable.

from src.core import constants
import src.config.config  # pylint: disable=unused-import
from src.core import utils
from src.retrieval import generation_store


class ModelWrapper:
    """Lightweight wrapper to hold projection head and backbone modules."""

    def __init__(self, head, backbone):
        """Initialize the wrapper with projection head and backbone."""
        self.projection_head = head
        self.backbone = backbone


def main():
    """Visualize the feature space, distance distributions, and angular properties of a trained model."""
    parser = argparse.ArgumentParser(
        description="Visualize embedding feature space for a trained model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m src.visualization.visualize_model "
               "--model-name test-model/vit_triplet_1\n"
               "  python3 -m src.visualization.visualize_model "
               "--model-name test-model/vit_triplet_1 "
               "--output-dir sm-data/data-out/visualizations/test-run\n")
    parser.add_argument('--model-name', type=str, required=True,
                        help='Embedding set name under sm-data/embeddings/; '
                             'ArcFace angle plots also load the trained model of this name.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output plots. Default: '
                             'timestamped directory under '
                             'sm-data/data-out/visualizations/.')
    args = parser.parse_args()

    utils.print_script_banner(__file__, args)

    embeddings, labels = load_embeddings_and_labels(args.model_name)

    # --- Resolve output directory. ---
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.output_dir = os.path.join(constants.VISUALIZATIONS_DIR, timestamp)
        print(f"Output directory not specified. Using: {args.output_dir}")
    else:
        args.output_dir = utils.resolve_directory_path(
            os.path.join(constants.VISUALIZATIONS_DIR, args.output_dir),
            constants.VISUALIZATIONS_DIR
        )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loaded {len(embeddings)} embeddings.")

    # Generate projection and distance plots for the loaded embeddings.
    project_embeddings(
        embeddings, labels,
        os.path.join(args.output_dir, "embedding_projection.png")
    )
    plot_distance_distributions(
        embeddings, labels,
        os.path.join(args.output_dir, "distance_distribution.png")
    )

    # Generate angular plot if the model uses ArcFace loss function.
    config_path = os.path.join(
        constants.TRAINED_MODELS_DIR, args.model_name, "config.json"
    )
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        loss_name = cfg.get("training", {}).get("loss", {}).get("name", "")
        if "arcface" in loss_name.lower():
            print("ArcFace model detected. Loading model for angular analysis...")
            try:
                head, backbone = utils.load_model(args.model_name)
                model = ModelWrapper(head, backbone)
                plot_angular_distributions(
                    model, embeddings, labels,
                    os.path.join(args.output_dir, "angular_distribution.png")
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Failed to load model for angular analysis: {e}")
                traceback.print_exc()
        else:
            print(f"Loss is '{loss_name}', skipping ArcFace angular analysis.")


def load_embeddings_and_labels(embedding_set_name):
    """Load the active generation's image embeddings and labels, exiting with status 1 when either file is missing."""
    embeddings_dir = generation_store.resolve_active_directory(embedding_set_name)
    emb_path = os.path.join(embeddings_dir, "stored_embeddings.pt")
    ds_path = os.path.join(embeddings_dir, "stored_dataset.pt")
    for path in (emb_path, ds_path):
        if not os.path.exists(path):
            print(f"Error: {path} not found; build the embeddings with src.retrieval.embeddings first.")
            sys.exit(1)

    embeddings = torch.load(emb_path, map_location='cpu', weights_only=False).numpy()
    dataset = torch.load(ds_path, map_location='cpu', weights_only=False)
    return embeddings, dataset.get_labels().numpy()


def project_embeddings(embeddings, labels, output_path="embedding_projection.png"):
    """Project embeddings to 2D using PCA then t-SNE and save the scatter plot."""
    if not SKLEARN_AVAILABLE:
        print("Skipping projection due to missing sklearn.")
        return

    utils.print_function_banner("project_embeddings")

    # Use PCA first to reduce to 50 dims if needed, then apply t-SNE.
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
        embeddings_reduced = pca.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_reduced)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter)
    plt.title("t-SNE Projection of Embeddings")
    plt.savefig(output_path)
    print(f"Saved embedding projection to {output_path}")


def plot_distance_distributions(embeddings, labels,
                                output_path="distance_distribution.png"):
    """Plot histograms of positive vs negative pair Euclidean distances."""
    utils.print_function_banner("plot_distance_distributions")
    print("Computing pairwise distances...")
    num_samples = embeddings.shape[0]
    if num_samples > 2000:
        print(f"Subsampling from {num_samples} to 2000 for distance plotting.")
        indices = np.random.choice(num_samples, 2000, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    dist_matrix = torch.cdist(
        torch.from_numpy(embeddings),
        torch.from_numpy(embeddings),
        p=2.0
    )

    # Build positive/negative masks for distance separation.
    labels_tensor = torch.from_numpy(labels)
    mask_pos = (labels_tensor.unsqueeze(1) == labels_tensor.unsqueeze(0))
    mask_pos.fill_diagonal_(False)

    mask_neg = ~mask_pos
    mask_neg.fill_diagonal_(False)

    pos_dists = dist_matrix[mask_pos].numpy()
    neg_dists = dist_matrix[mask_neg].numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(pos_dists, bins=50, alpha=0.5, label='Positive Pairs',
             density=True, color='green')
    plt.hist(neg_dists, bins=50, alpha=0.5, label='Negative Pairs',
             density=True, color='red')
    plt.title("Pairwise Distance Distributions")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(output_path)
    print(f"Saved distance distribution to {output_path}")


def plot_angular_distributions(model, embeddings, labels,
                               output_path="angular_distribution.png"):
    """Plot angular distribution analysis for ArcFace models using projection head weights."""
    utils.print_function_banner("plot_angular_distributions")
    head = None
    sub_centers = 1

    # Try to find the ArcFace head in the model.
    if hasattr(model, "projection_head"):
        head = model.projection_head
    elif isinstance(model, torch.nn.Sequential):
        # Assume last layer is the head for sequential models.
        head = model[-1]

    # Check for wrapped sub-center logic in the head.
    if head and hasattr(head, "wmetric_classify"):
        head = head.wmetric_classify
        if hasattr(head, "k"):
            sub_centers = head.k

    if head is None or not hasattr(head, "weight"):
        print("Could not find ArcFace head or weights. Skipping angular plot.")
        return

    print(f"Computing angular stats (k={sub_centers})...")
    centers = head.weight.detach().cpu()
    centers = torch.nn.functional.normalize(centers, p=2, dim=1)

    embs = torch.tensor(embeddings, dtype=torch.float32)
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)

    # Cosine similarity equals dot product of normalized vectors.
    cos_theta = torch.mm(embs, centers.T)

    # Handle sub-centers by reshaping and taking the max similarity.
    if sub_centers > 1:
        n_samples, total_centers = cos_theta.shape
        num_classes = total_centers // sub_centers
        cos_theta = cos_theta.view(n_samples, num_classes, sub_centers)
        cos_theta, _ = torch.max(cos_theta, dim=2)

    # Clamp for numerical stability in acos computation.
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta) * (180.0 / np.pi)

    # Gather theta for the true class center.
    rows = torch.arange(len(labels))
    target_thetas = theta[rows, labels]

    # Gather theta for the nearest incorrect class center.
    theta_clone = theta.clone()
    theta_clone[rows, labels] = float('inf')
    nearest_neg_thetas, _ = torch.min(theta_clone, dim=1)

    plt.figure(figsize=(10, 6))
    plt.hist(target_thetas.numpy(), bins=50, alpha=0.5,
             label='Angle to True Center', density=True, color='blue')
    plt.hist(nearest_neg_thetas.numpy(), bins=50, alpha=0.5,
             label='Angle to Nearest Wrong Center', density=True, color='orange')
    plt.title(f"Angular Distributions (Degrees) - k={sub_centers}")
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(output_path)
    print(f"Saved angular distribution to {output_path}")


if __name__ == "__main__":
    main()
