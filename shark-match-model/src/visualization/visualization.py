"""Embedding visualization via PCA, t-SNE, and NCA dimensionality reduction."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from src.core import utils
from src.retrieval import generation_store


def main():
    """Load cached embeddings and generate 3D PCA, t-SNE, and NCA visualizations."""
    parser = argparse.ArgumentParser(
        description="Visualize embedding feature space using PCA, t-SNE, "
                    "and NCA dimensionality reduction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m src.visualization.visualization "
               "--model-name lora --min-count 10\n"
               "  python3 -m src.visualization.visualization "
               "--model-name lora --centroid\n")
    parser.add_argument('--model-name', type=str, default='lora',
                        help='Name of the model directory under '
                             'sm-data/embeddings/ (default: "lora").')
    parser.add_argument('--centroid', action='store_true',
                        help='Load centroid embeddings instead of per-image.')
    parser.add_argument('--train-test', action='store_true',
                        help='Filter to only the test set embeddings.')
    parser.add_argument('--min-count', type=int, default=10,
                        help='Minimum images per individual to include in '
                             'visualization (default: 10).')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output plots. If not provided, '
                             'displays plots interactively.')

    args = parser.parse_args()

    utils.print_script_banner(__file__, args)

    # Load and filter embeddings.
    embeddings, labels = load_embeddings_labels(
        model_name=args.model_name,
        centroid=args.centroid,
        train_test=args.train_test
    )
    embeddings, labels = filter_embeddings(embeddings, labels, args.min_count)

    # Resolve output paths.
    nca_path = None
    pca_path = None
    tsne_path = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        nca_path = os.path.join(args.output_dir, "nca_embeddings.png")
        pca_path = os.path.join(args.output_dir, "pca_embeddings.png")
        tsne_path = os.path.join(args.output_dir, "tsne_embeddings.png")

    # Generate all three dimensionality reduction visualizations.
    visualize_embeddings_nca(embeddings, labels, output_path=nca_path)
    visualize_embeddings_pca(embeddings, labels, output_path=pca_path)
    visualize_embeddings_tsne(embeddings, labels, output_path=tsne_path)


def load_embeddings_labels(model_name, centroid=False, train_test=False):
    """Load cached embeddings and label arrays stored as PT files."""
    utils.print_function_banner("load_embeddings_labels",
                                f"Model: {model_name}, centroid={centroid}")
    device = utils.get_torch_device()
    embeddings_dir = generation_store.resolve_active_directory(model_name)

    if not centroid:
        print("Loading image embeddings.")
        embeddings = torch.load(
            os.path.join(embeddings_dir, "stored_embeddings.pt"),
            map_location=device, weights_only=False
        ).cpu().detach().numpy()
        dataset = torch.load(
            os.path.join(embeddings_dir, "stored_dataset.pt"),
            map_location=device, weights_only=False
        )
        ids = dataset.get_ids()
        labels = dataset.get_labels().cpu().detach().numpy()

        # Split to just the train or test embeddings.
        # Only test-set images are kept when train_test is True.
        if train_test:
            focus_embeddings = []
            focus_labels = []
            test_ids = dataset.get_test_ids()
            for index, media_id in enumerate(ids):
                if media_id in test_ids:
                    focus_embeddings.append(embeddings[index])
                    focus_labels.append(labels[index])
            embeddings = np.array(focus_embeddings)
            labels = np.array(focus_labels)

    else:
        # TODO: train/test loading for centroids.
        print("Loading centroid embeddings.")
        embeddings = torch.load(
            os.path.join(embeddings_dir, "stored_centroid_embeddings.pt"),
            map_location=device, weights_only=False
        ).cpu().detach().numpy()
        labels = torch.load(
            os.path.join(embeddings_dir, "stored_centroid_labels.pt"),
            map_location=device, weights_only=False
        ).cpu().detach().numpy()

    return embeddings, labels


def filter_embeddings(embeddings, labels, min_count):
    """Filter embeddings and labels to individuals meeting the minimum image count threshold."""
    utils.print_function_banner("filter_embeddings",
                                f"Minimum count: {min_count}")
    label_counts = pd.Series(labels).value_counts()
    frequent_labels = label_counts[label_counts >= min_count].index.tolist()

    filtered_indices = np.isin(labels, frequent_labels)
    filtered_embeddings = embeddings[filtered_indices]
    filtered_labels = labels[filtered_indices]

    print(f"Embeddings meeting min-count threshold: {len(filtered_embeddings)}")
    print(f"Unique labels represented: {len(set(filtered_labels))}")

    return filtered_embeddings, filtered_labels


def visualize_embeddings_nca(embeddings, labels, output_path=None):
    """Visualize embeddings using 3D Neighborhood Components Analysis (NCA)."""
    print("Running NCA dimensionality reduction...")
    nca = NeighborhoodComponentsAnalysis(n_components=3, random_state=1)
    reduced_embeddings = nca.fit_transform(embeddings, labels)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        reduced_embeddings[:, 2],
        c=labels, cmap="viridis"
    )
    ax.set_xlabel('NC1')
    ax.set_ylabel('NC2')
    ax.set_zlabel('NC3')
    plt.colorbar(scatter)
    plt.title('3D NCA of Embeddings')

    if output_path:
        plt.savefig(output_path)
        print(f"Saved NCA visualization to {output_path}")
    else:
        plt.show()


def visualize_embeddings_pca(embeddings, labels, output_path=None):
    """Visualize embeddings using 3D Principal Component Analysis (PCA)."""
    print("Running PCA dimensionality reduction...")
    pca = PCA(n_components=3, random_state=0)
    reduced_embeddings = pca.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        reduced_embeddings[:, 2],
        c=labels, cmap="viridis"
    )
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.colorbar(scatter)
    plt.title('3D PCA of Embeddings')

    if output_path:
        plt.savefig(output_path)
        print(f"Saved PCA visualization to {output_path}")
    else:
        plt.show()


def visualize_embeddings_tsne(embeddings, labels, output_path=None):
    """Visualize embeddings using 3D t-SNE dimensionality reduction."""
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=3, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        reduced_embeddings[:, 2],
        c=labels, cmap="viridis"
    )
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Embeddings')

    if output_path:
        plt.savefig(output_path)
        print(f"Saved t-SNE visualization to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
