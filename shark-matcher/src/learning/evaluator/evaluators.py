"""Retrieval evaluation functions and embedding computation for model checkpointing."""

import typing

import numpy as np
import sklearn.neighbors
import sklearn.preprocessing
import torch

from src.core import constants

K_THRESHOLDS = [1, 5, 10, 25, 50, 100]


def main():
    """Module entry point for ad-hoc evaluation debugging."""


def get_unique_ordered_labels(labels: typing.Iterable) -> typing.List:
    """Returns unique elements from the labels iterable preserving first-occurrence order."""
    unique_labels = []
    seen = set()
    for label in labels:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)
    return unique_labels


def compute_embeddings(model, dataset, device, embed_fn=None):
    """Computes and L2-normalizes embeddings for a dataset using the specified model and embedder."""
    if embed_fn is None:
        embed_fn = _default_image_only_embed

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=constants.MINI_BATCH_SIZE,
        shuffle=False
    )

    # Compute embeddings.
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            embeddings.append(embed_fn(model, batch, device).cpu())

    if not embeddings:
        return None, None

    embeddings = torch.cat(embeddings, dim=0)

    # L2-normalize embeddings to make them compatible with Euclidean distance
    # for cosine similarity ranking.
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    embeddings = embeddings.numpy()

    labels = dataset.get_labels().cpu().numpy()
    return embeddings, labels


def calculate_hits_for_indices(labels, neighbor_indices, neighbor_labels_source,
                               label_counts_dict=None):
    """Calculates average hits@k for a set of query labels against neighbor predictions."""
    # pylint: disable=too-many-locals
    hits_sums = {k: 0 for k in K_THRESHOLDS}
    num_samples = 0

    for index, neighbors in enumerate(neighbor_indices):
        label = labels[index]

        # Skip queries whose label has no possible match in the gallery
        # (singleton means only self is present).
        if label_counts_dict and label_counts_dict[label] == 1:
            continue

        # Gallery contains the query itself (train and val are concatenated),
        # so self sits at position 0; skip it.
        relevant_neighbors = neighbors[1:] if label_counts_dict else neighbors

        current_neighbor_labels = get_unique_ordered_labels(
            neighbor_labels_source[relevant_neighbors]
        )

        hits = __hits_at_k__(label, current_neighbor_labels)

        for i, k in enumerate(K_THRESHOLDS):
            hits_sums[k] += hits[i]

        num_samples += 1

    averages = {}
    for k, s in hits_sums.items():
        averages[f"hit_at_{k}"] = s / num_samples if num_samples > 0 else 0.0

    return averages


def hits_at_k(model, train_dataset, valid_dataset, device, embed_fn=None):
    """Compute hits@k metrics for a model across training and validation datasets."""
    # pylint: disable=too-many-locals
    # Compute training and validation embeddings.
    train_embeddings, train_labels = compute_embeddings(
        model, train_dataset, device, embed_fn=embed_fn
    )

    valid_embeddings, valid_labels = None, None
    if valid_dataset and len(valid_dataset) > 0:
        valid_embeddings, valid_labels = compute_embeddings(
            model, valid_dataset, device, embed_fn=embed_fn
        )
        if valid_embeddings is None:
            print("Warning: Validation dataset provided but no embeddings were generated.")

    # Build a combined gallery (train + val) so that val queries from held-out
    # individuals can find their peers, matching the retrieval-time setup.
    if valid_embeddings is not None:
        gallery_embeddings = np.concatenate([train_embeddings, valid_embeddings], axis=0)
        gallery_labels = np.concatenate([train_labels, valid_labels], axis=0)
    else:
        gallery_embeddings = train_embeddings
        gallery_labels = train_labels

    n_neighbors_val = min(200, len(gallery_embeddings))
    nearest_neighbors = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors_val, metric='euclidean'
    ).fit(gallery_embeddings)

    gallery_label_counts = np.unique(gallery_labels, return_counts=True)
    gallery_label_counts_dict = dict(zip(gallery_label_counts[0], gallery_label_counts[1]))

    # Compute training hits@k against the combined gallery.
    _, train_neighbor_indices = nearest_neighbors.kneighbors(train_embeddings)
    train_stats = calculate_hits_for_indices(
        train_labels, train_neighbor_indices, gallery_labels, gallery_label_counts_dict
    )
    results = {f"average_train_{k}": v for k, v in train_stats.items()}

    # Compute validation hits@k against the combined gallery.
    checkpoint_metric_val = 0.0
    if valid_embeddings is not None:
        _, valid_neighbor_indices = nearest_neighbors.kneighbors(valid_embeddings)
        valid_stats = calculate_hits_for_indices(
            valid_labels, valid_neighbor_indices, gallery_labels, gallery_label_counts_dict
        )
        results.update({f"average_valid_{k}": v for k, v in valid_stats.items()})
        checkpoint_metric_val = results.get(constants.CHECKPOINT_METRIC, 0.0)

    # Fill missing validation keys with 0.0 if not computed.
    for k in K_THRESHOLDS:
        if f"average_valid_hit_at_{k}" not in results:
            results[f"average_valid_hit_at_{k}"] = 0.0

    return checkpoint_metric_val, results


def __hits_at_k__(label: np.array, neighbor_labels: np.array) -> \
        typing.Tuple[int, int, int, int, int, int]:
    """Computes binary hit indicators across standard K thresholds for ranked neighbor labels."""
    hits = [0] * len(K_THRESHOLDS)

    for i in range(0, min(max(K_THRESHOLDS), len(neighbor_labels))):
        if label == neighbor_labels[i]:
            for j, threshold in enumerate(K_THRESHOLDS):
                if i < threshold:
                    hits[j] = 1
            break

    return tuple(hits)


def _default_image_only_embed(model, batch, device):
    """Default embedder: ``batch == (images, labels)``; call ``model(images)``."""
    images, _ = batch
    return model(images.to(device))


if __name__ == "__main__":
    main()
