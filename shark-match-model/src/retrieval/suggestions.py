"""Generate merge and unmatch data-quality suggestions from retrieval embeddings."""

import argparse
import os
import sys

import numpy as np
import torch

from src.core import constants
from src.core import path_utils

path_utils.ensure_shark_model_in_path()

# pylint: disable=wrong-import-position
import src.retrieval.evaluation
import src.retrieval.rerank
import src.retrieval.retrieval
from src.core import utils
# pylint: enable=wrong-import-position


def get_merge_suggestions(retriever, threshold=None, limit=None):
    """Identify pairs of individuals whose centroids are closest together as potential merge candidates."""
    utils.print_function_banner("get_merge_suggestions",
                                "Finding potential merges based on centroid distance.")

    data = retriever._get_model_data(None)  # pylint: disable=protected-access
    centroid_labels = data['centroid_labels']
    centroid_embeddings = data['centroid_embeddings']

    if centroid_labels is None or centroid_embeddings is None:
        print("ERROR: Centroid data not available for this model. "
              "Re-run embeddings generation to produce centroid files.")
        return []

    # Convert centroid labels to Python ints once upfront to avoid per-iteration tensor synchronization.
    centroid_labels = [int(l) for l in centroid_labels]
    n = len(centroid_labels)
    # pylint: disable=protected-access
    dist_matrix = src.retrieval.rerank._pairwise_l2(centroid_embeddings)

    seen_pairs = set()
    merge_candidates = []

    if threshold is not None:
        for i in range(n):
            label_a = centroid_labels[i]
            within = np.where(dist_matrix[i] <= threshold)[0]
            for neighbor_idx in within:
                if neighbor_idx == i:
                    continue
                label_b = centroid_labels[int(neighbor_idx)]
                pair_key = frozenset((label_a, label_b))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                merge_candidates.append(
                    (label_a, label_b, float(dist_matrix[i, neighbor_idx]))
                )
    else:
        # Each of the top-L pairs has both endpoints among each other's L nearest neighbors.
        k = min(limit + 1 if limit is not None else n, n)
        neighbor_indices = np.argsort(dist_matrix, axis=1, kind="stable")[:, :k]
        for i in range(n):
            label_a = centroid_labels[i]
            for neighbor_idx in neighbor_indices[i]:
                if neighbor_idx == i:
                    continue
                label_b = centroid_labels[int(neighbor_idx)]
                pair_key = frozenset((label_a, label_b))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                merge_candidates.append(
                    (label_a, label_b, float(dist_matrix[i, neighbor_idx]))
                )

    merge_candidates.sort(key=lambda x: x[2])

    if limit is not None:
        merge_candidates = merge_candidates[:limit]

    utils.print_key_value_section("Merge Suggestions Summary", {
        "Centroids evaluated": n,
        "Candidate pairs found": len(merge_candidates),
        "Distance threshold": threshold if threshold is not None else "None (all pairs)",
        "Limit": limit if limit is not None else "None"
    })

    return merge_candidates


MERGE_SUGGESTIONS_LIMIT = 100


def store_suggestions(retriever, model_dir):
    """Compute unmatch and merge suggestions for a model and persist both to suggestions.pt."""
    utils.print_function_banner(
        "store_suggestions",
        f"Computing and storing unmatch and merge suggestions into {model_dir}.")

    unmatch_candidates = get_unmatch_suggestions(
        retriever, retrieval_method=0, train_test=False, min_instances=2)
    merge_candidates = get_merge_suggestions(
        retriever, limit=MERGE_SUGGESTIONS_LIMIT)

    payload = {
        "unmatch_suggestions": unmatch_candidates,
        "merge_suggestions": merge_candidates,
    }

    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "suggestions.pt")
    torch.save(payload, out_path)
    print(f"Saved {len(unmatch_candidates)} unmatch and "
          f"{len(merge_candidates)} merge suggestions to {out_path}.")
    return payload


def load_suggestions(model_dir):
    """Load cached suggestions for a model, returning an empty payload if missing or invalid."""
    empty = {"unmatch_suggestions": [], "merge_suggestions": []}
    path = os.path.join(model_dir, "suggestions.pt")
    if not os.path.exists(path):
        return empty
    data = torch.load(path, weights_only=False)
    # Treat legacy bare-list suggestion caches as empty to avoid crashing.
    if not isinstance(data, dict):
        print(f"WARNING: Legacy suggestions.pt format at {path}; "
              "regenerate embeddings to populate merge suggestions.")
        return empty
    return {
        "unmatch_suggestions": data.get("unmatch_suggestions", []),
        "merge_suggestions": data.get("merge_suggestions", []),
    }


def get_unmatch_suggestions(retriever, retrieval_method=0, train_test=False,
                            min_instances=2, limit=None):
    """Identify query images with the worst retrieval rank (k-at-hit) as potential unmatch candidates."""
    utils.print_function_banner("get_unmatch_suggestions",
                                "Identifying hard-to-match images.")

    ids, _, labels, _ = src.retrieval.evaluation._get_test_instances(  # pylint: disable=protected-access
        retriever, train_test=train_test, min_instances=min_instances)

    # Compute k-at-hit directly per query to avoid redundant dict construction.
    print(f"Evaluating k-at-hit for {len(ids)} samples.")
    unmatch_candidates = []
    for image_id, label in zip(ids, labels):
        unique_neighbor_labels, _ = src.retrieval.evaluation._get_individual_neighbors(  # pylint: disable=protected-access
            retriever, retrieval_method, image_id, label)

        try:
            k_value = unique_neighbor_labels.index(label) + 1
        except ValueError:
            # Correct label never appeared in the neighbor list; treat as worst.
            k_value = len(unique_neighbor_labels) + 1

        unmatch_candidates.append((image_id, k_value, int(label)))

    unmatch_candidates.sort(key=lambda x: -x[1])

    total = len(unmatch_candidates)
    perfect = sum(1 for _, k, _ in unmatch_candidates if k == 1)
    utils.print_key_value_section("Unmatch Suggestions Summary", {
        "Images evaluated": total,
        "Hardest match (max k)": unmatch_candidates[0][1] if unmatch_candidates else 0,
        "Perfect matches (k=1)": f"{perfect} ({perfect / total * 100:.1f}%)"
        if total else 0
    })

    if limit is not None:
        unmatch_candidates = unmatch_candidates[:limit]

    return unmatch_candidates


def main():
    """Surface merge and unmatch suggestions for a set of stored embeddings."""
    parser = argparse.ArgumentParser(
        description="Surface merge and unmatch suggestions from stored embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Retrieval Methods (for unmatch analysis):\n"
               "  0: Image-Image (default)\n"
               "  1: Image-Individual Average\n"
               "  2: Centroid Mean Distance\n\n"
               "Examples:\n"
               "  python3 -m src.retrieval.suggestions "
               "--embeddings-name prod_model --split-strategy train-test\n"
               "  python3 -m src.retrieval.suggestions "
               "--embeddings-name prod_model --merge-threshold 0.5\n")
    parser.add_argument('--embeddings-name', type=str, default='prod_model',
                        help='Name of the embeddings directory to analyze '
                             '(directory under sm-data/embeddings/). '
                             'Default: "prod_model".')
    parser.add_argument('--retrieval-method', type=int, default=0,
                        help='Retrieval method ID for unmatch analysis: '
                             '0=Image-Image, 1=Image-Individual, 2=Centroid. '
                             'Default: 0.')
    parser.add_argument('--split-strategy', choices=['train-test', 'none'],
                        default='train-test',
                        help='Strategy for handling dataset splits. '
                             '"train-test" (default) evaluates only on the test set. '
                             '"none" evaluates across the entire dataset. '
                             'Note: train/test splitting may increase false positives '
                             'in merge suggestions.')
    parser.add_argument('--min-instances', type=int, default=2,
                        help='Minimum number of instances per individual required '
                             'for inclusion in unmatch analysis. Default: 2.')
    parser.add_argument('--merge-threshold', type=float, default=None,
                        help='Maximum centroid distance to include in merge suggestions. '
                             'If omitted, all pairs are returned (sorted by distance).')
    parser.add_argument('--num-results', type=int, default=30,
                        help='Number of results to display per table. Default: 30.')
    parser.add_argument('--save-suggestions', action='store_true',
                        help='After computing the report, also write the '
                             'results to <model_dir>/suggestions.pt, '
                             'overwriting any existing file. Merge '
                             'suggestions are capped at MERGE_SUGGESTIONS_LIMIT '
                             'in the cache regardless of CLI args. For '
                             'refreshing the review-tab cache using custom '
                             'CLI params without a full embedding '
                             'regeneration.')
    args = parser.parse_args()

    utils.print_script_banner(__file__, args)

    train_test = args.split_strategy == 'train-test'
    args.min_instances = max(args.min_instances, 2)

    retrieval_methods = {
        0: "image_image",
        1: "image_image_average",
        2: "loo_centroid",
    }

    print(f"Running suggestions for embeddings '{args.embeddings_name}'. "
          f"Train/test split: {train_test}, "
          f"retrieval method: {args.retrieval_method}: "
          f"{retrieval_methods[args.retrieval_method]}, "
          f"min instances: {args.min_instances}.")

    if not os.path.exists(constants.EMBEDDINGS_DIR):
        print(f"Error: Embeddings directory not found at {constants.EMBEDDINGS_DIR}")
        sys.exit(1)

    model_path = os.path.join(constants.EMBEDDINGS_DIR, args.embeddings_name)
    if not os.path.exists(model_path):
        print(f"Error: Embeddings directory '{args.embeddings_name}' not found at {model_path}.")
        sys.exit(1)

    try:
        retriever = src.retrieval.retrieval.Retriever([args.embeddings_name])
        if retriever.get_dataset() is None:
            print(f"Error: '{args.embeddings_name}' has an invalid or missing dataset.")
            sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error loading embeddings '{args.embeddings_name}': {e}")
        sys.exit(1)

    # --- Merge Suggestions ---
    merge_candidates = get_merge_suggestions(retriever, threshold=args.merge_threshold)

    num_merge = min(args.num_results, len(merge_candidates))
    print(f"\nTop {num_merge} Merge Candidates (closest centroid pairs):")
    merge_rows = [
        [label_a, label_b, f"{dist:.4f}"]
        for label_a, label_b, dist in merge_candidates[:num_merge]
    ]
    utils.print_table(["Individual A", "Individual B", "Centroid Distance"], merge_rows)

    # --- Unmatch Suggestions ---
    unmatch_candidates = get_unmatch_suggestions(
        retriever,
        retrieval_method=args.retrieval_method,
        train_test=train_test,
        min_instances=args.min_instances
    )

    num_unmatch = min(args.num_results, len(unmatch_candidates))
    print(f"\nTop {num_unmatch} Unmatch Candidates (hardest-to-match images):")
    unmatch_rows = [
        [image_id, k_value, label]
        for image_id, k_value, label in unmatch_candidates[:num_unmatch]
    ]
    utils.print_table(["Image ID", "K at Hit", "Label"], unmatch_rows)

    # Persist computed candidates to the review-tab cache, capping merge suggestions.
    if args.save_suggestions:
        payload = {
            "unmatch_suggestions": unmatch_candidates,
            "merge_suggestions": merge_candidates[:MERGE_SUGGESTIONS_LIMIT],
        }
        out_path = os.path.join(model_path, "suggestions.pt")
        torch.save(payload, out_path)
        print(f"\nSaved {len(payload['unmatch_suggestions'])} unmatch and "
              f"{len(payload['merge_suggestions'])} merge suggestions to {out_path}.")


if __name__ == "__main__":
    main()
