"""Functions used for hits@k evaluation and embeddings/neighbors analysis."""

import argparse
import random
import os
import sys
import numpy as np

from src.core import constants
from src.core import path_utils

path_utils.ensure_shark_model_in_path()

# pylint: disable=wrong-import-position
import src.learning.evaluator.evaluators
import src.retrieval.fine_evaluation
import src.retrieval.rerank
import src.retrieval.retrieval
from src.core import utils
# pylint: enable=wrong-import-position

class NoEvaluationQueriesError(ValueError):
    """Raised when the chosen split and filters leave no query image to evaluate."""


def _require_queries(ids):
    """Raise NoEvaluationQueriesError when no query image remains."""
    if len(ids) == 0:
        raise NoEvaluationQueriesError(
            "No evaluation queries remain for the chosen split and filters; if this embedding "
            "set has no test split, evaluate with --split-strategy none.")


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
def retrieval_hits_at_k(retriever, retrieval_method=0, train_test=False, min_instances=2,
                        verbose=False, random_guesses=False, min_quality=None,
                        max_instances=None, max_quality=None, reranker=None,
                        media_type=None):
    """Compute the hits at k for a set of images."""
    # Random guess simulation.
    if random_guesses:
        print("Running evaluation with simulated random guessing, not using the embeddings.")

    # Prepare for evaluation by getting the ids to test along with their filenames and labels.
    ids, _, labels, _ = _get_test_instances(retriever, train_test, min_instances=min_instances,
                                            min_quality=min_quality, max_instances=max_instances,
                                            max_quality=max_quality, media_type=media_type)
    _require_queries(ids)

    # Compute hits@k with nearest neighbors.
    average_hits = {k: 0.0 for k in src.learning.evaluator.evaluators.K_THRESHOLDS}
    num_samples = 0
    hits_dictionary = {k: [] for k in src.learning.evaluator.evaluators.K_THRESHOLDS + [0]}

    # Compute hits@k by evaluating results for each possible query.
    print(f"Evaluating hits at k for {len(ids)} samples.")
    for index, shark_id in enumerate(ids):
        query = shark_id
        label = labels[index]

        # Get the unique neighbor labels.
        unique_neighbor_labels, _ = _get_individual_neighbors(
            retriever, retrieval_method, query, label, reranker=reranker
        )
        # If random_guesses is True, shuffle the neighbors 10 times and average the hits.
        if random_guesses:
            num_random_trials = 10
            for _ in range(num_random_trials):
                shuffled_labels = list(unique_neighbor_labels)
                random.shuffle(shuffled_labels)

                hits = src.learning.evaluator.evaluators.__hits_at_k__(
                    labels[index], shuffled_labels
                )
                for i, k in enumerate(src.learning.evaluator.evaluators.K_THRESHOLDS):
                    average_hits[k] += hits[i]

            num_samples += num_random_trials # Increment samples by number of trials
        else:
            # Standard evaluation: Compare the desired label to the label of the
            # closest k neighbors.
            hits = src.learning.evaluator.evaluators.__hits_at_k__(
                labels[index], unique_neighbor_labels
            )

            if verbose:
                hit_found = False
                for i, k in enumerate(src.learning.evaluator.evaluators.K_THRESHOLDS):
                    if hits[i]:
                        hits_dictionary[k].append(query)
                        hit_found = True
                        break
                if not hit_found:
                    hits_dictionary[0].append(query)

            for i, k in enumerate(src.learning.evaluator.evaluators.K_THRESHOLDS):
                average_hits[k] += hits[i]
            num_samples += 1

    for k in src.learning.evaluator.evaluators.K_THRESHOLDS:
        average_hits[k] /= num_samples

    print(f"Number of samples taken in hits@k computation: {num_samples}")

    if verbose is True:
        print("\nQueries resulting in hits at various K values:")
        for key, value in hits_dictionary.items():
            print(key, value)
        print("\n")

    return {
        f"average_hit_at_{k}": average_hits[k]
        for k in src.learning.evaluator.evaluators.K_THRESHOLDS
    }

def retrieval_cmc(retriever, retrieval_method=0, train_test=False, min_instances=2,
                  min_quality=None, max_instances=None, max_quality=None, reranker=None,
                  media_type=None):
    """Compute the Area Under the CMC Curve (AUC-CMC) for image retrieval."""
    ids, _, labels, _ = _get_test_instances(retriever, train_test, min_instances,
                                            min_quality=min_quality, max_instances=max_instances,
                                            max_quality=max_quality, media_type=media_type)
    _require_queries(ids)

    max_rank = max(src.learning.evaluator.evaluators.K_THRESHOLDS)

    print(f"Computing AUC-CMC for {len(ids)} samples.")
    first_hit_ranks = []

    for index, shark_id in enumerate(ids):
        label = labels[index]
        unique_neighbor_labels, _ = _get_individual_neighbors(
            retriever, retrieval_method, shark_id, label, reranker=reranker
        )

        # Find rank of first correct match (1-indexed); None if not found within list.
        first_hit_rank = None
        for rank, neighbor_label in enumerate(unique_neighbor_labels, 1):
            if neighbor_label == label:
                first_hit_rank = rank
                break

        first_hit_ranks.append(first_hit_rank)

    # Build CMC curve: CMC(k) = fraction of queries with first hit at rank <= k.
    num_queries = len(first_hit_ranks)
    cmc_values = [
        sum(1 for r in first_hit_ranks if r is not None and r <= k) / num_queries
        for k in range(1, max_rank + 1)
    ]

    # AUC = mean of CMC values over all ranks (equivalent to uniform trapezoidal integral).
    auc_cmc = float(np.mean(cmc_values))
    print(f"Number of queries used in AUC-CMC computation: {num_queries}")
    return auc_cmc

def retrieval_map(retriever, retrieval_method=0, train_test=False, min_instances=2,
                  min_quality=None, max_instances=None, max_quality=None, reranker=None,
                  media_type=None):
    """Compute mean Average Precision (mAP) for image retrieval."""
    ids, _, labels, label_counts_dict = _get_test_instances(retriever, train_test, min_instances,
                                                            min_quality=min_quality,
                                                            max_instances=max_instances,
                                                            max_quality=max_quality,
                                                            media_type=media_type)

    print(f"Computing mAP for {len(ids)} samples.")
    total_ap = 0.0
    num_queries = 0

    for index, shark_id in enumerate(ids):
        query_label = labels[index]

        # Total relevant = all images of this individual in the gallery, minus the query itself.
        total_relevant = label_counts_dict.get(query_label, 0) - 1
        if total_relevant <= 0:
            continue

        # Get the raw image-level ranked neighbor list (not deduplicated by individual).
        if reranker is not None:
            neighbors = reranker.get_neighbors(shark_id)
        else:
            neighbors = retriever.get_neighbors(retrieval_method, shark_id)
        neighbor_ids = list(neighbors["ids"])
        neighbor_labels = list(neighbors["labels"])

        # Remove the query from the list — mirrors the logic in _get_individual_neighbors.
        try:
            query_idx = neighbor_ids.index(shark_id)
            neighbor_ids.pop(query_idx)
            neighbor_labels.pop(query_idx)
        except ValueError:
            print(f"Warning: query {shark_id} not found in its own neighbor list for mAP.")

        # Compute AP: accumulate precision at each rank where a correct image is found.
        num_hits = 0
        ap = 0.0
        for rank, label in enumerate(neighbor_labels, 1):
            if label == query_label:
                num_hits += 1
                ap += num_hits / rank
        ap /= total_relevant

        total_ap += ap
        num_queries += 1

    map_score = total_ap / num_queries if num_queries > 0 else 0.0
    print(f"Number of queries used in mAP computation: {num_queries}")
    return map_score


def retrieval_k_for_hit(retriever, retrieval_method, ids, labels, reranker=None):
    """Compute the k rank needed to get the first hit for each query."""
    utils.print_function_banner("retrieval_k_for_hit",
                                        "Compute the k value needed to get a hit for a set of images.")

    hits_dictionary = {}
    # Compute hits@k by evaluating results for each possible query.
    print(f"Evaluating hits at k for {len(ids)} samples.")
    for index, individual_id in enumerate(ids):
        query = individual_id
        label = labels[index]

        # Get the unique neighbor labels by funning the retrieval function needed and going
        # from nearest neighbor images to nearest neighbor individuals.
        unique_neighbor_labels, _ = _get_individual_neighbors(
            retriever, retrieval_method, query, label, reranker=reranker)

        # Compute the k value needed to get a hit.
        k_value = unique_neighbor_labels.index(label) + 1

        # Add to the hits dictionary.
        if hits_dictionary.get(str(k_value)) is None:
            hits_dictionary[str(k_value)] = [query]
        else:
            hits_dictionary[str(k_value)].append(query)

    # Sort the output dictionary to show highest k_value ids first.
    hits_dictionary = dict(sorted(hits_dictionary.items(), key=lambda item: int(item[0])))

    # Bucket results for user-friendly printing.
    buckets = {
        "1": 0,
        "2-5": 0,
        "6-10": 0,
        "11-25": 0,
        "26-50": 0,
        ">50": 0
    }

    for key, value in hits_dictionary.items():
        k = int(key)
        count = len(value)

        if k == 1:
            buckets["1"] += count
        elif 2 <= k <= 5:
            buckets["2-5"] += count
        elif 6 <= k <= 10:
            buckets["6-10"] += count
        elif 11 <= k <= 25:
            buckets["11-25"] += count
        elif 26 <= k <= 50:
            buckets["26-50"] += count
        else:
            buckets[">50"] += count

    utils.print_key_value_section("Hits @ k Distribution", buckets)

    return hits_dictionary

def compute_average_k(hits_dict):
    """Compute the average k value from a hits dictionary returned by retrieval_k_for_hit."""
    total_k = 0
    count = 0
    for k_str, query_ids in hits_dict.items():
        k = int(k_str)
        total_k += k * len(query_ids)
        count += len(query_ids)
    return total_k / count if count > 0 else 0.0


def compute_bucketed_average_k(hits_dict, threshold=50):
    """Compute Average K split into easy (<= threshold) and hard (> threshold) buckets."""
    easy_total, easy_count = 0, 0
    hard_total, hard_count = 0, 0
    for k_str, query_ids in hits_dict.items():
        k = int(k_str)
        n = len(query_ids)
        if k <= threshold:
            easy_total += k * n
            easy_count += n
        else:
            hard_total += k * n
            hard_count += n
    return {
        'avg_k_easy': easy_total / easy_count if easy_count > 0 else None,
        'avg_k_hard': hard_total / hard_count if hard_count > 0 else None,
        'count_easy': easy_count,
        'count_hard': hard_count,
    }


def compute_mrr(hits_dict):
    """Compute Mean Reciprocal Rank (MRR) from a hits dictionary returned by retrieval_k_for_hit."""
    total_rr = 0.0
    count = 0
    for k_str, query_ids in hits_dict.items():
        k = int(k_str)
        total_rr += (1.0 / k) * len(query_ids)
        count += len(query_ids)
    return total_rr / count if count > 0 else 0.0


# Helper functions.


# pylint: disable=too-many-locals
def _get_test_instances(retriever, train_test=False, min_instances=2, min_quality=None,
                        max_instances=None, max_quality=None, media_type=None):
    """Load the dataset and isolate the instance IDs to test based on filtering criteria."""
    # Load the stored dataset, add up label counts for hits computation.
    suffix = ""
    if getattr(retriever, "use_stacked", False):
        suffix = " (Stacked: Multimodal + PSL)"
    elif getattr(retriever, "use_multimodal", False):
        suffix = " (Multimodal)"
    elif getattr(retriever, "use_psl", False):
        suffix = " (PSL Reranked)"
    print(f"Loading dataset and embeddings{suffix}.")
    dataset = retriever.get_dataset()
    ids = dataset.get_ids()
    filenames = dataset.get_filenames()
    labels = dataset.get_labels().cpu().detach().numpy()
    label_counts = np.unique(labels, return_counts=True)
    label_counts_dict = dict(zip(label_counts[0], label_counts[1]))

    total_images = len(ids)
    drop_summary = {}

    # Filter based on min_instances: keep only ids from labels with at least
    # min_instances matched ids.
    before = len(ids)
    filtered_ids = []
    filtered_filenames = []
    filtered_labels = []
    for i, shark_id in enumerate(ids):
        label = labels[i]
        if label_counts_dict.get(label, 0) >= min_instances:
            filtered_ids.append(shark_id)
            filtered_filenames.append(filenames[i])
            filtered_labels.append(label)
    ids = filtered_ids
    filenames = filtered_filenames
    labels = np.array(filtered_labels)
    drop_summary[f"min_instances >= {min_instances}"] = before - len(ids)

    # Filter based on max_instances: keep only ids from labels with at most max_instances.
    if max_instances is not None:
        before = len(ids)
        filtered_ids = []
        filtered_filenames = []
        filtered_labels = []
        for i, shark_id in enumerate(ids):
            label = labels[i]
            if label_counts_dict.get(label, 0) <= max_instances:
                filtered_ids.append(shark_id)
                filtered_filenames.append(filenames[i])
                filtered_labels.append(label)
        ids = filtered_ids
        filenames = filtered_filenames
        labels = np.array(filtered_labels)
        drop_summary[f"max_instances <= {max_instances}"] = before - len(ids)

    # Filter queries by quality range. min_quality keeps unknown-quality images (existing
    # behaviour); max_quality excludes unknown-quality images from that bucket.
    if min_quality is not None or max_quality is not None:
        if min_quality is not None and min_quality.lower() not in constants.QUALITY_ORDER:
            raise ValueError(f"Unknown min_quality {min_quality!r}; expected one of "
                             f"{constants.QUALITY_ORDER}.")
        if max_quality is not None and max_quality.lower() not in constants.QUALITY_ORDER:
            raise ValueError(f"Unknown max_quality {max_quality!r}; expected one of "
                             f"{constants.QUALITY_ORDER}.")
        # Pin csv_store to the same CSVs the embeddings were generated from.
        csv_override = getattr(retriever, 'csv_dir_override', None)
        if not (isinstance(csv_override, str) or (isinstance(csv_override, os.PathLike) and not hasattr(csv_override, '_mock_return_value'))):
            csv_override = None
        src.retrieval.fine_evaluation._pin_csv_store_from_dataset(
            dataset, csv_dir_override=csv_override)
        import src.csv_store.media as csv_media  # pylint: disable=import-outside-toplevel
        media_quality_map = {m.id: m.quality for m in csv_media.get_medias(ids=list(ids))}
        min_idx = constants.QUALITY_ORDER.index(min_quality.lower()) if min_quality is not None else 0
        max_idx = (constants.QUALITY_ORDER.index(max_quality.lower()) if max_quality is not None
                   else len(constants.QUALITY_ORDER) - 1)

        before = len(ids)
        kept_ids = []
        kept_filenames = []
        kept_labels = []
        for i, shark_id in enumerate(ids):
            q = media_quality_map.get(shark_id)
            if q is not None and str(q).lower() in constants.QUALITY_ORDER:
                q_idx = constants.QUALITY_ORDER.index(str(q).lower())
                if q_idx < min_idx or q_idx > max_idx:
                    continue
            elif max_quality is not None:
                # Unknown quality is excluded from max_quality buckets.
                continue
            kept_ids.append(shark_id)
            kept_filenames.append(filenames[i])
            kept_labels.append(labels[i])
        ids = kept_ids
        filenames = kept_filenames
        labels = np.array(kept_labels)
        filter_parts = []
        if min_quality is not None:
            filter_parts.append(f"min_quality >= {min_quality}")
        if max_quality is not None:
            filter_parts.append(f"max_quality <= {max_quality}")
        drop_summary[", ".join(filter_parts)] = before - len(ids)

    # Filter queries by media type (e.g. 'topshot', 'underwater').
    if media_type is not None:
        csv_override = getattr(retriever, 'csv_dir_override', None)
        if not (isinstance(csv_override, str) or (isinstance(csv_override, os.PathLike) and not hasattr(csv_override, '_mock_return_value'))):
            csv_override = None
        src.retrieval.fine_evaluation._pin_csv_store_from_dataset(
            dataset, csv_dir_override=csv_override)
        import src.csv_store.media as csv_media  # pylint: disable=import-outside-toplevel
        media_type_map = {m.id: str(getattr(m, 'type', '')).lower() for m in csv_media.get_medias(ids=list(ids))}
        target_type = media_type.lower()
        before = len(ids)
        kept_ids = []
        kept_filenames = []
        kept_labels = []
        for i, shark_id in enumerate(ids):
            t = media_type_map.get(shark_id, '')
            if t == target_type:
                kept_ids.append(shark_id)
                kept_filenames.append(filenames[i])
                kept_labels.append(labels[i])
        ids = kept_ids
        filenames = kept_filenames
        labels = np.array(kept_labels)
        drop_summary[f"media_type == {media_type}"] = before - len(ids)

    # Filter to only the test set.
    if train_test:
        train_ids = dataset.get_train_ids()
        test_ids = dataset.get_test_ids()

        before = len(ids)
        focus_ids = []
        focus_filenames = []
        focus_labels = []
        for index, shark_id in enumerate(ids):
            if shark_id in test_ids:
                focus_ids.append(shark_id)
                focus_filenames.append(filenames[index])
                focus_labels.append(labels[index])
        ids = focus_ids
        filenames = focus_filenames
        labels = focus_labels
        drop_summary[
            f"test split (train={len(train_ids)}, test={len(test_ids)})"
        ] = before - len(ids)

    summary = {"Total images": total_images}
    for constraint, dropped in drop_summary.items():
        summary[f"Dropped: {constraint}"] = dropped
    summary["Remaining for evaluation"] = len(ids)
    utils.print_key_value_section("Dataset Filtering", summary)

    return ids, filenames, labels, label_counts_dict

def _get_individual_neighbors(retriever, retrieval_method, query, label, reranker=None):
    """Retrieve neighbor images and deduplicate to return unique ordered neighbor individuals."""
    # Get neighbors with chosen retrieval method, optionally re-ranked.
    if reranker is not None:
        neighbors = reranker.get_neighbors(query)
    else:
        neighbors = retriever.get_neighbors(retrieval_method, query)
    neighbor_ids = neighbors["ids"]
    neighbor_distances = neighbors["distances"]
    rounded_distances = [round(dist, 2) for dist in neighbor_distances]
    neighbor_labels = neighbors["labels"]

    try:
        # Remove the query from the neighbor results, confirm it is the first
        # neighbor when doing image-image retrieval.
        neighbors_query_index = neighbor_ids.index(query)

        # With image-image retrieval, query should be in position 0.
        # k-reciprocal rerank blends original distance with Jaccard, so the
        # query's own row may not land exactly at position 0.
        if retrieval_method == 0 and neighbors_query_index != 0 and reranker is None:
            print(f"Warning, query {query} not in index 0 of its returned neighbors. "
                  f"Index is {neighbors_query_index}.")

        neighbor_ids.pop(neighbors_query_index)
        removed_label = neighbor_labels.pop(neighbors_query_index)
        removed_distance =  rounded_distances.pop(neighbors_query_index)

        # Self-match distance is expected near zero except for LOO centroid and k-reciprocal reranking.
        assert removed_label == label
        if reranker is None and retrieval_method != 2 and removed_distance >= 0.01:
            print(f"Warning: query {query} self-match distance is {removed_distance:.4f} "
                  f"(expected < 0.01). This is normal for PSL-reranked neighbors where "
                  f"pseudo-distances reflect ADMM numerical precision.")
    except ValueError:
        print(f"Warning, query {query} not in its own set of neighbors.")

    # Remove duplicates to just return unique neighbor sharks, ordered by the
    # first appearance of each label in the neighbors list.
    seen_labels = set()
    unique_neighbor_labels = []
    unique_neighbor_distances = []

    for i, l in enumerate(neighbor_labels):
        if l not in seen_labels:
            unique_neighbor_labels.append(l)
            unique_neighbor_distances.append(rounded_distances[i])
            seen_labels.add(l)

    return unique_neighbor_labels, unique_neighbor_distances


RANDOM_GUESSES_NOT_APPLICABLE = "n/a (random guesses)"


def run_single_evaluation(model_name: str, args: argparse.Namespace, train_test: bool):
    """Encapsulates the evaluation pipeline for a single model."""
    print(f"\n{'='*20}\nEvaluating Model: {model_name}\n{'='*20}")

    use_psl = getattr(args, 'use_psl', False)
    use_multimodal = getattr(args, 'use_multimodal', False)
    use_stacked = getattr(args, 'stacked', False)
    rerank = getattr(args, 'rerank', 'none')
    max_instances = getattr(args, 'max_instances', None)
    max_quality = getattr(args, 'max_quality', None)

    # Inform if PSL is paired with centroid retrieval, as PSL operates only on image-image neighbors.
    if args.retrieval_method == 2:
        if use_psl:
            print("NOTE: --use-psl has no effect with retrieval method 2 (LOO "
                  "centroid). PSL reranks image-image neighbours; method 2 uses "
                  "centroid neighbours. Results are identical to baseline method 2.")
        if use_stacked:
            print("NOTE: With --stacked and retrieval method 2 the PSL component "
                  "has no effect (PSL reranks image neighbours; method 2 uses "
                  "centroid neighbours). Results reflect --use-multimodal on "
                  "centroid retrieval only.")

    if rerank != 'none':
        if args.retrieval_method not in (0, 2):
            print(f"ERROR: --rerank requires retrieval method 0 or 2; got "
                  f"{args.retrieval_method}.")
            return

    try:
        model_dir = str(src.retrieval.generation_store.resolve_active_directory(model_name))
    except src.retrieval.generation_store.GenerationValidationError as exc:
        print(f"ERROR: {exc}")
        return

    if use_stacked:
        needed = [
            os.path.join(model_dir, "stored_embeddings_multimodal.pt"),
            os.path.join(model_dir, "stored_image_neighbors_multimodal_psl.pt"),
        ]
        missing = [p for p in needed if not os.path.exists(p)]
        if missing:
            print(f"WARNING: Stacked neighbours not found for '{model_name}'. "
                  f"Run: embeddings.py --stacked  Skipping.")
            return

    if use_multimodal and not use_stacked:
        mm_path = os.path.join(model_dir, "stored_embeddings_multimodal.pt")
        if not os.path.exists(mm_path):
            print(f"WARNING: Multimodal files not found for '{model_name}'. "
                  f"Run: embeddings.py --use-multimodal  Skipping.")
            return

    if use_psl and not use_stacked:
        psl_path = os.path.join(model_dir, "stored_image_neighbors_psl.pt")
        if not os.path.exists(psl_path):
            print(f"WARNING: PSL neighbours not found for '{model_name}'. "
                  f"Run: embeddings.py --use-psl  Skipping.")
            return

    # Initialize the retriever.
    try:
        retriever = src.retrieval.retrieval.Retriever(
            [model_name], use_psl=use_psl, use_multimodal=use_multimodal,
            use_stacked=use_stacked,
            csv_dir_override=getattr(args, 'csv_dir', None))
        # Ensure retriever has a valid dataset before proceeding.
        if retriever.get_dataset() is None:
            print(f"Error: Model '{model_name}' appears to have an invalid or missing dataset.")
            return
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error loading model {model_name}: {e}")
        return

    # Build the reranker once if requested. The reranker caches per-query
    # neighbours internally so the gallery V matrix is built only once for
    # the entire evaluation.
    reranker = None
    if rerank == 'kreciprocal':
        reranker = src.retrieval.rerank.KReciprocalReranker(
            retriever, base_method=args.retrieval_method, model=model_name)

    # Standard Hits@K evaluation.
    media_type = getattr(args, 'media_type', None)
    ids, _, labels, _ = _get_test_instances(
        retriever, train_test=train_test, min_instances=args.min_instances,
        min_quality=args.min_quality, max_instances=max_instances, max_quality=max_quality,
        media_type=media_type)
    try:
        _require_queries(ids)
    except NoEvaluationQueriesError as error:
        print(f"WARNING: {error} Skipping '{model_name}'.")
        return
    result = retrieval_hits_at_k(
        retriever, retrieval_method=args.retrieval_method, train_test=train_test,
        min_instances=args.min_instances, verbose=args.verbose,
        random_guesses=args.random_guesses, min_quality=args.min_quality,
        max_instances=max_instances, max_quality=max_quality, reranker=reranker,
        media_type=media_type)

    # Prepare and print results table.
    table_headers = ["Metric", "Value"]
    table_rows = []
    keys_map = {
        "average_hit_at_1": "Hits@1",
        "average_hit_at_5": "Hits@5",
        "average_hit_at_10": "Hits@10",
        "average_hit_at_25": "Hits@25",
        "average_hit_at_50": "Hits@50",
        "average_hit_at_100": "Hits@100"
    }
    sorted_keys = ["average_hit_at_1", "average_hit_at_5", "average_hit_at_10",
                   "average_hit_at_25", "average_hit_at_50", "average_hit_at_100"]

    for key in sorted_keys:
        if key in result:
            table_rows.append([keys_map[key], f"{result[key]:.3f}"])
    for key, value in result.items():
        if key not in sorted_keys:
            table_rows.append([key, str(value)])

    # Only Hits@K is shuffled, so rank-based metrics would report the real ranking.
    if args.random_guesses:
        rank_metrics = ["Avg K", "Avg K (k<=50)", "Avg K (k>50)"]
        rank_metrics += [name for name, wanted in
                         (("MRR", args.mrr), ("mAP", args.map), ("AUC-CMC", args.cmc)) if wanted]
        table_rows.extend([name, RANDOM_GUESSES_NOT_APPLICABLE] for name in rank_metrics)
        print(f"\nResults for {model_name}:")
        utils.print_table(table_headers, table_rows)
        return

    # Compute Average K (mean rank of first correct match) using shared function.
    hits_dict = retrieval_k_for_hit(retriever, args.retrieval_method, ids, labels,
                                    reranker=reranker)
    avg_k = compute_average_k(hits_dict)
    bucketed = compute_bucketed_average_k(hits_dict)

    # Append Average K and its bucketed breakdown.
    table_rows.append(["Avg K", f"{avg_k:.2f}"])
    easy_val = bucketed['avg_k_easy']
    hard_val = bucketed['avg_k_hard']
    easy_str = f"{easy_val:.2f} (n={bucketed['count_easy']})" if easy_val is not None else f"N/A (n=0)"
    hard_str = f"{hard_val:.2f} (n={bucketed['count_hard']})" if hard_val is not None else f"N/A (n=0)"
    table_rows.append(["Avg K (k<=50)", easy_str])
    table_rows.append(["Avg K (k>50)", hard_str])

    if args.mrr:
        mrr = compute_mrr(hits_dict)
        table_rows.append(["MRR", f"{mrr:.3f}"])

    if args.map:
        map_score = retrieval_map(
            retriever, retrieval_method=args.retrieval_method, train_test=train_test,
            min_instances=args.min_instances, min_quality=args.min_quality,
            max_instances=max_instances, max_quality=max_quality, reranker=reranker,
            media_type=media_type)
        table_rows.append(["mAP", f"{map_score:.3f}"])

    if args.cmc:
        auc_cmc = retrieval_cmc(
            retriever, retrieval_method=args.retrieval_method, train_test=train_test,
            min_instances=args.min_instances, min_quality=args.min_quality,
            max_instances=max_instances, max_quality=max_quality, reranker=reranker,
            media_type=media_type)
        table_rows.append(["AUC-CMC", f"{auc_cmc:.3f}"])

    print(f"\nResults for {model_name}:")
    utils.print_table(table_headers, table_rows)

def main():
    """Evaluate retrieval quality of stored embeddings using Hits@K and ranking metrics."""
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality of stored embeddings "
                    "using Hits@K metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Retrieval Methods:\n"
               "  0: Image-Image (default)\n"
               "  1: Image-Individual Average\n"
               "  2: LOO Centroid (leave-one-out; unbiased centroid eval)\n\n"
               "Examples:\n"
               "  python3 -m src.retrieval.evaluation "
               "--embeddings-name test-model/vit_triplet_1 "
               "--retrieval-method 0\n"
               "  python3 -m src.retrieval.evaluation "
               "--embeddings-name test-model/vit_triplet_1 "
               "--retrieval-method 2 --split-strategy train-test\n")
    parser.add_argument('--embeddings-name', type=str, nargs='+',
                        default=['prod_model'],
                        help='Name(s) of the embeddings directory(ies) to evaluate (directories '
                             'under sm-data/embeddings/). Multiple names can '
                             'be specified for comparison. Default: "prod_model".')
    parser.add_argument('--retrieval-method', type=int, default=0,
                        help='Retrieval method ID: 0=Image-Image, '
                             '1=Image-Individual, 2=LOO Centroid (unbiased). '
                             'Default: 0.')
    parser.add_argument('--split-strategy', choices=['train-test', 'none'],
                        default='train-test',
                        help='Strategy for handling dataset splits. '
                             '"train-test" (default) evaluates only on the '
                             'test set. "none" evaluates across the entire '
                             'dataset.')
    parser.add_argument('--verbose', action='store_true',
                        help='Show which queries hit at each K value.')
    parser.add_argument('--min-instances', type=int, default=2,
                        help='Minimum number of instances per individual '
                             'required for inclusion in evaluation. '
                             'Default: 2 (need at least 2 to match).')
    parser.add_argument('--min-quality', type=str, default=None,
                        choices=['unmatchable', 'poor', 'medium', 'good'],
                        help='Skip evaluating queries whose media quality is '
                             'below this tier. Mirrors convert_data --min-quality '
                             'but only filters the query set; the gallery is '
                             'unchanged. Images with unknown quality are kept.')
    parser.add_argument('--random-guesses', action='store_true',
                        help='Shuffle neighbors randomly to establish a '
                             'Hits@K baseline for comparison. Avg K, MRR, mAP '
                             'and AUC-CMC are reported as n/a.')
    parser.add_argument('--map', action='store_true',
                        help='Also compute mean Average Precision (mAP). '
                             'Unlike hits@k, mAP rewards finding all images '
                             'of the correct individual and finding them earlier '
                             'in the ranked list.')
    parser.add_argument('--mrr', action='store_true',
                        help='Also compute Mean Reciprocal Rank (MRR). '
                             'MRR is the mean of 1/k where k is the rank '
                             'of the first correct match in the deduplicated-'
                             'by-individual neighbor list (same ranking as '
                             'Avg K). Bounded in [0, 1]; higher is better. '
                             'Standard ranking metric in retrieval/ReID '
                             'literature, complementary to Avg K.')
    parser.add_argument('--cmc', action='store_true',
                        help='Also compute the Area Under the CMC Curve (AUC-CMC). '
                             'The CMC curve plots the fraction of queries whose '
                             'correct match appears within the top-k unique neighbors; '
                             'AUC-CMC is the standard single-number summary in ReID '
                             'literature.')
    parser.add_argument('--fine-eval', action='store_true',
                        help='Run fine-grained evaluation functions instead '
                             'of standard Hits@K evaluation.')
    parser.add_argument('--num-results', type=int, default=30,
                        help='Number of top results to show in fine-evaluation '
                             'lists (e.g., highest variance queries). Default: 30.')
    parser.add_argument('--use-psl', action='store_true',
                        help='Evaluate using PSL-reranked neighbours '
                             '(stored_image_neighbors_psl.pt). Generate with '
                             'embeddings.py --use-psl if missing.')
    parser.add_argument('--use-multimodal', action='store_true',
                        help='Evaluate using multimodal embeddings and neighbours '
                             '(stored_embeddings_multimodal.pt + '
                             'stored_image_neighbors_multimodal.pt). Generate '
                             'with embeddings.py --use-multimodal if missing.')
    parser.add_argument('--stacked', action='store_true',
                        help='Evaluate using the stacked result: multimodal '
                             'embeddings with PSL-reranked multimodal neighbours '
                             '(stored_image_neighbors_multimodal_psl.pt). '
                             'Generate with embeddings.py --stacked if missing.')
    parser.add_argument('--rerank', choices=['none', 'kreciprocal'], default='none',
                        help='Optional post-retrieval re-ranking pass. '
                             '"kreciprocal" applies Zhong et al. (2017) '
                             'k-reciprocal re-ranking on top of the base '
                             'retrieval method. Only compatible with '
                             'retrieval methods 0 and 2. It reranks the fused '
                             'embeddings with --use-multimodal and replaces the '
                             'PSL ranking with --use-psl or --stacked. '
                             'Hyperparameters live as constants in '
                             'src/retrieval/rerank.py.')
    parser.add_argument('--csv-dir', type=str, default=None,
                        help='Explicit override for the source CSV directory '
                             '(e.g. path containing media.csv and individuals.csv).')
    parser.add_argument('--media-type', choices=['topshot', 'underwater'], default=None,
                        help='Filter test queries by media type (topshot or underwater).')
    args = parser.parse_args()

    utils.print_script_banner(__file__, args)

    # Min instances must be at least 2 for hits@k evaluation.
    args.min_instances = max(args.min_instances, 2)
    train_test = args.split_strategy == 'train-test'

    # Retrieval methods map.
    retrieval_methods = {
        0: "image_image",
        1: "image_image_average",
        2: "loo_centroid",
    }

    print(f"Running evaluation for embeddings {args.embeddings_name}. Test split: {train_test}, "
          f"retrieval method: {args.retrieval_method}: {retrieval_methods[args.retrieval_method]}, "
          f"min instances: {args.min_instances}, min quality: {args.min_quality}, "
          f"rerank: {args.rerank}, "
          f"num results (for fine eval): {args.num_results}.")

    # --- Early Validation ---
    if not os.path.exists(constants.EMBEDDINGS_DIR):
        print(f"Error: Embeddings directory not found at {constants.EMBEDDINGS_DIR}")
        sys.exit(1)

    # Resolve all embedding names to their subdirectories.
    all_resolved_names = []
    for embeddings_name in args.embeddings_name:
        resolved = utils.resolve_embedding_dirs(embeddings_name)
        for resolved_name in resolved:
            model_path = os.path.join(constants.EMBEDDINGS_DIR, resolved_name)
            if not os.path.exists(model_path):
                print(f"Warning: Embeddings directory '{resolved_name}' not found at "
                      f"{model_path}. Skipping.")
                continue
            all_resolved_names.append(resolved_name)

    if not all_resolved_names:
        print("Error: No valid embedding directories found.")
        sys.exit(1)

    # Standard per-model evaluation (hits@k, avg k, optional mAP).
    for resolved_name in all_resolved_names:
        run_single_evaluation(resolved_name, args, train_test)

    # Fine-grained cross-model analysis (consistent failures, volatile queries, summaries).
    if args.fine_eval:
        src.retrieval.fine_evaluation.cross_model_failure_analysis(
            all_resolved_names, retrieval_method=args.retrieval_method,
            train_test=train_test, min_instances=args.min_instances,
            min_quality=args.min_quality, top_n=args.num_results)

if __name__ == "__main__":
    main()
