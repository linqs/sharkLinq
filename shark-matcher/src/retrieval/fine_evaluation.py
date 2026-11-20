"""Functions for fine-grained analysis of retrieval results."""

import os
import sys

import numpy as np
import torch

from src.core import constants
import src.csv_store._loader as _csv_loader
import src.csv_store.individual
import src.csv_store.media
import src.retrieval.evaluation
import src.retrieval.generation_store
import src.retrieval.retrieval
from src.core import utils


def main():
    """Run interactive comparison of retrieval results across models for debugging."""
    test_retriever = src.retrieval.retrieval.Retriever("vit_baylearn")
    unknown_features_dict = count_unknown_features(
        test_retriever._get_model_data(None)['dataset'])
    ids, _, labels, _ = _get_filtered_test_instances(
        test_retriever, train_test=True, min_instances=2, min_time=None, max_time=None,
        allow_unknonw=False)
    hits_dict_vit = src.retrieval.evaluation.retrieval_k_for_hit(test_retriever, 0, ids, labels)

    test_retriever = src.retrieval.retrieval.Retriever("fusion_vit_all")
    ids, _, labels, _ = _get_filtered_test_instances(
        test_retriever, train_test=True, min_instances=2, min_time=None, max_time=None,
        allow_unknonw=False)
    hits_dict_miew = src.retrieval.evaluation.retrieval_k_for_hit(test_retriever, 0, ids, labels)

    compare_hits_dict_by_threshold(hits_dict_vit, hits_dict_miew, k_boundary=3,
                                   unknown_features=unknown_features_dict)
    compare_hits_dict_by_delta(hits_dict_vit, hits_dict_miew)


def count_unknown_features(dataset):
    """Count unknown or missing media instances for each metadata feature in the pinned CSV snapshot."""
    utils.print_function_banner("count_unknown_features", "Counting unknown features.")
    _pin_csv_store_from_dataset(dataset)

    individuals = src.csv_store.individual.get_individuals(ids=None, get_full=True)

    total_individuals = 0
    total_media = 0

    def is_unknown(val):
        if val is None:
            return True
        s = str(val).strip().lower()
        return s in ["", "u", "U", "none", "null", "unknown"]

    missing_counts = {
        "sex": 0,
        "length": 0,
        "location": 0,
        "time": 0,
        "type": 0,
        "quality": 0
    }

    missing_features_by_id = {}

    for source_individual in individuals:
        for individual in source_individual.associated_individuals:
            if not src.csv_store.individual.is_media_individual(individual):
                continue

            total_individuals += 1
            individual_missing = []
            if is_unknown(individual.sex):
                missing_counts["sex"] += 1
                individual_missing.append("sex")
            if is_unknown(individual.length):
                missing_counts["length"] += 1
                individual_missing.append("length")

            if individual_missing:
                if individual.id in missing_features_by_id:
                    current_missing = missing_features_by_id[individual.id]
                    missing_features_by_id[individual.id] = sorted(
                        list(set(current_missing + individual_missing)))
                else:
                    missing_features_by_id[individual.id] = sorted(individual_missing)

        medias = source_individual.associated_medias
        total_media += len(medias)

        for media in medias:
            media_missing = []
            if is_unknown(media.location):
                missing_counts["location"] += 1
                media_missing.append("location")
            if is_unknown(media.time):
                missing_counts["time"] += 1
                media_missing.append("time")
            if is_unknown(media.type):
                missing_counts["type"] += 1
                media_missing.append("type")
            if is_unknown(media.quality):
                missing_counts["quality"] += 1
                media_missing.append("quality")

            if media_missing:
                if media.id in missing_features_by_id:
                    current_missing = missing_features_by_id[media.id]
                    missing_features_by_id[media.id] = sorted(
                        list(set(current_missing + media_missing)))
                else:
                    missing_features_by_id[media.id] = sorted(media_missing)

    utils.print_key_value_section("CSV Store Overview", {
        "Total Individuals": total_individuals,
        "Total Media Instances": total_media
    })

    table_rows = []
    for feat in ["sex", "length"]:
        count = missing_counts[feat]
        pct = (count / total_individuals * 100) if total_individuals > 0 else 0
        table_rows.append([feat, f"{count}/{total_individuals}", f"{pct:.2f}%"])

    for feat in ["location", "time", "type", "quality"]:
        count = missing_counts[feat]
        pct = (count / total_media * 100) if total_media > 0 else 0
        table_rows.append([feat, f"{count}/{total_media}", f"{pct:.2f}%"])

    print("\nMissing/Unknown Feature Counts:")
    utils.print_table(["Feature", "Missing Count", "Percentage"], table_rows)

    return missing_features_by_id


def show_embedding_retrieval_info(retriever, queries, retrieval_method=0, min_instances=2,
                                  show=30, verbose=False):
    """Display detailed retrieval results for specific queries using the specified retrieval method."""
    utils.print_function_banner("show_embedding_retrieval_info",
                                f"Showing detailed retrieval info for {len(queries)} queries.")
    _pin_csv_store_from_retriever(retriever)
    ids, filenames, labels, _ = src.retrieval.evaluation._get_test_instances(
        retriever, False, min_instances=min_instances)

    for query in queries:
        try:
            query_index = ids.index(query)
        except ValueError:
            print(f"ERROR: Query ID {query} not found in the dataset or dataset split.")
            continue

        label = labels[query_index]
        query_name = filenames[query_index]
        print(f"\nResult for query id {query} ({query_name}) ---")

        if verbose:
            _medias = src.csv_store.media.get_medias([query], get_full=True)
            if not _medias:
                print(f"WARNING: Query image with id {query} not found in CSV store.")
                continue
            query_image = _medias[0]
            query_individual_id = query_image.individual["id"]
            query_individual = src.csv_store.individual.get_individuals(
                ids=[query_individual_id], get_full=True, get_source=True)[0]

            print(f"Associated individuals: {len(query_individual.associated_individuals)}")
            print(f"Associated medias: {len(query_individual.associated_medias)}")

        neighbors = retriever.get_neighbors(retrieval_method, query)

        neighbor_ids = neighbors["ids"]
        neighbor_labels = neighbors["labels"]
        neighbor_distances = neighbors["distances"]
        rounded_distances = [round(dist, 2) for dist in neighbor_distances]

        if neighbor_ids[0] != query or neighbor_labels[0] != labels[query_index]:
            print("WARNING: Top match is NOT the query itself.")

        table_rows = []
        for i in range(min(show, len(neighbor_ids))):
            is_match = "Yes" if neighbor_labels[i] == labels[query_index] else "No"
            table_rows.append([neighbor_ids[i], f"{rounded_distances[i]:.2f}", neighbor_labels[i], is_match])

        utils.print_table(["Neighbor ID", "Dist", "Label", "Match?"], table_rows)

        unique_neighbor_labels, unique_neighbor_distances = src.retrieval.evaluation._get_individual_neighbors(
            retriever, retrieval_method, query, label)

        print("\nUnique Individual Matches (Deduplicated):")
        unique_rows = []
        for i in range(min(1000, len(unique_neighbor_labels))):
            unique_rows.append([unique_neighbor_labels[i], f"{unique_neighbor_distances[i]:.2f}"])

        utils.print_table(["Label", "Distance"], unique_rows)


def compare_hits_dict_by_threshold(hits_dict_1, hits_dict_2, k_boundary=3, unknown_features=None):
    """Compare two hits dictionaries and list IDs whose rank <= k_boundary in one but not the other."""
    utils.print_function_banner("compare_hits_dict_by_threshold",
                                f"Comparing hits dictionaries (boundary K={k_boundary}).")

    def find_k_value(hits_dict, target_id):
        for k_str, ids in hits_dict.items():
            if target_id in ids:
                return int(k_str)
        return "Not Found"

    def get_ids_up_to_boundary(hits_dict, boundary):
        good_ids = set()
        for k_str, ids in hits_dict.items():
            k = int(k_str)
            if k <= boundary:
                good_ids.update(ids)
        return good_ids

    def get_row_data(id_val, k_val, unknown_fts):
        row = [id_val, k_val]
        if unknown_fts is not None:
            row.append(len(unknown_fts.get(id_val, [])))
        return row

    good_ids_1 = get_ids_up_to_boundary(hits_dict_1, k_boundary)
    good_ids_2 = get_ids_up_to_boundary(hits_dict_2, k_boundary)

    cols_1 = ["ID", "K in Dict 2"]
    cols_2 = ["ID", "K in Dict 1"]
    if unknown_features is not None:
        cols_1.append("Unknown Count")
        cols_2.append("Unknown Count")

    only_in_d1 = good_ids_1 - good_ids_2
    print(f"\nIn Dict 1 (K<={k_boundary}), NOT in Dict 2 (K<={k_boundary}) ({len(only_in_d1)}):")
    rows1 = [get_row_data(id_val, find_k_value(hits_dict_2, id_val), unknown_features)
             for id_val in only_in_d1]
    utils.print_table(cols_1, rows1)

    only_in_d2 = good_ids_2 - good_ids_1
    print(f"\nIn Dict 2 (K<={k_boundary}), NOT in Dict 1 (K<={k_boundary}) ({len(only_in_d2)}):")
    rows2 = [get_row_data(id_val, find_k_value(hits_dict_1, id_val), unknown_features)
             for id_val in only_in_d2]
    utils.print_table(cols_2, rows2)


def compare_hits_dict_by_delta(hits_dict_1, hits_dict_2):
    """Compare two hits dictionaries and return IDs sorted by the largest absolute delta in rank k."""
    utils.print_function_banner("compare_hits_dict_by_delta",
                                "Comparing hits dictionaries by K-value delta.")

    def flatten_hits(hits_dict):
        id_to_k = {}
        for k_str, ids in hits_dict.items():
            k = int(k_str)
            for id_val in ids:
                id_to_k[id_val] = k
        return id_to_k

    map_1 = flatten_hits(hits_dict_1)
    map_2 = flatten_hits(hits_dict_2)

    common_ids = set(map_1.keys()).intersection(set(map_2.keys()))

    delta_to_ids = {}
    for id_val in common_ids:
        delta = abs(map_1[id_val] - map_2[id_val])
        if delta not in delta_to_ids:
            delta_to_ids[delta] = []
        delta_to_ids[delta].append(id_val)

    sorted_deltas = dict(sorted(delta_to_ids.items(), key=lambda item: -1 * item[0]))

    print(f"\nTop K-Value Deltas ({len(common_ids)} IDs compared):")

    table_rows = []
    count = 0
    max_show = 30

    for delta, ids in sorted_deltas.items():
        if delta == 0:
            continue
        for id_val in ids:
            if count >= max_show:
                break
            table_rows.append([id_val, delta, map_1[id_val], map_2[id_val]])
            count += 1
        if count >= max_show:
            break

    if table_rows:
        utils.print_table(["ID", "Delta", "K in Dict 1", "K in Dict 2"], table_rows)
    else:
        print("No differences found or no common IDs.")

    return sorted_deltas


# pylint: disable=too-many-locals
def cross_model_failure_analysis(embeddings_names, retrieval_method=0,
                                  train_test=False, min_instances=2, min_quality=None,
                                  top_n=30):
    """Run per-query failure analysis across embedding sets to identify consistently failing and volatile queries."""
    utils.print_function_banner(
        "cross_model_failure_analysis",
        f"Analyzing failures across {len(embeddings_names)} embedding sets.")

    query_k_map = {}
    model_short_names = []

    # Assumes all embedding sets share the same dataset; label maps are built once from the first loaded set.
    label_count_map = {}
    query_id_label_map = {}

    for embeddings_name in embeddings_names:
        short_name = os.path.basename(embeddings_name)
        model_short_names.append(short_name)

        print(f"\n{'='*20}\nLoading embeddings: {embeddings_name}\n{'='*20}")

        try:
            _show_dataset_embeddings_consistency(embeddings_name)
            retriever = src.retrieval.retrieval.Retriever([embeddings_name])
            if retriever.get_dataset() is None:
                print(f"Error: '{embeddings_name}' has an invalid or missing dataset.")
                continue
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error loading embeddings {embeddings_name}: {e}")
            continue

        if not label_count_map:
            dataset = retriever.get_dataset()
            all_labels = dataset.get_labels().cpu().detach().numpy()
            all_ids = dataset.get_ids()
            unique_labels, counts = np.unique(all_labels, return_counts=True)
            label_count_map = dict(zip(unique_labels.tolist(), counts.tolist()))
            query_id_label_map = dict(zip(all_ids, all_labels.tolist()))

        ids, _, labels, _ = src.retrieval.evaluation._get_test_instances(
            retriever, train_test=train_test, min_instances=min_instances,
            min_quality=min_quality)

        hits_dict = src.retrieval.evaluation.retrieval_k_for_hit(retriever, retrieval_method, ids, labels)

        for k_str, query_ids in hits_dict.items():
            k_value = int(k_str)
            for query_id in query_ids:
                if query_id not in query_k_map:
                    query_k_map[query_id] = {}
                query_k_map[query_id][short_name] = k_value

    if not query_k_map:
        print("No results collected. Aborting cross-model analysis.")
        return

    query_stats = []
    for query_id, model_ks in query_k_map.items():
        k_values = list(model_ks.values())
        query_label = query_id_label_map.get(query_id)
        total_images = label_count_map.get(query_label, 0) if query_label is not None else 0
        query_stats.append({
            "id": query_id,
            "mean_k": np.mean(k_values),
            "std_k": np.std(k_values),
            "min_k": min(k_values),
            "max_k": max(k_values),
            "n_models": len(k_values),
            "total_images": total_images,
            "model_ks": model_ks
        })

    # Table 1: Consistent Failures (sorted by mean_k descending).
    failure_title = "Consistent Failures" if len(model_short_names) > 1 else "Hardest Matches"
    failure_desc = (f"Queries with highest mean k across {len(model_short_names)} models (always hard to match)."
                    if len(model_short_names) > 1 else "Queries with highest k values (hardest to match).")

    utils.print_function_banner(failure_title, failure_desc)

    consistent_failures = sorted(query_stats, key=lambda x: -x["mean_k"])
    headers = ["Query ID", "Mean K", "Std K", "Min K", "Max K", "Total Imgs"] + model_short_names
    rows = []
    for stats in consistent_failures[:top_n]:
        row = [stats["id"], f"{stats['mean_k']:.1f}", f"{stats['std_k']:.1f}",
               stats["min_k"], stats["max_k"], stats["total_images"]]
        for name in model_short_names:
            row.append(stats["model_ks"].get(name, "-"))
        rows.append(row)
    utils.print_table(headers, rows)

    # Table 2: Volatile Queries (sorted by std_k descending) — multiple models only.
    if len(model_short_names) > 1:
        utils.print_function_banner(
            "Volatile Queries",
            f"Queries with highest k variance across {len(model_short_names)} models "
            "(model-dependent failures).")

        volatile_queries = sorted(query_stats, key=lambda x: -x["std_k"])
        rows = []
        for stats in volatile_queries[:top_n]:
            row = [stats["id"], f"{stats['mean_k']:.1f}", f"{stats['std_k']:.1f}",
                   stats["min_k"], stats["max_k"], stats["total_images"]]
            for name in model_short_names:
                row.append(stats["model_ks"].get(name, "-"))
            rows.append(row)
        utils.print_table(headers, rows)

    # Summary statistics.
    all_mean_ks = [s["mean_k"] for s in query_stats]
    hit_at_1_count = sum(1 for s in query_stats if s["mean_k"] == 1.0)
    never_hit_at_1 = sum(1 for s in query_stats if s["min_k"] > 1)
    always_hit_at_1 = sum(1 for s in query_stats if s["max_k"] == 1)

    hard_failure_stats = [s for s in query_stats if s["min_k"] > 1]
    hard_with_lte_2 = sum(1 for s in hard_failure_stats if s["total_images"] <= 2)
    hard_with_lte_3 = sum(1 for s in hard_failure_stats if s["total_images"] <= 3)
    hard_with_lte_5 = sum(1 for s in hard_failure_stats if s["total_images"] <= 5)

    summary = {
        "Total queries evaluated": len(query_stats),
        "Models compared": len(model_short_names),
        "Always Hits@1 (max_k=1)": f"{always_hit_at_1} ({always_hit_at_1/len(query_stats)*100:.1f}%)",
        "Never Hits@1 (min_k>1)": f"{never_hit_at_1} ({never_hit_at_1/len(query_stats)*100:.1f}%)",
        "Mean k=1.0 across all models": f"{hit_at_1_count} ({hit_at_1_count/len(query_stats)*100:.1f}%)",
        "Overall mean k": f"{np.mean(all_mean_ks):.2f}",
    }

    if hard_failure_stats:
        n_hard = len(hard_failure_stats)
        summary["Hard failures with ≤2 images"] = f"{hard_with_lte_2}/{n_hard} ({hard_with_lte_2/n_hard*100:.1f}%)"
        summary["Hard failures with ≤3 images"] = f"{hard_with_lte_3}/{n_hard} ({hard_with_lte_3/n_hard*100:.1f}%)"
        summary["Hard failures with ≤5 images"] = f"{hard_with_lte_5}/{n_hard} ({hard_with_lte_5/n_hard*100:.1f}%)"

    utils.print_key_value_section("Cross-Model Summary", summary)

    # Table 3: Average K per Model — multiple models only.
    if len(model_short_names) > 1:
        utils.print_function_banner(
            "Average K Per Model",
            "Average K value for each model across all evaluated queries (lower is better).")

        model_avg_ks = {name: [] for name in model_short_names}
        for query_id, model_ks in query_k_map.items():
            for name, k_val in model_ks.items():
                model_avg_ks[name].append(k_val)

        model_hits_at_1 = {name: sum(1 for k in vals if k == 1)
                           for name, vals in model_avg_ks.items()}
        model_avg_ks = {name: np.mean(vals) for name, vals in model_avg_ks.items() if vals}
        sorted_model_avg_ks = sorted(model_avg_ks.items(), key=lambda item: item[1])

        utils.print_table(["Model Name", "Average K"],
                          [[name, f"{avg_k:.2f}"] for name, avg_k in sorted_model_avg_ks])

        # Table 4: Hits@1 Per Model.
        utils.print_function_banner(
            "Hits@1 Per Model",
            "Total number of queries correctly matched at rank 1 per model (higher is better).")

        sorted_model_hits_at_1 = sorted(model_hits_at_1.items(), key=lambda item: item[1], reverse=True)
        utils.print_table(["Model Name", "Hits@1 Count"],
                          [[name, str(hits)] for name, hits in sorted_model_hits_at_1])

    return query_stats


def _pin_csv_store_from_dataset(dataset, csv_dir_override=None):
    """Pin csv_store reads to the CSV directory recorded on the dataset or an explicit override."""
    if isinstance(csv_dir_override, str) or (isinstance(csv_dir_override, os.PathLike) and not hasattr(csv_dir_override, '_mock_return_value')):
        resolved = utils.resolve_directory_path(str(csv_dir_override), constants.DB_DATA_DIR)
        dataset.csv_dir = resolved
        _csv_loader.set_db_dir(resolved)
        return

    csv_dir = getattr(dataset, 'csv_dir', None)
    if not csv_dir:
        raise RuntimeError(
            "Dataset has no csv_dir attribute. Regenerate the embeddings "
            "with a current build of src.retrieval.embeddings (which now "
            "requires --csv-dir or a .source_csv_dir sidecar).")

    resolved = utils.resolve_relative_csv_dir(csv_dir) or csv_dir

    if not os.path.isdir(resolved):
        warning_msg = (
            f"\nWARNING: The CSV directory recorded on this dataset was not found on this machine:\n"
            f"  {csv_dir}\n"
            f"This occurs when embeddings or datasets were serialized on a different machine or path layout.\n"
            f"Please specify the correct directory using --csv-dir (e.g. --csv-dir database-format/thesis)."
        )
        print(warning_msg, file=sys.stderr)
        raise FileNotFoundError(
            f"Recorded CSV directory '{csv_dir}' not found on this machine. "
            f"Use --csv-dir to provide the local path."
        )

    dataset.csv_dir = resolved
    _csv_loader.set_db_dir(resolved)


def _pin_csv_store_from_retriever(retriever, model_name=None, csv_dir_override=None):
    """Pin csv_store from the active model's dataset on a Retriever."""
    data = retriever._get_model_data(model_name)
    if data is None:
        raise RuntimeError(
            f"Retriever has no loaded data for model {model_name!r}.")
    override = csv_dir_override or getattr(retriever, 'csv_dir_override', None)
    if not (isinstance(override, str) or (isinstance(override, os.PathLike) and not hasattr(override, '_mock_return_value'))):
        override = None
    _pin_csv_store_from_dataset(data['dataset'], csv_dir_override=override)


def _get_filtered_test_instances(retriever, train_test=False, min_instances=2, min_length=None,
                                 max_length=None, sex=None, locations=None, types=None,
                                 qualities=None, min_time=None, max_time=None, allow_unknonw=False):
    """Filter test instances based on media and individual properties from the pinned CSV snapshot."""
    _pin_csv_store_from_retriever(retriever)
    ids, filenames, labels, _ = src.retrieval.evaluation._get_test_instances(
        retriever, train_test=train_test, min_instances=min_instances)

    # Convert time strings to unix timestamps if necessary.
    min_time = utils.convert_date_to_unix(min_time)
    max_time = utils.convert_date_to_unix(max_time)

    # 1. Verify all retrieved IDs exist in the CSV store.
    db_media_ids = src.csv_store.media.get_matching_media_ids(ids=ids)

    if len(db_media_ids) != len(ids):
        missing_ids = list(set(ids) - set(db_media_ids))
        print(f"ERROR: The following {len(missing_ids)} IDs were not found in the media "
              "CSV snapshot. Please regenerate the snapshot.")
        return [], [], [], {}

    filtered_media_ids_set = set(ids)

    # 2. Filter based on Media properties.
    media_filters = {
        "locations": locations,
        "types": types,
        "qualities": qualities,
        "min_time": min_time,
        "max_time": max_time
    }

    if any(v is not None for v in media_filters.values()):
        matched_media_ids = src.csv_store.media.get_matching_media_ids(
            ids=list(filtered_media_ids_set),
            locations=locations,
            types=types,
            qualities=qualities,
            min_time=min_time,
            max_time=max_time,
            allow_unknown=allow_unknonw
        )
        filtered_media_ids_set = filtered_media_ids_set.intersection(set(matched_media_ids))

    # 3. Filter based on Individual properties.
    individual_filters = {
        "min_length": min_length,
        "max_length": max_length,
        "sex": sex
    }

    if any(v is not None for v in individual_filters.values()) and filtered_media_ids_set:
        media_to_individual_map = src.csv_store.media.get_individual_ids_by_media_ids(
            list(filtered_media_ids_set), dict_format=True)
        candidate_individual_ids = list(set(media_to_individual_map.values()))

        matched_individual_ids = src.csv_store.individual.get_matching_individual_ids(
            ids=candidate_individual_ids,
            min_length=min_length,
            max_length=max_length,
            sex=[sex] if sex is not None else None,
            allow_unknown=allow_unknonw,
            source_only=False
        )

        matched_individual_ids_set = set(matched_individual_ids)

        final_media_ids = set()
        for media_id in filtered_media_ids_set:
            individual_id = media_to_individual_map.get(media_id)
            if individual_id in matched_individual_ids_set:
                final_media_ids.add(media_id)

        filtered_media_ids_set = final_media_ids

    # 4. Reconstruct the output lists.
    final_ids = []
    final_filenames = []
    final_labels = []

    for i, original_id in enumerate(ids):
        if original_id in filtered_media_ids_set:
            final_ids.append(original_id)
            final_filenames.append(filenames[i])
            final_labels.append(labels[i])

    unique_labels, counts = np.unique(final_labels, return_counts=True)
    final_label_counts_dict = dict(zip(unique_labels, counts))

    print("\n\nNumber of ids fetched after filtering:", len(final_ids))

    return final_ids, final_filenames, np.array(final_labels), final_label_counts_dict


def _show_dataset_embeddings_consistency(model_name):
    """Check and print whether the stored dataset and stored embeddings files have matching lengths."""
    utils.print_function_banner(
        "show_dataset_embeddings_consistency",
        f"Checking dataset/embeddings consistency for model: {model_name}.")

    embeddings_dir = src.retrieval.generation_store.resolve_active_directory(model_name)
    dataset_path = os.path.join(embeddings_dir, "stored_dataset.pt")
    embeddings_path = os.path.join(embeddings_dir, "stored_embeddings.pt")

    if not os.path.exists(dataset_path):
        print(f"WARNING: Dataset file not found at {dataset_path}.")
        return
    if not os.path.exists(embeddings_path):
        print(f"WARNING: Embeddings file not found at {embeddings_path}.")
        return

    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    stored_embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=False)

    dataset_size = len(dataset)
    embeddings_size = len(stored_embeddings)

    utils.print_key_value_section(f"Consistency Check: {model_name}", {
        "Dataset instances": dataset_size,
        "Embedding vectors": embeddings_size,
        "In sync": "Yes" if dataset_size == embeddings_size else "NO"
    })

    if dataset_size == embeddings_size:
        print("Dataset and embeddings are consistent.")
    else:
        print(f"WARNING: Dataset ({dataset_size}) and embeddings ({embeddings_size}) are "
              f"out of sync by {abs(dataset_size - embeddings_size)} instances.")

        filenames = dataset.get_filenames()
        if dataset_size > embeddings_size:
            extra_count = dataset_size - embeddings_size
            print(f"The dataset has {extra_count} more instance(s) than the embeddings.")
            extra_filenames = filenames[embeddings_size:]
            print("Filenames in dataset without embeddings:")
            for fn in extra_filenames:
                print(f"  - {fn}")
        else:
            extra_count = embeddings_size - dataset_size
            print(f"The embeddings have {extra_count} more vector(s) than the dataset.")
            print("These orphaned embeddings have no corresponding dataset entry.")


if __name__ == "__main__":
    main()
