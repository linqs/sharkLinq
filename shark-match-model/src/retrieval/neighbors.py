"""Computes and stores the nearest neighbors for embeddings."""

import os

import numpy as np
import torch
import tqdm

from src.core import constants
import src.retrieval.rerank
from src.core import utils


def main():
    """Run nearest neighbor computation for ad-hoc debugging."""
    pass


def store_neighbors(model_dir, centroid=False, transformed=False,
                    use_psl_reranking=False, metadata=None, psl_seed=42,
                    psl_weights=None, psl_top_k=100, psl_active_rules=None,
                    suffix=""):
    """Compute and store nearest neighbors for image or centroid embeddings in model_dir."""
    utils.print_function_banner(
        "store_neighbors",
        f"Storing neighbors for model dir: {model_dir}. Centroid: {centroid}, "
        f"transformed: {transformed}, "
        f"use_psl_reranking: {use_psl_reranking}."
    )

    # Load dataset on CPU.
    dataset = torch.load(
        os.path.join(model_dir, "stored_dataset.pt"),
        map_location="cpu", weights_only=False)

    # Load the regular image or centroid embeddings, either transformed or original.
    if not centroid:
        file_name = f"stored_embeddings{suffix}.pt"
        if transformed:
            file_name = f"stored_transformed_embeddings{suffix}.pt"

        embeddings = torch.load(
            os.path.join(model_dir, file_name),
            map_location="cpu", weights_only=False).cpu().detach().numpy()
        labels = dataset.get_labels().cpu().detach().numpy()
    else:
        file_name = f"stored_centroid_embeddings{suffix}.pt"
        if transformed:
            file_name = f"stored_transformed_centroid_embeddings{suffix}.pt"

        embeddings = torch.load(
            os.path.join(model_dir, file_name),
            map_location="cpu", weights_only=False).cpu().detach().numpy()
        labels = torch.load(
            os.path.join(model_dir, f"stored_centroid_labels{suffix}.pt"),
            map_location="cpu", weights_only=False).cpu().detach().numpy()

    # Check that the number of embeddings matches the number of labels.
    assert len(embeddings) == len(labels)

    if len(embeddings) == 0:
        print("No embeddings found. Returning empty neighbors list.")
        # Save empty results to maintain consistency
        os.makedirs(model_dir, exist_ok=True)
        if not centroid:
            torch.save({}, os.path.join(model_dir, f"stored_image_neighbors{suffix}.pt"))
        else:
            torch.save({}, os.path.join(model_dir, f"stored_centroid_neighbors{suffix}.pt"))
        return

    print(f"Computing neighbors for {len(embeddings)}.", )

    # Compute neighbors using pure-NumPy Euclidean distance.
    # pylint: disable=protected-access
    dist_matrix = src.retrieval.rerank._pairwise_l2(embeddings)
    neighbor_results = np.argsort(dist_matrix, axis=1, kind="stable")
    distances = np.take_along_axis(dist_matrix, neighbor_results, axis=1)

    # Check that each label has a neighbor result list.
    assert len(labels) == len(neighbor_results)

    # Get the neighbors for each id in the dataset in the desired dict format.
    neighbors = _computed_neighbors_to_dict(
        dataset, labels, distances, neighbor_results, centroid=centroid)

    # Optionally apply PSL reranking to image neighbors.
    if use_psl_reranking and not centroid and metadata:
        neighbors = _apply_psl_reranking(neighbors, metadata,
                                         psl_seed=psl_seed,
                                         psl_weights=psl_weights,
                                         psl_top_k=psl_top_k,
                                         psl_active_rules=psl_active_rules)

    # Save the neighbors.
    os.makedirs(model_dir, exist_ok=True)
    if not centroid:
        print(f"Saving image neighbors.")
        torch.save(neighbors, os.path.join(model_dir,
                                           f"stored_image_neighbors{suffix}.pt"))
    else:
        print(f"Saving centroid neighbors.")
        torch.save(neighbors, os.path.join(model_dir,
                                           f"stored_centroid_neighbors{suffix}.pt"))


def update_centroid_neighbors(model_dir, update_result, transformed=False):
    """Incrementally update centroid neighbors, computing distances only for affected centroids."""
    utils.print_function_banner(
        "update_centroid_neighbors",
        f"Incrementally updating centroid neighbors in dir: {model_dir}. "
        f"transformed: {transformed}. "
        f"Affected: {update_result.affected_labels}.")

    centroid_file = ("stored_transformed_centroid_embeddings.pt" if transformed
                     else "stored_centroid_embeddings.pt")
    current_centroids = torch.load(
        os.path.join(model_dir, centroid_file),
        map_location="cpu", weights_only=False).cpu().detach().numpy()
    current_centroid_labels_np = torch.load(
        os.path.join(model_dir, "stored_centroid_labels.pt"),
        map_location="cpu", weights_only=False).cpu().detach().numpy()

    dataset = torch.load(
        os.path.join(model_dir, "stored_dataset.pt"),
        map_location="cpu", weights_only=False)
    dataset_labels_np = dataset.get_labels().cpu().detach().numpy()
    dataset_ids = dataset.get_ids()
    dataset_filenames = dataset.get_filenames()

    old_neighbors_path = os.path.join(model_dir, "stored_centroid_neighbors.pt")
    if not os.path.exists(old_neighbors_path):
        store_neighbors(model_dir, centroid=True, transformed=transformed)
        return
    old_neighbors_dict = torch.load(
        old_neighbors_path, map_location="cpu", weights_only=False)

    affected_set = set(update_result.affected_labels)

    # Centroid layout: one row per label, in sorted-label order.
    n_total = current_centroids.shape[0]
    label_at_idx = [int(l) for l in current_centroid_labels_np]
    label_to_idx = {lbl: i for i, lbl in enumerate(label_at_idx)}
    assert n_total == len(label_at_idx)

    # Map label -> image indices belonging to that label.
    images_per_label = {}
    for i, lbl in enumerate(dataset_labels_np):
        images_per_label.setdefault(int(lbl), []).append(i)

    # A representative image filename per label, used to look up the old
    # cached expansion (any image of an unaffected centroid works).
    rep_fn_per_label = {}
    for i in range(len(dataset_ids)):
        lbl = int(dataset_labels_np[i])
        rep_fn_per_label.setdefault(lbl, dataset_filenames[i])

    # Determine which centroid rows are affected. A centroid is affected if
    # its label is in affected_set (vector changed) or it lacks a cached
    # representative (newly added / no usable old entry).
    affected_indices = []
    for idx, lbl in enumerate(label_at_idx):
        rep = rep_fn_per_label.get(lbl)
        if (lbl in affected_set or rep is None
                or old_neighbors_dict.get(rep) is None):
            affected_indices.append(idx)
    affected_indices_set = set(affected_indices)

    # Single pairwise L2 pass over affected rows covering all (affected, any) pairs by symmetry.
    if affected_indices:
        affected_emb = current_centroids[affected_indices]
        # pylint: disable=protected-access
        dist_matrix = src.retrieval.rerank._pairwise_l2(
            affected_emb, current_centroids
        )
        affected_row_of = {idx: dist_matrix[i]
                           for i, idx in enumerate(affected_indices)}
    else:
        affected_row_of = {}

    def fresh_distance(a_idx, b_idx):
        """Symmetric lookup; one of a_idx/b_idx must be affected."""
        if a_idx in affected_row_of:
            return float(affected_row_of[a_idx][b_idx])
        return float(affected_row_of[b_idx][a_idx])

    # Build per-query sorted neighbor list, indexed by centroid row index.
    sorted_neighbors_per_idx = {}
    for q_idx, q_label in enumerate(label_at_idx):
        if q_idx in affected_indices_set:
            dist_row = affected_row_of[q_idx]
            sorted_neighbors_per_idx[q_idx] = sorted(
                ((j, float(dist_row[j])) for j in range(n_total)),
                key=lambda p: p[1])
            continue

        # Unaffected query: reconstruct from the old cached expansion.
        rep_fn = rep_fn_per_label.get(q_label)
        old_entry = old_neighbors_dict.get(rep_fn) if rep_fn is not None else None
        if old_entry is None:
            print(f"WARNING: No cached centroid neighbor entry for "
                  f"label={q_label}. Computing from scratch.")
            q_vec = current_centroids[q_idx:q_idx + 1]
            # pylint: disable=protected-access
            dist_row_vec = src.retrieval.rerank._pairwise_l2(
                q_vec, current_centroids
            )[0]
            sorted_neighbors_per_idx[q_idx] = sorted(
                ((j, float(dist_row_vec[j])) for j in range(n_total)),
                key=lambda p: p[1])
            continue

        old_match_labels = old_entry["match_labels"]
        old_match_distances = old_entry["match_distances"]

        # Walk the old expansion; contiguous (label, distance) runs come from
        # one neighbor centroid.
        pairs = []
        seen_labels = set()
        k = 0
        while k < len(old_match_labels):
            label = int(old_match_labels[k])
            dist = old_match_distances[k]
            run_end = k + 1
            while (run_end < len(old_match_labels)
                   and int(old_match_labels[run_end]) == label
                   and old_match_distances[run_end] == dist):
                run_end += 1

            if label in seen_labels:
                k = run_end
                continue
            seen_labels.add(label)

            neighbor_idx = label_to_idx.get(label)
            if neighbor_idx is None:
                # Centroid was deleted in the new state.
                k = run_end
                continue

            if neighbor_idx in affected_indices_set:
                pairs.append((neighbor_idx, fresh_distance(q_idx, neighbor_idx)))
            else:
                pairs.append((neighbor_idx, float(dist)))
            k = run_end

        # Newly present labels missing from the old expansion.
        for new_lbl, new_idx in label_to_idx.items():
            if new_lbl in seen_labels:
                continue
            if new_idx in affected_indices_set:
                pairs.append((new_idx, fresh_distance(q_idx, new_idx)))

        pairs.sort(key=lambda p: p[1])
        sorted_neighbors_per_idx[q_idx] = pairs

    # Expand each centroid's neighbor list to all its member images.
    expansion_per_label = {}
    for q_lbl, q_idx in label_to_idx.items():
        match_ids, match_names, match_labels_out, match_distances = [], [], [], []
        for neighbor_idx, dist in sorted_neighbors_per_idx[q_idx]:
            neighbor_label = label_at_idx[neighbor_idx]
            for img_idx in images_per_label.get(neighbor_label, []):
                match_ids.append(dataset_ids[img_idx])
                match_names.append(dataset_filenames[img_idx])
                match_labels_out.append(neighbor_label)
                match_distances.append(dist)
        expansion_per_label[q_lbl] = (
            match_ids, match_names, match_labels_out, match_distances)

    # Image-keyed result dict.
    new_dict = {}
    for i, img_id in enumerate(dataset_ids):
        lbl = int(dataset_labels_np[i])
        expansion = expansion_per_label.get(lbl)
        if expansion is None:
            continue
        m_ids, m_names, m_labels, m_dists = expansion
        fn = dataset_filenames[i]
        new_dict[fn] = {
            "query": img_id,
            "query_name": fn,
            "matches": list(m_ids),
            "match_names": list(m_names),
            "match_distances": list(m_dists),
            "match_labels": list(m_labels),
        }

    os.makedirs(model_dir, exist_ok=True)
    print("Saving incrementally updated centroid neighbors.")
    torch.save(new_dict, os.path.join(model_dir, "stored_centroid_neighbors.pt"))


def update_image_neighbors(model_dir, new_filenames, transformed=False,
                           use_psl_reranking=False, metadata=None,
                           psl_seed=42, psl_weights=None, psl_top_k=100,
                           psl_active_rules=None):
    """Incrementally update cached image neighbors by computing distances only for new embeddings."""
    utils.print_function_banner(
        "update_image_neighbors",
        f"Updating image neighbors for embeddings in dir: {model_dir}. "
        f"New images: {len(new_filenames)}, transformed: {transformed}, "
        f"use_psl_reranking: {use_psl_reranking}."
    )

    emb_dir = model_dir

    # Load the previously cached neighbor dictionary if present.
    stored_neighbors_path = os.path.join(emb_dir, "stored_image_neighbors.pt")
    if not os.path.exists(stored_neighbors_path):
        store_neighbors(
            model_dir,
            centroid=False,
            transformed=transformed,
            use_psl_reranking=use_psl_reranking,
            metadata=metadata,
            psl_seed=psl_seed,
            psl_weights=psl_weights,
            psl_top_k=psl_top_k,
            psl_active_rules=psl_active_rules,
        )
        return

    # Load the current dataset and embeddings on CPU.
    dataset = torch.load(
        os.path.join(emb_dir, "stored_dataset.pt"),
        map_location="cpu", weights_only=False)

    file_name = "stored_transformed_embeddings.pt" if transformed else "stored_embeddings.pt"
    all_embeddings = torch.load(
        os.path.join(emb_dir, file_name),
        map_location="cpu", weights_only=False).cpu().detach().numpy()

    stored_neighbors = torch.load(
        stored_neighbors_path, map_location="cpu", weights_only=False)

    # Current dataset metadata.
    ids = dataset.get_ids()
    filenames = dataset.get_filenames()
    labels = dataset.get_labels().cpu().detach().numpy()

    n_total = len(filenames)

    if n_total == 0:
        print("No embeddings found. Saving empty neighbors.")
        torch.save({}, stored_neighbors_path)
        return

    # Create a staging temp file inside EMBEDDINGS_DIR for atomic write.
    import tempfile
    import shutil
    staging_fd, staging_path = tempfile.mkstemp(
        prefix="neighbors_staging_", suffix=".pt", dir=emb_dir)
    os.close(staging_fd)

    try:
        # Build index maps for the current dataset.
        filename_to_idx = {fn: i for i, fn in enumerate(filenames)}
        new_filenames_set = set(new_filenames)
        current_filenames_set = set(filenames)

        # Identify old filenames that have cached neighbor data.
        old_filenames = [fn for fn in filenames if fn not in new_filenames_set]
        # Identify filenames that were deleted (in old cache but not in current dataset).
        deleted_filenames_set = set(stored_neighbors.keys()) - current_filenames_set

        has_deletions = len(deleted_filenames_set) > 0
        has_additions = len(new_filenames) > 0

        print(f"Total images: {n_total}, new: {len(new_filenames)}, "
              f"carried over: {len(old_filenames)}, deleted: {len(deleted_filenames_set)}.")

        # -- Compute distances between new embeddings and all embeddings in pure NumPy. --
        new_distances = None
        new_fn_to_row = {}
        if has_additions:
            new_indices = [filename_to_idx[fn] for fn in new_filenames]
            new_emb = all_embeddings[new_indices]  # [M, D]

            print(f"Computing {len(new_filenames)} × {n_total} pairwise distances in NumPy.")
            # pylint: disable=protected-access
            new_distances = src.retrieval.rerank._pairwise_l2(
                new_emb, all_embeddings
            )

            # Build a map from new filename to its row index in new_distances.
            new_fn_to_row = {fn: row_i for row_i, fn in enumerate(new_filenames)}

        # -- Build the updated neighbor dictionary. --
        neighbors = {}

        for i, query_name in tqdm.tqdm(enumerate(filenames), total=n_total,
                                        desc="Updating neighbor dictionary"):
            query_id = ids[i]

            if query_name in new_filenames_set:
                # --- New image: build neighbor list from scratch using its distance row. ---
                row = new_distances[new_fn_to_row[query_name]]
                sorted_indices = np.argsort(row)

                neighbors[query_name] = {
                    "query": query_id,
                    "query_name": query_name,
                    "matches": [ids[j] for j in sorted_indices],
                    "match_names": [filenames[j] for j in sorted_indices],
                    "match_distances": [float(row[j]) for j in sorted_indices],
                    "match_labels": [int(labels[j]) for j in sorted_indices],
                }
            else:
                # --- Old image: merge new distances into its existing sorted neighbor list. ---
                old_entry = stored_neighbors.get(query_name)
                if old_entry is None:
                    # Edge case: old image with no cached neighbors (shouldn't happen but handle it).
                    print(f"WARNING: No cached neighbors for {query_name}. "
                          "Computing from scratch.")
                    # pylint: disable=protected-access
                    dists = src.retrieval.rerank._pairwise_l2(
                        all_embeddings[i:i + 1], all_embeddings
                    )[0]
                    sorted_indices = np.argsort(dists)
                    neighbors[query_name] = {
                        "query": query_id,
                        "query_name": query_name,
                        "matches": [ids[j] for j in sorted_indices],
                        "match_names": [filenames[j] for j in sorted_indices],
                        "match_distances": [float(dists[j]) for j in sorted_indices],
                        "match_labels": [int(labels[j]) for j in sorted_indices],
                    }
                    continue

                # Start from the existing cached lists.
                old_match_names = old_entry["match_names"]
                old_match_ids = old_entry["matches"]
                old_match_dists = old_entry["match_distances"]
                old_match_labels = old_entry["match_labels"]

                # Fast path: additions only, no deletions — just copy to append.
                if not has_deletions:
                    kept_names = list(old_match_names)
                    kept_ids = list(old_match_ids)
                    kept_dists = list(old_match_dists)
                    kept_labels = list(old_match_labels)
                else:
                    # Filter out deleted entries.
                    kept_names, kept_ids, kept_dists, kept_labels = [], [], [], []
                    for k in range(len(old_match_names)):
                        if old_match_names[k] not in deleted_filenames_set:
                            kept_names.append(old_match_names[k])
                            kept_ids.append(old_match_ids[k])
                            kept_dists.append(old_match_dists[k])
                            kept_labels.append(old_match_labels[k])

                if has_additions:
                    # Append distances to the M new images.
                    # The distance from old image i to new image j is in new_distances[j, i]
                    # (new_distances is [M, N] where row=new image, col=all images).
                    query_col_idx = i  # This image's column index in the all_embeddings tensor.
                    for fn in new_filenames:
                        new_idx = filename_to_idx[fn]
                        row_in_new = new_fn_to_row[fn]
                        kept_names.append(fn)
                        kept_ids.append(ids[new_idx])
                        kept_dists.append(float(new_distances[row_in_new, query_col_idx]))
                        kept_labels.append(int(labels[new_idx]))

                # Re-sort by distance using argsort (efficient on nearly-sorted data).
                dists_array = np.array(kept_dists, dtype=np.float32)
                sort_order = np.argsort(dists_array, kind='mergesort')

                neighbors[query_name] = {
                    "query": query_id,
                    "query_name": query_name,
                    "matches": [kept_ids[j] for j in sort_order],
                    "match_names": [kept_names[j] for j in sort_order],
                    "match_distances": [float(dists_array[j]) for j in sort_order],
                    "match_labels": [kept_labels[j] for j in sort_order],
                }

        # Optionally apply PSL reranking.
        if use_psl_reranking and metadata:
            neighbors = _apply_psl_reranking(neighbors, metadata,
                                             psl_seed=psl_seed,
                                             psl_weights=psl_weights,
                                             psl_top_k=psl_top_k,
                                             psl_active_rules=psl_active_rules)

        # Save to staging file first.
        print("Saving updated image neighbors to staging file.")
        torch.save(neighbors, staging_path)

        # Atomic commit: move staging file to live path.
        print("Neighbor update successful. Committing staged file to live path.")
        shutil.move(staging_path, stored_neighbors_path)

    except Exception as e:
        print(f"ERROR: Neighbor update failed: {e}. Rolling back staged changes.")
        if os.path.exists(staging_path):
            os.remove(staging_path)
        raise

    finally:
        # Clean up the staging file if it still exists (e.g. after failed move).
        if os.path.exists(staging_path):
            os.remove(staging_path)


def store_psl_neighbors(model_dir, metadata, psl_seed=42,
                        psl_weights=None, psl_top_k=100, psl_active_rules=None,
                        suffix=""):
    """Apply PSL reranking to cached image neighbors and save as a separate file."""
    utils.print_function_banner(
        "store_psl_neighbors",
        f"Applying PSL reranking in: {model_dir} (suffix='{suffix}')."
    )

    base_path = os.path.join(model_dir, f"stored_image_neighbors{suffix}.pt")
    if not os.path.exists(base_path):
        store_neighbors(model_dir, centroid=False, transformed=False, suffix=suffix)
    psl_path = os.path.join(model_dir, f"stored_image_neighbors{suffix}_psl.pt")

    neighbors = torch.load(base_path, map_location="cpu", weights_only=False)

    neighbors = _apply_psl_reranking(neighbors, metadata, psl_seed=psl_seed,
                                     psl_weights=psl_weights, psl_top_k=psl_top_k,
                                     psl_active_rules=psl_active_rules)

    print("Saving PSL-reranked image neighbors.")
    torch.save(neighbors, psl_path)


def _expand_centroid_neighbors(query_label, embedding_labels, neighbor_results,
                               distances, label_to_images_map, ids, filenames,
                               label_to_centroid_idx):
    """Build the per-image expanded neighbor list for a single query label."""
    query_centroid_index = label_to_centroid_idx.get(query_label)
    if query_centroid_index is None:
        return None

    neighbor_centroid_indices = neighbor_results[query_centroid_index]
    neighbor_distances_result = distances[query_centroid_index]

    match_ids, match_names, match_labels, match_distances = [], [], [], []
    for i, neighbor_centroid_idx in enumerate(neighbor_centroid_indices):
        neighbor_label = embedding_labels[neighbor_centroid_idx]
        distance = neighbor_distances_result[i]
        for image_dataset_index in label_to_images_map.get(neighbor_label, []):
            match_ids.append(ids[image_dataset_index])
            match_names.append(filenames[image_dataset_index])
            match_labels.append(neighbor_label)
            match_distances.append(distance)

    return {
        "match_ids": match_ids,
        "match_names": match_names,
        "match_labels": match_labels,
        "match_distances": match_distances,
    }


def _computed_neighbors_to_dict(dataset, labels, distances, neighbor_results, centroid=False):
    """Convert sklearn kneighbors output into a filename-keyed dictionary of neighbor lists."""
    ids = dataset.get_ids()
    filenames = dataset.get_filenames()
    dataset_labels = dataset.get_labels().cpu().detach().numpy()

    embedding_labels = [int(label) for label in labels]

    return_dict = {}
    if not centroid:
        # Image-image: one row per image, straight copy of ordered lists.
        for i, query_id in tqdm.tqdm(enumerate(ids), total=len(ids),
                                     desc="Building neighbor dictionary"):
            query_name = filenames[i]
            neighbor_result = neighbor_results[i]
            match_ids = [ids[index] for index in neighbor_result]
            match_names = [filenames[index] for index in neighbor_result]
            match_labels = [int(dataset_labels[index]) for index in neighbor_result]
            match_distances = list(distances[i])
            return_dict[query_name] = {
                "query": query_id,
                "query_name": query_name,
                "matches": match_ids,
                "match_names": match_names,
                "match_distances": match_distances,
                "match_labels": match_labels,
            }
        return return_dict

    # Centroid path: build per-label expanded lists once, then fan out to every
    # image of the label.
    label_to_images_map = {}
    for i, image_id in enumerate(ids):
        label = int(dataset_labels[i])
        label_to_images_map.setdefault(label, []).append(i)

    # One centroid per label; map label -> its row index in neighbor_results.
    label_to_centroid_idx = {lbl: i for i, lbl in enumerate(embedding_labels)}

    label_to_neighbors = {}
    for query_label in tqdm.tqdm(list(label_to_centroid_idx.keys()),
                                 desc="Pre-computing centroid neighbors"):
        label_to_neighbors[query_label] = _expand_centroid_neighbors(
            query_label, embedding_labels, neighbor_results, distances,
            label_to_images_map, ids, filenames, label_to_centroid_idx)

    for i, query_id in tqdm.tqdm(enumerate(ids), total=len(ids),
                                 desc="Assembling neighbor dictionary"):
        query_name = filenames[i]
        query_label = int(dataset_labels[i])
        neighbor_data = label_to_neighbors.get(query_label)
        if not neighbor_data:
            print(f"WARNING: No neighbor data found for query_id {query_id} "
                  f"with label {query_label}. Skipping.")
            continue
        return_dict[query_name] = {
            "query": query_id,
            "query_name": query_name,
            "matches": neighbor_data["match_ids"],
            "match_names": neighbor_data["match_names"],
            "match_distances": neighbor_data["match_distances"],
            "match_labels": neighbor_data["match_labels"],
        }

    return return_dict


def _apply_psl_reranking(neighbors, metadata, psl_seed=42,
                         psl_weights=None, psl_top_k=100, psl_active_rules=None):
    """Apply PSL biological constraint reranking to all entries in a neighbors dictionary."""
    from src.psl.psl_engine import PSLReranker

    reranker = PSLReranker(psl_top_k=psl_top_k, seed=psl_seed,
                           weights=psl_weights, active_rules=psl_active_rules)
    total = len(neighbors)
    reranked_count = 0

    for query_name in tqdm.tqdm(neighbors, total=total, desc="PSL reranking neighbors"):
        entry = neighbors[query_name]

        match_labels = entry["match_labels"]
        match_distances = entry["match_distances"]
        match_ids = entry["matches"]
        match_names = entry["match_names"]

        if len(match_labels) < 2:
            continue

        reordered_indices, reordered_distances = reranker.rerank(
            query_name=query_name,
            neighbor_names=match_names,
            neighbor_distances=match_distances,
            metadata_by_filename=metadata,
        )

        # Apply the reordering.
        entry["matches"] = [match_ids[i] for i in reordered_indices]
        entry["match_names"] = [match_names[i] for i in reordered_indices]
        entry["match_distances"] = reordered_distances
        entry["match_labels"] = [match_labels[i] for i in reordered_indices]

        reranked_count += 1

    print(f"[PSL] Reranked {reranked_count}/{total} neighbor entries.")
    return neighbors


if __name__ == "__main__":
    main()
