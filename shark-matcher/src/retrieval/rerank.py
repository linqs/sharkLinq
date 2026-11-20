"""k-Reciprocal re-ranking (Zhong et al., 2017) post-processing adapter for image retrieval."""

import numpy as np

# Outer k: size of the candidate's k-reciprocal neighbour set used to build the sparse feature vectors.
KRECIPROCAL_K1 = 20
# Inner k: number of nearest items whose feature vectors get averaged into the query's feature.
KRECIPROCAL_K2 = 6
# Blend weight on the Jaccard term: final = (1 - lambda) * original + lambda * jaccard.
KRECIPROCAL_LAMBDA = 0.3


class KReciprocalReranker:
    """k-reciprocal re-ranking adapter over a Retriever with caching of reranked neighbors."""

    def __init__(self, retriever, base_method, model=None):
        if base_method not in (0, 2):
            raise ValueError(
                f"k-reciprocal rerank only supports retrieval methods 0 "
                f"(image-image) and 2 (LOO centroid); got {base_method}.")

        self.retriever = retriever
        self.base_method = base_method
        self.model = model

        # pylint: disable=protected-access
        data = retriever._get_model_data(model)
        if data is None:
            raise ValueError(f"Failed to load model data for {model!r}.")
        self._data = data

        if base_method == 0:
            self._setup_image_gallery()
        else:
            self._setup_centroid_gallery()

        self._cache = {}

    def get_neighbors(self, query_id, model=None):
        """Return reranked neighbors dictionary matching the shape of Retriever.get_neighbors."""
        if model is not None and model != self.model:
            raise ValueError(
                f"Reranker was built for model {self.model!r}; received {model!r}.")

        cached = self._cache.get(query_id)
        if cached is None:
            if self.base_method == 0:
                cached = self._rerank_image_query(query_id)
            else:
                cached = self._rerank_centroid_query(query_id)
            self._cache[query_id] = cached

        return {
            "ids": list(cached["ids"]),
            "distances": list(cached["distances"]),
            "labels": list(cached["labels"]),
        }

    def _setup_image_gallery(self):
        """Build N x N image distance matrix and the full gallery V matrix."""
        emb = self._data['embeddings']
        self._gallery_ids = list(self._data['ids'])
        self._gallery_labels = self._data['labels']
        self._id_to_idx = {int(_id): i for i, _id in enumerate(self._gallery_ids)}

        n = len(emb)
        print(f"[KReciprocal] base_method=0; building gallery distance matrix "
              f"({n} x {n}) and V matrix. k1={KRECIPROCAL_K1}, "
              f"k2={KRECIPROCAL_K2}, lambda={KRECIPROCAL_LAMBDA}.")
        self._gallery_dist = _pairwise_l2(emb)
        # Diag=0 (Zhong contract) so each image is its own nearest neighbour.
        np.fill_diagonal(self._gallery_dist, 0.0)

        V, initial_rank = _build_v_matrix(self._gallery_dist, KRECIPROCAL_K1)
        self._V = _smooth_v_matrix(V, initial_rank, KRECIPROCAL_K2)
        print(f"[KReciprocal] V matrix ready.")

    def _setup_centroid_gallery(self):
        """Build L x L centroid distance matrix; augment per query."""
        # pylint: disable=protected-access
        if 'loo_state' not in self._data:
            self.retriever._build_loo_state(self._data)
        self._loo = self._data['loo_state']

        centroids = self._loo['centroid_matrix']
        n = len(centroids)
        print(f"[KReciprocal] base_method=2; building centroid distance matrix "
              f"({n} x {n}). k1={KRECIPROCAL_K1}, k2={KRECIPROCAL_K2}, "
              f"lambda={KRECIPROCAL_LAMBDA}.")
        self._gallery_dist = _pairwise_l2(centroids)
        np.fill_diagonal(self._gallery_dist, 0.0)
        print(f"[KReciprocal] Gallery ready; per-query augmentation will follow.")

    def _rerank_image_query(self, query_id):
        q_idx = self._id_to_idx.get(int(query_id))
        if q_idx is None:
            print(f"WARNING: Query image id {query_id} not in gallery.")
            return {"ids": [], "distances": [], "labels": []}

        jaccard = _jaccard_distance_sparse(self._V[q_idx], self._V)
        original = self._gallery_dist[q_idx]
        final = (1.0 - KRECIPROCAL_LAMBDA) * original \
                + KRECIPROCAL_LAMBDA * jaccard

        order = np.argsort(final)
        ids_out = [int(self._gallery_ids[i]) for i in order]
        distances_out = [float(final[i]) for i in order]
        labels_out = [int(self._gallery_labels[i]) for i in order]
        return {"ids": ids_out, "distances": distances_out, "labels": labels_out}

    def _rerank_centroid_query(self, query_id):
        q_idx = self._loo['id_to_emb_idx'].get(int(query_id))
        if q_idx is None:
            print(f"WARNING: Query image id {query_id} not in gallery.")
            return {"ids": [], "distances": [], "labels": []}

        # LOO-corrected query-to-centroid distances. Mirrors
        # Retriever.get_loo_centroid_neighbors exactly so the original-distance
        # blend term matches method 2's evaluation contract.
        emb = self._data['embeddings']
        labels = self._data['labels']
        q_emb = emb[q_idx]
        q_label = int(labels[q_idx])

        dists = np.linalg.norm(self._loo['centroid_matrix'] - q_emb, axis=1)
        n_L = self._loo['count_by_label'][q_label]
        if n_L >= 2:
            loo_sum = self._loo['sum_by_label'][q_label] - q_emb
            loo_norm = np.linalg.norm(loo_sum)
            if loo_norm > 0:
                loo_centroid = loo_sum / loo_norm
                dists[self._loo['label_to_idx'][q_label]] = float(
                    np.linalg.norm(loo_centroid - q_emb))

        reranked = kreciprocal_rerank(dists, self._gallery_dist)

        # Sort centroids and expand to image-level rows so downstream code
        # (dedup-by-label, mAP, self-removal) sees the same shape as the
        # non-reranked method 2 output.
        order = np.argsort(reranked)
        label_order = self._loo['label_order']
        images_by_label = self._loo['images_by_label']
        ids_arr = self._data['ids']
        labels_arr = self._data['labels']

        ids_out = []
        distances_out = []
        labels_out = []
        for c_idx in order:
            label = label_order[c_idx]
            d = float(reranked[c_idx])
            for img_idx in images_by_label[label]:
                ids_out.append(int(ids_arr[img_idx]))
                distances_out.append(d)
                labels_out.append(int(labels_arr[img_idx]))

        return {"ids": ids_out, "distances": distances_out, "labels": labels_out}


def kreciprocal_rerank(query_to_gallery, gallery_to_gallery):
    """Pure-function k-reciprocal reranking for a single external query against a gallery."""
    query_to_gallery = np.asarray(query_to_gallery, dtype=np.float32)
    gallery_to_gallery = np.asarray(gallery_to_gallery, dtype=np.float32)
    n = gallery_to_gallery.shape[0]
    total = n + 1

    dist = np.zeros((total, total), dtype=np.float32)
    dist[0, 1:] = query_to_gallery
    dist[1:, 0] = query_to_gallery
    dist[1:, 1:] = gallery_to_gallery
    # Diagonal stays 0 so each item is its own nearest neighbour (Zhong).

    V, initial_rank = _build_v_matrix(dist, KRECIPROCAL_K1)
    V = _smooth_v_matrix(V, initial_rank, KRECIPROCAL_K2)

    jaccard = _jaccard_distance_sparse(V[0], V[1:])
    final = (1.0 - KRECIPROCAL_LAMBDA) * query_to_gallery \
            + KRECIPROCAL_LAMBDA * jaccard
    return final.astype(np.float32)


def l2_normalize(X, axis=-1, eps=1e-12):
    """Normalize array vectors to unit L2 length using float64 intermediate precision."""
    arr = np.asarray(X)
    arr64 = arr.astype(np.float64, copy=False)
    norms = np.linalg.norm(arr64, axis=axis, keepdims=True)
    normalized = arr64 / np.maximum(norms, eps)
    if arr.dtype == np.float32:
        return normalized.astype(np.float32)
    return normalized


def compute_centroids(embeddings, labels):
    """Compute sorted unique labels and unit-L2 mean centroid embeddings per label."""
    emb = np.asarray(embeddings)
    emb64 = emb.astype(np.float64, copy=False)
    lbls = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(lbls)
    if unique_labels.size == 0:
        out_dtype = np.float32 if emb.dtype == np.float32 else np.float64
        return np.empty((0, emb64.shape[1]), dtype=out_dtype), unique_labels
    centroids = np.zeros((unique_labels.size, emb64.shape[1]), dtype=np.float64)
    for idx, label in enumerate(unique_labels):
        mean_vec = emb64[lbls == label].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0.0:
            mean_vec = mean_vec / norm
        centroids[idx] = mean_vec
    if emb.dtype == np.float32:
        return centroids.astype(np.float32), unique_labels
    return centroids, unique_labels


def _pairwise_l2(X, Y=None):
    """Compute the pairwise L2 distance matrix between rows of X and Y (or X and itself)."""
    X64 = np.asarray(X, dtype=np.float64)
    x_sq = (X64 * X64).sum(axis=1)
    # macOS Accelerate sets spurious floating-point flags in matmul even for finite inputs.
    if Y is None:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scaled_gram = 2.0 * (X64 @ X64.T)
        dist = x_sq[:, None] + x_sq[None, :] - scaled_gram
        np.fill_diagonal(dist, 0.0)
    else:
        Y64 = np.asarray(Y, dtype=np.float64)
        y_sq = (Y64 * Y64).sum(axis=1)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scaled_gram = 2.0 * (X64 @ Y64.T)
        dist = x_sq[:, None] + y_sq[None, :] - scaled_gram
    np.maximum(dist, 0.0, out=dist)
    return np.sqrt(dist).astype(np.float32)


def _build_v_matrix(dist_matrix, k1):
    """Build the sparse k-reciprocal feature matrix V from a distance matrix."""
    n = dist_matrix.shape[0]
    initial_rank = np.argsort(dist_matrix, axis=1).astype(np.int32)
    V = np.zeros((n, n), dtype=np.float32)

    half_k1 = int(round(k1 / 2.0))

    for i in range(n):
        # k1 nearest of i (includes i itself at position 0 since diag=0).
        forward = initial_rank[i, :k1 + 1]
        # For each forward neighbour, is i in its own k1 nearest?
        backward = initial_rank[forward, :k1 + 1]
        mask = (backward == i).any(axis=1)
        k_reciprocal = forward[mask]

        # Set expansion: pull in each k-reciprocal neighbour's (k1/2)-reciprocal
        # set when the two sets overlap heavily (Zhong's 2/3 threshold).
        k_expansion = k_reciprocal
        for c in k_reciprocal:
            c_forward = initial_rank[c, :half_k1 + 1]
            c_backward = initial_rank[c_forward, :half_k1 + 1]
            c_mask = (c_backward == c).any(axis=1)
            c_reciprocal = c_forward[c_mask]

            if c_reciprocal.size == 0:
                continue
            overlap = np.intersect1d(c_reciprocal, k_reciprocal,
                                     assume_unique=False).size
            if overlap > (2.0 / 3.0) * c_reciprocal.size:
                k_expansion = np.union1d(k_expansion, c_reciprocal)

        # Gaussian weights, L1-normalized.
        weights = np.exp(-dist_matrix[i, k_expansion]).astype(np.float32)
        s = weights.sum()
        if s > 0:
            V[i, k_expansion] = weights / s

    return V, initial_rank


def _smooth_v_matrix(V, initial_rank, k2):
    """Local query expansion smoothing on V: each row is averaged with its top-k2 nearest items' rows."""
    if k2 <= 1:
        return V
    n = V.shape[0]
    V_smoothed = np.zeros_like(V)
    for i in range(n):
        V_smoothed[i] = V[initial_rank[i, :k2], :].mean(axis=0)
    return V_smoothed


def _jaccard_distance_sparse(V_query, V):
    """Compute Jaccard distance from a single query feature vector V_query to every row of V."""
    n_gallery = V.shape[0]
    nonzero_q = np.where(V_query > 0)[0]
    if nonzero_q.size == 0:
        return np.ones(n_gallery, dtype=np.float32)

    sum_min = np.zeros(n_gallery, dtype=np.float32)
    for j in nonzero_q:
        sum_min += np.minimum(V_query[j], V[:, j])

    sum_q = V_query.sum()
    sum_g = V.sum(axis=1)
    sum_max = sum_q + sum_g - sum_min

    return (1.0 - sum_min / (sum_max + 1e-9)).astype(np.float32)
