"""Probabilistic Soft Logic (PSL) inference engine for biological constraint reranking."""

import pandas as pd
from pslpython.model import Model
from pslpython.partition import Partition

import src.csv_store._loader as csv_loader
import src.csv_store.individual as csv_individual
import src.csv_store.media as csv_media


def load_metadata_from_csvs(csv_dir):
    """Load per-image metadata keyed by filename from CSVs, applying source-individual fallback for sex."""
    print(f"[PSL] Loading per-image metadata from: {csv_dir}")

    # Step 1: per-image data via multimodal path.
    medias = csv_media.load_medias_with_individuals(csv_dir=csv_dir)

    metadata = {}
    for m in medias:
        time_days = None
        if m.time is not None:
            try:
                time_days = float(m.time) / 86400.0
            except (ValueError, TypeError):
                pass
        sex = (m.individual.sex or "").upper() or None if m.individual else None
        metadata[m.filename] = {
            "sex": sex,
            "length": (float(m.individual.length)
                       if m.individual and m.individual.length is not None
                       else None),
            "location": m.location,
            "time_days": time_days,
            "_source_id": m.individual.source if m.individual else None,
        }

    # Step 2: source-individual sex fallback.
    n_before = sum(1 for v in metadata.values() if v["sex"])

    csv_loader.set_db_dir(csv_dir)
    sources = csv_individual.get_individuals(
        ids=None, get_full=True, source_only=True)
    source_sex = {}
    for source in sources:
        sex = next(
            (ind.sex for ind in source.associated_individuals if ind.sex),
            None,
        )
        if sex:
            source_sex[source.id] = sex.upper()

    # Source sex overrides media-individual sex when available to match historical labeling behavior.
    # Prioritizing media-individual sex introduces approximately 26 M/F conflicts that harm reranking performance.
    for entry in metadata.values():
        src_id = entry["_source_id"]
        if src_id is not None and src_id in source_sex:
            entry["sex"] = source_sex[src_id]

    # Remove internal helper key before returning.
    for entry in metadata.values():
        del entry["_source_id"]

    n_sex = sum(1 for v in metadata.values() if v["sex"])
    n_len = sum(1 for v in metadata.values() if v["length"] is not None)
    n_loc = sum(1 for v in metadata.values() if v["location"])
    n_time = sum(1 for v in metadata.values() if v["time_days"] is not None)
    print(f"[PSL] Loaded metadata for {len(metadata)} images "
          f"(sex: {n_sex} [{n_before} media + {n_sex - n_before} source fallback], "
          f"length: {n_len}, location: {n_loc}, time: {n_time}).")

    return metadata


class PSLReranker:
    """Applies Hinge-Loss Markov Random Field inference to rerank nearest neighbors using biological constraints."""

    def __init__(self, psl_top_k=100, seed=42, weights=None, active_rules=None):
        """Initialize the PSL reranker with specified top-K limit, seed, weights, and active rule set."""
        from src.psl.psl_rules import DEFAULT_WEIGHTS  # pylint: disable=import-outside-toplevel

        self.psl_top_k = psl_top_k
        self.seed = seed
        self.weights = weights if weights is not None else dict(DEFAULT_WEIGHTS)
        self.active_rules = active_rules
        print(f"[PSL] Initialising PSLReranker (top_k={psl_top_k}, seed={seed}, "
              f"weights={self.weights}).")

    @staticmethod
    def euclidean_to_truth(dist):
        """Convert L2 distance between normalized embeddings to a PSL truth value in [0, 1]."""
        cos_sim = 1.0 - (dist * dist) / 2.0
        return max(0.0, (cos_sim + 1.0) / 2.0)

    @classmethod
    def euclidean_to_pseudo_distance(cls, dist):
        """Map an L2 distance into the [0, 1] pseudo-distance space used for PSL reranked entries."""
        return 1.0 - cls.euclidean_to_truth(dist)

    def rerank(self, query_name, neighbor_names, neighbor_distances,
               metadata_by_filename):
        """Rerank the top-K neighbors for a single query using PSL inference and biological constraints."""
        n_total = len(neighbor_names)
        k = min(self.psl_top_k, n_total)

        if k < 2:
            # Nothing meaningful to rerank.
            return list(range(n_total)), list(neighbor_distances)

        # Slice the top-K to rerank.
        top_names = neighbor_names[:k]
        top_dists = neighbor_distances[:k]

        # Build a fresh PSL model for this query.
        from src.psl import psl_rules  # pylint: disable=import-outside-toplevel

        model = Model("psl_rerank")

        predicates = psl_rules.build_predicates()
        for pred in predicates.values():
            model.add_predicate(pred)

        for rule in psl_rules.build_rules(weights=self.weights, learnable=False,
                                          active_rules=self.active_rules):
            model.add_rule(rule)

        # Use string IDs for PSL atoms: "q" for query, "nX" for neighbors.
        q_id = "q"

        neural_rows = []
        target_rows = []
        all_atoms = set()
        all_atoms.add(q_id)

        for i in range(k):
            n_id = f"n{i}"
            all_atoms.add(n_id)
            truth = self.euclidean_to_truth(top_dists[i])
            neural_rows.append([q_id, n_id, truth])
            target_rows.append([q_id, n_id])

        # Sex observations for all atoms (hard 0/0 for unknown — matches old engine).
        male_rows = []
        female_rows = []

        q_male, q_female = psl_rules.encode_sex(metadata_by_filename, query_name)
        male_rows.append([q_id, q_male])
        female_rows.append([q_id, q_female])

        for i in range(k):
            n_id = f"n{i}"
            n_male, n_female = psl_rules.encode_sex(
                metadata_by_filename, top_names[i])
            male_rows.append([n_id, n_male])
            female_rows.append([n_id, n_female])

        # Length observations — pairwise SimilarLength(q, nX).
        # Unknown length encodes as 1.0 so length_disagreement is silent.
        length_rows = []
        for i in range(k):
            n_id = f"n{i}"
            sim = psl_rules.encode_length_pair(
                metadata_by_filename, query_name, top_names[i])
            length_rows.append([q_id, n_id, sim])

        # Load data — pslpython 2.4 requires integer-indexed DataFrames.
        predicates["NeuralSim"].add_data(
            Partition.OBSERVATIONS, pd.DataFrame(neural_rows))
        predicates["IsMale"].add_data(
            Partition.OBSERVATIONS, pd.DataFrame(male_rows))
        predicates["IsFemale"].add_data(
            Partition.OBSERVATIONS, pd.DataFrame(female_rows))
        predicates["SimilarLength"].add_data(
            Partition.OBSERVATIONS, pd.DataFrame(length_rows))
        predicates["SameShark"].add_data(
            Partition.TARGETS, pd.DataFrame(target_rows))

        # Run inference.
        results = model.infer(
            psl_options={'random.seed': str(self.seed)})
        result_df = results[model.get_predicate("SameShark")]

        # Extract q->nX truth values.
        psl_scores = {}
        for _, row in result_df.iterrows():
            src = str(row.iloc[0])
            dst = str(row.iloc[1])
            if src == q_id and dst.startswith("n"):
                idx = int(dst[1:])
                psl_scores[idx] = float(row.iloc[2])

        # Sort top-K by descending PSL truth.
        top_k_indices = list(range(k))
        top_k_indices.sort(key=lambda i: psl_scores.get(i, 0.0), reverse=True)

        # Assemble reordered top-K and tail pseudo-distances in [0, 1] so lower values indicate greater similarity.
        final_indices = top_k_indices + list(range(k, n_total))
        top_k_pseudo = [1.0 - psl_scores.get(i, 0.0) for i in top_k_indices]
        tail_pseudo = [self.euclidean_to_pseudo_distance(d)
                       for d in neighbor_distances[k:]]
        final_distances = top_k_pseudo + tail_pseudo

        return final_indices, final_distances
