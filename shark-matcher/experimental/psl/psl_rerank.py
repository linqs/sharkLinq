"""Applies PSL biological-constraint reranking to an existing embeddings directory and writes stored_image_neighbors_psl.pt."""

import argparse
import os
import sys

import torch

from src.core import constants
from src.core import path_utils

path_utils.ensure_shark_model_in_path()

# pylint: disable=wrong-import-position
import src.retrieval.neighbors as neighbors_mod
from src.core import utils
from src.psl.psl_engine import load_metadata_from_csvs
from src.psl.train_psl import learn_weights, _load_embeddings_data
# pylint: enable=wrong-import-position




def main():
    """Apply PSL reranking to an existing embeddings directory."""
    parser = argparse.ArgumentParser(
        description="Apply PSL reranking to cached image neighbors, writing "
                    "stored_image_neighbors_psl.pt alongside the base file. "
                    "Learns rule weights from the training split automatically "
                    "before applying reranking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m experimental.psl.psl_rerank "
               "--embeddings-name best_unmatch_new_out\n"
               "  python3 -m experimental.psl.psl_rerank "
               "--embeddings-name best_unmatch_new_out "
               "--psl-top-k 100\n")
    parser.add_argument('--embeddings-name', type=str, required=True,
                        help='Name of the embeddings directory under '
                             'sm-data/embeddings/ (e.g. best_unmatch_new_out). '
                             'Must contain stored_image_neighbors.pt.')
    parser.add_argument('--psl-top-k', type=int, default=100,
                        help='Number of top neighbors to rerank (default: 100).')
    parser.add_argument('--psl-seed', type=int, default=42,
                        help='Random seed for deterministic PSL inference '
                             '(default: 42).')
    args = parser.parse_args()

    utils.print_script_banner(__file__, args)

    embeddings_dir = os.path.join(constants.EMBEDDINGS_DIR, args.embeddings_name)
    if not os.path.isdir(embeddings_dir):
        print(f"ERROR: Embeddings directory not found: {embeddings_dir}")
        sys.exit(1)

    base_path = os.path.join(embeddings_dir, "stored_image_neighbors.pt")
    if not os.path.exists(base_path):
        print(f"ERROR: Base neighbors file not found: {base_path}")
        print("  Run embeddings.py first to generate stored_image_neighbors.pt.")
        sys.exit(1)

    dataset, stored_neighbors = _load_embeddings_data(embeddings_dir)
    csv_dir = getattr(dataset, 'csv_dir', None)
    if not csv_dir or not os.path.isdir(csv_dir):
        print("ERROR: No csv_dir found on stored dataset. "
              "Re-generate embeddings so csv_dir is pinned.")
        sys.exit(1)

    metadata = load_metadata_from_csvs(csv_dir)

    print("[PSL] Learning weights from training split...")
    psl_weights = learn_weights(
        dataset, stored_neighbors, metadata,
        psl_top_k=args.psl_top_k, seed=args.psl_seed)

    neighbors_mod.store_psl_neighbors(
        model_dir=embeddings_dir,
        metadata=metadata,
        psl_seed=args.psl_seed,
        psl_weights=psl_weights,
        psl_top_k=args.psl_top_k,
    )

    psl_path = os.path.join(embeddings_dir, "stored_image_neighbors_psl.pt")
    print(f"\n[PSL] Done. PSL neighbors written to: {psl_path}")
    print(f"  Evaluate with: python3 -m src.retrieval.evaluation "
          f"--embeddings-name {args.embeddings_name} --use-psl")


if __name__ == "__main__":
    main()
