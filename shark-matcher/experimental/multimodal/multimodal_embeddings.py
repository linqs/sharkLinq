"""Orchestrates late-fusion hyperparameter tuning, embedding generation, and neighbor indexing."""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

from src.core import constants
from src.core import path_utils

path_utils.ensure_shark_model_in_path()

from src.core import utils
import src.multimodal.optimize_late_fusion as optimizer
import src.multimodal.fusion_dataset as fusion_dataset
import src.retrieval.neighbors as neighbors

def main():
    """Optimize late-fusion weights, generate multimodal embeddings, and compute neighbor tables for a base model."""
    parser = argparse.ArgumentParser(
        description="Optimize and generate multimodal embeddings in the base directory."
    )
    parser.add_argument("--embeddings-dir", required=True,
                        help="Directory of the base embeddings (e.g. sm-data/embeddings/<name>).")
    parser.add_argument("--csv-dir", type=str, default=None,
                        help="Source database-format CSV directory.")
    args = parser.parse_args()
    utils.print_script_banner(__file__, args)
    
    # 1. Validation
    # Use the name relative to EMBEDDINGS_DIR if provided as an absolute path
    if os.path.isabs(args.embeddings_dir):
        if not args.embeddings_dir.startswith(constants.EMBEDDINGS_DIR):
            print(f"ERROR: --embeddings-dir must be inside {constants.EMBEDDINGS_DIR}")
            sys.exit(1)
        embeddings_name = os.path.relpath(args.embeddings_dir, constants.EMBEDDINGS_DIR)
        model_dir = args.embeddings_dir
    else:
        # Resolve using utils
        resolved_dirs = utils.resolve_embedding_dirs(args.embeddings_dir)
        if not resolved_dirs:
            print(f"ERROR: Could not resolve embeddings dir '{args.embeddings_dir}'")
            sys.exit(1)
        embeddings_name = resolved_dirs[0]
        model_dir = os.path.join(constants.EMBEDDINGS_DIR, embeddings_name)

    if not os.path.exists(model_dir):
        print(f"ERROR: Base embeddings directory {model_dir} not found.")
        sys.exit(1)
        
    if args.csv_dir:
        constants.CSV_DIR = args.csv_dir

    # 2. Call the optimizer
    print(f"\n--- Optimizing Late Fusion Weights for {embeddings_name} ---")
    
    # Set up optimizer parameters
    metric = "hits_at_10"
    n_trials = 1000
    tune_image = False
    n_folds = 5
    representation_b = False
    representation_c = False
    seed = 42
    low = 0.001
    high = 1.0
    fit_stats_split = "train"
    device = utils.get_torch_device()

    # Load dataset and embeddings
    stored_dataset = torch.load(
        os.path.join(model_dir, "stored_dataset.pt"),
        map_location="cpu", weights_only=False,
    )
    image_embeddings = torch.load(
        os.path.join(model_dir, "stored_embeddings.pt"),
        map_location="cpu", weights_only=False,
    )

    import src.multimodal.metadata_features as metadata_features
    spec = metadata_features._resolve_metadata_spec(
        None,
        representation_b=representation_b,
        representation_c=representation_c,
    )

    fusion_ds = fusion_dataset.FusionDataset(
        stored_dataset, image_embeddings, spec, split=None,
        fit_split=fit_stats_split,
    )
    slices = metadata_features._group_slices(spec, fusion_ds.norm_stats)
    group_order = tuple(slices.keys())

    full_ids = list(fusion_ds.get_ids())
    id_to_index = {mid: i for i, mid in enumerate(full_ids)}
    labels = fusion_ds.get_labels().to(device)

    print(f"Building {n_folds}-fold cross-validation data...")
    seed_data = optimizer._make_kfold_seed_data(
        fusion_ds, k=n_folds, seed=seed, slices=slices, device=device,
        id_to_index=id_to_index, labels=labels, min_instances=2,
    )

    # Run search
    (best_v, best_w, best_sc), rs_history = optimizer._random_search(
        seed_data, metric, n_trials, low, high, seed,
        group_order=group_order, tune_image=tune_image,
        tied_metadata=False,
    )
    
    # Run coordinate refine
    (best_v, best_w, best_sc), cr_history = optimizer._coordinate_refine(
        best_w, seed_data, metric,
        group_order=group_order, tune_image=tune_image,
        tied_metadata=False,
    )
    
    alphas = optimizer._weights_to_alphas(best_w)
    print(f"Optimized alphas: {alphas}")
    
    # Save the config
    config_path = os.path.join(model_dir, "late_fusion_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "config_name": "multimodal_tuned",
            "alphas": alphas,
            "weights": best_w,
            "score": best_sc,
            "metadata_spec": spec.to_dict()
        }, f, indent=2)
    print(f"Saved optimized config to {config_path}")

    # 3. Produce fused embeddings
    print(f"\n--- Generating Multimodal Embeddings ---")
    img = fusion_ds.image_embeddings.float()
    meta = fusion_ds.metadata.float()
    img_l2 = F.normalize(img, p=2, dim=1, eps=1e-8)
    group_l2 = {
        g: F.normalize(meta[:, a:b], p=2, dim=1, eps=1e-8)
        for g, (a, b) in slices.items()
    }
    
    streams = []
    a_img = float(alphas.get("alpha_image", 0.0))
    if a_img > 0.0:
        streams.append(a_img * img_l2)
        
    for group in group_order:
        if group == "image" or group not in group_l2:
            continue
        a = float(alphas.get(f"alpha_{group}", 0.0))
        if a <= 0.0:
            continue
        streams.append(a * group_l2[group])
        
    fused = torch.cat(streams, dim=1)
    fused = F.normalize(fused, p=2, dim=1, eps=1e-8)
    
    multimodal_emb_path = os.path.join(model_dir, "stored_embeddings_multimodal.pt")
    torch.save(fused, multimodal_emb_path)
    print(f"Saved fused embeddings to {multimodal_emb_path} (shape {tuple(fused.shape)})")
    
    # 4. Generate neighbors
    print(f"\n--- Generating Multimodal Neighbors ---")
    neighbors.store_neighbors(model_dir, centroid=False, transformed=False, suffix="_multimodal")
    
    # 5. Generate Centroid Embeddings and Neighbors
    print(f"\n--- Generating Multimodal Centroids ---")
    import src.retrieval.embeddings as ret_embeddings
    
    ret_embeddings.store_centroid_embeddings(model_dir, suffix="_multimodal")
    
    print("Saving centroid neighbors.")
    neighbors.store_neighbors(
        model_dir,
        centroid=True,
        transformed=False,
        suffix="_multimodal",
    )
    
    print("\nMultimodal pipeline completed successfully.")
    
if __name__ == "__main__":
    main()
