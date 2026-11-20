"""Generates and persists fused embeddings from a trained fusion model and live CSV metadata."""

import os
import argparse
import json
import torch
import torch.utils.data
import tqdm

from src.core import constants
import src.multimodal.fusion_dataset
import src.multimodal.metadata_features
import experimental.multimodal.trained_fusion.fusion_model
import src.retrieval.neighbors
from src.core import utils


def _load_config_and_defaults(fusion_model_name, image_embeddings_name):
    utils.print_function_banner("_load_config_and_defaults",
                                f"Loading trained fusion model configuration: "
                                f"{fusion_model_name}.")
    fusion_model_dir = os.path.join(constants.TRAINED_MODELS_DIR, fusion_model_name)
    fusion_model_path = os.path.join(fusion_model_dir, "best-fusion_model.pt")
    config_path = os.path.join(fusion_model_dir, "config.json")

    saved_config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        print(f"Loaded model config from {config_path}")
    else:
        print(f"WARNING: Config file not found at {config_path}")

    if image_embeddings_name is None:
        image_embeddings_name = saved_config.get("data_sources", {}).get("image_embeddings_name")
        if image_embeddings_name is None:
            raise ValueError(
                "image_embeddings_name not provided and not found in saved config.")
        print(f"Using image_embeddings_name from saved config: {image_embeddings_name}")

    return saved_config, fusion_model_path, image_embeddings_name


def generate_fused_embeddings(fusion_model_name: str,
                              image_embeddings_name: str = None,
                              fit_stats_split: str = "train"):
    utils.print_function_banner(
        "generate_fused_embeddings",
        f"Generating fused embeddings for: {fusion_model_name} "
        f"(fit_stats_split={fit_stats_split}).")
    device = utils.get_torch_device()
    print(f"Using device: {device}")

    try:
        saved_config, fusion_model_path, image_embeddings_name = \
            _load_config_and_defaults(fusion_model_name, image_embeddings_name)
    except (ValueError, FileNotFoundError) as e:
        print(e)
        return

    embeddings_dir = os.path.join(constants.EMBEDDINGS_DIR, image_embeddings_name)
    stored_dataset = torch.load(
        os.path.join(embeddings_dir, "stored_dataset.pt"),
        map_location='cpu', weights_only=False)
    image_embeddings = torch.load(
        os.path.join(embeddings_dir, "stored_embeddings.pt"),
        map_location='cpu', weights_only=False)

    spec = src.multimodal.metadata_features.MetadataFeatureSpec.from_dict(
        saved_config.get("metadata_features"))
    dataset = src.multimodal.fusion_dataset.FusionDataset(
        stored_dataset, image_embeddings, spec, split=None,
        fit_split=fit_stats_split,
    )

    if len(dataset) == 0:
        print("ERROR: No samples loaded; cannot generate fused embeddings.")
        return

    image_dim = dataset.image_dim()
    metadata_dim = dataset.metadata_dim()
    print(f"Image dim: {image_dim}  Metadata dim: {metadata_dim}  N: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model_config = saved_config.get("model", {})
    model = experimental.multimodal.trained_fusion.fusion_model.MultimodalFusionModel(
        image_dim=image_dim,
        metadata_dim=metadata_dim,
        shared_dim=model_config.get("shared_dim", 512),
        final_dim=model_config.get("final_dim", 256),
        fusion_method=model_config.get("fusion_method", "concatenate"),
        expert_dropout=model_config.get("expert_dropout", 0.0),
        projection_dropout=model_config.get("projection_dropout", 0.0),
        expert_depth=model_config.get("expert_depth", 1),
        residual_hidden_dim=model_config.get("residual_hidden_dim", 256),
    ).to(device)
    model.load_state_dict(torch.load(fusion_model_path, map_location=device))
    model.eval()
    print("Fusion model loaded successfully.")

    all_fused_embeddings = []
    print("Generating fused embeddings...")
    with torch.no_grad():
        for image_embeds, metadata_embeds, _ in tqdm.tqdm(
                dataloader, desc="Generating Fused Embeddings"):
            image_embeds = image_embeds.to(device)
            metadata_embeds = metadata_embeds.to(device)
            fused_embeddings = model(image_embeds, metadata_embeds)
            all_fused_embeddings.append(fused_embeddings.cpu())

    final_embeddings_tensor = torch.cat(all_fused_embeddings, dim=0)
    output_dir = os.path.join(constants.EMBEDDINGS_DIR, fusion_model_name)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(dataset, os.path.join(output_dir, "stored_dataset.pt"))
    torch.save(final_embeddings_tensor, os.path.join(output_dir, "stored_embeddings.pt"))

    print("Pre-computing nearest neighbors...")
    src.retrieval.neighbors.store_neighbors(
        model_dir=output_dir, centroid=False, transformed=False)

    print(f"\nDone. To evaluate:\n  python3 -m src.retrieval.evaluation "
          f"--embeddings-name {fusion_model_name} "
          f"--retrieval-method 0 --split-strategy train-test --mrr")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fused embeddings from a trained fusion model.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fusion-model-name', type=str, required=True,
                        help='Name of the trained fusion model directory '
                             'under sm-data/trained-models/.')
    parser.add_argument('--image-embeddings-name', type=str, default=None,
                        help='Override for the image embeddings directory. '
                             'Default: read from saved config.')
    parser.add_argument('--fit-stats-split', choices=('train', 'all'),
                        default='train',
                        help='Subset used to fit metadata normalization '
                             'statistics. Default "train" matches the '
                             'training-time stats and prevents leakage.')
    args = parser.parse_args()
    utils.print_script_banner(__file__, args)

    generate_fused_embeddings(args.fusion_model_name, args.image_embeddings_name,
                              fit_stats_split=args.fit_stats_split)


if __name__ == "__main__":
    main()
