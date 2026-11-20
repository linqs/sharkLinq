"""PAIR-X explainable AI visualization generator for shark fin image pairs."""

import argparse
import datetime
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.core import constants
from src.core import path_utils
from src.core import utils

# Ensure pairx is importable via the constants path discovery pattern.
path_utils.ensure_pairx_in_path()

# PAIR-X is an optional sibling checkout resolved by ensure_pairx_in_path().
from pairx.core import explain  # pylint: disable=wrong-import-position,import-error
from pairx.helpers import list_layer_keys  # pylint: disable=wrong-import-position,import-error

import src.individual_dataset  # pylint: disable=wrong-import-position


class SharkMatcherModel(nn.Module):
    """Combines a feature extraction backbone and projection head into a PAIR-X-compatible module."""

    def __init__(self, backbone, projection_head):
        """Initializes the wrapper model with backbone and projection head."""
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        self.device = next(self.backbone.parameters()).device

    def forward(self, x):
        """Pass input image tensor through backbone and projection head."""
        x = self.backbone(x)
        x = self.projection_head(x)
        return x


def load_shark_model(model_name, device):
    """Load backbone and projection head weights into a PAIR-X-compatible SharkMatcherModel."""
    utils.print_function_banner("load_shark_model",
                                f"Loading shark-matcher model: {model_name}.")
    projection_head, model_backbone = utils.load_model(model_name)

    # Combine them into the wrapper model.
    model = SharkMatcherModel(model_backbone, projection_head)
    model.to(device)
    model.eval()
    return model

def disable_inplace(model):
    """Disable inplace activation operations recursively to avoid autograd errors."""
    for m in model.modules():
        if hasattr(m, 'inplace'):
            m.inplace = False


def list_model_layers(model):
    """Print available model layer keys for visualization inspection."""
    print("\nAvailable model layers for visualization:")
    print("-" * 50)
    print("The 'depth' refers to module nesting, not computational flow.")
    print("For Vision Transformer (ViT) models, layers are sequential.")
    print("A lower index (e.g., 'backbone.encoder.layer.0') is an EARLY layer.")
    print("A higher index (e.g., 'backbone.encoder.layer.23') is a LATE layer.")
    print("\nTo understand what the model uses for its final embedding, it is best to")
    print("visualize the later layers (e.g., layers 16 through 23 for a 24-layer model).")
    print("-" * 50)
    # max_depth=7 is sufficient to list the transformer blocks by name.
    list_layer_keys(model, max_depth=7)
    print("-" * 50)


def get_image_data(dataset, filename, device):
    """Fetch an image tensor and display-ready numpy array from the dataset."""
    print(f"Loading {filename}...")

    if filename not in dataset.filenames:
        raise ValueError(f"File {filename} not found in dataset.")

    idx = dataset.filenames.index(filename)

    # Get transformed tensor from dataset (C, H, W).
    img_tensor, _ = dataset[idx]

    # Convert to (1, C, H, W) batch for model.
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Create display image (H, W, 3) uint8 from the tensor.
    # Since dataset scales to [0, 1], scale back to [0, 255].
    img_np = (img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255
              ).clip(0, 255).astype(np.uint8)

    return img_tensor, img_np


def resolve_model_and_dataset_config(args, device):
    """Load model weights and resolve dataset configuration parameters."""
    output_image_size = None
    normalization = None

    if args.model_name:
        model = load_shark_model(args.model_name, device)
        disable_inplace(model)
        # Load dataset config properties dynamically if we are not forcing the original image format
        if not args.original_image:
            config_path = os.path.join(constants.TRAINED_MODELS_DIR, args.model_name, "config.json")
            if os.path.exists(config_path):
                try:
                    config = utils.load_json_file(config_path)
                    data_config = config.get("data", {})
                    if "image_size" in data_config:
                        output_image_size = tuple(data_config["image_size"])
                    if "normalization" in data_config:
                        normalization = data_config["normalization"]
                except Exception as e:
                    print(f"Error loading config for dataset parameters: {e}")
    else:
        print("No model name specified. Loading default MiewID model "
              "from pairx.example_loaders.")
        # pylint: disable-next=import-outside-toplevel,import-error
        from pairx.example_loaders import wildme_multispecies_miewid
        model, _, _ = wildme_multispecies_miewid(device)
        model.to(device)
        model.eval()
        disable_inplace(model)
        
    return model, output_image_size, normalization


def generate_visualization(model, img_0, img_1, img_np_0, img_np_1,
                           layer_keys_to_viz, output_directory, device,
                           metric_fn=None, k_lines=20, k_colors=10):
    """Run PAIR-X explanation and save visualized activation maps to disk."""
    utils.print_function_banner("generate_visualization",
                                f"Generating PAIR-X visualization for layers: "
                                f"{layer_keys_to_viz}.")

    # Run the PAIR-X explain function to obtain one image per layer key.
    img_0.requires_grad_()
    img_1.requires_grad_()
    pairx_visualizations = explain(
        img_0,
        img_1,
        img_np_0,
        img_np_1,
        model,
        layer_keys_to_viz,
        k_lines=k_lines,
        k_colors=k_colors,
        metric_fn=metric_fn
    )

    # Create a timestamped subdirectory for the output.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    final_output_dir = os.path.join(output_directory, timestamp)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Saving output to: {final_output_dir}")

    for viz_img, layer_key in zip(pairx_visualizations, layer_keys_to_viz):
        img_to_save = Image.fromarray(viz_img)
        save_path = os.path.join(
            final_output_dir,
            f"explanation_{layer_key.replace('.', '_')}.png"
        )
        img_to_save.save(save_path)
        print(f"Saved visualization for layer '{layer_key}' to {save_path}")


def main():
    """Parse arguments, load model and images, and generate PAIR-X visualizations."""
    parser = argparse.ArgumentParser(
        description="Generate PAIR-X visualizations for the shark-matcher model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m experimental.pairx_visualization.pairx_vis "
               "--model-name experiment_best_split1_trans_color --list-layers\n"
               "  python3 -m experimental.pairx_visualization.pairx_vis "
               "--model-name experiment_best_split1_trans_color "
               "--visualize backbone.base_model.model.backbone.encoder.layer.21 "
               "--image-filenames AN15093004.jpeg AN15100101.jpeg "
               "--output-directory pairx_test_1 --seed 42\n")
    parser.add_argument('--model-name', type=str, default=None,
                        help='Name of the trained model to load. If not '
                             'provided, loads the default MiewID model.')
    parser.add_argument('--list-layers', action='store_true',
                        help='List all available layer keys in the model and exit.')
    parser.add_argument('--visualize', nargs='+', type=str,
                        help='Space-separated list of layer keys to visualize '
                             '(e.g., backbone.blocks.5).')
    parser.add_argument('--output-directory', type=str, default='pairx_output',
                        help='Subdirectory name under sm-data/data-out/pairx-output/ for '
                             'saving results. Default: "pairx_output".')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--image-filenames', nargs=2, type=str,
                        default=['AN15093004.jpeg', 'AN15100101.jpeg'],
                        help='Two filenames from the dataset to visualize.')
    parser.add_argument('--function', type=str, default=None,
                        help='Metric function name. Defaults to cosine '
                             'similarity if not specified.')
    parser.add_argument('--k-lines', type=int, default=20,
                        help='Number of highest activation lines to draw '
                             '(default: 20).')
    parser.add_argument('--k-colors', type=int, default=10,
                        help='Number of matched pixel nodes localized '
                             '(default: 10).')
    parser.add_argument('--original-image', action='store_true',
                        help='If provided, loads the image without resolving '
                             'specific scaling and squashing rules defined in '
                             'the model config (reverting to padding logic).')

    args = parser.parse_args()

    utils.print_script_banner(__file__, args)

    # Set random seeds for reproducibility.
    utils.seed_everything(args.seed)

    # Resolve device.
    device = utils.get_torch_device()

    # Resolve model and dataset configuration parameters.
    model, output_image_size, normalization = resolve_model_and_dataset_config(args, device)

    # Handle --list-layers mode.
    if args.list_layers:
        list_model_layers(model)
        sys.exit(0)

    # Load images from the dataset using the dynamically resolved parameters
    full_ann_path = os.path.join(constants.ANNOTATIONS_DIR, "annotations", constants.FULL_ANNOTATIONS)
    dataset = src.individual_dataset.IndividualDataset(
        constants.IMAGES_DIR, full_ann_path, device=device,
        output_image_size=output_image_size,
        normalization=normalization
    )

    img_filename_0, img_filename_1 = args.image_filenames
    img_0, img_np_0 = get_image_data(dataset, img_filename_0, device)
    img_1, img_np_1 = get_image_data(dataset, img_filename_1, device)

    print(f"\nImages loaded and transformed for model input.")
    print(f"Image 1 tensor shape: {img_0.shape}")
    print(f"Image 2 tensor shape: {img_1.shape}")

    # Resolve metric function.
    metric_fn = None
    if args.function and args.function != 'miewID':
        print(f"Warning: Custom function '{args.function}' requested but not "
              f"implemented. Using default cosine similarity.")
    else:
        print("Using default function 'miewID' (Cosine Similarity).")

    # Generate visualizations.
    if args.visualize:
        output_directory = os.path.join(
            constants.PAIRX_OUTPUT_DIR, args.output_directory
        )
        generate_visualization(
            model, img_0, img_1, img_np_0, img_np_1,
            args.visualize, output_directory, device, 
            metric_fn=metric_fn, k_lines=args.k_lines, k_colors=args.k_colors
        )


if __name__ == "__main__":
    main()
