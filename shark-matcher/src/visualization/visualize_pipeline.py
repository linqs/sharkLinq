# pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
"""Visualize the training data pipeline: augmentations and batch sampling."""

import argparse
import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from src.core import constants
import src.config.config
import src.individual_dataset
import src.learning.augmented_dataset
from src.core import utils


def main():
    """Visualize augmentations and batch sampling for a trained model's config."""
    parser = argparse.ArgumentParser(
        description="Visualize data augmentations and batch sampling "
                    "for a trained model's configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m src.visualization.visualize_pipeline "
               "--model-name test-model/vit_triplet_1\n"
               "  python3 -m src.visualization.visualize_pipeline "
               "--model-name test-model/vit_triplet_1 "
               "--seed 42 --output-dir test-run\n")
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of the model directory (relative to '
                             'sm-data/trained-models/) containing a '
                             'config.json.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output plots. Default: '
                             'timestamped directory under '
                             'sm-data/data-out/visualizations/.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible visualizations '
                             '(default: 42).')
    args = parser.parse_args()

    utils.print_script_banner(__file__, args)

    # Resolve output directory.
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.output_dir = os.path.join(constants.VISUALIZATIONS_DIR, timestamp)
        print(f"Output directory not specified. Using: {args.output_dir}")
    else:
        args.output_dir = utils.resolve_directory_path(
            os.path.join(constants.VISUALIZATIONS_DIR, args.output_dir) if args.output_dir else constants.VISUALIZATIONS_DIR, 
            constants.VISUALIZATIONS_DIR
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load and validate model config. ---
    config_path = os.path.join(
        constants.TRAINED_MODELS_DIR, args.model_name, "config.json"
    )
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}.")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # --- Create dataset based on config. ---
    print("Creating dataset from config...")

    if not os.path.exists(constants.IMAGES_DIR):
        print(f"Error: Images directory not found at {constants.IMAGES_DIR}.")
        return

    train_annotations, valid_annotations = utils.get_annotation_paths(None)

    data_config = config.get("data", {})
    image_size = data_config.get("image_size", [224, 224])
    image_size = tuple(image_size) if image_size else None
    normalization = data_config.get("normalization")

    # Use the raw individual dataset (no transforms applied).
    # The visualization functions apply transforms manually to show effects.
    dataset = src.individual_dataset.IndividualDataset(
        constants.IMAGES_DIR,
        train_annotations,
        train_annotations_path=train_annotations,
        test_annotations_path=valid_annotations,
        channels=3,
        output_image_size=image_size,
        normalization=normalization,
        device=torch.device("cpu")
    )

    print(f"Dataset created with {len(dataset)} images.")
    print("Generating visualizations...")

    # Extract augmentations from config or fall back to defaults.
    aug_list_config = config.get("augmentations")
    if not aug_list_config:
        aug_list_config = src.config.config.AUGMENTATIONS

    visualize_augmentations(
        dataset, aug_list_config,
        seed=args.seed,
        output_path=os.path.join(args.output_dir, "augmentations.png")
    )
    visualize_isolated_augmentations(
        dataset, aug_list_config,
        output_path=os.path.join(
            args.output_dir, "augmentations_isolated.png"
        )
    )

    # Build the contrastive wrapper for batch visualization sampling.
    augmentations = [
        src.config.config.AUGMENTATION_CONSTRUCTORS[name](**params)
        for name, params in aug_list_config
    ]

    training_config = config.get("training", {})
    wrapper_dataset = src.learning.augmented_dataset.AugmentedDataset(
        dataset,
        augmentations,
        num_augmentations=training_config.get("num_augmentations", 2),
        rebalance_augmentations=training_config.get(
            "rebalance_augmentations", False
        )
    )

    visualize_batch(
        wrapper_dataset,
        sampler_source=dataset,
        seed=args.seed,
        output_path=os.path.join(args.output_dir, "batch_sample.png")
    )

    print("Done.")


def visualize_augmentations(dataset, aug_list_config, num_views=5, seed=None,
                            output_path="augmentations.png"):
    """Visualize augmented views of a single image alongside the original."""
    utils.print_function_banner("visualize_augmentations")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Get the first image from the dataset.
    img, _ = dataset[0]

    aug_list = []
    for name, params in aug_list_config:
        aug_list.append(
            src.config.config.AUGMENTATION_CONSTRUCTORS[name](**params)
        )

    transform = torchvision.transforms.Compose(aug_list)

    # Original plus augmented views in a row.
    images = [img.clone()]
    for _ in range(num_views):
        images.append(transform(img))

    grid = torchvision.utils.make_grid(
        images, nrow=num_views + 1, normalize=True
    )

    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Original + {num_views} Augmented Views")
    plt.axis('off')
    plt.savefig(output_path)
    print(f"Saved augmentation visualization to {output_path}")


def visualize_isolated_augmentations(dataset, aug_list_config,
                                     output_path="augmentations_isolated.png"):
    """Visualize each augmentation type acting alone on the same image."""
    utils.print_function_banner("visualize_isolated_augmentations")
    img, _ = dataset[0]

    images = [img.clone()]
    labels = ["Original"]

    for name, params in aug_list_config:
        constructor = src.config.config.AUGMENTATION_CONSTRUCTORS.get(name)
        if constructor:
            transform = constructor(**params)
            aug_img = transform(img)
            images.append(aug_img)
            labels.append(name)

    grid = torchvision.utils.make_grid(
        images, nrow=len(images), normalize=True
    )

    plt.figure(figsize=(20, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Isolated Augmentations: " + " -> ".join(labels))
    plt.axis('off')
    plt.savefig(output_path)
    print(f"Saved isolated augmentations to {output_path}")


def visualize_batch(dataset, sampler_source=None, batch_size=32,
                    num_positive=2, seed=None,
                    output_path="batch_sample.png"):
    """Visualize a batch from the sampler; pending migration to PKSampler."""
    raise NotImplementedError(
        "visualize_batch must be migrated to PKSampler."
    )


if __name__ == "__main__":
    main()
