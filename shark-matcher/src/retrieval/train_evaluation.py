"""Helper script to analyze and display metrics from training logs."""

import argparse
import csv
import os

from src.core import constants
from src.core import utils


def main():
    """Display training loss and Hits@k metrics from trained model history logs."""
    parser = argparse.ArgumentParser(
        description="Display metrics from training logs.")
    parser.add_argument('--model-name', type=str, required=True, nargs='+',
                        help='The name(s) of the model directory to search for training history.')
    parser.add_argument('--multiple', type=int, default=None,
                        help='If provided, present result tables for every epoch that is a multiple of this value.')
    parser.add_argument('--valid-only', action='store_true',
                        help='If provided, only show validation results for all K values. '
                             'Setting this is equivalent to --split-strategy train-test.')
    parser.add_argument('--split-strategy', choices=['train-test', 'none'],
                        default='train-test',
                        help='Strategy for handling dataset splits. '
                             '"train-test" (default) evaluates only on the '
                             'test set. "none" evaluates across the entire '
                             'dataset.')
    args = parser.parse_args()

    # Resolve valid_only where train-test corresponds to valid_only=True.
    valid_only = args.valid_only or (args.split_strategy == 'train-test')

    for model_name in args.model_name:
        # Resolve model name to subdirectories using the shared utility.
        resolved_dirs = utils.resolve_model_dirs(model_name)

        for resolved_name in resolved_dirs:
            model_dir = os.path.join(constants.TRAINED_MODELS_DIR, resolved_name)
            csv_path = os.path.join(model_dir, "training_history.csv")

            if not os.path.exists(csv_path):
                print(f"\nError: Could not find training_history.csv "
                      f"for model {resolved_name} in {model_dir}.")
                continue

            evaluate_history(resolved_name, csv_path, args, valid_only)


def evaluate_history(display_name, csv_path, args, valid_only):
    """Parse and display training history metrics for a single CSV file."""
    history = parse_training_history(csv_path)
    if not history:
        return

    if args.multiple:
        for data in history:
            if (data["epoch"] + 1) % args.multiple == 0:
                print_epoch_table(display_name, data, valid_only)
    else:
        # Default: Find the row with the highest validation Hits@1.
        best_epoch_data = max(history, key=lambda x: x["valid_hits"][1])
        print_epoch_table(display_name, best_epoch_data, valid_only)


def parse_training_history(csv_path):
    """Parse the training history CSV file and return a list of row dictionaries."""
    if not os.path.exists(csv_path):
        print(f"Error: Could not find training history at {csv_path}.")
        return []

    history = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].isdigit():
                continue
            
            # Map column indices to epoch, loss, and Hits@k metrics.
            try:
                data = {
                    "epoch": int(row[0]),
                    "loss": float(row[1]),
                    "train_hits": {
                        1: float(row[2]),
                        5: float(row[3]),
                        10: float(row[4]),
                        25: float(row[5]),
                        50: float(row[6]),
                        100: float(row[7]),
                    },
                    "valid_hits": {
                        1: float(row[8]),
                        5: float(row[9]),
                        10: float(row[10]),
                        25: float(row[11]),
                        50: float(row[12]),
                        100: float(row[13]),
                    }
                }
                history.append(data)
            except (IndexError, ValueError):
                continue
    return history


def print_epoch_table(model_name, epoch_data, valid_only=False):
    """Print a formatted table for a single epoch's evaluation results."""
    table_headers = ["Metric", "Value"]
    table_rows = []

    if valid_only:
        # Standard format like evaluation.py.
        sorted_ks = [1, 5, 10, 25, 50, 100]
        for k in sorted_ks:
            val = epoch_data["valid_hits"].get(k, 0.0)
            table_rows.append([f"Hits@{k}", f"{val:.3f}"])
    else:
        # Custom format for training history.
        table_rows.append(["Loss", f"{epoch_data['loss']:.4f}"])
        
        ks = [1, 5, 25]
        for k in ks:
            train_val = epoch_data["train_hits"].get(k, 0.0)
            valid_val = epoch_data["valid_hits"].get(k, 0.0)
            table_rows.append([f"Train Hits@{k}", f"{train_val:.3f}"])
            table_rows.append([f"Valid Hits@{k}", f"{valid_val:.3f}"])

    print(f"\nResults for {model_name} (Epoch {epoch_data['epoch']}):")
    utils.print_table(table_headers, table_rows)


if __name__ == "__main__":
    main()
