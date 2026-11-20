# pylint: disable=import-error, wrong-import-position, no-name-in-module
"""Orchestrator script to execute the 6 core steps of the Shark Matcher pipeline."""

import argparse
import os
import subprocess
import sys
import time

# Ensure constants can be imported.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import constants
from src.core import path_utils

path_utils.ensure_shark_model_in_path()

from src.core import utils

def run_step(command, step_name):
    """Run a subprocess command and check for errors."""
    print(f"\n{'='*60}")
    print(f"=== Pipeline Step: {step_name} ===")
    print(f"{'='*60}")
    print(f"Running command: {' '.join(command)}")

    result = subprocess.run(command, check=False, cwd=constants.THIS_DIR)

    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline failed at step: {step_name}")
        print(f"Command returned non-zero exit status: {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[SUCCESS] Step completed: {step_name}")

def main():
    """Orchestrate the end-to-end pipeline from raw data to model evaluation."""
    parser = argparse.ArgumentParser(
        description="Orchestrator script to automatically run the 6-step "
                    "Shark Matcher pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m src.cli.run_pipeline \\\n"
               "      --input-images my-photos \\\n"
               "      --input-csv pre2019-sample-data \\\n"
               "      --model-name my-test-model \\\n"
               "      --configs vit_triplet_1 miewid_arcface\n"
    )
    parser.add_argument('--input-images', type=str, required=True,
                        help='Absolute or relative path to the directory '
                             'containing raw images.')
    parser.add_argument('--input-csv', type=str, required=True,
                        help='Absolute or relative path to the directory '
                             'containing database CSV files.')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Base name for the model and annotations version '
                             '(e.g., "my-model" will create "my-model-annotations").')
    parser.add_argument("--configs", nargs="+", help="One or more model configurations to train (defined in src/config/config.py).")
    parser.add_argument('--no-test', action='store_true',
                        help='If specified, do not split a test set during '
                             'annotation generation.')
    parser.add_argument('--skip-train', action='store_true',
                        help='If specified, skip the model training step (Step 4) '
                             'and use an existing trained model.')
    parser.add_argument('--split-strategy', choices=['train-test', 'none'],
                        default='train-test',
                        help='Strategy for handling dataset splits during '
                             'embedding generation and evaluation. '
                             '"train-test" (default) evaluates on the test set. '
                             '"none" evaluates on the full dataset.')
    parser.add_argument('--stacked', action='store_true',
                        help='Generate the stacked embedding artifacts '
                             '(multimodal fusion plus PSL reranking) in Step 5. '
                             'Requires pslpython and a Java runtime; see '
                             'src/psl/README.md. Default: plain embeddings.')
    parser.add_argument('--no-email', action='store_true',
                        help='Do not send completion notifications for this '
                             'run, and suppress them in the training step too. '
                             'Recipients come from the comma-separated '
                             'SHARK_NOTIFICATION_EMAILS environment variable.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Let the training step replace trained weights that '
                             'already exist for this model name.')
    args = parser.parse_args()

    if not args.skip_train and not args.configs:
        print("\n[ERROR] --configs must be specified when training is enabled.")
        sys.exit(1)

    utils.print_script_banner(__file__, args)

    pipeline_start_time = time.time()

    python_exec = sys.executable
    backend_dir = constants.BACKEND_DIR
    # Resolve dataset paths using the utilities natively imported via utils.
    input_images_dir = utils.resolve_directory_path(
        os.path.join(constants.RAW_DATA_DIR, args.input_images) if args.input_images else constants.RAW_DATA_DIR, 
        constants.BASE_IMAGES_DIR
    )
    input_csv_dir = utils.resolve_directory_path(
        os.path.join(constants.DB_DATA_DIR, args.input_csv) if args.input_csv else constants.DB_DATA_DIR, 
        constants.DB_DATA_DIR
    )

    model_name = args.model_name
    annotations_version = f"{model_name}-annotations"
    
    # If --no-test is specified, default --split-strategy to 'none' if user didn't override it.
    split_strategy = args.split_strategy
    if args.no_test and args.split_strategy == 'train-test':
        print("Note: --no-test specified, defaulting --split-strategy to 'none'.")
        split_strategy = 'none'

    # 1. Format Images
    cmd_format = [
        python_exec, "-m", "src.cli.convert_images",
        "--input-directory", input_images_dir
        # Note: --output-directory defaults to constants.IMAGES_DIR (sm-data/images/)
    ]
    run_step(cmd_format, "1. Format Images")

    # 2. Prepare Annotations
    cmd_prepare = [
        python_exec, "-m", "src.cli.convert_data",
        "--input-directory", input_csv_dir,
        "--output-directory", annotations_version
    ]
    if args.no_test:
        cmd_prepare.append("--no-test")
    run_step(cmd_prepare, "2. Prepare Annotations")

    # 3. Validate Data
    cmd_validate = [
        python_exec, os.path.join(backend_dir, "data_scripts", "analyze_data.py"),
        "--verbose",
        "--annotations-version", annotations_version
    ]
    run_step(cmd_validate, "3. Validate Data")

    # 4. Train Model
    if not args.skip_train:
        cmd_train = [
            python_exec, "-m", "src.learning.trainer.new_trainer",
            "--model-name", model_name,
            "--annotations-version", annotations_version,
        ]
        if args.no_email:
            # Pass --no-email to silence training-step email notifications.
            cmd_train.append("--no-email")
        if args.overwrite:
            cmd_train.append("--overwrite")
        cmd_train += ["--configs"] + args.configs
        run_step(cmd_train, "4. Train Model")

        # When training new models, we explicitly iterate over the requested configs.
        model_paths = [f"{model_name}/{cfg}" for cfg in args.configs]
    else:
        print("\n[SKIP] Skipping Step 4: Train Model as requested by --skip-train.")

        # If configs are provided even with --skip-train, we use them to build paths.
        if args.configs:
            model_paths = [f"{model_name}/{cfg}" for cfg in args.configs]
        else:
            # Resolve existing base or fine-tuned model directories when training is skipped without explicit configs.
            model_paths = utils.resolve_model_dirs(model_name)

    # 5. Generate Embeddings across resolved models.
    for model_path in model_paths:
        cmd_embed = [
            python_exec, "-m", "src.retrieval.embeddings",
            "--model-name", model_path,
            "--annotations-version", annotations_version,
        ]
        if args.stacked:
            cmd_embed.append("--stacked")
        run_step(cmd_embed, f"5. Generate Embeddings ({model_path})")

    # 6. Evaluate Embeddings
    cmd_eval = [
        python_exec, "-m", "src.retrieval.evaluation",
        "--split-strategy", split_strategy,
        "--embeddings-name"
    ] + model_paths

    run_step(cmd_eval, f"6. Evaluate Embeddings ({', '.join(model_paths)})")

    print(f"\n{'='*60}")
    print(f"Pipeline successfully completed for model: {model_name}")
    print(f"Generated annotations: sm-data/annotations/{annotations_version}")
    print(f"Models saved to: sm-data/trained-models/{model_name}/")
    print(f"{'='*60}\n")

    # Send run-level completion summary to configured notification subscribers unless silenced by --no-email.
    utils.notify_completion(
        pipeline_start_time,
        os.path.join(constants.TRAINED_MODELS_DIR, model_name),
        no_email=args.no_email,
        subject_prefix="pipeline",
    )

if __name__ == "__main__":
    main()
