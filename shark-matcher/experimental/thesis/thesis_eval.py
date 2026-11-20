"""Run specific evaluations for the thesis."""

import argparse
import os
import sys
import io
from datetime import datetime
from contextlib import redirect_stdout

from src.core import constants
from src.core import path_utils

path_utils.ensure_shark_model_in_path()
from src.core import utils
import src.retrieval.evaluation as evaluation

def run_eval(model_name, use_psl, use_multimodal, retrieval_method, min_quality=None,
             min_instances=2, rerank='none', use_stacked=False, csv_dir=None):
    """Run one thesis evaluation configuration and return its captured printed report."""
    # Mock argparse.Namespace expected by run_single_evaluation
    args = argparse.Namespace(
        embeddings_name=[model_name],
        retrieval_method=retrieval_method,
        split_strategy='train-test',
        verbose=False,
        min_instances=max(min_instances, 2),  # required by evaluation.py logic
        min_quality=min_quality,
        random_guesses=False,
        map=True,
        mrr=True,
        cmc=True,
        fine_eval=False,
        num_results=30,
        use_psl=use_psl,
        use_multimodal=use_multimodal,
        stacked=use_stacked,
        rerank=rerank,
        csv_dir=csv_dir,
    )

    f = io.StringIO()
    with redirect_stdout(f):
        evaluation.run_single_evaluation(model_name, args, train_test=True)
    return f.getvalue()


def _build_variants(args):
    """Compose the ordered list of evaluation variants to emit."""
    base_variants = [("baseline", False, False, False)]
    
    add_psl = args.use_psl or args.stacked
    add_multimodal = args.use_multimodal or args.stacked
    
    if add_multimodal:
        base_variants.append(("multimodal", False, True, False))
    if add_psl:
        base_variants.append(("psl", True, False, False))
    if args.stacked:
        base_variants.append(("stacked", False, False, True))
        
    variants = []
    for label, use_psl, use_mm, use_stacked in base_variants:
        variants.append((label, use_psl, use_mm, 'none', use_stacked))
        if args.rerank != 'none':
            variants.append((f"{label}+{args.rerank}", use_psl, use_mm, args.rerank, use_stacked))
            
    return variants


def main():
    """Run thesis evaluations for standard, poor quality, and min-instances=5 scenarios."""
    parser = argparse.ArgumentParser(description="Run thesis evaluation suite.")
    parser.add_argument('--embeddings-name', type=str, nargs='+',
                        default=['prod_model'],
                        help='Name(s) of the embeddings directory(ies) to evaluate (directories '
                             'under sm-data/embeddings/). Multiple names can '
                             'be specified for comparison. Default: "prod_model".')
    parser.add_argument('--use-psl', action='store_true',
                        help='Additive overlay: also evaluate with PSL-reranked '
                             'neighbours (stored_image_neighbors_psl.pt) alongside '
                             'the baseline.')
    parser.add_argument('--use-multimodal', action='store_true',
                        help='Additive overlay: also evaluate with multimodal '
                             'embeddings/neighbours alongside the baseline.')
    parser.add_argument('--stacked', action='store_true',
                        help='Additive overlay: also evaluate with the stacked '
                             'result (multimodal embeddings + PSL-reranked '
                             'multimodal neighbours) alongside the baseline.')
    parser.add_argument('--centroid', action='store_true',
                        help='Method selector: when set, the base retrieval '
                             'method is 2 (LOO centroid) instead of 0 '
                             '(image-image). Method 0 and method 2 are '
                             'mutually exclusive within a single output file.')
    parser.add_argument('--rerank', choices=['none', 'kreciprocal'], default='none',
                        help='Additive overlay: also evaluate with the named '
                             'post-retrieval re-ranking pass alongside the '
                             'baseline. "kreciprocal" applies Zhong et al. '
                             '(2017) k-reciprocal re-ranking. Hyperparameters '
                             'live in src/retrieval/rerank.py.')
    parser.add_argument('--csv-dir', type=str, default=None,
                        help='Explicit override for the source CSV directory '
                             '(e.g. path containing media.csv and individuals.csv).')

    args = parser.parse_args()

    utils.print_script_banner(__file__, args)

    # Ensure output directory exists
    os.makedirs(constants.THESIS_RESULTS_DIR, exist_ok=True)

    # Resolve all embedding names to their subdirectories.
    all_resolved_names = []
    for embeddings_name in args.embeddings_name:
        resolved = utils.resolve_embedding_dirs(embeddings_name)
        for resolved_name in resolved:
            model_path = os.path.join(constants.EMBEDDINGS_DIR, resolved_name)
            if not os.path.exists(model_path):
                print(f"Warning: Embeddings directory '{resolved_name}' not found at "
                      f"{model_path}. Skipping.")
                continue
            all_resolved_names.append(resolved_name)

    if not all_resolved_names:
        print("Error: No valid embedding directories found.")
        sys.exit(1)

    retrieval_method = 2 if args.centroid else 0
    method_label = f"Method {retrieval_method}"
    variants = _build_variants(args)

    # Scenarios drive the outer loop so baseline and overlays sit adjacent
    # within each (scenario, model) cell of the output.
    scenarios = [
        ("Standard evaluation", {}),
        ("Evaluation with min-quality 'poor'", {"min_quality": "poor"}),
        ("Evaluation with min-instances 5", {"min_instances": 5}),
    ]

    for resolved_name in all_resolved_names:
        print(f"Running thesis evaluations for {resolved_name}...")
        output_parts = []

        for variant_label, use_psl, use_multimodal, rerank, use_stacked in variants:
            for scenario_label, scenario_kwargs in scenarios:
                print(f"  -> [{variant_label}] {scenario_label} ({method_label})")
                header = (f"=== [{variant_label}] {scenario_label} "
                          f"({method_label}) for {resolved_name} ===")
                output_parts.append(header)
                
                # Check for incompatible overlays
                if rerank == 'kreciprocal' and (use_psl or use_stacked):
                    msg = ("Skipping evaluation: k-reciprocal re-ranking is currently "
                           "not compatible with PSL pseudo-distances (it ignores them and "
                           "rebuilds the distance matrix from base embeddings).")
                    print(f"     {msg}")
                    output_parts.append(msg + "\n")
                else:
                    out = run_eval(resolved_name, use_psl, use_multimodal,
                                   retrieval_method=retrieval_method,
                                   rerank=rerank, use_stacked=use_stacked,
                                   csv_dir=args.csv_dir,
                                   **scenario_kwargs)
                    output_parts.append(out)

        full_output = "\n\n".join(output_parts)

        # Filename suffix accumulates every active flag so multiple-overlay
        # runs land in distinct files.
        safe_name = resolved_name.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_parts = []
        if args.centroid:
            suffix_parts.append("centroid")
        if args.use_psl:
            suffix_parts.append("psl")
        if args.use_multimodal:
            suffix_parts.append("multimodal")
        if args.stacked:
            suffix_parts.append("stacked")
        if args.rerank == 'kreciprocal':
            suffix_parts.append("kreciprocal")
        suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
        out_filename = f"{safe_name}{suffix}_{timestamp}.txt"
        out_path = os.path.join(constants.THESIS_RESULTS_DIR, out_filename)

        with open(out_path, "w") as f:
            f.write(full_output)

        print(f"Saved results for {resolved_name} to {out_path}\n")

if __name__ == '__main__':
    main()
