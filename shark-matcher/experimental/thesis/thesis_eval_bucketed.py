"""Run bucketed evaluations for the thesis across shark frequency and image quality."""

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

# Scenarios mapping labels to run_eval kwargs for query-side frequency and quality buckets.
_BUCKETED_SCENARIOS = [
    ("2 Images/Shark",          {"min_instances": 2, "max_instances": 2}),
    ("3-5 Images/Shark",        {"min_instances": 3, "max_instances": 5}),
    ("6+ Images/Shark",         {"min_instances": 6}),
    ("Unmatchable/Poor Quality", {"max_quality": "poor"}),
    ("Medium/Good Quality",     {"min_quality": "medium"}),
]

_BUCKETED_SCENARIOS_FOUR_QUALITY = [
    ("2 Images/Shark",          {"min_instances": 2, "max_instances": 2}),
    ("3-5 Images/Shark",        {"min_instances": 3, "max_instances": 5}),
    ("6+ Images/Shark",         {"min_instances": 6}),
    ("Unmatchable Quality",     {"min_quality": "unmatchable", "max_quality": "unmatchable"}),
    ("Poor Quality",            {"min_quality": "poor", "max_quality": "poor"}),
    ("Medium Quality",          {"min_quality": "medium", "max_quality": "medium"}),
    ("Good Quality",            {"min_quality": "good", "max_quality": "good"}),
]

_BUCKETED_SCENARIOS_MEDIA_TYPE = [
    ("Topshot Type",            {"media_type": "topshot"}),
    ("Underwater Type",         {"media_type": "underwater"}),
]

_BUCKETED_SCENARIOS_STACKED = (
    _BUCKETED_SCENARIOS_FOUR_QUALITY + _BUCKETED_SCENARIOS_MEDIA_TYPE
)


def run_eval(model_name, use_psl, use_multimodal, retrieval_method,
             min_instances=2, max_instances=None,
             min_quality=None, max_quality=None,
             rerank='none', use_stacked=False,
             csv_dir=None, media_type=None):
    """Run one bucketed thesis evaluation configuration and return its captured printed report."""
    args = argparse.Namespace(
        embeddings_name=[model_name],
        retrieval_method=retrieval_method,
        split_strategy='train-test',
        verbose=False,
        min_instances=max(min_instances, 2),  # required by evaluation.py logic
        max_instances=max_instances,
        min_quality=min_quality,
        max_quality=max_quality,
        media_type=media_type,
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
    """Run bucketed evaluations for the thesis across shark frequency and image quality."""
    parser = argparse.ArgumentParser(description="Run bucketed thesis evaluation suite.")
    parser.add_argument('--embeddings-name', type=str, nargs='+',
                        default=['prod_model'],
                        help='Name(s) of the embeddings directory(ies) to evaluate (directories '
                             'under sm-data/embeddings/). Multiple names can '
                             'be specified for comparison. Default: "prod_model".')
    parser.add_argument('--use-psl', action='store_true',
                        help='Additive overlay: also evaluate with PSL-reranked '
                             'neighbours alongside the baseline.')
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
                             '(image-image).')
    parser.add_argument('--rerank', choices=['none', 'kreciprocal'], default='none',
                        help='Additive overlay: also evaluate with the named '
                             'post-retrieval re-ranking pass alongside the baseline.')
    parser.add_argument('--csv-dir', type=str, default=None,
                        help='Explicit override for the source CSV directory '
                             '(e.g. path containing media.csv and individuals.csv).')
    parser.add_argument('--four-quality-buckets', action='store_true',
                        help='Evaluate 4 discrete quality buckets (unmatchable, '
                             'poor, medium, good) instead of 2 lumped buckets.')
    parser.add_argument('--media-type-only', action='store_true',
                        help='Evaluate only media type buckets (topshot, underwater).')
    parser.add_argument('--media-type-buckets', action='store_true',
                        help='Include media type buckets alongside frequency and quality buckets.')
    parser.add_argument('--stacked-bucketing', action='store_true',
                        help='Run stacked bucketing across 4 quality, 3 quantity, and 2 media type buckets.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom directory to write evaluation result text files.')

    args = parser.parse_args()

    utils.print_script_banner(__file__, args)

    target_dir = args.output_dir if args.output_dir else constants.THESIS_RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)

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
    if args.stacked_bucketing:
        scenarios = _BUCKETED_SCENARIOS_STACKED
    elif args.media_type_only:
        scenarios = _BUCKETED_SCENARIOS_MEDIA_TYPE
    elif args.media_type_buckets:
        base_scenarios = (_BUCKETED_SCENARIOS_FOUR_QUALITY if args.four_quality_buckets
                          else _BUCKETED_SCENARIOS)
        scenarios = base_scenarios + _BUCKETED_SCENARIOS_MEDIA_TYPE
    elif args.four_quality_buckets:
        scenarios = _BUCKETED_SCENARIOS_FOUR_QUALITY
    else:
        scenarios = _BUCKETED_SCENARIOS

    for resolved_name in all_resolved_names:
        print(f"Running bucketed evaluations for {resolved_name}...")
        output_parts = []

        for variant_label, use_psl, use_multimodal, rerank, use_stacked in variants:
            for scenario_label, scenario_kwargs in scenarios:
                print(f"  -> [{variant_label}] {scenario_label} ({method_label})")
                header = (f"=== [{variant_label}] {scenario_label} "
                          f"({method_label}) for {resolved_name} ===")
                output_parts.append(header)

                if rerank == 'kreciprocal' and (use_psl or use_stacked):
                    msg = ("Skipping evaluation: k-reciprocal re-ranking is currently "
                           "not compatible with PSL pseudo-distances.")
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

        safe_name = resolved_name.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_parts = ["stacked_bucketing"] if args.stacked_bucketing else ["bucketed"]
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
        if args.four_quality_buckets and not args.stacked_bucketing:
            suffix_parts.append("4qual")
        if (args.media_type_only or args.media_type_buckets) and not args.stacked_bucketing:
            suffix_parts.append("mediatype")
        suffix = "_" + "_".join(suffix_parts)
        out_filename = f"{safe_name}{suffix}_{timestamp}.txt"
        out_path = os.path.join(target_dir, out_filename)

        with open(out_path, "w") as f:
            f.write(full_output)

        print(f"Saved bucketed results for {resolved_name} to {out_path}\n")

if __name__ == '__main__':
    main()
