# pylint: disable=import-error, wrong-import-position, no-name-in-module, too-many-locals, too-many-branches, too-many-statements
"""Learns optimal PSL rule weights from training split nearest neighbors using Maximum Likelihood Estimation."""

import argparse
import datetime
import os
import sys

import pandas as pd
import torch

from src.core import constants
from src.core import path_utils

path_utils.ensure_shark_model_in_path()

# pylint: disable=wrong-import-position
import src.psl.psl_rules as psl_rules
from src.psl.psl_engine import PSLReranker, load_metadata_from_csvs
from src.core import utils
from src.retrieval import generation_store
# pylint: enable=wrong-import-position

from pslpython.model import Model
from pslpython.partition import Partition


# ─── Rule Config Helpers ─────────────────────────────────────────────────────

# Rules always forced active regardless of --active-rules (derived from
# NEURAL_PRIOR_CONFIG so it stays in sync when rules are toggled off there).
_NEURAL_PRIOR_RULES = {k for k, v in psl_rules.NEURAL_PRIOR_CONFIG.items() if v}

# Convenience aliases mapping a short group name to the full rule names.
RULE_GROUPS = {
    "length": ["length_disagreement"],
    "all":    ["length_disagreement"],
}


def _parse_active_rules(active_rules_str):
    """Build an active_rules dictionary from a comma-separated rule list or group alias."""
    if not active_rules_str:
        return None

    # Expand group aliases first.
    tokens = [t.strip() for t in active_rules_str.split(",") if t.strip()]
    expanded = []
    for token in tokens:
        if token in RULE_GROUPS:
            expanded.extend(RULE_GROUPS[token])
        else:
            expanded.append(token)

    enabled = set(expanded) | _NEURAL_PRIOR_RULES
    all_rule_names = [name for name, _ in psl_rules.RULE_TEMPLATES]
    return {name: (name in enabled) for name in all_rule_names}


# ─── Data Loading ────────────────────────────────────────────────────────────

def _load_embeddings_data(embeddings_dir):
    """Load stored dataset and cached image neighbors from an embeddings directory."""
    dataset_path = os.path.join(embeddings_dir, "stored_dataset.pt")
    neighbors_path = os.path.join(embeddings_dir, "stored_image_neighbors.pt")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Stored dataset not found at {dataset_path}")
    if not os.path.exists(neighbors_path):
        raise FileNotFoundError(
            f"Stored image neighbors not found at {neighbors_path}. "
            "Run embeddings.py first to generate them.")

    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    neighbors = torch.load(neighbors_path, map_location="cpu",
                           weights_only=False)

    return dataset, neighbors



# ─── Training ────────────────────────────────────────────────────────────────

def learn_weights(dataset, neighbors, metadata, psl_top_k, seed, active_rules=None):
    """Run PSL maximum likelihood weight learning across top-K neighbors from training queries. Neural prior rules are excluded from MLE and restored to default weights after learning."""
    # Identify train-split queries by filename (neighbor keys are filenames).
    train_ids = set(dataset.get_train_ids())
    all_ids = dataset.get_ids()
    all_filenames = dataset.get_filenames()
    all_labels = dataset.get_labels().cpu().numpy()

    filename_to_label = dict(zip(all_filenames, all_labels))

    # Train filenames — neighbor keys are filenames, not integer IDs.
    train_filenames = set(
        fn for mid, fn in zip(all_ids, all_filenames) if mid in train_ids)

    train_queries = [qname for qname in neighbors if qname in train_filenames]
    print(f"[PSL] Training on {len(train_queries)} train queries "
          f"(out of {len(neighbors)} total) with top_k={psl_top_k}.")

    if not train_queries:
        print("ERROR: No train queries found. Ensure the dataset has a "
              "train/test split (stored_dataset must contain train_ids).")
        sys.exit(1)

    # Neural priors are excluded from MLE because their weights collapse toward zero on ranking negatives.
    if active_rules is None:
        active_rules = psl_rules.BIOLOGICAL_CONFIG
    # Neural prior keys always set to False last so they can't be overridden
    # by an active_rules dict that sets them True (e.g. from _parse_active_rules).
    training_rules = {
        **active_rules,
        **{k: False for k in psl_rules.NEURAL_PRIOR_CONFIG},
    }

    learnable_names = psl_rules.get_rule_names(active_rules=training_rules)
    if not learnable_names:
        # All active rules are in NEURAL_PRIOR_CONFIG (fixed). Nothing to learn —
        # return DEFAULT_WEIGHTS directly without running MLE.
        print("[PSL] No learnable biological rules — skipping MLE, "
              "using DEFAULT_WEIGHTS for all rules.")
        learned_weights = {}
        for name in psl_rules.NEURAL_PRIOR_CONFIG:
            learned_weights[name] = psl_rules.DEFAULT_WEIGHTS[name]
        return learned_weights

    model = Model("psl_weight_learn")

    predicates = psl_rules.build_predicates()
    for pred in predicates.values():
        model.add_predicate(pred)

    rules = psl_rules.build_rules(learnable=True, active_rules=training_rules)
    rule_names = learnable_names
    for rule in rules:
        model.add_rule(rule)

    # ── Build observation DataFrames ─────────────────────────────
    neural_rows = []
    target_rows = []
    truth_rows = []
    length_rows = []
    sex_atoms = {}  # atom_id -> (male_val, female_val)

    for q_idx, qname in enumerate(train_queries):
        entry = neighbors[qname]
        n_names = entry["match_names"]
        n_dists = entry["match_distances"]
        n_labels = entry["match_labels"]

        q_label = filename_to_label.get(qname)
        if q_label is None:
            continue

        # Unique prefix per query to avoid atom collisions.
        q_atom = f"q{q_idx}"
        k = min(psl_top_k, len(n_names))

        # Record sex for the query atom.
        if q_atom not in sex_atoms:
            sex_atoms[q_atom] = psl_rules.encode_sex(metadata, qname)

        for i in range(k):
            n_atom = f"q{q_idx}_n{i}"
            truth = PSLReranker.euclidean_to_truth(n_dists[i])
            neural_rows.append([q_atom, n_atom, truth])
            target_rows.append([q_atom, n_atom])

            # Ground truth: same individual = 1.0, else 0.0.
            same = 1.0 if n_labels[i] == q_label else 0.0
            truth_rows.append([q_atom, n_atom, same])

            # Record sex for the neighbor atom.
            if n_atom not in sex_atoms:
                sex_atoms[n_atom] = psl_rules.encode_sex(
                    metadata, n_names[i])

            # Length similarity (unknown -> 1.0, rule silent on unknowns).
            length_rows.append([q_atom, n_atom,
                                psl_rules.encode_length_pair(
                                    metadata, qname, n_names[i])])

        if (q_idx + 1) % 500 == 0:
            print(f"  Processed {q_idx + 1}/{len(train_queries)} queries...")

    # Build sex observation DataFrames.
    male_rows = [[atom, vals[0]] for atom, vals in sex_atoms.items()]
    female_rows = [[atom, vals[1]] for atom, vals in sex_atoms.items()]

    print(f"[PSL] Built {len(neural_rows)} NeuralSim observations, "
          f"{len(truth_rows)} SameShark truth labels, "
          f"{len(sex_atoms)} sex observations, "
          f"{len(length_rows)} length observations.")

    # ── Load data into predicates ────────────────────────────────
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
    predicates["SameShark"].add_data(
        Partition.TRUTH, pd.DataFrame(truth_rows))

    # ── Run weight learning ──────────────────────────────────────
    print("\n[PSL] Running weight learning (MLE)...")
    model.learn(psl_options={"random.seed": str(seed)})

    # ── Extract learned weights ──────────────────────────────────
    learned_weights = {}
    model_rules = model.get_rules()
    for i, name in enumerate(rule_names):
        learned_weights[name] = model_rules[i].weight()

    # Restore neural prior to fixed DEFAULT_WEIGHTS — these were excluded
    # from MLE because their learned value always collapses to ~0.
    for name in psl_rules.NEURAL_PRIOR_CONFIG:
        learned_weights[name] = psl_rules.DEFAULT_WEIGHTS[name]

    return learned_weights


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    """Parse arguments, load training data, and run PSL rule weight learning."""
    parser = argparse.ArgumentParser(
        description="Learn PSL rule weights from real training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 -m src.psl.train_psl "
               "--embeddings-name prod_model\n"
               "  python3 -m src.psl.train_psl "
               "--embeddings-name my-model/vit_triplet_1 "
               "--psl-name sex_rules_v1 --psl-top-k 50\n"
               "  python3 -m src.psl.train_psl "
               "--embeddings-name prod_model --dry-run\n")
    parser.add_argument('--embeddings-name', type=str, required=True,
                        help='Name of the pre-computed embeddings directory '
                             'under sm-data/embeddings/ (produced by '
                             'src.retrieval.embeddings). Its active generation '
                             'must hold stored_dataset.pt and '
                             'stored_image_neighbors.pt, which embeddings '
                             'writes with --store-neighbors, --use-psl or '
                             '--stacked.')
    parser.add_argument('--psl-name', type=str, default=None,
                        help='Name for the output PSL rules directory under '
                             'sm-data/psl-rules/. Defaults to a timestamp '
                             'if not specified.')
    parser.add_argument('--psl-top-k', type=int, default=100,
                        help='Number of top neighbors per query to include '
                             'in weight learning (default: 100).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for PSL solver (default: 42).')
    parser.add_argument('--active-rules', type=str, default=None,
                        help='Comma-separated rule names to enable (beyond the '
                             'neural prior, which is always on). Accepts full '
                             'rule names (e.g. "sex_disagreement_mf,'
                             'sex_disagreement_fm") or the group aliases '
                             'length and all. If omitted, falls back to '
                             'psl_rules.BIOLOGICAL_CONFIG.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate paths and print configuration '
                             'without running weight learning.')
    args = parser.parse_args()

    utils.print_script_banner(__file__, args)

    # ── Resolve paths ────────────────────────────────────────────
    embedding_set_root = os.path.join(constants.EMBEDDINGS_DIR,
                                      args.embeddings_name)
    if not os.path.exists(embedding_set_root):
        print(f"ERROR: Embeddings directory not found at {embedding_set_root}")
        sys.exit(1)
    embeddings_dir = str(generation_store.resolve_active_directory(args.embeddings_name))

    psl_name = args.psl_name
    if psl_name is None:
        psl_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(constants.PSL_DIR, psl_name)

    # ── Summary ──────────────────────────────────────────────────
    config_summary = {
        "Embeddings Directory": embeddings_dir,
        "Output Directory": output_dir,
        "PSL Name": psl_name,
        "Top-K Neighbors": args.psl_top_k,
        "Seed": args.seed,
        "Active Biological Rules": args.active_rules or "BIOLOGICAL_CONFIG",
    }
    utils.print_key_value_section("PSL Weight Learning Configuration",
                                  config_summary)

    if args.dry_run:
        # Check if required files exist.
        required_files = [
            ("stored_dataset.pt",
             os.path.exists(os.path.join(embeddings_dir,
                                         "stored_dataset.pt"))),
            ("stored_image_neighbors.pt",
             os.path.exists(os.path.join(embeddings_dir,
                                         "stored_image_neighbors.pt"))),
        ]
        path_checks = {name: "OK" if ok else "MISSING"
                       for name, ok in required_files}
        utils.print_key_value_section("Path Verification", path_checks)
        print("Dry run complete. Exiting without training.\n")
        return

    # ── Load data ────────────────────────────────────────────────
    print("\nLoading stored dataset and neighbors...")
    dataset, neighbors = _load_embeddings_data(embeddings_dir)

    csv_dir = getattr(dataset, 'csv_dir', None)
    if not csv_dir or not os.path.isdir(csv_dir):
        print("ERROR: No csv_dir found on stored dataset. "
              "Re-generate embeddings so csv_dir is pinned.")
        sys.exit(1)
    metadata = load_metadata_from_csvs(csv_dir)

    # ── Print default weights ────────────────────────────────────
    print("\nDefault weights (before learning):")
    default_weights = psl_rules.DEFAULT_WEIGHTS
    for name, weight in default_weights.items():
        print(f"  {name}: {weight}")

    # ── Learn weights ────────────────────────────────────────────
    active_rules = _parse_active_rules(args.active_rules)
    learned_weights = learn_weights(
        dataset, neighbors, metadata,
        psl_top_k=args.psl_top_k, seed=args.seed, active_rules=active_rules)

    # ── Print comparison ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("WEIGHT LEARNING RESULTS")
    print("=" * 60)

    table_headers = ["Rule", "Default", "Learned", "Delta"]
    table_rows = []
    for name in psl_rules.get_rule_names(active_rules=active_rules):
        default = default_weights.get(name, 0.0)
        learned = learned_weights.get(name, 0.0)
        delta = learned - default
        table_rows.append([name, f"{default:.4f}", f"{learned:.4f}",
                           f"{delta:+.4f}"])
    utils.print_table(table_headers, table_rows)

    # ── Save ─────────────────────────────────────────────────────
    psl_rules.save_weights(learned_weights, output_dir)
    print(f"\nLearned weights saved to: {output_dir}")
    print(f"Use with: --psl-rules {psl_name}")


if __name__ == "__main__":
    main()
