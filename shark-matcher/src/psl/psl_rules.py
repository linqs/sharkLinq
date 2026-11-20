"""PSL rule templates, predicate schemas, soft encodings, and weight serialization."""

import json
import os

import torch
from pslpython.predicate import Predicate
from pslpython.rule import Rule

LENGTH_DEADBAND_FEET = 6.0
LENGTH_RAMP_FEET = 6.0
_WEIGHTS_FILENAME = "psl_rules.json"
_WEIGHTS_FILENAME_LEGACY = "psl_rules.pt"

# Default weights used prior to MLE learning and fixed for neural prior and sex disagreement rules.
DEFAULT_WEIGHTS = {
    # Neural prior — hand-tuned, excluded from MLE (see NEURAL_PRIOR_CONFIG).
    "neural_prior_forward": 0.23,
    "neural_prior_reverse": 0.23,
    # Sex disagreement rules fire only when both sexes are confirmed and disagree.
    "sex_disagreement_mf": 20.0,
    "sex_disagreement_fm": 20.0,
    # Learnable via MLE because same-individual pairs can exhibit measurement noise and growth differences.
    "length_disagreement": 3.0,
}
# Fixed neural priors preserve ordering, while biological rules are calibrated via MLE on the training split.
NEURAL_PRIOR_CONFIG = {
    "neural_prior_forward": True,
    # Two-sided prior preserves neural ordering without saturation.
    "neural_prior_reverse": True,
    # Sex disagreement rules use fixed weights and are excluded from MLE.
    "sex_disagreement_mf": True,
    "sex_disagreement_fm": True,
}
BIOLOGICAL_CONFIG = {
    # Unknown length yields SimilarLength=1.0 so the rule stays silent on missing metadata.
    "length_disagreement": True,
}
# Combined view used by inference.
DEFAULT_CONFIG = {**NEURAL_PRIOR_CONFIG, **BIOLOGICAL_CONFIG}
# Each entry is (rule_name, psl_rule_string_template).
RULE_TEMPLATES = [
    # Two-sided neural prior anchors SameShark from above and below to preserve ordering without saturation.
    ("neural_prior_forward",
     "{weight}: NeuralSim(I1, I2) -> SameShark(I1, I2) ^2"),
    ("neural_prior_reverse",
     "{weight}: SameShark(I1, I2) -> NeuralSim(I1, I2) ^2"),
    # Fires only when both sexes are confirmed and disagree; stays silent when either sex is unknown.
    ("sex_disagreement_mf",
     "{weight}: IsMale(I1) & IsFemale(I2) -> !SameShark(I1, I2) ^2"),
    ("sex_disagreement_fm",
     "{weight}: IsFemale(I1) & IsMale(I2) -> !SameShark(I1, I2) ^2"),
    # Fires only when both lengths are known and exceed the similarity threshold; silent on unknowns.
    ("length_disagreement",
     "{weight}: ~SimilarLength(I1, I2) -> ~SameShark(I1, I2) ^2"),
]


def build_predicates():
    """Construct predicate objects shared across training and inference."""
    return {
        "NeuralSim":      Predicate("NeuralSim",      size=2),
        "IsMale":         Predicate("IsMale",         size=1),
        "IsFemale":       Predicate("IsFemale",       size=1),
        "SimilarLength":  Predicate("SimilarLength",  size=2),
        "SameShark":      Predicate("SameShark",      size=2),
    }


def build_rules(weights=None, learnable=True, active_rules=None):
    """Construct PSL Rule objects from templates and weights for active rules."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if active_rules is None:
        active_rules = DEFAULT_CONFIG

    rules = []
    for name, template in RULE_TEMPLATES:
        if not active_rules.get(name, True):
            continue
        w = weights.get(name, DEFAULT_WEIGHTS[name])
        rule_str = template.format(weight=w)
        # All templates carry a weight prefix, so weighted=True always.
        # Learning vs. fixed inference is controlled by the caller via
        # model.learn() vs model.infer(), not this flag.
        rules.append(Rule(rule_str, weighted=True))
    return rules


def get_rule_names(active_rules=None):
    """Return the ordered list of active rule names."""
    if active_rules is None:
        active_rules = DEFAULT_CONFIG
    return [name for name, _ in RULE_TEMPLATES
            if active_rules.get(name, True)]


def encode_sex(metadata, key):
    """Encode an image's sex as soft PSL truth values for Male and Female predicates."""
    meta = metadata.get(key, {})
    sex = (meta.get("sex") or "").upper()
    if sex == "M":
        return (1.0, 0.0)
    if sex == "F":
        return (0.0, 1.0)
    # Unknown / missing — hard zero. Provides no IsMale/IsFemale observation
    # so the sex disagreement rules stay completely silent for this image.
    return (0.0, 0.0)


def encode_length_pair(metadata, key1, key2):
    """Compute a soft SimilarLength truth value for an image pair using deadband ramp encoding."""
    len1 = (metadata.get(key1) or {}).get("length")
    len2 = (metadata.get(key2) or {}).get("length")
    if len1 is None or len2 is None:
        return 1.0
    diff = abs(float(len1) - float(len2))
    above_deadband = max(0.0, diff - LENGTH_DEADBAND_FEET)
    return max(0.0, 1.0 - above_deadband / LENGTH_RAMP_FEET)


def load_weights(psl_dir):
    """Load a weights dictionary from psl_rules.json or legacy psl_rules.pt in the specified directory."""
    json_path = os.path.join(psl_dir, _WEIGHTS_FILENAME)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            weights = json.load(f)
        print(f"[PSL] Loaded learned weights from {json_path}")
        return weights

    pt_path = os.path.join(psl_dir, _WEIGHTS_FILENAME_LEGACY)
    if os.path.exists(pt_path):
        print(f"[PSL] Loading legacy weights from {pt_path}")
        weights = torch.load(pt_path, map_location="cpu", weights_only=False)
        return weights

    raise FileNotFoundError(
        f"PSL weights not found in {psl_dir} "
        f"(looked for {_WEIGHTS_FILENAME} and {_WEIGHTS_FILENAME_LEGACY})"
    )


def save_weights(weights, output_dir):
    """Save a learned weights dictionary to psl_rules.json in the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, _WEIGHTS_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in weights.items()}, f, indent=2)
    print(f"[PSL] Saved learned weights to {path}")
