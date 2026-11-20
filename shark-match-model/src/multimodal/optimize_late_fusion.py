"""Tunes per-group late-fusion alpha weights by optimizing retrieval metrics on validation folds."""

import argparse
import copy
import json
import math
import os
import random
import time
import typing

import torch
import torch.nn.functional as F

from src.core import constants
import src.multimodal.fusion_dataset
import src.multimodal.metadata_features as metadata_features
import src.multimodal.multimodal_config as multimodal_config
from src.core import utils

# Default group order (representation_a); overwritten when representation_b derives slice keys.
_GROUP_ORDER_DEFAULT = ("sex", "location", "type", "quality", "length", "time")
_METRICS = {
    "hits_at_1": ("hits", 1),
    "hits_at_5": ("hits", 5),
    "hits_at_10": ("hits", 10),
    "hits_at_25": ("hits", 25),
    "hits_at_100": ("hits", 100),
    "avg_k": ("avg_k", None),
}


def _precompute_similarity(
        fusion_ds, query_indices: torch.Tensor, slices: dict, device: torch.device,
        no_group_l2: bool = False,
):
    """Precompute per-group query-to-gallery similarity matrices on the specified device."""
    img = fusion_ds.image_embeddings.to(device).float()
    meta = fusion_ds.metadata.to(device).float()
    img_l2 = F.normalize(img, p=2, dim=1, eps=1e-8)

    streams: typing.Dict[str, torch.Tensor] = {}
    streams["image"] = img_l2 @ img_l2[query_indices].t()  # (N, Q)
    streams["image"] = streams["image"].t().contiguous()    # (Q, N)
    for g, (a, b) in slices.items():
        if no_group_l2:
            meta_g = meta[:, a:b]
        else:
            meta_g = F.normalize(meta[:, a:b], p=2, dim=1, eps=1e-8)
        sim = meta_g[query_indices] @ meta_g.t()            # (Q, N)
        streams[g] = sim
    return streams


def _make_kfold_seed_data(
        fusion_ds, k: int, seed: int, slices: dict, device: torch.device,
        id_to_index: dict, labels: torch.Tensor,
        min_instances: int = 2, no_group_l2: bool = False,
) -> list:
    """Partition eligible training individuals into K validation folds with full-corpus galleries and return precomputed similarity tuples."""
    train_ids = set(fusion_ds.get_train_ids())
    label_by_id = {mid: fusion_ds.get_label_by_id(mid) for mid in fusion_ds.get_ids()}

    label_to_train_ids: typing.Dict[int, list] = {}
    for mid in fusion_ds.get_ids():
        if mid in train_ids:
            lbl = label_by_id[mid]
            label_to_train_ids.setdefault(lbl, []).append(mid)

    eligible = sorted([
        lbl for lbl, ids in label_to_train_ids.items() if len(ids) >= min_instances
    ])

    rng = random.Random(seed)
    shuffled = list(eligible)
    rng.shuffle(shuffled)

    fold_labels: typing.List[typing.List[int]] = [[] for _ in range(k)]
    for i, lbl in enumerate(shuffled):
        fold_labels[i % k].append(lbl)

    seed_data = []
    for fold_i in range(k):
        fold_ids = sorted(
            mid for lbl in fold_labels[fold_i] for mid in label_to_train_ids[lbl]
        )
        query_indices = torch.tensor(
            [id_to_index[mid] for mid in fold_ids], dtype=torch.long, device=device,
        )
        sims = _precompute_similarity(
            fusion_ds, query_indices, slices, device, no_group_l2=no_group_l2,
        )
        seed_data.append((sims, labels, query_indices))

    return seed_data


def _score(weights: dict, sims: dict, labels: torch.Tensor,
           query_indices: torch.Tensor, metric: str, k_values=(1, 5, 10, 25, 50, 100),
           num_results: int = 100) -> dict:
    """Evaluate retrieval on validation queries given a weight dict and return metric scores."""
    fused = None
    for g, sim in sims.items():
        w = float(weights.get(g, 0.0))
        if w <= 0.0:
            continue
        contrib = w * sim
        fused = contrib if fused is None else fused + contrib
    if fused is None:
        return {"hits_at_1": 0.0, "hits_at_5": 0.0, "hits_at_10": 0.0,
                "hits_at_25": 0.0, "hits_at_50": 0.0, "hits_at_100": 0.0,
                "avg_k": float("inf")}

    Q = query_indices.shape[0]
    if Q == 0:
        return {"hits_at_1": 0.0, "hits_at_5": 0.0, "hits_at_10": 0.0,
                "hits_at_25": 0.0, "hits_at_50": 0.0, "hits_at_100": 0.0,
                "avg_k": float("inf")}
    fused = fused.clone()
    arange_q = torch.arange(Q, device=fused.device)
    fused[arange_q, query_indices] = -float("inf")  # exclude self.

    topk = min(num_results, fused.shape[1])
    top_indices = torch.topk(fused, topk, dim=1).indices  # (Q, topk)
    top_labels = labels[top_indices]                        # (Q, topk)
    query_labels = labels[query_indices].unsqueeze(1)       # (Q, 1)
    matches = top_labels == query_labels                    # (Q, topk)

    out = {}
    for k in k_values:
        out[f"hits_at_{k}"] = matches[:, :k].any(dim=1).float().mean().item()

    # Avg K: rank of first hit (1-indexed). If no hit in topk, count as topk+1.
    first_hit = matches.float().argmax(dim=1)              # (Q,) — 0 if no match
    has_hit = matches.any(dim=1)
    rank = torch.where(has_hit, first_hit + 1, torch.full_like(first_hit, topk + 1))
    out["avg_k"] = rank.float().mean().item()
    return out


def _score_avg(weights: dict, seed_data: list, metric: str) -> dict:
    """Average per-metric scores across validation pools from seed_data tuples."""
    scores = [_score(weights, sims, labels, qi, metric)
              for (sims, labels, qi) in seed_data]
    keys = scores[0].keys()
    return {k: sum(s[k] for s in scores) / len(scores) for k in keys}


def _objective_value(score: dict, metric: str) -> float:
    kind, k = _METRICS[metric]
    if kind == "hits":
        return score[f"hits_at_{k}"]
    if kind == "avg_k":
        return -score["avg_k"]  # maximize -avg_k.
    raise ValueError(metric)


def _sample_log_uniform(rng: random.Random, low: float, high: float) -> float:
    log_low, log_high = math.log(low), math.log(high)
    return math.exp(log_low + (log_high - log_low) * rng.random())


def _random_search(seed_data, metric, n_trials, low, high, seed,
                   group_order=_GROUP_ORDER_DEFAULT, tune_image: bool = False,
                   tied_metadata: bool = False):
    """Perform random search over per-group weights, optionally tying metadata groups."""
    rng = random.Random(seed)
    history = []
    best = (-float("inf"), {"image": 1.0}, None)
    # Edge anchors: evaluate image-only and uniform-low baselines.
    for w in [{"image": 1.0}, {"image": 1.0, **{g: 0.05 for g in group_order}}]:
        sc = _score_avg(w, seed_data, metric)
        history.append({"weights": w, "score": sc})
        v = _objective_value(sc, metric)
        if v > best[0]:
            best = (v, w, sc)
    for _ in range(n_trials):
        w = {"image": _sample_log_uniform(rng, low, high) if tune_image else 1.0}
        if tied_metadata:
            shared = _sample_log_uniform(rng, low, high)
            for g in group_order:
                w[g] = shared
        else:
            for g in group_order:
                w[g] = _sample_log_uniform(rng, low, high)
        sc = _score_avg(w, seed_data, metric)
        history.append({"weights": w, "score": sc})
        v = _objective_value(sc, metric)
        if v > best[0]:
            best = (v, w, sc)
    return best, history


def _coordinate_refine(start_weights, seed_data, metric,
                       group_order=_GROUP_ORDER_DEFAULT,
                       n_passes=4, multipliers=(0.25, 0.5, 0.75, 1.5, 2.0),
                       tune_image: bool = False, tied_metadata: bool = False):
    """Perform coordinate refinement on per-group weights, optionally tying metadata groups."""
    weights = dict(start_weights)
    best_sc = _score_avg(weights, seed_data, metric)
    best_v = _objective_value(best_sc, metric)
    history = [{"weights": dict(weights), "score": best_sc}]

    # Build the coordinate iteration order. With tied metadata we expose a
    # single synthetic "__meta__" coordinate that scales every metadata group
    # together; otherwise each group is its own coordinate.
    if tied_metadata:
        meta_coord = ("__meta__",)
        coord_order = (("image",) + meta_coord) if tune_image else meta_coord
    else:
        coord_order = (("image",) + tuple(group_order)) if tune_image else tuple(group_order)

    def _set_coord(trial: dict, coord: str, value: float) -> None:
        if coord == "__meta__":
            for gg in group_order:
                trial[gg] = value
        else:
            trial[coord] = value

    def _get_coord(weights: dict, coord: str) -> float:
        if coord == "__meta__":
            # All metadata groups share the same value when tied; read any.
            return float(weights.get(group_order[0], 0.0))
        return float(weights.get(coord, 0.0))

    for _ in range(n_passes):
        improved = False
        for g in coord_order:
            base = _get_coord(weights, g)
            candidates = []
            if base > 0:
                for m in multipliers:
                    candidates.append(base * m)
            # Keep image weight positive in tune-image mode so the model never becomes metadata-only.
            if g != "image":
                candidates.append(0.0)
            candidates.append(0.05 if base == 0 else base)
            for cand in candidates:
                if cand < 0.0 or cand > 4.0:
                    continue
                if g == "image" and cand <= 0.0:
                    continue
                trial = dict(weights)
                _set_coord(trial, g, cand)
                sc = _score_avg(trial, seed_data, metric)
                v = _objective_value(sc, metric)
                if v > best_v + 1e-6:
                    best_v = v
                    best_sc = sc
                    weights = trial
                    improved = True
                history.append({"weights": dict(trial), "score": sc})
        if not improved:
            break
    return (best_v, weights, best_sc), history


def _carve_validation(fusion_ds, fraction: float, min_instances: int, seed: int):
    """Move whole individuals with at least min_instances images out of train into validation."""
    train_ids = set(fusion_ds.get_train_ids())
    label_by_id = {mid: fusion_ds.get_label_by_id(mid) for mid in fusion_ds.get_ids()}

    label_to_train_ids: typing.Dict[int, list] = {}
    for mid in fusion_ds.get_ids():
        if mid in train_ids:
            lbl = label_by_id[mid]
            label_to_train_ids.setdefault(lbl, []).append(mid)
    eligible = sorted([
        lbl for lbl, ids in label_to_train_ids.items() if len(ids) >= min_instances
    ])
    rng = random.Random(seed)
    n_valid_labels = max(1, int(round(len(eligible) * fraction)))
    valid_labels = set(rng.sample(eligible, n_valid_labels))

    valid_ids = []
    for lbl in valid_labels:
        valid_ids.extend(label_to_train_ids[lbl])
    valid_ids = sorted(valid_ids)
    return valid_ids, valid_labels


def _weights_to_alphas(weights: dict) -> dict:
    """Convert per-group cosine-sim weights w_g (=alpha_g^2) back to alpha_g."""
    alphas = {}
    for k, w in weights.items():
        alphas[f"alpha_{k}"] = round(math.sqrt(max(0.0, float(w))), 4)
    return alphas


def _format_config_snippet(config_name: str, alphas: dict, group_order=None) -> str:
    """Emit a Python dictionary snippet formatted for late-fusion configuration files."""
    if group_order is None:
        group_order = _GROUP_ORDER_DEFAULT
    keys = ["alpha_image"] + [f"alpha_{g}" for g in group_order]
    lines = [f'    "{config_name}": ' + "{"]
    for k in keys:
        v = alphas.get(k, 0.0)
        lines.append(f'        "{k}": {v},')
    lines.append("    },")
    return "\n".join(lines)
