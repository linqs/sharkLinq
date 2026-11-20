"""Implements Triplet Loss with configurable negative mining strategies."""

import typing

import torch


def create_triplet_loss(margin: float = 0.2, mining_strategy: str = "batch_hard", normalize_embeddings: bool = False) \
        -> typing.Callable:
    """Creates a triplet loss function supporting batch-hard and semi-hard negative mining."""
    # pylint: disable=too-many-locals
    def batch_hard_triplet_loss(embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if normalize_embeddings:
            # L2-normalize embeddings so distances reflect angular separation on the hypersphere.
            embedding = torch.nn.functional.normalize(embedding, p=2.0, dim=1)

        # Compute pairwise distances between embeddings.
        pairwise_dist = torch.cdist(embedding, embedding, p=2.0)

        # Create a mask for positive and negative pairs.
        # shape: (batch_size, batch_size)
        mask_positive = (labels.unsqueeze(1) == labels.unsqueeze(0))
        # An anchor should not be its own positive.
        mask_positive.fill_diagonal_(False)
        mask_negative = ~mask_positive
        mask_negative.fill_diagonal_(False) # Also ignore self for negatives.

        # --- Find the hardest positive for each anchor ---
        # Set distances for non-positive pairs to a very small number.
        anchor_positive_dist = pairwise_dist.clone()
        anchor_positive_dist[~mask_positive] = float('-inf')
        # The max distance for each row is the hardest positive.
        hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1)

        # --- Find the negative for each anchor based on mining strategy ---
        if mining_strategy == "batch_hard":
            # Batch Hard: find the closest negative for each anchor.
            anchor_negative_dist = pairwise_dist.clone()
            anchor_negative_dist[~mask_negative] = float('inf')
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1)

        elif mining_strategy == "semi_hard":
            # Select negatives satisfying pos_dist < neg_dist < pos_dist + margin, falling back to hard negatives.
            pos_dist_expanded = hardest_positive_dist.unsqueeze(1)  # [batch_size, 1]

            # Mask valid negative pairs within the semi-hard margin window.
            semi_hard_mask = mask_negative & \
                             (pairwise_dist > pos_dist_expanded) & \
                             (pairwise_dist < pos_dist_expanded + margin)

            # Find the closest semi-hard negative for each anchor.
            anchor_negative_dist_semi_hard = pairwise_dist.clone()
            anchor_negative_dist_semi_hard[~semi_hard_mask] = float('inf')
            semi_hard_negative_dist, _ = torch.min(anchor_negative_dist_semi_hard, dim=1)

            # Fall back to hard negatives for anchors without valid semi-hard negatives.
            has_semi_hard = semi_hard_negative_dist != float('inf')
            anchor_negative_dist_hard = pairwise_dist.clone()
            anchor_negative_dist_hard[~mask_negative] = float('inf')
            hard_negative_dist, _ = torch.min(anchor_negative_dist_hard, dim=1)

            # Combine: use semi-hard where available, hard otherwise.
            hardest_negative_dist = torch.where(
                has_semi_hard,
                semi_hard_negative_dist,
                hard_negative_dist
            )
        else:
            raise ValueError(
                f"Unknown mining_strategy: {mining_strategy}. "
                "Must be 'batch_hard' or 'semi_hard'"
            )

        # Handle cases where no positive or negative pairs are found in the batch
        # (e.g., if a sample has no other positive samples in its batch).
        has_valid_positive = hardest_positive_dist != float('-inf')
        has_valid_negative = hardest_negative_dist != float('inf')
        has_valid_triplet = has_valid_positive & has_valid_negative

        hardest_positive_dist[~has_valid_positive] = 0
        hardest_negative_dist[~has_valid_negative] = 0

        # Compute the triplet loss.
        triplet_losses = torch.nn.functional.relu(
            hardest_positive_dist - hardest_negative_dist + margin
        )
        loss = torch.sum(triplet_losses)

        # Mining diagnostics.
        batch_size = embedding.shape[0]
        num_valid = has_valid_triplet.sum().item()
        num_active = (triplet_losses[has_valid_triplet] > 0).sum().item() if num_valid > 0 else 0

        diagnostics = {
            "Mining/FractionValidTriplets": num_valid / batch_size if batch_size > 0 else 0.0,
            "Mining/FractionActiveTriplets": num_active / num_valid if num_valid > 0 else 0.0,
            "Mining/MeanPositiveDist": hardest_positive_dist[has_valid_positive].mean().item()
                if has_valid_positive.any() else 0.0,
            "Mining/MeanNegativeDist": hardest_negative_dist[has_valid_negative].mean().item()
                if has_valid_negative.any() else 0.0,
        }

        return loss, diagnostics

    return batch_hard_triplet_loss

