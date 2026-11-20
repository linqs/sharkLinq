"""Evaluation adapter delegating multimodal fusion Hits@K computation to the shared evaluator."""
import torch

import src.learning.evaluator.evaluators


def _fusion_embed(model: torch.nn.Module, batch, device: torch.device) -> torch.Tensor:
    """Embedder for FusionDataset batches: ``model(image_emb, metadata)``."""
    image_embeds, metadata_embeds, _ = batch
    return model(image_embeds.to(device), metadata_embeds.to(device))


def multimodal_hits_at_k(
        model: torch.nn.Module,
        train_dataset,
        valid_dataset,
        device: torch.device,
):
    """Compute Hits@K for a multimodal fusion model via the shared evaluator."""
    return src.learning.evaluator.evaluators.hits_at_k(
        model, train_dataset, valid_dataset, device, embed_fn=_fusion_embed
    )
