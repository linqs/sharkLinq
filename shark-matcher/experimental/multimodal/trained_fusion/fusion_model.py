"""Trained multimodal fusion architectures comparing concatenation and residual heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusionModel(nn.Module):
    """Trained fusion head operating over frozen image and engineered metadata features."""

    def __init__(
            self,
            image_dim: int,
            metadata_dim: int,
            shared_dim: int = 512,
            final_dim: int = 256,
            fusion_method: str = "concatenate",
            expert_dropout: float = 0.1,
            projection_dropout: float = 0.1,
            expert_depth: int = 1,
            residual_hidden_dim: int = 256,
    ):
        super().__init__()
        self.image_dim = image_dim
        self.metadata_dim = metadata_dim
        self.fusion_method = fusion_method

        if fusion_method == "residual":
            # Zero-initialize the final linear layer so the initial forward pass matches the base image embeddings.
            self.residual_mlp = nn.Sequential(
                nn.LayerNorm(image_dim + metadata_dim),
                nn.Linear(image_dim + metadata_dim, residual_hidden_dim),
                nn.ReLU(),
                nn.Dropout(projection_dropout),
                nn.Linear(residual_hidden_dim, image_dim),
            )
            nn.init.zeros_(self.residual_mlp[-1].weight)
            nn.init.zeros_(self.residual_mlp[-1].bias)
            return

        if fusion_method != "concatenate":
            raise ValueError(
                f"Unsupported fusion_method '{fusion_method}'. "
                f"This module supports only 'concatenate' and 'residual'."
            )

        # Visual Feature Expert.
        vfe_layers = [nn.Linear(image_dim, shared_dim), nn.ReLU(),
                      nn.Dropout(expert_dropout)]
        for _ in range(expert_depth - 1):
            vfe_layers += [nn.Linear(shared_dim, shared_dim), nn.ReLU(),
                           nn.Dropout(expert_dropout)]
        vfe_layers.append(nn.LayerNorm(shared_dim))
        self.vfe = nn.Sequential(*vfe_layers)

        # Textual Metadata Expert (input dim adapts to the metadata vector).
        tme_layers = [nn.Linear(metadata_dim, shared_dim), nn.ReLU(),
                      nn.Dropout(expert_dropout)]
        for _ in range(expert_depth - 1):
            tme_layers += [nn.Linear(shared_dim, shared_dim), nn.ReLU(),
                           nn.Dropout(expert_dropout)]
        tme_layers.append(nn.LayerNorm(shared_dim))
        self.tme = nn.Sequential(*tme_layers)

        # Final projection head.
        proj_input_dim = shared_dim * 2
        self.projection_head = nn.Sequential(
            nn.LayerNorm(proj_input_dim),
            nn.Linear(proj_input_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(projection_dropout),
            nn.Linear(shared_dim, final_dim),
        )

    def forward(self, image_embedding: torch.Tensor,
                metadata_embedding: torch.Tensor) -> torch.Tensor:
        if self.fusion_method == "residual":
            delta_input = torch.cat([image_embedding, metadata_embedding], dim=1)
            delta = self.residual_mlp(delta_input)
            return F.normalize(image_embedding + delta, p=2, dim=1)

        i_prime = self.vfe(image_embedding)
        t_prime = self.tme(metadata_embedding)
        fused = torch.cat((i_prime, t_prime), dim=1)
        out = self.projection_head(fused)
        return F.normalize(out, p=2, dim=1)
