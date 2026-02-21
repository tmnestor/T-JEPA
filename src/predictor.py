"""Transformer predictor for T-JEPA.

Predicts target feature embeddings from context encoder output.
Architecture: project down -> add positional embeddings -> create mask tokens
at target positions -> concat with context -> transform -> slice targets -> project back.
"""

import torch
import torch.nn as nn

from src.config import PredictorConfig
from src.utils import get_sincos_pos_embed


class TransformerPredictor(nn.Module):
    """Predict target embeddings from context embeddings."""

    def __init__(
        self,
        config: PredictorConfig,
        num_features: int,
        encoder_output_dim: int,
    ) -> None:
        """Initialize predictor.

        Args:
            config: Predictor configuration.
            num_features: Total number of features (without [REG]).
            encoder_output_dim: Dimension of encoder output.
        """
        super().__init__()
        self.config = config
        self.num_features = num_features
        self.encoder_output_dim = encoder_output_dim
        self.embed_dim = config.embed_dim

        # Project encoder output down to predictor dimension
        self.input_proj = nn.Linear(encoder_output_dim, config.embed_dim)

        # Positional embeddings for all feature positions (+1 for [REG])
        pe = get_sincos_pos_embed(num_features + 1, config.embed_dim)
        self.pos_embed = nn.Parameter(pe.unsqueeze(0), requires_grad=True)

        # Learnable mask token for target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Transformer
        pred_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=pred_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(config.embed_dim)

        # Project back to encoder output dimension
        self.output_proj = nn.Linear(config.embed_dim, encoder_output_dim)

    def forward(
        self,
        context_encoded: torch.Tensor,
        context_mask: torch.Tensor,
        target_masks: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Predict target embeddings.

        Args:
            context_encoded: Encoder output for context features.
                Shape (batch, 1 + num_context, encoder_output_dim).
                Position 0 is [REG] token.
            context_mask: Indices of context features (num_context,).
            target_masks: List of target mask index tensors, each (num_target,).

        Returns:
            List of predicted target embeddings, each (batch, num_target, encoder_output_dim).
        """
        batch_size = context_encoded.size(0)
        device = context_encoded.device

        # Project context to predictor dim
        context = self.input_proj(context_encoded)

        # Add positional embeddings to context
        # Context positions: [REG]=0, then context_mask+1
        context_mask = context_mask.to(device)
        context_positions = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                context_mask + 1,
            ]
        )
        context_pos = self.pos_embed[:, context_positions]
        context = context + context_pos

        predictions = []
        for target_mask in target_masks:
            target_mask = target_mask.to(device)
            num_targets = target_mask.size(0)

            # Create mask tokens for target positions
            mask_tokens = self.mask_token.expand(batch_size, num_targets, -1)

            # Add positional embeddings for target positions
            target_positions = target_mask + 1  # +1 for [REG] offset
            target_pos = self.pos_embed[:, target_positions]
            mask_tokens = mask_tokens + target_pos

            # Concat: [context_tokens, mask_tokens]
            combined = torch.cat([context, mask_tokens], dim=1)

            # Transform
            transformed = self.transformer(combined)
            transformed = self.norm(transformed)

            # Slice out only the target predictions (last num_targets tokens)
            target_preds = transformed[:, -num_targets:]

            # Project back to encoder output dim
            target_preds = self.output_proj(target_preds)
            predictions.append(target_preds)

        return predictions
