"""Transformer encoder for T-JEPA.

Pipeline: tokenize -> positional encode -> (optional mask) -> TransformerEncoder -> LayerNorm -> project.
"""

import torch
import torch.nn as nn

from src.config import EncoderConfig
from src.tokenizer import Tokenizer
from src.utils import PositionalEncoding


class Encoder(nn.Module):
    """T-JEPA encoder: tokenizer + transformer encoder."""

    def __init__(
        self,
        config: EncoderConfig,
        num_numerical: int,
        num_categorical: int,
        cat_cardinalities: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_features = num_numerical + num_categorical

        # Tokenizer
        self.tokenizer = Tokenizer(
            num_numerical=num_numerical,
            num_categorical=num_categorical,
            cat_cardinalities=cat_cardinalities,
            embed_dim=config.hidden_dim,
        )

        # Positional encoding: +1 for [REG] token
        self.pos_encoding = PositionalEncoding(
            num_positions=self.num_features + 1,
            embed_dim=config.hidden_dim,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

        # Output projection
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.out_norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        x_num: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
        mask_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode tabular features.

        Args:
            x_num: Numerical features (batch, num_numerical).
            x_cat: Categorical features (batch, num_categorical).
            mask_indices: Feature indices to keep (context mask). Shape (num_kept,).
                If None, all features are used.

        Returns:
            Encoded representations (batch, seq_len, output_dim).
            seq_len = 1 + num_kept_features (includes [REG] token).
        """
        # Tokenize: (batch, 1 + num_features, hidden_dim)
        tokens = self.tokenizer(x_num, x_cat)

        # Add positional encoding
        tokens = self.pos_encoding(tokens)

        # Apply context mask: keep [REG] (pos 0) + selected features
        if mask_indices is not None:
            mask_indices = mask_indices.to(tokens.device)
            # Shift mask indices by 1 to account for [REG] token at position 0
            reg_and_mask = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=tokens.device),
                    mask_indices + 1,
                ]
            )
            idx = reg_and_mask.unsqueeze(0).unsqueeze(-1).expand(tokens.size(0), -1, tokens.size(2))
            tokens = torch.gather(tokens, dim=1, index=idx)

        # Transformer
        encoded = self.transformer(tokens)

        # Project
        encoded = self.norm(encoded)
        encoded = self.proj(encoded)
        encoded = self.out_norm(encoded)

        return encoded
