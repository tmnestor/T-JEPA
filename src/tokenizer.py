"""Feature tokenizer for tabular data.

Converts raw numerical and categorical features into token embeddings.
Each feature becomes one token in the sequence.
"""

import torch
import torch.nn as nn


class Tokenizer(nn.Module):
    """Tokenize tabular features into embeddings.

    Numerical features are projected via a learned weight matrix.
    Categorical features use nn.Embedding lookup tables.
    A [REG] token is prepended to the sequence for aggregation.
    """

    def __init__(
        self,
        num_numerical: int,
        num_categorical: int,
        cat_cardinalities: list[int] | None,
        embed_dim: int,
    ) -> None:
        """Initialize tokenizer.

        Args:
            num_numerical: Number of numerical features.
            num_categorical: Number of categorical features.
            cat_cardinalities: List of category counts per categorical feature.
            embed_dim: Output embedding dimension per token.
        """
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.embed_dim = embed_dim
        self.num_features = num_numerical + num_categorical

        # Numerical: each scalar -> embed_dim via learned weight + bias
        if num_numerical > 0:
            self.num_weights = nn.Parameter(torch.empty(num_numerical, embed_dim))
            self.num_biases = nn.Parameter(torch.zeros(num_numerical, embed_dim))
            nn.init.kaiming_uniform_(self.num_weights)

        # Categorical: embedding table per feature
        if num_categorical > 0 and cat_cardinalities is not None:
            self.cat_embeddings = nn.ModuleList(
                [
                    nn.Embedding(card + 1, embed_dim, padding_idx=0)  # +1 for unknown
                    for card in cat_cardinalities
                ]
            )

        # [REG] token (learnable)
        self.reg_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.reg_token, std=0.02)

    def forward(
        self,
        x_num: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tokenize features.

        Args:
            x_num: Numerical features of shape (batch, num_numerical).
            x_cat: Categorical features of shape (batch, num_categorical) as long indices.

        Returns:
            Token embeddings of shape (batch, 1 + num_features, embed_dim).
            Position 0 is the [REG] token.
        """
        batch_size = x_num.size(0) if x_num is not None else x_cat.size(0)
        tokens = []

        if x_num is not None and self.num_numerical > 0:
            # (batch, num_numerical) -> (batch, num_numerical, embed_dim)
            num_tokens = x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + self.num_biases.unsqueeze(0)
            tokens.append(num_tokens)

        if x_cat is not None and self.num_categorical > 0:
            cat_tokens = []
            for i, emb in enumerate(self.cat_embeddings):
                cat_tokens.append(emb(x_cat[:, i]))
            # (batch, num_categorical, embed_dim)
            cat_tokens = torch.stack(cat_tokens, dim=1)
            tokens.append(cat_tokens)

        # Concat all feature tokens: (batch, num_features, embed_dim)
        feature_tokens = torch.cat(tokens, dim=1)

        # Prepend [REG] token
        reg = self.reg_token.expand(batch_size, -1, -1)
        return torch.cat([reg, feature_tokens], dim=1)
