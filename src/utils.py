"""Utility functions for T-JEPA."""

import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_sincos_pos_embed(num_positions: int, embed_dim: int) -> torch.Tensor:
    """Generate sinusoidal positional embeddings.

    Args:
        num_positions: Number of positions (features).
        embed_dim: Embedding dimension.

    Returns:
        Tensor of shape (num_positions, embed_dim).
    """
    position = torch.arange(num_positions).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))

    pe = torch.zeros(num_positions, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: embed_dim // 2])
    return pe


class PositionalEncoding(nn.Module):
    """Learnable positional encoding added to token embeddings."""

    def __init__(self, num_positions: int, embed_dim: int) -> None:
        super().__init__()
        pe = get_sincos_pos_embed(num_positions, embed_dim)
        self.pos_embed = nn.Parameter(pe.unsqueeze(0), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Tensor with positional encoding added.
        """
        return x + self.pos_embed[:, : x.size(1)]


def apply_masks_from_idx(x: torch.Tensor, mask_indices: list[torch.Tensor]) -> list[torch.Tensor]:
    """Extract features at specified mask indices.

    Args:
        x: Input tensor of shape (batch, num_features, embed_dim).
        mask_indices: List of tensors, each of shape (num_masked,) with feature indices.

    Returns:
        List of tensors, each of shape (batch, num_masked, embed_dim).
    """
    outputs = []
    for indices in mask_indices:
        indices = indices.to(x.device)
        # indices shape: (num_masked,) -> gather across seq dim
        idx = indices.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, x.size(2))
        selected = torch.gather(x, dim=1, index=idx)
        outputs.append(selected)
    return outputs
