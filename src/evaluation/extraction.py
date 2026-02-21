"""Feature extraction from frozen T-JEPA target encoder."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.tjepa import TJEPA
from src.utils import get_device


@torch.no_grad()
def extract_representations(
    model: TJEPA,
    dataloader: DataLoader,
    has_num: bool = True,
    has_cat: bool = False,
    device: torch.device | None = None,
) -> np.ndarray:
    """Extract representations from the frozen target encoder.

    Uses the target encoder (EMA-updated), removes the [REG] token,
    and flattens feature embeddings.

    Args:
        model: Trained T-JEPA model.
        dataloader: DataLoader yielding (x_num,) or (x_num, x_cat) or (x_cat,) tuples.
        has_num: Whether data has numerical features.
        has_cat: Whether data has categorical features.
        device: Device to use.

    Returns:
        Array of shape (n_samples, num_features * output_dim).
    """
    device = device or get_device()
    model.to(device)
    model.eval()

    all_reps = []

    for batch in dataloader:
        x_num = None
        x_cat = None
        idx = 0

        if has_num:
            x_num = batch[idx].to(device)
            idx += 1
        if has_cat:
            x_cat = batch[idx].to(device)

        # Forward through target encoder (no mask = full input)
        h = model.target_encoder(x_num, x_cat)

        # Remove [REG] token at position 0
        h_features = h[:, 1:]

        # Flatten: (batch, num_features, output_dim) -> (batch, num_features * output_dim)
        reps = h_features.reshape(h_features.size(0), -1)
        all_reps.append(reps.cpu().numpy())

    return np.concatenate(all_reps, axis=0)
