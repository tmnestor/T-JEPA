"""Tests for the mask collator."""

import torch

from src.config import MaskConfig
from src.mask import MaskCollator


def test_mask_collator_basic():
    """Test that mask collator produces valid masks."""
    config = MaskConfig()
    collator = MaskCollator(num_features=8, config=config)

    # Fake batch of 4 samples with 8 numerical features
    batch = [(torch.randn(8),) for _ in range(4)]
    collated, enc_masks, pred_masks = collator(batch)

    # Check collated shape
    assert collated[0].shape == (4, 8)

    # Check we got encoder and predictor masks
    assert len(enc_masks) >= 1
    assert len(pred_masks) >= 1


def test_masks_non_overlapping():
    """Test that context and target masks don't overlap."""
    config = MaskConfig(num_pred_masks=2)
    collator = MaskCollator(num_features=8, config=config)

    batch = [(torch.randn(8),) for _ in range(2)]

    for _ in range(20):  # Run multiple times due to randomness
        _, enc_masks, pred_masks = collator(batch)

        context_set = set(enc_masks[0].tolist())
        for pred_mask in pred_masks:
            target_set = set(pred_mask.tolist())
            overlap = context_set & target_set
            assert len(overlap) == 0, f"Context and target overlap: {overlap}"


def test_masks_cover_valid_indices():
    """Test that all mask indices are within valid range."""
    num_features = 10
    config = MaskConfig(num_pred_masks=3)
    collator = MaskCollator(num_features=num_features, config=config)

    batch = [(torch.randn(num_features),) for _ in range(2)]

    for _ in range(20):
        _, enc_masks, pred_masks = collator(batch)
        for mask in enc_masks + pred_masks:
            assert mask.min() >= 0
            assert mask.max() < num_features


def test_mask_collator_with_two_tensors():
    """Test collator with samples containing num and cat tensors."""
    config = MaskConfig()
    collator = MaskCollator(num_features=6, config=config)

    batch = [(torch.randn(4), torch.randint(0, 5, (2,))) for _ in range(3)]
    collated, enc_masks, pred_masks = collator(batch)

    assert collated[0].shape == (3, 4)
    assert collated[1].shape == (3, 2)
