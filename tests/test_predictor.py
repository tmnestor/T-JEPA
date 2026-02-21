"""Tests for the transformer predictor."""

import torch

from src.config import PredictorConfig
from src.predictor import TransformerPredictor


def _make_predictor(num_features=8, encoder_output_dim=32, embed_dim=16, num_layers=2, num_heads=2):
    config = PredictorConfig(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0,
    )
    return TransformerPredictor(config, num_features=num_features, encoder_output_dim=encoder_output_dim)


def test_predictor_single_target_mask():
    """Test predictor with one target mask."""
    pred = _make_predictor()
    # Context: [REG] + 3 features
    context = torch.randn(4, 4, 32)
    context_mask = torch.tensor([0, 2, 5])
    target_masks = [torch.tensor([1, 3, 4, 6, 7])]

    outputs = pred(context, context_mask, target_masks)
    assert len(outputs) == 1
    # (batch=4, 5 targets, encoder_output_dim=32)
    assert outputs[0].shape == (4, 5, 32)


def test_predictor_multiple_target_masks():
    """Test predictor with multiple target masks."""
    pred = _make_predictor()
    context = torch.randn(4, 4, 32)
    context_mask = torch.tensor([0, 2, 5])
    target_masks = [torch.tensor([1, 3]), torch.tensor([4, 6, 7])]

    outputs = pred(context, context_mask, target_masks)
    assert len(outputs) == 2
    assert outputs[0].shape == (4, 2, 32)
    assert outputs[1].shape == (4, 3, 32)


def test_predictor_gradient_flow():
    """Test gradient flow through predictor."""
    pred = _make_predictor()
    context = torch.randn(2, 3, 32)
    context_mask = torch.tensor([1, 4])
    target_masks = [torch.tensor([0, 2, 3])]

    outputs = pred(context, context_mask, target_masks)
    loss = outputs[0].sum()
    loss.backward()
    assert pred.mask_token.grad is not None
