"""Tests for the transformer encoder."""

import torch

from src.config import EncoderConfig
from src.encoder import Encoder


def _make_encoder(num_numerical=8, num_categorical=0, hidden_dim=32, num_layers=2, num_heads=2):
    config = EncoderConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0,
        ff_dim=hidden_dim,
        output_dim=hidden_dim,
    )
    return Encoder(config, num_numerical=num_numerical, num_categorical=num_categorical)


def test_encoder_full_input():
    """Test encoder with all features (no masking)."""
    enc = _make_encoder()
    x_num = torch.randn(4, 8)
    out = enc(x_num=x_num)
    # (batch=4, 1 + 8, output_dim=32)
    assert out.shape == (4, 9, 32)


def test_encoder_with_mask():
    """Test encoder with context mask."""
    enc = _make_encoder()
    x_num = torch.randn(4, 8)
    mask = torch.tensor([0, 2, 5])  # keep 3 features
    out = enc(x_num=x_num, mask_indices=mask)
    # (batch=4, 1 + 3, 32)
    assert out.shape == (4, 4, 32)


def test_encoder_gradient_flow():
    """Test that gradients flow through encoder."""
    enc = _make_encoder()
    x_num = torch.randn(2, 8)
    out = enc(x_num=x_num)
    loss = out.sum()
    loss.backward()
    # Check tokenizer weights have gradients
    assert enc.tokenizer.num_weights.grad is not None
