"""Tests for the feature tokenizer."""

import torch

from src.tokenizer import Tokenizer


def test_tokenizer_numerical_only():
    """Test tokenizer with only numerical features."""
    tok = Tokenizer(num_numerical=8, num_categorical=0, cat_cardinalities=None, embed_dim=64)
    x_num = torch.randn(4, 8)
    out = tok(x_num=x_num)
    # Expected: (batch=4, 1 + 8 features, 64)
    assert out.shape == (4, 9, 64)


def test_tokenizer_categorical_only():
    """Test tokenizer with only categorical features."""
    tok = Tokenizer(num_numerical=0, num_categorical=3, cat_cardinalities=[5, 10, 3], embed_dim=32)
    x_cat = torch.tensor([[1, 2, 1], [3, 5, 2], [0, 0, 0], [4, 9, 3]])
    out = tok(x_cat=x_cat)
    # Expected: (4, 1 + 3, 32)
    assert out.shape == (4, 4, 32)


def test_tokenizer_mixed():
    """Test tokenizer with both numerical and categorical features."""
    tok = Tokenizer(num_numerical=5, num_categorical=2, cat_cardinalities=[4, 6], embed_dim=16)
    x_num = torch.randn(2, 5)
    x_cat = torch.tensor([[1, 2], [3, 4]])
    out = tok(x_num=x_num, x_cat=x_cat)
    # Expected: (2, 1 + 5 + 2, 16)
    assert out.shape == (2, 8, 16)


def test_tokenizer_reg_token_is_learnable():
    """Test that [REG] token is a learnable parameter."""
    tok = Tokenizer(num_numerical=3, num_categorical=0, cat_cardinalities=None, embed_dim=8)
    assert tok.reg_token.requires_grad is True
