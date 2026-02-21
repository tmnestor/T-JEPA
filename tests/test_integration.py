"""Integration tests for T-JEPA: end-to-end with small config."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import EncoderConfig, MaskConfig, PredictorConfig, TJEPAConfig, TrainingConfig
from src.data.preprocessing import PretrainingDataset
from src.mask import MaskCollator
from src.tjepa import TJEPA
from src.utils import set_seed


def _small_config() -> TJEPAConfig:
    """Create a small config for fast testing."""
    return TJEPAConfig(
        encoder=EncoderConfig(
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
            ff_dim=16,
            output_dim=16,
        ),
        predictor=PredictorConfig(
            embed_dim=8,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
        ),
        mask=MaskConfig(
            context_ratio_min=0.2,
            context_ratio_max=0.4,
            target_ratio_min=0.2,
            target_ratio_max=0.5,
            num_pred_masks=2,
        ),
        training=TrainingConfig(
            batch_size=8,
            epochs=2,
            lr=1e-3,
            warmup_epochs=1,
        ),
    )


def test_forward_numerical_only():
    """Test full forward pass with numerical features only."""
    set_seed(42)
    config = _small_config()
    num_features = 8

    model = TJEPA(config, num_numerical=num_features, num_categorical=0)

    # Create batch
    x_num = torch.randn(8, num_features)
    mask_collator = MaskCollator(num_features, config.mask)

    batch = [(x_num[i],) for i in range(x_num.size(0))]
    collated, enc_masks, pred_masks = mask_collator(batch)

    loss, preds, targets = model(collated[0], None, enc_masks, pred_masks)

    assert loss.item() > 0
    assert len(preds) == len(targets)
    for p, t in zip(preds, targets, strict=False):
        assert p.shape == t.shape


def test_forward_mixed_features():
    """Test forward pass with both numerical and categorical features."""
    set_seed(42)
    config = _small_config()
    num_numerical = 5
    num_categorical = 3
    cat_cards = [4, 6, 3]
    num_features = num_numerical + num_categorical

    model = TJEPA(
        config,
        num_numerical=num_numerical,
        num_categorical=num_categorical,
        cat_cardinalities=cat_cards,
    )

    x_num = torch.randn(8, num_numerical)
    x_cat = torch.randint(1, 4, (8, num_categorical))

    mask_collator = MaskCollator(num_features, config.mask)
    batch = [(x_num[i], x_cat[i]) for i in range(x_num.size(0))]
    collated, enc_masks, pred_masks = mask_collator(batch)

    loss, preds, targets = model(collated[0], collated[1], enc_masks, pred_masks)

    assert loss.item() > 0


def test_ema_update():
    """Test that EMA update changes target encoder weights."""
    set_seed(42)
    config = _small_config()
    model = TJEPA(config, num_numerical=4, num_categorical=0)

    # Get a target encoder param before EMA
    target_param_before = model.target_encoder.tokenizer.num_weights.clone()

    # Modify context encoder
    with torch.no_grad():
        model.context_encoder.tokenizer.num_weights.add_(1.0)

    # EMA update
    model.ema_update(momentum=0.9)

    target_param_after = model.target_encoder.tokenizer.num_weights

    # Should have changed
    assert not torch.allclose(target_param_before, target_param_after)


def test_loss_decreases():
    """Test that loss decreases over a few training steps."""
    set_seed(42)
    config = _small_config()
    num_features = 6

    model = TJEPA(config, num_numerical=num_features, num_categorical=0)

    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-3,
    )

    mask_collator = MaskCollator(num_features, config.mask)

    # Create a small dataset
    dataset = PretrainingDataset(x_num=np.random.randn(32, num_features).astype(np.float32))
    loader = DataLoader(dataset, batch_size=8, collate_fn=mask_collator)

    losses = []
    for _epoch in range(3):
        epoch_loss = 0.0
        for batch_data, enc_masks, pred_masks in loader:
            loss, _, _ = model(batch_data[0], None, enc_masks, pred_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.ema_update(0.996)
            epoch_loss += loss.item()
        losses.append(epoch_loss)

    # Loss should generally trend down (allow some noise)
    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"


def test_dataloader_integration():
    """Test that PretrainingDataset works with MaskCollator and DataLoader."""
    num_features = 8
    config = _small_config()

    dataset = PretrainingDataset(x_num=np.random.randn(20, num_features).astype(np.float32))
    mask_collator = MaskCollator(num_features, config.mask)
    loader = DataLoader(dataset, batch_size=4, collate_fn=mask_collator)

    for batch_data, enc_masks, pred_masks in loader:
        assert batch_data[0].shape[1] == num_features
        assert len(enc_masks) >= 1
        assert len(pred_masks) >= 1
        break
