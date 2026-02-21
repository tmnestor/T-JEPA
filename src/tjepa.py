"""T-JEPA model: orchestrates encoder, predictor, and EMA target encoder."""

import copy

import torch
import torch.nn as nn

from src.config import TJEPAConfig
from src.encoder import Encoder
from src.predictor import TransformerPredictor
from src.utils import apply_masks_from_idx


class TJEPA(nn.Module):
    """T-JEPA: Joint Embedding Predictive Architecture for tabular data.

    The model has three components:
    1. Context encoder: processes masked input features
    2. Target encoder: processes full input (no grad, EMA-updated)
    3. Predictor: predicts target embeddings from context embeddings
    """

    def __init__(
        self,
        config: TJEPAConfig,
        num_numerical: int,
        num_categorical: int,
        cat_cardinalities: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        num_features = num_numerical + num_categorical

        # Context encoder
        self.context_encoder = Encoder(
            config=config.encoder,
            num_numerical=num_numerical,
            num_categorical=num_categorical,
            cat_cardinalities=cat_cardinalities,
        )

        # Target encoder (EMA copy, no grad)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = TransformerPredictor(
            config=config.predictor,
            num_features=num_features,
            encoder_output_dim=config.encoder.output_dim,
        )

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        """Update target encoder via exponential moving average.

        Args:
            momentum: EMA momentum (0 < m <= 1).
                target = m * target + (1-m) * context
        """
        for target_param, context_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
            strict=False,
        ):
            target_param.data.mul_(momentum).add_(context_param.data, alpha=1.0 - momentum)

    def forward(
        self,
        x_num: torch.Tensor | None,
        x_cat: torch.Tensor | None,
        enc_masks: list[torch.Tensor],
        pred_masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass for T-JEPA pretraining.

        Args:
            x_num: Numerical features (batch, num_numerical) or None.
            x_cat: Categorical features (batch, num_categorical) or None.
            enc_masks: List with 1 tensor of context feature indices.
            pred_masks: List of N tensors of target feature indices.

        Returns:
            Tuple of (loss, predictions, targets).
        """
        # 1. Target encoder forward (no gradient)
        with torch.no_grad():
            h_target_full = self.target_encoder(x_num, x_cat)
            # Remove [REG] token (position 0), keep feature tokens only
            h_target_features = h_target_full[:, 1:]
            # Extract target features at pred_mask positions
            h_targets = apply_masks_from_idx(h_target_features, pred_masks)

        # 2. Context encoder forward (with mask)
        context_mask = enc_masks[0]
        z_context = self.context_encoder(x_num, x_cat, mask_indices=context_mask)

        # 3. Predictor: predict target embeddings
        z_predictions = self.predictor(z_context, context_mask, pred_masks)

        # 4. Compute loss: MSE between predictions and targets
        loss = torch.tensor(0.0, device=z_context.device)
        for z_pred, h_tgt in zip(z_predictions, h_targets, strict=False):
            loss = loss + nn.functional.mse_loss(z_pred, h_tgt)
        loss = loss / len(z_predictions)

        return loss, z_predictions, h_targets
