"""Mask collator for T-JEPA.

Generates non-overlapping context and target feature masks for each batch.
Used as a DataLoader collate_fn.
"""

import torch

from src.config import MaskConfig


class MaskCollator:
    """Generate context and target masks for T-JEPA pretraining.

    For each batch, generates:
    - 1 context (encoder) mask: subset of features visible to the encoder
    - N target (predictor) masks: subsets of remaining features to predict

    Context and target masks are non-overlapping.
    """

    def __init__(self, num_features: int, config: MaskConfig) -> None:
        """Initialize mask collator.

        Args:
            num_features: Total number of features in the dataset.
            config: Masking configuration.
        """
        self.num_features = num_features
        self.config = config

    def __call__(
        self, batch: list[tuple[torch.Tensor, ...]]
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Collate batch and generate masks.

        Args:
            batch: List of samples from dataset (tuples of tensors).

        Returns:
            Tuple of (collated_batch, encoder_masks, predictor_masks).
            - collated_batch: Stacked tensors for each element in the sample tuple.
            - encoder_masks: List with 1 tensor of context feature indices.
            - predictor_masks: List of N tensors of target feature indices.
        """
        # Collate: stack each element across the batch
        collated = []
        for i in range(len(batch[0])):
            collated.append(torch.stack([sample[i] for sample in batch]))

        # Generate masks
        enc_masks, pred_masks = self._generate_masks()

        return tuple(collated), enc_masks, pred_masks

    def _generate_masks(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Generate context and target masks.

        Returns:
            (encoder_masks, predictor_masks) where each is a list of index tensors.
        """
        all_indices = torch.randperm(self.num_features)

        # Determine context size
        ctx_ratio = (
            torch.empty(1)
            .uniform_(
                self.config.context_ratio_min,
                self.config.context_ratio_max,
            )
            .item()
        )
        ctx_size = max(1, int(ctx_ratio * self.num_features))

        context_indices = all_indices[:ctx_size].sort().values
        remaining_indices = all_indices[ctx_size:]

        # Split remaining into target masks
        pred_masks = []
        num_remaining = remaining_indices.size(0)

        if num_remaining > 0 and self.config.num_pred_masks > 0:
            # Determine target size per mask
            tgt_ratio = (
                torch.empty(1)
                .uniform_(
                    self.config.target_ratio_min,
                    self.config.target_ratio_max,
                )
                .item()
            )
            tgt_size_per_mask = max(1, int(tgt_ratio * self.num_features / self.config.num_pred_masks))

            # Shuffle remaining and split into pred masks
            remaining_shuffled = remaining_indices[torch.randperm(num_remaining)]

            for i in range(self.config.num_pred_masks):
                start = i * tgt_size_per_mask
                end = min(start + tgt_size_per_mask, num_remaining)
                if start < num_remaining:
                    mask = remaining_shuffled[start:end].sort().values
                    pred_masks.append(mask)

        # Ensure at least one target mask
        if not pred_masks:
            # Use at least 1 remaining feature
            if num_remaining > 0:
                pred_masks.append(remaining_indices[:1].sort().values)
            else:
                # Edge case: give one feature to target
                pred_masks.append(context_indices[-1:])
                context_indices = context_indices[:-1]
                if context_indices.size(0) == 0:
                    context_indices = all_indices[:1].sort().values

        enc_masks = [context_indices]
        return enc_masks, pred_masks
