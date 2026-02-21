"""T-JEPA configuration dataclasses."""

from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    """Transformer encoder configuration."""

    hidden_dim: int = 64
    num_layers: int = 16
    num_heads: int = 2
    dropout: float = 0.002
    ff_dim: int = 64
    output_dim: int = 64


@dataclass
class PredictorConfig:
    """Transformer predictor configuration."""

    embed_dim: int = 16
    num_layers: int = 16
    num_heads: int = 4
    dropout: float = 0.002


@dataclass
class MaskConfig:
    """Masking configuration for context/target splits."""

    context_ratio_min: float = 0.136
    context_ratio_max: float = 0.368
    target_ratio_min: float = 0.156
    target_ratio_max: float = 0.622
    num_pred_masks: int = 4
    num_enc_masks: int = 1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 512
    epochs: int = 100
    lr: float = 3.658e-4
    weight_decay: float = 0.04
    warmup_epochs: int = 10
    ema_start: float = 0.996
    ema_end: float = 1.0
    patience: int = 15
    checkpoint_dir: str = "checkpoints"
    use_amp: bool = False


@dataclass
class DataConfig:
    """Dataset configuration."""

    dataset: str = "california"
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    num_workers: int = 0


@dataclass
class TJEPAConfig:
    """Top-level T-JEPA configuration."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
