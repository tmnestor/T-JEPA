"""Checkpoint management and early stopping for T-JEPA."""

from pathlib import Path

import torch
from rich.console import Console

console = Console()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str | Path,
) -> None:
    """Save training checkpoint.

    Args:
        model: T-JEPA model.
        optimizer: Optimizer state.
        epoch: Current epoch.
        loss: Current loss value.
        path: Save path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load training checkpoint.

    Args:
        path: Checkpoint path.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
        device: Device to map tensors to.

    Returns:
        Checkpoint dict with 'epoch' and 'loss'.
    """
    path = Path(path)
    if not path.exists():
        console.print(f"[bold red]Checkpoint not found: {path}[/bold red]")
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {"epoch": checkpoint["epoch"], "loss": checkpoint["loss"]}


class EarlyStopping:
    """Early stopping based on validation loss."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience
