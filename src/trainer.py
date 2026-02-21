"""T-JEPA training loop."""

import csv
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from torch.utils.data import DataLoader

from src.checkpoint import EarlyStopping, save_checkpoint
from src.config import TJEPAConfig
from src.scheduler import CosineWDSchedule, WarmupCosineSchedule, momentum_schedule
from src.tjepa import TJEPA
from src.utils import get_device

console = Console()


class TJEPATrainer:
    """Training loop for T-JEPA pretraining."""

    def __init__(
        self,
        model: TJEPA,
        config: TJEPAConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device()

        self.model.to(self.device)

        # Optimizer: separate param groups for weight decay
        wd_params = []
        no_wd_params = []
        for name, param in model.context_encoder.named_parameters():
            if "bias" in name or "norm" in name or "pos_embed" in name:
                no_wd_params.append(param)
            else:
                wd_params.append(param)
        for name, param in model.predictor.named_parameters():
            if "bias" in name or "norm" in name or "pos_embed" in name:
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": wd_params, "weight_decay": config.training.weight_decay, "apply_wd": True},
                {"params": no_wd_params, "weight_decay": 0.0, "apply_wd": False},
            ],
            lr=config.training.lr,
        )

        # Schedulers
        total_steps = config.training.epochs * len(train_loader)
        warmup_steps = config.training.warmup_epochs * len(train_loader)

        self.lr_scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        self.wd_scheduler = CosineWDSchedule(
            self.optimizer,
            start_wd=config.training.weight_decay,
            end_wd=config.training.weight_decay * 0.1,
            total_steps=total_steps,
        )
        self.momentum_iter = iter(
            momentum_schedule(
                config.training.ema_start,
                config.training.ema_end,
                total_steps,
            )
        )

        self.early_stopping = EarlyStopping(patience=config.training.patience)

    def train(self) -> float:
        """Run full training loop.

        Returns:
            Best validation loss (or last train loss if no val set).
        """
        best_loss = float("inf")
        dataset_name = self.config.data.dataset
        checkpoint_dir = Path(self.config.training.checkpoint_dir) / dataset_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history CSV
        history_path = checkpoint_dir / "training_history.csv"
        history_fields = ["epoch", "train_loss", "val_loss", "lr", "best_loss"]
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history_fields)
            writer.writeheader()

        ds = self.train_loader.dataset
        has_num = hasattr(ds, "x_num") and ds.x_num is not None
        has_cat = hasattr(ds, "x_cat") and ds.x_cat is not None

        for epoch in range(self.config.training.epochs):
            train_loss = self._train_epoch(epoch, has_num, has_cat)

            # Validation
            val_loss = None
            if self.val_loader is not None:
                val_loss = self._validate(has_num, has_cat)

            current_loss = val_loss if val_loss is not None else train_loss

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            msg = f"Epoch {epoch + 1}/{self.config.training.epochs} | train_loss={train_loss:.6f}"
            if val_loss is not None:
                msg += f" | val_loss={val_loss:.6f}"
            msg += f" | lr={lr:.2e}"
            console.print(msg)

            # Checkpointing
            if current_loss < best_loss:
                best_loss = current_loss
                save_checkpoint(self.model, self.optimizer, epoch, current_loss, checkpoint_dir / "best.pt")

            # Append to training history
            with open(history_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=history_fields)
                writer.writerow(
                    {
                        "epoch": epoch + 1,
                        "train_loss": f"{train_loss:.6f}",
                        "val_loss": f"{val_loss:.6f}" if val_loss is not None else "",
                        "lr": f"{lr:.2e}",
                        "best_loss": f"{best_loss:.6f}",
                    }
                )

            # Early stopping
            if self.early_stopping(current_loss):
                console.print(f"[yellow]Early stopping at epoch {epoch + 1}[/yellow]")
                break

        # Save final checkpoint
        save_checkpoint(self.model, self.optimizer, epoch, current_loss, checkpoint_dir / "last.pt")
        console.print(f"[green]Training complete. Best loss: {best_loss:.6f}[/green]")
        console.print(f"[bold]Checkpoints: {checkpoint_dir}/[/bold]")
        console.print(f"[bold]Training history: {history_path}[/bold]")
        return best_loss

    def _train_epoch(self, epoch: int, has_num: bool, has_cat: bool) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Epoch {epoch + 1}", total=len(self.train_loader))

            for batch_data, enc_masks, pred_masks in self.train_loader:
                # Unpack batch based on available features
                x_num = None
                x_cat = None
                idx = 0
                if has_num:
                    x_num = batch_data[idx].to(self.device)
                    idx += 1
                if has_cat:
                    x_cat = batch_data[idx].to(self.device)

                # Forward
                loss, _, _ = self.model(x_num, x_cat, enc_masks, pred_masks)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.context_encoder.parameters()) + list(self.model.predictor.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()

                # EMA update
                momentum = next(self.momentum_iter, self.config.training.ema_end)
                self.model.ema_update(momentum)

                # Scheduler steps
                self.lr_scheduler.step()
                self.wd_scheduler.step()

                total_loss += loss.item()
                num_batches += 1
                progress.advance(task)

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def _validate(self, has_num: bool, has_cat: bool) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch_data, enc_masks, pred_masks in self.val_loader:
            x_num = None
            x_cat = None
            idx = 0
            if has_num:
                x_num = batch_data[idx].to(self.device)
                idx += 1
            if has_cat:
                x_cat = batch_data[idx].to(self.device)

            loss, _, _ = self.model(x_num, x_cat, enc_masks, pred_masks)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)
