"""T-JEPA pretraining entry point."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console
from torch.utils.data import DataLoader

from src.config import TJEPAConfig
from src.data.adult import AdultIncome
from src.data.california import CaliforniaHousing
from src.data.preprocessing import DataPreprocessor, PretrainingDataset
from src.mask import MaskCollator
from src.tjepa import TJEPA
from src.trainer import TJEPATrainer
from src.utils import get_device, set_seed

app = typer.Typer()
console = Console()

DATASETS = {
    "california": CaliforniaHousing,
    "adult": AdultIncome,
}


@app.command()
def train(
    dataset: str = typer.Option("california", help="Dataset name: california or adult"),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    batch_size: int = typer.Option(512, help="Batch size"),
    lr: float = typer.Option(3.658e-4, help="Learning rate"),
    seed: int = typer.Option(42, help="Random seed"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Checkpoint directory"),
) -> None:
    """Pretrain T-JEPA on a tabular dataset."""
    set_seed(seed)
    device = get_device()
    console.print(f"[bold]Device: {device}[/bold]")

    # Load dataset
    if dataset not in DATASETS:
        console.print(
            f"[bold red]Unknown dataset: {dataset}. Choose from: {list(DATASETS.keys())}[/bold red]"
        )
        raise typer.Exit(1) from None

    tab_dataset = DATASETS[dataset]()
    console.print(f"[bold]Dataset: {tab_dataset.name} ({tab_dataset.num_features} features)[/bold]")

    # Preprocess
    train_df, val_df, test_df = tab_dataset.get_splits(seed=seed)
    preprocessor = DataPreprocessor(tab_dataset)
    splits = preprocessor.fit_transform(train_df, val_df, test_df)

    has_num = splits["train"]["x_num"] is not None
    has_cat = splits["train"]["x_cat"] is not None

    console.print(f"[bold]Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}[/bold]")

    # Datasets
    train_ds = PretrainingDataset(
        x_num=splits["train"]["x_num"],
        x_cat=splits["train"]["x_cat"],
    )
    val_ds = PretrainingDataset(
        x_num=splits["val"]["x_num"],
        x_cat=splits["val"]["x_cat"],
    )

    # Config
    config = TJEPAConfig()
    config.training.epochs = epochs
    config.training.batch_size = batch_size
    config.training.lr = lr
    config.training.checkpoint_dir = checkpoint_dir
    config.data.dataset = dataset
    config.data.seed = seed

    # Mask collator
    mask_collator = MaskCollator(
        num_features=tab_dataset.num_features,
        config=config.mask,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mask_collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=mask_collator,
        num_workers=0,
    )

    # Model
    num_numerical = len(tab_dataset.numerical_columns)
    num_categorical = len(tab_dataset.categorical_columns)
    cat_cards = tab_dataset.cat_cardinalities

    model = TJEPA(
        config=config,
        num_numerical=num_numerical,
        num_categorical=num_categorical,
        cat_cardinalities=cat_cards,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[bold]Parameters: {total_params:,} total, {trainable_params:,} trainable[/bold]")

    # Train
    trainer = TJEPATrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    best_loss = trainer.train()
    console.print(f"[bold green]Done! Best loss: {best_loss:.6f}[/bold green]")


if __name__ == "__main__":
    app()
