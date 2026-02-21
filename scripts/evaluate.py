"""T-JEPA downstream evaluation entry point."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint
from src.config import TJEPAConfig
from src.data.adult import AdultIncome
from src.data.california import CaliforniaHousing
from src.data.preprocessing import DataPreprocessor, PretrainingDataset
from src.evaluation.downstream import evaluate_downstream
from src.evaluation.extraction import extract_representations
from src.tjepa import TJEPA
from src.utils import get_device, set_seed

app = typer.Typer()
console = Console()

DATASETS = {
    "california": CaliforniaHousing,
    "adult": AdultIncome,
}


@app.command()
def evaluate(
    checkpoint: str = typer.Option("checkpoints/best.pt", help="Path to checkpoint"),
    dataset: str = typer.Option("california", help="Dataset name"),
    seed: int = typer.Option(42, help="Random seed"),
    probe_epochs: int = typer.Option(100, help="Epochs for MLP probe"),
    batch_size: int = typer.Option(512, help="Batch size for feature extraction"),
) -> None:
    """Evaluate T-JEPA representations on downstream task."""
    set_seed(seed)
    device = get_device()
    console.print(f"[bold]Device: {device}[/bold]")

    # Load dataset
    if dataset not in DATASETS:
        console.print(f"[bold red]Unknown dataset: {dataset}[/bold red]")
        raise typer.Exit(1) from None

    tab_dataset = DATASETS[dataset]()
    console.print(f"[bold]Dataset: {tab_dataset.name}[/bold]")

    # Preprocess
    train_df, val_df, test_df = tab_dataset.get_splits(seed=seed)
    preprocessor = DataPreprocessor(tab_dataset)
    splits = preprocessor.fit_transform(train_df, val_df, test_df)

    has_num = splits["train"]["x_num"] is not None
    has_cat = splits["train"]["x_cat"] is not None

    # Build model
    config = TJEPAConfig()
    num_numerical = len(tab_dataset.numerical_columns)
    num_categorical = len(tab_dataset.categorical_columns)
    cat_cards = tab_dataset.cat_cardinalities

    model = TJEPA(
        config=config,
        num_numerical=num_numerical,
        num_categorical=num_categorical,
        cat_cardinalities=cat_cards,
    )

    # Load checkpoint
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        console.print(f"[bold red]Checkpoint not found: {checkpoint_path}[/bold red]")
        raise typer.Exit(1) from None

    info = load_checkpoint(checkpoint_path, model, device=device)
    console.print(f"[bold]Loaded checkpoint from epoch {info['epoch']} (loss={info['loss']:.6f})[/bold]")

    # Extract features
    console.print("[bold]Extracting representations...[/bold]")

    train_ds = PretrainingDataset(
        x_num=splits["train"]["x_num"],
        x_cat=splits["train"]["x_cat"],
    )
    test_ds = PretrainingDataset(
        x_num=splits["test"]["x_num"],
        x_cat=splits["test"]["x_cat"],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    train_features = extract_representations(
        model,
        train_loader,
        has_num=has_num,
        has_cat=has_cat,
        device=device,
    )
    test_features = extract_representations(
        model,
        test_loader,
        has_num=has_num,
        has_cat=has_cat,
        device=device,
    )

    console.print(f"[bold]Feature dim: {train_features.shape[1]}[/bold]")

    # Downstream evaluation
    console.print("[bold]Training downstream MLP probe...[/bold]")
    results = evaluate_downstream(
        train_features=train_features,
        train_labels=splits["train"]["y"],
        test_features=test_features,
        test_labels=splits["test"]["y"],
        task_type=tab_dataset.task_type,
        epochs=probe_epochs,
        device=device,
    )

    # Baseline comparison: raw features
    console.print("\n[bold]Baseline (raw features)...[/bold]")
    import numpy as np

    raw_train = []
    raw_test = []
    if has_num:
        raw_train.append(splits["train"]["x_num"])
        raw_test.append(splits["test"]["x_num"])
    if has_cat:
        raw_train.append(splits["train"]["x_cat"].astype(np.float32))
        raw_test.append(splits["test"]["x_cat"].astype(np.float32))

    raw_train_features = np.concatenate(raw_train, axis=1)
    raw_test_features = np.concatenate(raw_test, axis=1)

    baseline_results = evaluate_downstream(
        train_features=raw_train_features,
        train_labels=splits["train"]["y"],
        test_features=raw_test_features,
        test_labels=splits["test"]["y"],
        task_type=tab_dataset.task_type,
        epochs=probe_epochs,
        device=device,
    )

    # Summary
    console.print("\n[bold]--- Results Summary ---[/bold]")
    for metric, value in results.items():
        baseline_value = baseline_results[metric]
        diff = value - baseline_value
        sign = "+" if diff > 0 else ""
        console.print(
            f"T-JEPA {metric}: {value:.4f} | Baseline: {baseline_value:.4f} | Diff: {sign}{diff:.4f}"
        )


if __name__ == "__main__":
    app()
