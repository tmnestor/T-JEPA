"""Downstream MLP evaluation for T-JEPA representations."""

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import DataLoader, TensorDataset

from src.data.base import TaskType
from src.utils import get_device

console = Console()


class DownstreamMLP(nn.Module):
    """Simple MLP probe for downstream evaluation."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate_downstream(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    task_type: TaskType,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Train MLP on extracted features and evaluate.

    Args:
        train_features: Training features from T-JEPA encoder.
        train_labels: Training labels.
        test_features: Test features.
        test_labels: Test labels.
        task_type: Type of task (regression, binary, multiclass).
        epochs: Training epochs for the probe.
        lr: Learning rate.
        batch_size: Batch size.
        device: Device.

    Returns:
        Dict with metric name -> value (e.g. {"r2": 0.85} or {"accuracy": 0.91}).
    """
    device = device or get_device()
    input_dim = train_features.shape[1]

    if task_type == TaskType.REGRESSION:
        output_dim = 1
        criterion = nn.MSELoss()
    elif task_type == TaskType.BINARY:
        output_dim = 2
        criterion = nn.CrossEntropyLoss()
    else:
        output_dim = int(train_labels.max()) + 1
        criterion = nn.CrossEntropyLoss()

    model = DownstreamMLP(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DataLoaders
    x_train = torch.tensor(train_features, dtype=torch.float32)
    if task_type == TaskType.REGRESSION:
        y_train = torch.tensor(train_labels, dtype=torch.float32)
    else:
        y_train = torch.tensor(train_labels, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    # Train
    model.train()
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            if task_type == TaskType.REGRESSION:
                loss = criterion(preds.squeeze(-1), y_batch)
            else:
                loss = criterion(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # Evaluate
    model.eval()
    x_test = torch.tensor(test_features, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_preds = model(x_test).cpu().numpy()

    if task_type == TaskType.REGRESSION:
        score = r2_score(test_labels, test_preds.squeeze(-1))
        console.print(f"[green]R² score: {score:.4f}[/green]")
        return {"r2": score}
    else:
        pred_classes = test_preds.argmax(axis=1)
        score = accuracy_score(test_labels, pred_classes)
        console.print(f"[green]Accuracy: {score:.4f}[/green]")
        return {"accuracy": score}
