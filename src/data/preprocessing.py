"""Data preprocessing and PyTorch dataset wrappers."""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

from src.data.base import TabularDataset, TaskType


class PretrainingDataset(Dataset):
    """PyTorch Dataset for T-JEPA pretraining (no labels needed)."""

    def __init__(
        self,
        x_num: np.ndarray | None = None,
        x_cat: np.ndarray | None = None,
    ) -> None:
        self.x_num = torch.tensor(x_num, dtype=torch.float32) if x_num is not None else None
        self.x_cat = torch.tensor(x_cat, dtype=torch.long) if x_cat is not None else None

        if self.x_num is not None:
            self.n_samples = self.x_num.size(0)
        else:
            self.n_samples = self.x_cat.size(0)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        items = []
        if self.x_num is not None:
            items.append(self.x_num[idx])
        if self.x_cat is not None:
            items.append(self.x_cat[idx])
        return tuple(items)


class SupervisedDataset(Dataset):
    """PyTorch Dataset for downstream supervised evaluation."""

    def __init__(
        self,
        x_num: np.ndarray | None,
        x_cat: np.ndarray | None,
        y: np.ndarray,
        task_type: TaskType,
    ) -> None:
        self.x_num = torch.tensor(x_num, dtype=torch.float32) if x_num is not None else None
        self.x_cat = torch.tensor(x_cat, dtype=torch.long) if x_cat is not None else None

        if task_type == TaskType.REGRESSION:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = torch.tensor(y, dtype=torch.long)

        if self.x_num is not None:
            self.n_samples = self.x_num.size(0)
        else:
            self.n_samples = self.x_cat.size(0)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        items = []
        if self.x_num is not None:
            items.append(self.x_num[idx])
        if self.x_cat is not None:
            items.append(self.x_cat[idx])
        items.append(self.y[idx])
        return tuple(items)


class DataPreprocessor:
    """Preprocess tabular data: scale numerical, encode categorical, split."""

    def __init__(self, dataset: TabularDataset) -> None:
        self.dataset = dataset
        self.num_scaler = StandardScaler() if dataset.numerical_columns else None
        self.cat_encoders: dict[str, LabelEncoder] = {}
        self.target_scaler = StandardScaler() if dataset.task_type == TaskType.REGRESSION else None

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[str, dict]:
        """Fit on train, transform all splits.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            test_df: Test DataFrame.

        Returns:
            Dict with 'train', 'val', 'test' keys, each containing
            'x_num', 'x_cat', 'y' arrays.
        """
        result = {}

        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            x_num = None
            x_cat = None

            # Numerical features
            if self.dataset.numerical_columns:
                num_data = df[self.dataset.numerical_columns].values.astype(np.float32)
                if split_name == "train":
                    x_num = self.num_scaler.fit_transform(num_data).astype(np.float32)
                else:
                    x_num = self.num_scaler.transform(num_data).astype(np.float32)

            # Categorical features
            if self.dataset.categorical_columns:
                cat_data = []
                for col in self.dataset.categorical_columns:
                    if split_name == "train":
                        enc = LabelEncoder()
                        encoded = enc.fit_transform(df[col].astype(str))
                        self.cat_encoders[col] = enc
                    else:
                        enc = self.cat_encoders[col]
                        # Handle unseen categories
                        col_values = df[col].astype(str)
                        encoded = np.zeros(len(col_values), dtype=np.int64)
                        known_mask = col_values.isin(enc.classes_)
                        encoded[known_mask] = enc.transform(col_values[known_mask])
                        # Unknown categories get 0 (will map to padding_idx in embedding)

                    cat_data.append(encoded + 1)  # +1 so 0 is reserved for unknown
                x_cat = np.column_stack(cat_data).astype(np.int64)

            # Target
            y = df[self.dataset.target_column].values
            if split_name == "train" and self.target_scaler is not None:
                y = self.target_scaler.fit_transform(y.reshape(-1, 1).astype(np.float32)).ravel()
            elif self.target_scaler is not None:
                y = self.target_scaler.transform(y.reshape(-1, 1).astype(np.float32)).ravel()

            result[split_name] = {"x_num": x_num, "x_cat": x_cat, "y": y}

        # Set cat cardinalities
        if self.dataset.categorical_columns:
            self.dataset.cat_cardinalities = [
                len(self.cat_encoders[col].classes_) + 1  # +1 for unknown
                for col in self.dataset.categorical_columns
            ]

        return result
