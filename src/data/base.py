"""Base dataset interface for tabular datasets."""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd


class TaskType(Enum):
    """Type of downstream task."""

    REGRESSION = "regression"
    BINARY = "binary"
    MULTICLASS = "multiclass"


class TabularDataset(ABC):
    """Abstract base class for tabular datasets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Type of downstream task."""

    @property
    @abstractmethod
    def numerical_columns(self) -> list[str]:
        """Names of numerical feature columns."""

    @property
    @abstractmethod
    def categorical_columns(self) -> list[str]:
        """Names of categorical feature columns."""

    @property
    @abstractmethod
    def target_column(self) -> str:
        """Name of the target column."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and return the raw dataset as a DataFrame.

        Returns:
            DataFrame with all features and the target column.
        """

    @property
    def num_features(self) -> int:
        """Total number of features."""
        return len(self.numerical_columns) + len(self.categorical_columns)

    @property
    def cat_cardinalities(self) -> list[int] | None:
        """Cardinalities of categorical features (set after preprocessing)."""
        return getattr(self, "_cat_cardinalities", None)

    @cat_cardinalities.setter
    def cat_cardinalities(self, value: list[int]) -> None:
        self._cat_cardinalities = value

    def get_splits(
        self, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/val/test.

        Args:
            val_ratio: Fraction for validation.
            test_ratio: Fraction for test.
            seed: Random seed.

        Returns:
            (train_df, val_df, test_df).
        """
        df = self.load()
        rng = np.random.RandomState(seed)
        n = len(df)
        indices = rng.permutation(n)

        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)

        test_idx = indices[:test_size]
        val_idx = indices[test_size : test_size + val_size]
        train_idx = indices[test_size + val_size :]

        train = df.iloc[train_idx].reset_index(drop=True)
        val = df.iloc[val_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        return train, val, test
