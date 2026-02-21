"""Adult Income dataset (binary classification, mixed features)."""

import pandas as pd
from sklearn.datasets import fetch_openml

from src.data.base import TabularDataset, TaskType


class AdultIncome(TabularDataset):
    """Adult Income dataset (Census Income).

    Mixed numerical and categorical features, binary classification target.
    """

    @property
    def name(self) -> str:
        return "adult"

    @property
    def task_type(self) -> TaskType:
        return TaskType.BINARY

    @property
    def numerical_columns(self) -> list[str]:
        return ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

    @property
    def categorical_columns(self) -> list[str]:
        return [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

    @property
    def target_column(self) -> str:
        return "income"

    def load(self) -> pd.DataFrame:
        data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
        df = data.frame

        # Rename target column
        if "class" in df.columns:
            df = df.rename(columns={"class": "income"})

        # Binarize target
        target = df[self.target_column].astype(str).str.strip().str.rstrip(".")
        df[self.target_column] = (target.isin([">50K", ">50K."])).astype(int)

        # Drop rows with missing values
        df = df.dropna().reset_index(drop=True)

        return df
