"""California Housing dataset (regression, 8 numerical features)."""

import pandas as pd
from sklearn.datasets import fetch_california_housing

from src.data.base import TabularDataset, TaskType


class CaliforniaHousing(TabularDataset):
    """California Housing dataset from sklearn.

    8 numerical features, regression target (median house value).
    No download required — bundled with sklearn.
    """

    @property
    def name(self) -> str:
        return "california"

    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION

    @property
    def numerical_columns(self) -> list[str]:
        return [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]

    @property
    def categorical_columns(self) -> list[str]:
        return []

    @property
    def target_column(self) -> str:
        return "MedHouseVal"

    def load(self) -> pd.DataFrame:
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        return df
