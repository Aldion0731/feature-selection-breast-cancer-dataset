from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .feature_selection_methods import FeatureSelectionMethod
from .models import CancerModel


@dataclass
class Metrics:
    accuracy: float
    roc_score: float
    precision: float
    recall: float
    f1: float
    num_features: int


def evaluate_model(model: CancerModel, X: np.ndarray, y_true: np.ndarray) -> Metrics:
    predicted = model.predict(X)

    accuracy = accuracy_score(y_true, predicted)
    roc_score = roc_auc_score(y_true, predicted)
    precision = precision_score(y_true, predicted)
    recall = recall_score(y_true, predicted)
    f1 = f1_score(y_true, predicted)

    num_features = X.shape[1]

    return Metrics(accuracy, roc_score, precision, recall, f1, num_features)


@dataclass
class MetricsManager:
    comparison_df: pd.DataFrame = pd.DataFrame([])

    def update(
        self, metrics: Metrics, feature_selection_method: FeatureSelectionMethod
    ) -> None:
        updated = pd.DataFrame(asdict(metrics), index=[f"{feature_selection_method}"])
        self.comparison_df = pd.concat([self.comparison_df, updated])
        self.__remove_duplicates()

    def __remove_duplicates(
        self,
    ) -> None:  # Allows multiple calls to .update() without producing duplicate results
        self.comparison_df.reset_index(inplace=True)
        self.comparison_df.drop_duplicates(inplace=True)
        self.comparison_df.set_index("index", drop=True, inplace=True)
        self.comparison_df.index.name = None
