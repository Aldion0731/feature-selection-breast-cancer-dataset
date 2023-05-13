from typing import Protocol

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class CancerModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


def train_rf_classifier(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    model = create_random_forest_classifier()
    model.fit(X, y)
    return model


def create_random_forest_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(criterion="entropy", random_state=100)
