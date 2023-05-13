from typing import Any

import numpy as np
from sklearn.feature_selection import RFE, SelectFromModel

from .models import CancerModel


def train_rfe_wrapper(
    model: CancerModel, X: np.ndarray, y: np.ndarray, n_features_to_select: int
) -> RFE:
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    return rfe


def select_features_from_model(
    model: CancerModel, features: list[str], model_params: dict[str, Any]
) -> list[str]:
    feature_selector = SelectFromModel(model, **model_params)
    return feature_selector.get_feature_names_out(features).tolist()
