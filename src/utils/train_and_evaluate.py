import pandas as pd

from .metrics import Metrics, evaluate_model
from .models import train_rf_classifier
from .preprocess import preprocess_data


def train_and_evaluate_rfc(X: pd.DataFrame, y: pd.Series) -> Metrics:
    processed_data = preprocess_data(X, y)

    rf_classifier = train_rf_classifier(
        processed_data.X_train_scaled, processed_data.y_train
    )
    return evaluate_model(
        rf_classifier, processed_data.X_test_scaled, processed_data.y_test
    )
