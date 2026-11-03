"""Model pipelines: preprocessing + estimator in a single sklearn Pipeline."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing.features import build_preprocessor


def build_model_pipeline(model_name="logistic", categorical_features=None, numeric_features=None, random_state=42):
    if model_name == "logistic":
        estimator = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(random_state=random_state)
    else:
        raise ValueError(f"unknown model: {model_name}")

    return Pipeline([
        ("preprocessor", build_preprocessor(categorical_features, numeric_features)),
        ("model", estimator),
    ])
