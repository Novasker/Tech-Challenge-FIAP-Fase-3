"""Preprocessing integrated into the model pipeline (no leakage: fit on train only)."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ["rede", "nome_uf", "amazonia_legal"]
NUMERIC_FEATURES = ["school_size", "municipality_size"]


def build_preprocessor(categorical_features=None, numeric_features=None):
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    numeric_features = numeric_features or NUMERIC_FEATURES

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
